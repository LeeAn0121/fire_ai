"""
화재 탐지 추론 스크립트 (핫 리로드 + 카메라별 알람 상태 추적)

알람 로직:
  - 10초 슬라이딩 윈도우 내 화재 감지 비율(내부누적)이 임계값 초과 시 알람 ON
  - 10초 이상 프레임 공백 발생 시 Soft Reset (알람/히스토리 초기화)
  - 오탐 페널티: 사람억제(person-fire 박스 중첩), 미탐감쇠(알람 ON 중 미감지)
"""

import time
import yaml
import cv2
import re
import sys
import logging
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from ultralytics import YOLO

BASE_DIR   = Path(__file__).parent

# ── 로그 설정 ────────────────────────────────────────────────────────
def setup_logging():
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = log_dir / f"{timestamp}.log"
    
    # 터미널과 파일에 동시 출력
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# ── 알람 파라미터 ─────────────────────────────────────────────────────
TIME_WINDOW        = 10.0   # 슬라이딩 윈도우 (초)
ALARM_ON_THRESH    = 0.30   # 0.70 -> 0.30 (감지 기준 완화로 미탐 방지)
ALARM_OFF_THRESH   = 0.15   # 0.40 -> 0.15
DECAY_FACTOR       = 0.98   # 0.95 -> 0.98 (미탐 시 알람 유지력 강화)
DISPLAY_EMA_ALPHA  = 0.50   
SOFT_RESET_THRESH  = 30.0   # 10s -> 30s (CPU 지연 대응)

_TS_RE = re.compile(r"(\d{17})")  # YYYYMMDDHHMMSSMMM


def load_config():
    with open(BASE_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_ts(stem: str) -> float | None:
    """파일명 stem에서 타임스탬프(unix float) 추출. 실패 시 None."""
    m = _TS_RE.search(stem)
    if not m:
        return None
    s = m.group(1)
    try:
        dt = datetime(
            int(s[0:4]), int(s[4:6]),  int(s[6:8]),
            int(s[8:10]), int(s[10:12]), int(s[12:14]),
            int(s[14:17]) * 1000,
        )
        return dt.timestamp()
    except ValueError:
        return None


def _iou(box_a, box_b) -> float:
    """두 박스(xyxy) IoU 계산"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── 알람 상태 자료구조 ─────────────────────────────────────────────────

@dataclass
class _FrameRec:
    ts:         float
    fire_count: int
    max_conf:   float   # 화재 박스 최대 신뢰도 (0~1), 감지 없으면 0.0


@dataclass
class CameraAlarmState:
    camera_id:      str
    history:        deque      = field(default_factory=deque)
    alarm_on:       bool       = False
    last_ts:        float|None = None
    display_ratio:  float      = 0.0   # 표시비율  (EMA 기반, %)
    internal_accum: float      = 0.0   # 내부누적  (윈도우 fire-frame 비율, %)


class AlarmManager:
    """카메라별 알람 상태 관리자 (모듈 전역 싱글턴)"""

    def __init__(self):
        self._states: dict[str, CameraAlarmState] = {}

    def get(self, camera_id: str) -> CameraAlarmState:
        if camera_id not in self._states:
            self._states[camera_id] = CameraAlarmState(camera_id=camera_id)
        return self._states[camera_id]

    def process(
        self,
        camera_id: str,
        ts: float,
        fire_count: int,
        max_conf: float,
        person_suppressed: bool,
    ) -> tuple[bool, str, float, float]:
        """
        알람 상태 갱신.
        Returns: (alarm_on, penalty_label, display_ratio%, internal_accum%)
        """
        st = self.get(camera_id)
        penalty = "없음"

        # ── 1. Soft Reset ────────────────────────────────────────────
        if st.last_ts is not None and (ts - st.last_ts) > SOFT_RESET_THRESH:
            print(
                f"[DEBUG] 알람리셋(Soft) | camera_id={camera_id}"
                f" | thresh({int(SOFT_RESET_THRESH)}s) 이탈로 인한 알람/히스토리 연속성 단절"
            )
            st.history.clear()
            st.alarm_on       = False
            st.display_ratio  = 0.0
            st.internal_accum = 0.0

        st.last_ts = ts

        # ── 2. 오탐 페널티 결정 ──────────────────────────────────────
        if person_suppressed:
            penalty    = "사람억제"
            fire_count = 0
            max_conf   = 0.0
        elif st.alarm_on and fire_count == 0:
            penalty = "미탐감쇠"

        # ── 3. 히스토리 갱신 ─────────────────────────────────────────
        st.history.append(_FrameRec(ts=ts, fire_count=fire_count, max_conf=max_conf))
        cutoff = ts - TIME_WINDOW
        while st.history and st.history[0].ts < cutoff:
            st.history.popleft()

        # ── 4. 메트릭 계산 ───────────────────────────────────────────
        if penalty == "미탐감쇠":
            st.internal_accum *= DECAY_FACTOR
            st.display_ratio  *= DECAY_FACTOR
        else:
            n           = len(st.history)
            fire_frames = sum(1 for f in st.history if f.fire_count > 0)
            st.internal_accum = fire_frames / n * 100 if n else 0.0
            cur_signal        = max_conf * 100
            st.display_ratio  = (
                DISPLAY_EMA_ALPHA * cur_signal
                + (1 - DISPLAY_EMA_ALPHA) * st.display_ratio
            )

        # ── 5. 알람 토글 (히스테리시스) ──────────────────────────────
        if not st.alarm_on and st.internal_accum >= ALARM_ON_THRESH * 100:
            st.alarm_on = True
        elif st.alarm_on and st.internal_accum < ALARM_OFF_THRESH * 100:
            st.alarm_on = False

        return st.alarm_on, penalty, st.display_ratio, st.internal_accum


# 모듈 전역 AlarmManager (pipeline 등 외부 임포트 시 공유)
alarm_manager = AlarmManager()


# ── Detector ──────────────────────────────────────────────────────────

class Detector:
    def __init__(self, cfg):
        self.cfg         = cfg
        self.dcfg        = cfg["detect"]
        self.model_path  = BASE_DIR / cfg["model"]["path"]
        self.last_mtime  = self._get_mtime()

        print(f"[감지기] 모델 로드 중: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.person_model = self._load_person_model()

    # ── 모델 관리 ────────────────────────────────────────────────────

    def _get_mtime(self):
        return self.model_path.stat().st_mtime if self.model_path.exists() else 0

    def _load_person_model(self):
        if not self.dcfg.get("person_filter", False):
            return None
        p_path = BASE_DIR / self.dcfg["person_model"]
        if p_path.exists():
            print(f"[감지기] 사람 필터링 모델 로드: {p_path.name}")
            return YOLO(str(p_path))
        return None

    def check_model_update(self):
        """config 변경 또는 모델 파일 갱신 시 핫 리로드"""
        current_cfg  = load_config()
        new_path     = BASE_DIR / current_cfg["model"]["path"]

        if new_path != self.model_path:
            self.model_path = new_path
            print(f"[감지기] 모델 경로 변경: {new_path.name}")
            self.model      = YOLO(str(self.model_path))
            self.last_mtime = self._get_mtime()
            self.cfg        = current_cfg
            self.dcfg       = current_cfg["detect"]
            return

        cur_mtime = self._get_mtime()
        if cur_mtime > self.last_mtime:
            print("[감지기] 모델 파일 업데이트 감지 → 다시 로드")
            self.model      = YOLO(str(self.model_path))
            self.last_mtime = cur_mtime

    # ── 사람 억제 판단 ────────────────────────────────────────────────

    def _check_person_suppression(self, orig_img, fire_boxes_xyxy: list) -> bool:
        """사람 박스와 화재 박스가 충분히 겹치면 True 반환"""
        if not self.person_model or not fire_boxes_xyxy:
            return False
        threshold = self.dcfg.get("overlap_threshold", 0.7)
        p_results = self.person_model.predict(
            source=orig_img, conf=0.25, verbose=False
        )
        person_boxes = [
            b.xyxy[0].tolist()
            for b in p_results[0].boxes
            if p_results[0].names[int(b.cls)] == "person"
        ]
        for f_box in fire_boxes_xyxy:
            for p_box in person_boxes:
                if _iou(f_box, p_box) >= threshold:
                    return True
        return False

    # ── 이미지 처리 ──────────────────────────────────────────────────

    def process_image(
        self,
        img_path: Path,
        base_out_dir: Path,
        camera_id: str = "default",
    ) -> dict:
        """
        이미지 1장 추론 후 알람 상태 갱신.
        Returns: {'alarm_on': bool, 'fire_count': int, 'classes': set}
        """
        results = self.model.predict(
            source=str(img_path),
            conf=self.dcfg["conf"],
            iou=self.dcfg["iou"],
            imgsz=self.dcfg["imgsz"],
            device=self.dcfg["device"],
            verbose=False,
        )
        r = results[0]

        # 화재 박스 수집
        fire_boxes  = []
        max_conf    = 0.0
        for box in r.boxes:
            cls_name = r.names[int(box.cls)]
            conf     = float(box.conf)
            if cls_name != "person":   # 화재 클래스만 (person 클래스가 섞인 경우 제외)
                fire_boxes.append(box.xyxy[0].tolist())
                max_conf = max(max_conf, conf)

        fire_count = len(fire_boxes)

        # 사람 억제 판단
        person_suppressed = self._check_person_suppression(r.orig_img, fire_boxes)

        # 타임스탬프 파싱 (실패 시 현재 시각)
        ts = _parse_ts(img_path.stem) or time.time()

        # 알람 상태 갱신
        alarm_on, penalty, display_ratio, internal_accum = alarm_manager.process(
            camera_id, ts, fire_count, max_conf, person_suppressed
        )

        # 결과 저장
        plotted  = r.plot()
        out_dir  = base_out_dir / ("fire" if fire_count > 0 else "no_detection")
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / img_path.name), plotted)

        alarm_icon = "🔴 ON " if alarm_on else "🟢 OFF"
        print(
            f"[DEBUG] 알람: {alarm_icon} | 화재객체: {fire_count:2d}"
            f" | 오탐페널티: {penalty:6s}"
            f" | 표시비율: {display_ratio:5.1f}%"
            f" | 내부누적: {internal_accum:5.1f}%"
            f" | 결과저장: 저장"
            f" | 파일: {img_path.name}"
            f" | camera_id: {camera_id}"
        )

        detected_classes = (
            {r.names[int(b.cls)] for b in r.boxes} if r.boxes else {"no_detection"}
        )
        return {
            "alarm_on":      alarm_on,
            "fire_count":    fire_count,
            "max_conf":      max_conf,
            "penalty":       penalty,
            "display_ratio": display_ratio,
            "internal_accum": internal_accum,
            "classes":       detected_classes,
            "camera_id":     camera_id,
            "filename":      img_path.name,
        }

    # ── 영상 처리 ────────────────────────────────────────────────────

    def process_video(self, vid_path: Path, base_out_dir: Path, camera_id: str = "default") -> dict:
        """영상 1개 처리 - 감지된 프레임을 클래스별 폴더에 저장"""
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"[감지기] 영상 열기 실패: {vid_path.name}")
            return {}

        stride      = self.dcfg.get("vid_stride", 2)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx   = 0
        stats: dict[str, int] = {}

        print(f"[감지기] 영상 처리 중: {vid_path.name} ({total_frames}프레임, stride={stride})")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            # 비디오 프레임 기반 타임스탬프 계산 (CPU 속도 영향 무시)
            ts = frame_idx / fps

            results = self.model.predict(
                source=frame,
                conf=self.dcfg["conf"],
                iou=self.dcfg["iou"],
                imgsz=self.dcfg["imgsz"],
                device=self.dcfg["device"],
                verbose=False,
            )
            r           = results[0]
            frame_name  = f"{vid_path.stem}_f{frame_idx:06d}.jpg"
            plotted     = r.plot()

            fire_boxes = [b.xyxy[0].tolist() for b in r.boxes]
            fire_count = len(fire_boxes)
            max_conf   = max((float(b.conf) for b in r.boxes), default=0.0)
            person_sup = self._check_person_suppression(r.orig_img, fire_boxes)

            alarm_on, penalty, display_ratio, internal_accum = alarm_manager.process(
                camera_id, ts, fire_count, max_conf, person_sup
            )

            # 프레임별 실시간 출력 추가 (최대 확신도 포함)
            label = "fire" if fire_count > 0 else "no_detection"
            status_text = "FIRE DETECT" if fire_count > 0 else "no detection"
            logging.info(f"[감지기] {frame_name} / {status_text} / fire {fire_count}개 / max_conf:{max_conf:.2f}")

            out_dir = base_out_dir / label / vid_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / frame_name), plotted)
            stats[label] = stats.get(label, 0) + 1

            frame_idx += 1

        cap.release()
        return stats

    # ── 폴더 처리 ────────────────────────────────────────────────────

    def run_on_folder(self, source_dir=None):
        """
        폴더 내 이미지/영상 처리.
        구조: source_dir/{camera_id}/images/*.jpg  (camera_id 서브폴더 지원)
              source_dir/images/*.jpg               (단일 카메라)
        """
        if source_dir is None:
            source_dir = BASE_DIR / self.dcfg["source"]
        source_dir = Path(source_dir)
        has_work   = False

        # camera_id 서브폴더 탐색 (없으면 source_dir 자체를 단일 카메라로 처리)
        cam_dirs = [d for d in source_dir.iterdir() if d.is_dir()] if source_dir.exists() else []
        if not cam_dirs:
            cam_dirs = [source_dir]

        for cam_dir in cam_dirs:
            camera_id = cam_dir.name if cam_dir != source_dir else "default"

            img_dir = cam_dir / "images"
            if img_dir.exists():
                images = [f for f in img_dir.glob("*") if f.suffix.lower() in IMAGE_EXTS]
                if images:
                    has_work = True
                    print(f"[감지기] [{camera_id}] {len(images)}장 처리 시작")
                    base_out = BASE_DIR / self.dcfg["output"] / "images"
                    stats: dict[str, int] = {}
                    for img in sorted(images):
                        ret = self.process_image(img, base_out, camera_id)
                        for cls in ret["classes"]:
                            stats[cls] = stats.get(cls, 0) + 1
                    summary = ", ".join(f"{k}: {v}장" for k, v in sorted(stats.items()))
                    print(f"[감지기] [{camera_id}] 이미지 완료 → {base_out}  [{summary}]")

            vid_dir = cam_dir / "videos"
            if vid_dir.exists():
                videos = [f for f in vid_dir.glob("*") if f.suffix.lower() in VIDEO_EXTS]
                if videos:
                    has_work = True
                    print(f"[감지기] [{camera_id}] {len(videos)}개 영상 처리 시작")
                    base_out = BASE_DIR / self.dcfg["output"] / "videos"
                    total_stats: dict[str, int] = {}
                    for vid in videos:
                        vstats = self.process_video(vid, base_out, camera_id)
                        for cls, cnt in vstats.items():
                            total_stats[cls] = total_stats.get(cls, 0) + cnt
                    summary = ", ".join(f"{k}: {v}프레임" for k, v in sorted(total_stats.items()))
                    print(f"[감지기] [{camera_id}] 영상 완료 → {base_out}  [{summary}]")

        if not has_work:
            print("[감지기] 처리할 파일 없음 (대기 중...)")


# ── pipeline.py 에서 호출하는 단일 파일 처리 인터페이스 ──────────────

_detector: Detector | None = None


def _get_detector(cfg) -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector(cfg)
    return _detector


def detect_single(file_path: str, cfg) -> str | None:
    """
    단일 파일 추론 (pipeline.py 에서 호출).
    Returns: 저장된 출력 경로 문자열, 또는 None.
    """
    detector  = _get_detector(cfg)
    img_path  = Path(file_path)
    camera_id = img_path.parent.name  # 부모 디렉터리명을 camera_id로 사용

    if img_path.suffix.lower() not in IMAGE_EXTS:
        return None

    base_out = BASE_DIR / cfg["detect"]["output"] / "images"
    ret      = detector.process_image(img_path, base_out, camera_id)
    return str(base_out / ("fire" if ret["fire_count"] > 0 else "no_detection") / img_path.name)


def run(cfg=None):
    """전체 폴더 탐지 실행 (pipeline.py 에서 호출)"""
    if cfg is None:
        cfg = load_config()
    detector = _get_detector(cfg)
    detector.run_on_folder()


# ── 단독 실행 ─────────────────────────────────────────────────────────

def main():
    log_file = setup_logging()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="모델 경로 (예: model/release/result_v7.pt)")
    parser.add_argument("--source",  type=str, default=None, help="입력 폴더 경로")
    parser.add_argument("--conf",    type=float, default=None, help="신뢰도 임계값")
    parser.add_argument("--device",  type=str, default=None, help="장치 (0, 1, 'cpu')")
    parser.add_argument("--name",    type=str, default=None, help="결과 저장 이름")
    args = parser.parse_args()

    logging.info(f"로그 시작: {log_file}")
    cfg = load_config()

    # 명령줄 인자로 config 값 덮어쓰기
    if args.weights:
        cfg["model"]["path"] = args.weights
    if args.source:
        cfg["detect"]["source"] = args.source
    if args.conf is not None:
        cfg["detect"]["conf"] = args.conf
    if args.device is not None:
        # 숫자인 경우 int로 변환 (GPU 번호)
        if args.device.isdigit():
            cfg["detect"]["device"] = int(args.device)
        else:
            cfg["detect"]["device"] = args.device
    if args.name:
        cfg["detect"]["output_name"] = args.name  # Detector 내부에서 사용할 이름 저장

    detector = Detector(cfg)

    # 단발성 실행인지, 루프 모드인지 결정
    # source가 명시적으로 들어온 경우 한 번만 실행하고 종료
    if args.source:
        print(f"[감지기] 단일 경로 탐지 모드 시작: {args.source}")
        detector.run_on_folder(args.source)
        print("[감지기] 탐지 완료.")
    else:
        print("[감지기] 실시간 감시 모드 시작 (poll_interval)...")
        while True:
            detector.check_model_update()
            detector.run_on_folder()
            time.sleep(cfg["pipeline"]["poll_interval"])


if __name__ == "__main__":
    main()
