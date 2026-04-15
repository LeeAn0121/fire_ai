"""
자동 파이프라인
- input/ 폴더를 감시하여 새 파일 자동 탐지
- data/train/ 에 새 이미지 누적되면 자동 재학습 (옵션)

사용법:
  python pipeline.py               # 자동 감시 루프 시작
  python pipeline.py --once        # 현재 input/ 폴더 한 번만 처리
  python pipeline.py --train       # 즉시 학습 실행
  python pipeline.py --detect      # 즉시 탐지 실행
"""

import sys
import time
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def setup_logging():
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("pipeline")


def load_config():
    with open(BASE_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_all_media(directory: Path) -> set:
    """디렉터리 내 모든 이미지/비디오 파일 경로 수집"""
    files = set()
    if not directory.exists():
        return files
    for ext in IMAGE_EXTS | VIDEO_EXTS:
        files.update(directory.rglob(f"*{ext}"))
        files.update(directory.rglob(f"*{ext.upper()}"))
    return files


class Pipeline:
    def __init__(self):
        self.cfg = load_config()
        self.log = setup_logging()
        self.pcfg = self.cfg["pipeline"]

        self.input_dir = BASE_DIR / self.cfg["detect"]["source"]
        self.data_train_dir = BASE_DIR / self.cfg["data"]["train_images"]

        self.known_input_files: set = set()
        self.known_train_count: int = 0
        self.new_data_since_last_train: int = 0

    def _snapshot_input(self):
        """현재 input 폴더 파일 목록 저장"""
        self.known_input_files = get_all_media(self.input_dir)
        self.log.info(f"[init] input 폴더 파일 {len(self.known_input_files)}개 기준 설정")

    def _snapshot_train(self):
        train_imgs = list(self.data_train_dir.glob("*.jpg")) + list(self.data_train_dir.glob("*.png"))
        self.known_train_count = len(train_imgs)
        self.log.info(f"[init] train 이미지 {self.known_train_count}장 기준 설정")

    def detect_new_files(self, new_files: list):
        """새 파일에 대해 탐지 실행"""
        from detect import detect_single
        for f in new_files:
            self.log.info(f"[detect] 새 파일 탐지: {Path(f).name}")
            result = detect_single(f, self.cfg)
            if result:
                self.log.info(f"[detect] 결과 저장: {result}")

    def check_and_retrain(self):
        """새 학습 데이터 누적량 확인 후 재학습 트리거"""
        if not self.pcfg.get("retrain_on_new_data", False):
            return

        train_imgs = list(self.data_train_dir.glob("*.jpg")) + list(self.data_train_dir.glob("*.png"))
        current_count = len(train_imgs)
        added = current_count - self.known_train_count

        if added > 0:
            self.new_data_since_last_train += added
            self.known_train_count = current_count
            self.log.info(f"[pipeline] 새 학습 이미지 누적: {self.new_data_since_last_train}장")

        threshold = self.pcfg.get("min_new_images", 20)
        if self.new_data_since_last_train >= threshold:
            self.log.info(f"[pipeline] 재학습 트리거 ({self.new_data_since_last_train}장 누적)")
            self.run_train()
            self.new_data_since_last_train = 0

    def run_train(self):
        self.log.info("[train] 학습 시작")
        from train import train
        try:
            train(self.cfg)
            self.log.info("[train] 학습 완료")
        except Exception as e:
            self.log.error(f"[train] 학습 실패: {e}")

    def run_detect_all(self):
        self.log.info("[detect] 전체 탐지 시작")
        from detect import run
        try:
            run(cfg=self.cfg)
            self.log.info("[detect] 탐지 완료")
        except Exception as e:
            self.log.error(f"[detect] 탐지 실패: {e}")

    def watch_loop(self):
        """input 폴더를 주기적으로 감시하며 새 파일 자동 처리"""
        self._snapshot_input()
        self._snapshot_train()

        interval = self.pcfg.get("poll_interval", 5)
        self.log.info(f"[pipeline] 감시 루프 시작 (주기: {interval}초) — Ctrl+C로 종료")
        self.log.info(f"[pipeline] 감시 폴더: {self.input_dir}")

        try:
            while True:
                current_files = get_all_media(self.input_dir)
                new_files = current_files - self.known_input_files

                if new_files and self.pcfg.get("auto_detect", True):
                    self.log.info(f"[pipeline] 새 파일 {len(new_files)}개 감지")
                    self.detect_new_files([str(f) for f in new_files])
                    self.known_input_files = current_files

                self.check_and_retrain()
                time.sleep(interval)

        except KeyboardInterrupt:
            self.log.info("[pipeline] 감시 루프 종료")


def main():
    parser = argparse.ArgumentParser(description="Fire AI 자동 파이프라인")
    parser.add_argument("--once", action="store_true", help="현재 input 폴더 한 번만 처리")
    parser.add_argument("--train", action="store_true", help="즉시 학습 실행")
    parser.add_argument("--detect", action="store_true", help="즉시 탐지 실행")
    args = parser.parse_args()

    p = Pipeline()

    if args.train:
        p.run_train()
    elif args.detect:
        p.run_detect_all()
    elif args.once:
        p.log.info("[once] 현재 input 폴더 탐지 실행")
        p.run_detect_all()
    else:
        p.watch_loop()


if __name__ == "__main__":
    main()
