"""
TFLite 변환 스크립트
- model/best.pt → model/best.tflite
- int8 양자화 (크기↓ 속도↑) 또는 float16 선택 가능

사용법:
  python export_tflite.py              # 기본 (int8 양자화)
  python export_tflite.py --float16    # float16 변환
  python export_tflite.py --float32    # 양자화 없이 변환 (크지만 정확)
  python export_tflite.py --imgsz 320  # 입력 해상도 지정 (기본 320)
"""

import argparse
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).parent


def load_config():
    with open(BASE_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def export(imgsz=320, mode="int8"):
    cfg = load_config()
    trained_path = BASE_DIR / cfg["model"]["trained"]

    if not trained_path.exists():
        print(f"[export] 학습된 모델 없음: {trained_path}")
        print("[export] 먼저 python train.py 를 실행하세요.")
        return None

    print(f"[export] 모델 로드: {trained_path}")
    print(f"[export] 변환 설정: imgsz={imgsz}, mode={mode}")

    model = YOLO(str(trained_path))

    export_kwargs = dict(
        format="tflite",
        imgsz=imgsz,
    )

    if mode == "int8":
        export_kwargs["int8"] = True
        print("[export] int8 양자화 적용 — 캘리브레이션 데이터: data/train/images")
        export_kwargs["data"] = str(BASE_DIR / cfg["data"]["yaml"])
    elif mode == "float16":
        export_kwargs["half"] = True
    # float32: 추가 옵션 없음

    result_path = model.export(**export_kwargs)
    print(f"[export] 변환 완료: {result_path}")

    # SavedModel 폴더 안의 .tflite 파일 탐색
    src = Path(result_path)
    if src.is_dir():
        tflite_files = list(src.glob("*.tflite"))
        if not tflite_files:
            print(f"[export] .tflite 파일을 찾을 수 없음: {src}")
            return None
        src = tflite_files[0]

    dest = BASE_DIR / "model" / "best.tflite"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dest))
    print(f"[export] 저장: {dest}")
    print(f"[export] 파일 크기: {dest.stat().st_size / 1024 / 1024:.2f} MB")

    return dest


def main():
    parser = argparse.ArgumentParser(description="TFLite 변환")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--int8", action="store_true", default=True,
                       help="int8 양자화 (기본값, 가장 작고 빠름)")
    group.add_argument("--float16", action="store_true",
                       help="float16 변환 (int8보다 정확, 약간 느림)")
    group.add_argument("--float32", action="store_true",
                       help="float32 변환 (가장 정확, 가장 느림)")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="입력 이미지 크기 (기본값: 320, RK3288은 256 권장)")
    args = parser.parse_args()

    if args.float16:
        mode = "float16"
    elif args.float32:
        mode = "float32"
    else:
        mode = "int8"

    export(imgsz=args.imgsz, mode=mode)


if __name__ == "__main__":
    main()
