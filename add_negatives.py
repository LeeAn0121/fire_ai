"""
오탐 프레임 → negative 샘플 변환 스크립트

사용법:
    python add_negatives.py                  # output/frames/ 전체
    python add_negatives.py output/frames/영상명  # 특정 영상 폴더만
"""

import sys
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
TRAIN_IMG_DIR = BASE_DIR / "data/merged/train/images"
TRAIN_LBL_DIR = BASE_DIR / "data/merged/train/labels"


def add_negatives(source_dir: Path):
    jpgs = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.png"))
    if not jpgs:
        print(f"[negatives] 이미지 없음: {source_dir}")
        return

    added = 0
    skipped = 0
    for img_path in jpgs:
        dest_img = TRAIN_IMG_DIR / img_path.name
        dest_lbl = TRAIN_LBL_DIR / (img_path.stem + ".txt")

        if dest_img.exists():
            skipped += 1
            continue

        shutil.copy2(str(img_path), dest_img)
        dest_lbl.touch()   # 빈 라벨 파일 = "화재 없음"
        added += 1

    print(f"[negatives] 추가 {added}장, 중복 스킵 {skipped}장 → {TRAIN_IMG_DIR}")
    if added > 0:
        print(f"[negatives] 재학습 후 오탐이 줄어듭니다. python train.py 를 실행하세요.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
    else:
        source = BASE_DIR / "output/frames"

    if not source.exists():
        print(f"[negatives] 폴더 없음: {source}")
        sys.exit(1)

    add_negatives(source)
