"""
Roboflow Universe 공개 화재 데이터셋 다운로드 & 병합 스크립트

사용법:
    python download_datasets.py --api-key YOUR_API_KEY
    python download_datasets.py --api-key YOUR_API_KEY --classes fire smoke
    python download_datasets.py --api-key YOUR_API_KEY --list
    python download_datasets.py --skip-download --classes fire   # 병합만 재실행

데이터셋 추가 방법:
    DATASETS 목록에 Roboflow Universe URL + class_map 항목을 추가하면 됩니다.
    URL 형식: https://universe.roboflow.com/{workspace}/{project}/dataset/{version}
"""

import argparse
import os
import shutil
import yaml
from pathlib import Path

# ── 다운로드할 Roboflow Universe 공개 화재 데이터셋 ───────────────────────────
# url    : universe.roboflow.com 의 dataset/버전 URL (버전 번호 필수)
# desc   : 설명 (로그용)
# class_map : 원본 클래스명 → 통합 클래스명 (제외할 클래스는 아예 넣지 않음)
DATASETS = [
    {
        "url": "https://universe.roboflow.com/middle-east-tech-university/fire-and-smoke-detection-hiwia/dataset/2",
        "desc": "METU Fire & Smoke (~6400장)",
        "class_map": {"fire": "fire", "Fire": "fire", "smoke": "smoke", "Smoke": "smoke"},
    },
    {
        "url": "https://universe.roboflow.com/fire-dataset-tp9jt/fire-detection-sejra/dataset/1",
        "desc": "Fire Detection Sejra (~9000장)",
        "class_map": {"fire": "fire", "Fire": "fire", "flame": "fire", "smoke": "smoke", "Smoke": "smoke"},
    },
    {
        "url": "https://universe.roboflow.com/aj-garcia-736tc/fire-dataset-for-yolov8/dataset/1",
        "desc": "Fire Dataset YOLOv8 (~4000장)",
        "class_map": {"fire": "fire", "Fire": "fire", "smoke": "smoke"},
    },
    {
        "url": "https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11/dataset/1",
        "desc": "Fire Smoke YOLOv11 (~4400장)",
        "class_map": {"fire": "fire", "smoke": "smoke"},
    },
    {
        "url": "https://universe.roboflow.com/fire-and-smoke-dpzyd/fire-smoke-yolo/dataset/1",
        "desc": "Fire Smoke YOLO (~9900장)",
        "class_map": {"fire": "fire", "smoke": "smoke"},
    },
    {
        "url": "https://universe.roboflow.com/yolofire-wbkwv/fire-detection-yolo-1gn9t/dataset/1",
        "desc": "Fire Detection YOLO (~7900장)",
        "class_map": {"fire": "fire", "Fire": "fire", "smoke": "smoke"},
    },
]

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DOWNLOAD_DIR = BASE_DIR / "data" / "_roboflow_raw"
MERGED_DIR = BASE_DIR / "data" / "merged"

DOWNLOAD_FORMAT = "yolov8"  # YOLO11도 동일 포맷 사용


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def _url_to_folder_name(url: str) -> str:
    """URL → 다운로드 캐시 폴더명"""
    # https://universe.roboflow.com/ws/proj/dataset/2 → ws_proj_v2
    parts = url.rstrip("/").split("/")
    workspace, project, version = parts[-5], parts[-4], parts[-1]
    return f"{workspace}__{project}__v{version}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default="", help="Roboflow API 키")
    p.add_argument(
        "--classes", nargs="+", default=["fire", "smoke"],
        help="포함할 통합 클래스 (기본: fire smoke). 예: --classes fire",
    )
    p.add_argument("--list", action="store_true", help="다운로드 대상 목록만 출력")
    p.add_argument("--skip-download", action="store_true", help="다운로드 건너뛰고 병합만 실행")
    p.add_argument("--val-split", type=float, default=0.15, help="검증셋 비율 (기본: 0.15)")
    return p.parse_args()


def print_dataset_list():
    print("\n[다운로드 대상 데이터셋]")
    total = 0
    for i, ds in enumerate(DATASETS, 1):
        print(f"  {i}. {ds['desc']}")
        print(f"     {ds['url']}")
        print(f"     클래스 매핑: {ds['class_map']}")
    print(f"\n  포맷: {DOWNLOAD_FORMAT} (YOLO Object Detection)\n")


# ── 다운로드 ──────────────────────────────────────────────────────────────────
def download_all(api_key: str):
    import roboflow
    roboflow.login(api_key=api_key)          # API 키 전역 설정
    from roboflow import download_dataset

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    success, failed = [], []

    for ds in DATASETS:
        folder = DOWNLOAD_DIR / _url_to_folder_name(ds["url"])
        if folder.exists():
            print(f"[skip] 이미 다운로드됨: {folder.name}")
            success.append((ds, folder))
            continue

        print(f"[download] {ds['desc']} ...")
        try:
            dataset = download_dataset(ds["url"], DOWNLOAD_FORMAT, location=str(folder))
            print(f"  -> 저장: {folder}")
            success.append((ds, folder))
        except Exception as e:
            print(f"  [오류] {ds['url']}\n         {e}")
            failed.append(ds)

    if failed:
        print(f"\n[경고] {len(failed)}개 실패 — 나머지 성공분만 병합합니다.")
    return success


def collect_downloaded() -> list[tuple]:
    result = []
    for ds in DATASETS:
        folder = DOWNLOAD_DIR / _url_to_folder_name(ds["url"])
        if folder.exists():
            result.append((ds, folder))
    return result


# ── 병합 ──────────────────────────────────────────────────────────────────────
def read_yaml(ds_path: Path) -> dict:
    for name in ["data.yaml", "dataset.yaml"]:
        p = ds_path / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def remap_label(
    src: Path, dst: Path,
    src_classes: list[str],
    class_map: dict[str, str],
    global_index: dict[str, int],
) -> bool:
    lines_out = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            orig_id = int(parts[0])
            if orig_id >= len(src_classes):
                continue
            unified = class_map.get(src_classes[orig_id])
            if unified is None or unified not in global_index:
                continue
            lines_out.append(f"{global_index[unified]} " + " ".join(parts[1:]))

    if not lines_out:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return True


def merge_datasets(downloaded: list[tuple], target_classes: list[str], val_split: float):
    import random

    global_index = {cls: i for i, cls in enumerate(target_classes)}

    # 출력 폴더 초기화
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            (MERGED_DIR / split / sub).mkdir(parents=True, exist_ok=True)

    ds_stats = []
    total = 0
    skipped = 0

    for ds, ds_path in downloaded:
        info = read_yaml(ds_path)
        src_classes = info.get("names", [])
        if isinstance(src_classes, dict):
            src_classes = [src_classes[k] for k in sorted(src_classes)]

        print(f"\n[merge] {ds['desc']}")
        print(f"  원본 클래스: {src_classes}")

        count = 0
        prefix = _url_to_folder_name(ds["url"]) + "__"

        for split_name in ["train", "valid", "test"]:
            img_dir = ds_path / split_name / "images"
            lbl_dir = ds_path / split_name / "labels"
            if not img_dir.exists():
                continue
            for img in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
                lbl = lbl_dir / (img.stem + ".txt")
                if not lbl.exists():
                    skipped += 1
                    continue
                dst_img = MERGED_DIR / "train" / "images" / (prefix + img.name)
                dst_lbl = MERGED_DIR / "train" / "labels" / (prefix + img.stem + ".txt")
                shutil.copy2(img, dst_img)
                if remap_label(lbl, dst_lbl, src_classes, ds["class_map"], global_index):
                    count += 1
                    total += 1
                else:
                    dst_img.unlink(missing_ok=True)
                    skipped += 1

        ds_stats.append((ds["desc"], count))
        print(f"  병합: {count}장")

    # train / val 분리
    all_imgs = list((MERGED_DIR / "train" / "images").glob("*"))
    random.seed(42)
    random.shuffle(all_imgs)
    val_n = max(1, int(len(all_imgs) * val_split))

    for img in all_imgs[:val_n]:
        lbl = MERGED_DIR / "train" / "labels" / (img.stem + ".txt")
        shutil.move(str(img), MERGED_DIR / "val" / "images" / img.name)
        if lbl.exists():
            shutil.move(str(lbl), MERGED_DIR / "val" / "labels" / lbl.name)

    train_n = total - val_n

    # data.yaml 저장
    yaml_path = MERGED_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "train": str(MERGED_DIR / "train" / "images"),
                "val": str(MERGED_DIR / "val" / "images"),
                "nc": len(target_classes),
                "names": target_classes,
            },
            f, allow_unicode=True, default_flow_style=False,
        )

    print("\n" + "=" * 55)
    print("[병합 완료]")
    for desc, cnt in ds_stats:
        print(f"  {desc}: {cnt}장")
    print(f"  라벨 없음/제외: {skipped}장")
    print(f"  train {train_n}장 / val {val_n}장")
    print(f"  클래스 {len(target_classes)}개: {target_classes}")
    print(f"  저장: {yaml_path}")
    print("=" * 55)
    return yaml_path


# ── config.yaml 업데이트 ───────────────────────────────────────────────────────
def update_config(merged_yaml: Path):
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def rel(p): return str(p.relative_to(BASE_DIR)).replace("\\", "/")

    cfg["data"]["yaml"] = rel(merged_yaml)
    cfg["data"]["train_images"] = rel(MERGED_DIR / "train" / "images")
    cfg["data"]["train_labels"] = rel(MERGED_DIR / "train" / "labels")
    cfg["data"]["val_images"] = rel(MERGED_DIR / "val" / "images")
    cfg["data"]["val_labels"] = rel(MERGED_DIR / "val" / "labels")

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    print(f"[config] config.yaml 업데이트 완료")


# ── 진입점 ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.list:
        print_dataset_list()
        return

    if not args.skip_download:
        if not args.api_key:
            print("[오류] --api-key 가 필요합니다.")
            print("  https://app.roboflow.com → Settings → Roboflow API → Private API Key")
            return
        download_all(args.api_key)

    downloaded = collect_downloaded()
    if not downloaded:
        print("[오류] 다운로드된 데이터셋이 없습니다.")
        return

    print(f"\n[merge] 통합 클래스: {args.classes}")
    merged_yaml = merge_datasets(downloaded, args.classes, args.val_split)
    update_config(merged_yaml)

    print("\n[완료] 학습 시작:")
    print("  python train.py")


if __name__ == "__main__":
    main()
