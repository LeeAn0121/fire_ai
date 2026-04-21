"""
화재 탐지 YOLO 모델 학습 스크립트
- train/val 자동 분리 (8:2)
- config.yaml 기반 설정
- 학습 완료 후 model/release/result_vN.pt 자동 버전 관리
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
# expandable_segments: RTX 5060 Ti(Blackwell)에서 미지원 → 비활성화
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import shutil, random, yaml
from pathlib import Path
import torch
from ultralytics import YOLO

# RTX Ampere+ (30xx/40xx/50xx): TF32 활성화로 matmul 속도 향상
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
# benchmark=True 는 AutoBatch와 충돌하므로 비활성화

BASE_DIR = Path(__file__).parent

def load_config():
    with open(BASE_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_next_version(output_dir: Path) -> Path:
    """새로운 버전 번호를 부여한 저장 경로 반환"""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("result_v*.pt"))
    if not existing:
        return output_dir / "result_v1.pt"
    last_ver = int(existing[-1].stem.replace("result_v", ""))
    return output_dir / f"result_v{last_ver + 1}.pt"

def split_data(cfg):
    """학습 데이터를 학습용(train)과 검증용(val)으로 자동 분리"""
    train_img_dir = BASE_DIR / cfg["data"]["train_images"]
    val_img_dir   = BASE_DIR / cfg["data"]["val_images"]
    train_lbl_dir = BASE_DIR / cfg["data"]["train_labels"]
    val_lbl_dir   = BASE_DIR / cfg["data"]["val_labels"]

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 이미 분리되어 있는지 확인
    existing_val = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
    if existing_val:
        print(f"[데이터] 검증 폴더에 {len(existing_val)}장의 이미지가 이미 존재함 → 분리 생략")
        return

    images = sorted(train_img_dir.glob("*.jpg")) + sorted(train_img_dir.glob("*.png"))
    if not images:
        print("[데이터] 학습 폴더에 이미지가 없습니다.")
        return

    random.seed(42)
    random.shuffle(images)
    split_size = max(1, int(len(images) * cfg["data"]["val_split"]))
    val_images = images[:split_size]

    for img_path in val_images:
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        shutil.move(str(img_path), val_img_dir / img_path.name)
        if lbl_path.exists():
            shutil.move(str(lbl_path), val_lbl_dir / lbl_path.name)

    print(f"[데이터] 분리 완료: 학습용 {len(images) - len(val_images)}장 / 검증용 {len(val_images)}장")

def update_yaml_paths(cfg):
    """data.yaml 파일의 경로를 절대 경로로 갱신"""
    yaml_path = BASE_DIR / cfg["data"]["yaml"]
    with open(yaml_path, encoding="utf-8") as f:
        content = yaml.safe_load(f)

    content["train"] = str(BASE_DIR / cfg["data"]["train_images"])
    content["val"]   = str(BASE_DIR / cfg["data"]["val_images"])

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True, default_flow_style=False)
    print(f"[데이터] {yaml_path.name} 경로 업데이트 완료")

def train(cfg=None, base_model_override=None):
    if cfg is None:
        cfg = load_config()

    tcfg = cfg["train"]
    mcfg = cfg["model"]

    base_name = base_model_override or mcfg["base"]

    # 0. GPU 환경 점검
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"[학습] GPU 사용 중: {gpu_name} ({gpu_mem}GB)")
    else:
        print("[학습] 경고: GPU를 사용할 수 없어 CPU로 학습합니다.")

    # 1. 데이터 준비
    split_data(cfg)
    update_yaml_paths(cfg)

    # 2. 모델 로드
    base_model = YOLO(str(BASE_DIR / base_name))
    print(f"[학습] 베이스 모델 로드: {base_name}")

    # 3. 학습 실행 (최적화 설정 적용)
    print(f"[학습] 학습 시작 (에폭: {tcfg['epochs']}, 배치: {tcfg['batch']})")

    train_kwargs = dict(
        data=str(BASE_DIR / cfg["data"]["yaml"]),
        epochs=tcfg["epochs"],
        batch=tcfg["batch"],
        imgsz=tcfg["imgsz"],
        device=tcfg["device"],
        patience=tcfg["patience"],
        workers=tcfg["workers"],
        cache=tcfg.get("cache", False),
        project=str(BASE_DIR / tcfg["project"]),
        name=tcfg["name"],
        exist_ok=tcfg["exist_ok"],
        amp=tcfg["amp"],
        rect=tcfg.get("rect", False),
        close_mosaic=tcfg.get("close_mosaic", 10),
        multi_scale=tcfg.get("multi_scale", False),
        mosaic=tcfg.get("mosaic", 1.0),
        scale=tcfg.get("scale", 0.5),
        cos_lr=tcfg.get("cos_lr", True),
        val=tcfg.get("val", True),
        max_det=tcfg.get("max_det", 300),
        plots=tcfg.get("plots", False),
    )

    while True:
        try:
            results = base_model.train(**train_kwargs)
            break
        except RuntimeError as e:
            error_text = str(e)
            if "DataLoader worker" in error_text and train_kwargs["workers"] > 0:
                print(
                    f"[학습] DataLoader worker 비정상 종료 감지 -> "
                    f"workers={train_kwargs['workers']}에서 workers=0으로 재시도합니다."
                )
                train_kwargs["workers"] = 0
                continue

            oom_signals = ("out of memory", "cuda error: out of memory", "cuda out of memory")
            if any(signal in error_text.lower() for signal in oom_signals) and train_kwargs["batch"] > 1:
                next_batch = max(1, train_kwargs["batch"] // 2)
                if next_batch == train_kwargs["batch"]:
                    raise
                print(
                    f"[학습] 메모리 부족 감지 -> "
                    f"batch={train_kwargs['batch']}에서 batch={next_batch}로 재시도합니다."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                train_kwargs["batch"] = next_batch
                continue

            raise

    # 4. 결과 저장 및 버전 관리
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        output_dir = BASE_DIR / mcfg["output_dir"]
        target_path = get_next_version(output_dir)
        shutil.copy2(str(best_weights), str(target_path))
        print(f"\n[학습 완료] 새 모델 저장됨: {target_path}")
        
        # config.yaml의 model.path 업데이트
        _update_config_model_path(target_path)
    else:
        print("[학습] 결과물(best.pt)을 찾을 수 없습니다.")

def _update_config_model_path(new_path: Path):
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        content = f.read()
    
    rel_path = new_path.relative_to(BASE_DIR).as_posix()
    import re
    # model: 섹션 하위의 path: 값을 업데이트 (주석 유무 무관)
    new_content = re.sub(r"(?m)^(\s{2}path:\s*)[\w/.\-]+", rf"\g<1>{rel_path}", content)
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[학습] config.yaml의 모델 경로를 {rel_path}로 업데이트했습니다.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="베이스 모델 파일명 (예: yolo11m.pt)")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="순차 학습할 모델 목록 (예: yolo11m.pt yolo26m.pt)")
    args = parser.parse_args()

    models_to_train = args.models or ([args.model] if args.model else [None])
    cfg = load_config()

    for i, model_name in enumerate(models_to_train):
        if len(models_to_train) > 1:
            print(f"\n{'='*60}")
            print(f"[순차 학습] {i+1}/{len(models_to_train)}: {model_name}")
            print(f"{'='*60}")
        train(cfg=cfg, base_model_override=model_name)
