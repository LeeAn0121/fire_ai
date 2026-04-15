# Fire AI — 실시간 화재 탐지 시스템

YOLO 기반 화재/연기 탐지 모델 학습 및 실시간 추론 파이프라인.  
카메라별 슬라이딩 윈도우 알람 상태 추적, 사람 오탐 억제, 자동 재학습을 지원합니다.

---

## 요구 사항

| 항목 | 버전 |
|------|------|
| Python | 3.11 이상 |
| CUDA | 12.x 이상 |
| GPU | NVIDIA (권장: RTX 4060 Ti / 5060 Ti 이상) |
| RAM | 16GB 이상 권장 |

---

## 초기 설치

### 1. 저장소 클론

```bash
git clone https://github.com/<your-username>/fire-ai.git
cd fire-ai
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python pyyaml
```

> CUDA 버전에 따라 `cu124` 부분을 `cu121` 등으로 변경하세요.  
> CUDA 버전 확인: `nvidia-smi`

### 4. 베이스 모델 다운로드

학습 시작 전 베이스 모델 파일(`.pt`)을 프로젝트 루트에 배치합니다.

```
fire-ai/
├── yolo11m.pt      ← 베이스 모델
├── yolo11n.pt      ← 사람 필터용 모델 (person_filter 사용 시)
└── yolo26m.pt      ← 베이스 모델 (대형)
```

> Ultralytics 공식 모델: https://docs.ultralytics.com/models/

### 5. 데이터셋 준비

```
data/merged/
├── train/
│   ├── images/     ← 학습 이미지 (.jpg / .png)
│   └── labels/     ← YOLO 형식 라벨 (.txt)
└── data.yaml       ← 클래스 정의 파일
```

`data.yaml` 예시:

```yaml
nc: 2
names: ["fire", "smoke"]
```

> train/val 분리는 `train.py` 실행 시 자동으로 처리됩니다 (8:2 비율).

---

## 설정 파일 (`config.yaml`)

주요 설정 항목:

```yaml
train:
  epochs: 50
  batch: -1          # -1 = GPU 메모리에 맞게 자동 결정
  imgsz: 640
  device: 0          # GPU 번호 (CPU 사용 시 "cpu")
  cache: disk        # 데이터 로딩 캐시 (disk / ram / false)

model:
  base: yolo11m.pt   # 학습 베이스 모델
  output_dir: model/release

detect:
  conf: 0.45         # 탐지 신뢰도 임계값
  person_filter: true
  person_model: yolo11n.pt
```

---

## 사용법

### 모델 학습

```bash
# 단일 모델
python train.py --model yolo11m.pt

# 복수 모델 순차 학습
python train.py --models yolo11m.pt yolo26m.pt
```

학습 완료 후 `model/release/result_vN.pt`로 자동 저장됩니다.

### 실시간 탐지 (단독 실행)

```bash
python detect.py
```

`input/` 폴더를 감시하며 새 이미지를 자동 탐지합니다.  
카메라 서브폴더 구조를 지원합니다:

```
input/
└── K2RIO23Z005/      ← camera_id (폴더명)
    └── images/
        └── 20260415095225732.jpg
```

### 자동 파이프라인 실행

```bash
# 감시 루프 시작 (권장)
python pipeline.py

# 현재 input/ 폴더 한 번만 처리
python pipeline.py --once

# 즉시 학습 실행
python pipeline.py --train
```

---

## 알람 로직

| 항목 | 설명 |
|------|------|
| 슬라이딩 윈도우 | 최근 10초 내 프레임 기준 |
| 알람 ON 조건 | 내부누적 ≥ 70% |
| 알람 OFF 조건 | 내부누적 < 40% |
| Soft Reset | 프레임 간격 10초 초과 시 히스토리/알람 초기화 |
| 사람억제 | 화재 박스와 사람 박스 IoU ≥ 0.7 시 화재 카운트 0으로 처리 |
| 미탐감쇠 | 알람 ON 상태에서 화재 미감지 시 누적값을 0.95 계수로 감쇠 |

---

## 출력 구조

```
output/
├── images/
│   ├── fire/           ← 화재 감지된 이미지
│   └── no_detection/   ← 감지 없는 이미지
└── videos/
    └── fire/
        └── {영상명}/   ← 화재 감지된 프레임
```

---

## 디렉터리 구조

```
fire-ai/
├── config.yaml         # 전체 설정
├── train.py            # 모델 학습
├── detect.py           # 추론 및 알람
├── pipeline.py         # 자동화 파이프라인
├── data/
│   └── merged/         # 데이터셋 (gitignore)
├── model/
│   └── release/        # 학습된 모델 (gitignore)
├── runs/               # 학습 로그 (gitignore)
└── input/              # 탐지 입력 폴더 (gitignore)
```
