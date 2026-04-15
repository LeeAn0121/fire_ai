import os
import yaml
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def load_config():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def run_analysis():
    cfg = load_config()
    
    # 모델 경로 확인
    model_path = Path(cfg["model"]["path"])
    if not model_path.exists():
        print(f"[analysis] 오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    model = YOLO(str(model_path))
    val_img_dir = Path(cfg["data"]["val_images"])
    val_lbl_dir = Path(cfg["data"]["val_labels"])
    
    output_dir = Path("analysis/false_negatives")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[analysis] 검증 시작 (모델: {model_path.name})")
    print(f"[analysis] 대상: {val_img_dir}")
    
    images = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
    if not images:
        print("[analysis] 오류: 검증용 이미지가 없습니다.")
        return

    fn_count = 0
    total = len(images)
    
    for i, img_path in enumerate(images):
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists(): continue
        
        # 1. 정답(GT) 로드
        gt_boxes = []
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5: continue
                cls, x, y, nw, nh = map(float, parts[:5])
                x1 = (x - nw/2) * w
                y1 = (y - nh/2) * h
                x2 = (x + nw/2) * w
                y2 = (y + nh/2) * h
                gt_boxes.append([x1, y1, x2, y2])
        
        if not gt_boxes: continue

        # 2. 예측(Pred) 실행
        results = model.predict(
            source=str(img_path), 
            conf=cfg["detect"]["conf"], 
            imgsz=cfg["detect"]["imgsz"], 
            device=cfg["detect"]["device"],
            verbose=False
        )[0]
        pred_boxes = results.boxes.xyxy.cpu().numpy()
        
        # 3. 미탐(False Negative) 체크
        missed_gt = []
        for gt in gt_boxes:
            matched = False
            for pred in pred_boxes:
                if get_iou(gt, pred) > 0.3:
                    matched = True
                    break
            if not matched:
                missed_gt.append(gt)
        
        # 4. 미탐된 이미지만 저장 (정답 녹색, 예측 빨간색)
        if missed_gt:
            fn_count += 1
            # 시각화용 복사본
            vis_img = img.copy()
            # 모든 정답 표시 (미탐은 두껍게)
            for gt in gt_boxes:
                is_missed = any(np.array_equal(gt, m) for m in missed_gt)
                color = (0, 255, 0) # Green
                thickness = 3 if is_missed else 1
                cv2.rectangle(vis_img, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), color, thickness)
            
            # 모든 예측 표시 (빨간색)
            for pred in pred_boxes:
                cv2.rectangle(vis_img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255), 1)
            
            # 텍스트 추가
            cv2.putText(vis_img, f"Missed: {len(missed_gt)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imwrite(str(output_dir / img_path.name), vis_img)
            
        if (i+1) % 100 == 0:
            print(f"  진행 중... ({i+1}/{total})")
            
    print(f"\n{'='*50}")
    print(f"[analysis] 분석 완료")
    print(f"  - 전체 검증 이미지: {total}장")
    print(f"  - 미탐 발생 이미지: {fn_count}장")
    print(f"  - 결과 저장 경로: {output_dir.absolute()}")
    print(f"  * 녹색 상자: 정답 (두꺼운 상자가 놓친 것)")
    print(f"  * 빨간 상자: 모델의 예측")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_analysis()
