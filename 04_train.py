"""실습 04. 무작위 초기화 YOLO26의 1 epoch 학습 연결 점검. 성능 실험이 아닙니다."""
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
BASE = Path(__file__).resolve().parent
# Windows에서도 읽기 worker가 스크립트를 재실행하지 않도록 진입점을 구분합니다.
if __name__ == "__main__":
    torch.set_num_threads(1)
    config = {"train": str(BASE/"data/images/train"), "val": str(BASE/"data/images/val"),
              "names": ["square", "circle", "triangle"]}
    config_path = BASE / "p04_dataset.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    model = YOLO("yolo26n.yaml")  # 구조만 생성합니다. COCO 사전학습 .pt와 다릅니다.
    # >>> B05
    # 요구: config_path로 1 epoch, imgsz=160, batch=4, CPU 학습을 호출하세요.
    # 나머지 옵션은 제공한 그대로 유지하세요. test 경로는 전달하지 않습니다.
    model.train(
        data=config_path,
        epochs=1,
        imgsz=160,
        batch=4,
        device="cpu",
        workers=0,
        project=str(BASE / "p04_runs"),
        name="smoke",
    )
    # <<< B05
    print("task:", model.task, "names:", model.names, "device:", model.device)
    print("학습 결과:", BASE/"p04_runs/smoke")

""" 출력 결과 값
New https://pypi.org/project/ultralytics/8.4.203 available  Update with 'pip install -U ultralytics'
Ultralytics 8.4.142  Python-3.13.16 torch-2.14.0+cu130 CPU (Intel Core Ultra 9 275HX)
engine\trainer: agnostic_nms=False, amp=False, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=C:\WorkSpace\20261106\06_Fri_TestFile_correct\p04_dataset.yaml, degrees=0.0, deterministic=True, device=cpu, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=1, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=160, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo26n.yaml, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=smoke, nbs=64, nms=None, opset=None, optimize=False, optimizer=SGD, overlap_mask=True, patience=100, perspective=0.0, plots=False, pose=12.0, pretrained=False, profile=False, project=C:\WorkSpace\20261106\06_Fri_TestFile_correct\p04_runs, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=C:\WorkSpace\20261106\06_Fri_TestFile_correct\p04_runs\smoke, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=42, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=0, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5, 3, True]        
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    119808  ultralytics.nn.modules.block.C3k2            [384, 128, 1, True]           
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     34304  ultralytics.nn.modules.block.C3k2            [256, 64, 1, True]            
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     95232  ultralytics.nn.modules.block.C3k2            [192, 128, 1, True]           
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    463104  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True, 0.5, True]
 23        [16, 19, 22]  1    242346  ultralytics.nn.modules.head.Detect           [3, 1, True, [64, 128, 256]]  
YOLO26n summary: 260 layers, 2,504,970 parameters, 2,504,970 gradients, 5.9 GFLOPs

train: Fast image access  (ping: 0.00.0 ms, read: 108.640.4 MB/s, size: 40.4 KB)
train: Scanning C:\WorkSpace\20261106\06_Fri_TestFile_correct\data\labels\train... 12 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 12/12 2.1Kit/s 0.0s
train: New cache created: C:\WorkSpace\20261106\06_Fri_TestFile_correct\data\labels\train.cache
val: Fast image access  (ping: 0.00.0 ms, read: 676.4139.3 MB/s, size: 41.2 KB)
val: Scanning C:\WorkSpace\20261106\06_Fri_TestFile_correct\data\labels\val... 6 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 6/6 2.4Kit/s 0.0s
val: New cache created: C:\WorkSpace\20261106\06_Fri_TestFile_correct\data\labels\val.cache
optimizer: SGD(lr=0.01, momentum=0.937) with parameter groups 114 weight(decay=0.0), 126 weight(decay=0.0005), 126 bias(decay=0.0)
Using 12 train, 6 val images for fraction=1.0 at imgsz=160
Using 0 dataloader workers
Logging results to C:\WorkSpace\20261106\06_Fri_TestFile_correct\p04_runs\smoke
Starting training for 1 epochs...

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
        1/1         0G      3.213      4.178    0.04309         19        160: 100% ━━━━━━━━━━━━ 3/3 9.9it/s 0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 19.6it/s 0.1s
                   all          6         18          0          0          0          0

1 epochs completed in 0.000 hours.
Optimizer stripped from C:\WorkSpace\20261106\06_Fri_TestFile_correct\p04_runs\smoke\weights\last.pt, 5.3MB
Optimizer stripped from C:\WorkSpace\20261106\06_Fri_TestFile_correct\p04_runs\smoke\weights\best.pt, 5.3MB

Validating C:\WorkSpace\20261106\06_Fri_TestFile_correct\p04_runs\smoke\weights\best.pt...
Ultralytics 8.4.142  Python-3.13.16 torch-2.14.0+cu130 CPU (Intel Core Ultra 9 275HX)
YOLO26n summary (fused): 120 layers, 2,375,421 parameters, 0 gradients, 5.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 30.8it/s 0.0s
                   all          6         18          0          0          0          0
                square          6          6          0          0          0          0
                circle          6          6          0          0          0          0
              triangle          6          6          0          0          0          0
Speed: 0.0ms preprocess, 4.1ms inference, 0.0ms loss, 0.3ms postprocess per image
task: detect names: {0: 'square', 1: 'circle', 2: 'triangle'} device: cpu
학습 결과: C:\WorkSpace\20261106\06_Fri_TestFile_correct\p04_runs\smoke """

# 20261106, subject 04  TST-004

# 1. 주요 결과

# 본 실습에서는 3교시에서 확인한 YOLO 포맷의 0.0~1.0 사이 정규화된 상대 좌표(cx, cy, w, h)를 입력 이미지의 실제 해상도 크기와 연산하여, 
# 화면 드로잉에 필요한 절대 픽셀 좌표계(x1, y1, x2, y2)로 복원 및 시각화 검증을 수행하였습니다.

# 2. 결론 

# 본 실습을 통해 상대 비율로 저장되어 있던 텍스트 라벨 데이터를 디지털 이미지 행렬의 절대 픽셀 좌표로 성공적으로 복원하였습니다.
# 이 과정은 AI 모델이 예측한 결과나 학습용 정답 데이터를 인간이 육안으로 판별할 수 있도록 
#프레임 위에 사각형(Bounding Box)을 그리는 시각화 파이프라인의 핵심 기술임을 검증하였습니다.