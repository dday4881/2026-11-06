# """실습 06. 강사가 준비한 실제 도형 YOLO weight로 valid 이미지만 추론합니다."""
# from pathlib import Path
# import cv2
# import torch
# import pandas as pd
# from ultralytics import YOLO
# BASE = Path(__file__).resolve().parent
# torch.set_num_threads(1)
# # data/demo.pt는 교실 사전 준비 도구로 생성합니다. 임의 모델로 자동 대체하지 않습니다.
# model = YOLO(str(BASE / "data/demo.pt"))
# rows = []; counts = []
# # >>> B07
# # 요구: data/images/val을 conf=0.25, imgsz=160, CPU로 predict 하세요.
# # source는 입력 경로, conf/imgsz/device는 선택 인자입니다.
# pass  # TODO B07: 위 요구사항의 핵심 코드를 작성하세요.
# # <<< B07
# for result in results:
#     name = Path(result.path).stem
#     counts.append({"image": name, "detections": len(result.boxes)})
#     cv2.imwrite(str(BASE / ("p06_" + name + ".png")), result.plot())
#     for box, cls, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
#         rows.append([name, int(cls), score, *box])
# pd.DataFrame(rows, columns=["image", "class_id", "confidence", "x1", "y1", "x2", "y2"]).to_csv(BASE/"p06_predictions.csv", index=False)
# pd.DataFrame(counts).to_csv(BASE/"p06_counts.csv", index=False)
# print("처리 이미지:", len(counts), "탐지 수:", len(rows))

"""실습 06. 강사가 준비한 실제 도형 YOLO weight로 valid 이미지만 추론합니다."""
from pathlib import Path  # 경로 처리를 위한 라이브러리입니다.
import cv2                # 이미지 저장을 위한 OpenCV 라이브러리입니다.
import torch              # CPU 쓰레드 제어를 위한 PyTorch 라이브러리입니다.
import pandas as pd       # 데이터를 모아 CSV로 저장하기 위한 판다스입니다.
from ultralytics import YOLO  # YOLO 모델을 다루기 위한 라이브러리입니다.

# 현재 파일의 부모 폴더 경로를 기준 경로로 설정합니다.
BASE = Path(__file__).resolve().parent

# CPU 연산 시 사용할 쓰레드 수를 1개로 제한하여 불필요한 자원 낭비를 막습니다.
torch.set_num_threads(1)

# data/demo.pt는 교실 사전 준비 도구로 생성합니다. 임의 모델로 자동 대체하지 않습니다.
# 강사님이 준비한 YOLO 학습 모델 파일(.pt)을 불러옵니다.
model = YOLO(str(BASE / "data/demo.pt"))

rows = []; counts = []  # 추론 결과 상세 정보와 이미지별 탐지 수를 저장할 빈 리스트입니다.

# >>> B07
# 요구: data/images/val을 conf=0.25, imgsz=160, CPU로 predict 하세요.
# source는 입력 경로, conf/imgsz/device는 선택 인자입니다.
# pass  # TODO B07: 위 요구사항의 핵심 코드를 작성하세요.

# [수정] model.predict() 메서드를 사용해 추론을 수행하고, 결과를 아래 루프에서 사용하는 'results' 변수에 할당합니다.
# 경로 오류를 방지하기 위해 BASE 경로와 결합한 뒤 문자열로 전달하며, 요구된 인자(conf, imgsz, device)를 올바른 파이썬 문법으로 지정합니다.
results = model.predict(source=str(BASE / "data/images/val"), conf=0.25, imgsz=160, device="cpu")
# <<< B07

# 추론된 결과 리스트(results)를 하나씩 꺼내어 처리합니다.
for result in results:
    # 현재 처리 중인 이미지 파일의 이름(확장자 제외)을 추출합니다.
    name = Path(result.path).stem
    
    # 이미지 이름과 해당 이미지에서 탐지된 물체의 개수(len(result.boxes))를 사전에 담아 저장합니다.
    counts.append({"image": name, "detections": len(result.boxes)})
    
    # result.plot()으로 바운딩 박스가 그려진 시각화 이미지를 얻고, 이를 'p06_이미지이름.png'로 저장합니다.
    cv2.imwrite(str(BASE / ("p06_" + name + ".png")), result.plot())
    
    # 탐지된 모든 박스의 좌표(xyxy), 클래스(cls), 신뢰도(conf)를 GPU 메모리에서 CPU로 옮긴 후 리스트로 변환하여 묶어 가져옵니다.
    for box, cls, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        # [이미지이름, 클래스ID, 신뢰도점수, x1, y1, x2, y2] 형태로 행 데이터를 리스트에 추가합니다.
        rows.append([name, int(cls), score, *box])

# 모든 탐지 결과 데이터를 표(DataFrame)로 만들어 'p06_predictions.csv'로 저장합니다.
pd.DataFrame(rows, columns=["image", "class_id", "confidence", "x1", "y1", "x2", "y2"]).to_csv(BASE/"p06_predictions.csv", index=False)

# 이미지별 탐지 수 데이터를 표로 만들어 'p06_counts.csv'로 저장합니다.
pd.DataFrame(counts).to_csv(BASE/"p06_counts.csv", index=False)

# 최종적으로 처리된 이미지 장수와 총 탐지된 물체의 수를 콘솔에 출력합니다.
print("처리 이미지:", len(counts), "탐지 수:", len(rows))

""" 출력 결과 값
image 1/6 C:\WorkSpace\20261106\06_Fri_TestFile\data\images\val\basic_val_000.png: 128x160 (no detections), 31.0ms
image 2/6 C:\WorkSpace\20261106\06_Fri_TestFile\data\images\val\basic_val_001.png: 128x160 (no detections), 5.7ms
image 3/6 C:\WorkSpace\20261106\06_Fri_TestFile\data\images\val\basic_val_002.png: 128x160 (no detections), 7.0ms
image 4/6 C:\WorkSpace\20261106\06_Fri_TestFile\data\images\val\basic_val_003.png: 128x160 (no detections), 8.0ms
image 5/6 C:\WorkSpace\20261106\06_Fri_TestFile\data\images\val\basic_val_004.png: 128x160 (no detections), 4.4ms
image 6/6 C:\WorkSpace\20261106\06_Fri_TestFile\data\images\val\basic_val_005.png: 128x160 (no detections), 6.7ms
Speed: 1.8ms preprocess, 10.5ms inference, 0.3ms postprocess per image at shape (1, 3, 128, 160)
처리 이미지: 6 탐지 수: 0   """


# 20261106, subject 06  TST-006

# 1. 주요 결과

# 본 실험에서는 외부에서 반입된 모델 가중치(demo.pt)와 학습 과정에 참여하지 않은 독립된 검증 데이터셋(data/images/val)을 연동하여 
# 객체 탐지 추론을 수행하고, 연산 속도 및 탐지 결과를 파싱하여 저장하였습니다.
# 추론 및 연산 환경 정보
# 검증 대상 이미지: 총 6장 (basic_val_000.png ~ basic_val_005.png)
# 추론 해상도(imgsz): 128 × 160 (입력값 160 가이드 기반 내부 32배수 패딩 적용 완료)
# 연산 장치(device): 일괄 CPU 추론 (쓰레드 수: 1 제한)

# 최종 처리 성능 및 산출 지표
# 평균 전처리 속도 (Preprocess): 1.8 ms
# 평균 순방향 연산 속도 (Inference): 10.5 ms
# 평균 후처리 속도 (Postprocess): 0.3 ms
# 최종 총 탐지 객체 수 (Detections): 0 건

# 2. 결론 

# 본 실습을 통해 이미지 로드 → YOLO 모델 포워딩 → CSV 결과 배출로 이어지는 객체 탐지 전체 추론 파이프라인의 시스템 무결성을 성공적으로 검증하였습니다.
# 다만 실제 탐지력이 0%로 수렴하여 모델의 실효성 평가가 불가능하므로 지표 개선을 위한 후속 조치가 필요합니다.

#3. 향후 계획

# 임계값 완화 테스트 (Threshold Tuning): conf 인자 값을 0.25에서 0.05 또는 0.01로 대폭 낮추어 재실행한 뒤, 
# 모델이 아주 미세하게라도 물체를 잡고 있는지(Log-it 수준 검증) 디버깅하려고 합니다.
# 라벨 데이터 비교 검증: 생성된 p06_predictions.csv가 현재 비어 있으므로, 
# 원본 정답지(boxes_gt.csv)의 좌표 구역과 모델이 인지하는 피처 맵을 강제로 오버랩하여 가중치가 정상 주입되었는지 재검토할 예정입니다.