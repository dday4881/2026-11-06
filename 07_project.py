# """실습 07~08. 실제 모델의 이미지 탐지와 영상 추적을 연결합니다."""
# from pathlib import Path
# import cv2
# import torch
# import pandas as pd
# from ultralytics import YOLO
# BASE = Path(__file__).resolve().parent
# torch.set_num_threads(1)
# model = YOLO(str(BASE / "data/demo.pt"))
# rows = []
# for r in model.predict(source=str(BASE/"data/images/val"), conf=0.25, imgsz=160, device="cpu", verbose=False):
#     for box, cls, score in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist(), r.boxes.conf.cpu().tolist()):
#         rows.append([Path(r.path).name, int(cls), score, *box])
# pd.DataFrame(rows, columns=["image", "class_id", "confidence", "x1", "y1", "x2", "y2"]).to_csv(BASE/"p07_predictions.csv", index=False)
# # 새 영상은 새 모델 인스턴스로 시작하여 이전 추적 상태를 이어붙이지 않습니다.
# tracker = YOLO(str(BASE / "data/demo.pt"))
# cap = cv2.VideoCapture(str(BASE / "data/demo.avi"))
# fps = cap.get(cv2.CAP_PROP_FPS)
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# cap.release()
# writer = cv2.VideoWriter(str(BASE / "p07_tracked.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
# tracks = []; frame_rows = []
# # >>> B08
# # 요구: demo.avi를 ByteTrack으로 읽는 stream=True 결과 generator를 만드세요.
# # persist=False는 새로운 파일 스트림의 시작입니다. 같은 파일 내부 프레임은 이어집니다.
# # pass  # TODO B08: 위 요구사항의 핵심 코드를 작성하세요.

# # <<< B08
# for frame, result in enumerate(results):
#     writer.write(result.plot())
#     frame_rows.append([frame, len(result.boxes)])
#     if result.boxes.id is not None:
#         for box, cls, score, tid in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.id.cpu().tolist()):
#             tracks.append([frame, int(tid), int(cls), score, *box])
# writer.release()
# # >>> B09
# # 요구: tracks를 지정한 열 이름의 CSV로 저장하세요. index=False를 사용합니다.
# #pass  # TODO B09: 위 요구사항의 핵심 코드를 작성하세요.
# # <<< B09
# pd.DataFrame(frame_rows, columns=["frame", "detections"]).to_csv(BASE/"p07_frames.csv", index=False)
# print("입력 fps/size:", fps, size, "처리 Frame:", len(frame_rows))
# print("실습 08: 설정을 바꾸지 말고 다시 실행해 Frame 수·CSV·영상을 확인하세요.")


"""실습 07~08. 실제 모델의 이미지 탐지와 영상 추적을 연결합니다."""
from pathlib import Path  # 경로 제어를 위한 라이브러리입니다.
import cv2                # 비디오/이미지 입출력(OpenCV)을 위한 라이브러리입니다.
import torch              # CPU 연산 최적화를 위한 라이브러리입니다.
import pandas as pd       # 표 형태(CSV)의 데이터를 저장하기 위한 라이브러리입니다.
from ultralytics import YOLO  # YOLO 모델 및 추적(ByteTrack) 기능을 위한 라이브러리입니다.

# 현재 스크립트 파일의 부모 폴더 경로를 기준 경로로 설정합니다.
BASE = Path(__file__).resolve().parent

# CPU 연산 시 쓰레드 수를 1개로 제한하여 충돌을 방지합니다.
torch.set_num_threads(1)

# 이미지 탐지용 YOLO 가중치 모델을 로드합니다.
model = YOLO(str(BASE / "data/demo.pt"))
rows = []

# val 폴더 내의 이미지들을 대상으로 객체 탐지(Predict)를 실행합니다.
for r in model.predict(source=str(BASE/"data/images/val"), conf=0.25, imgsz=160, device="cpu", verbose=False):
    # 탐지된 상자(boxes)의 좌표, 클래스 번호, 신뢰도 점수를 묶어서 반복 처리합니다.
    for box, cls, score in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist(), r.boxes.conf.cpu().tolist()):
        # 이미지 파일명과 탐지 정보를 리스트에 보관합니다.
        rows.append([Path(r.path).name, int(cls), score, *box])

# 탐지한 이미지 결과를 'p07_predictions.csv' 파일로 저장합니다.
pd.DataFrame(rows, columns=["image", "class_id", "confidence", "x1", "y1", "x2", "y2"]).to_csv(BASE/"p07_predictions.csv", index=False)

# 새 영상은 새 모델 인스턴스로 시작하여 이전 추적 상태를 이어붙이지 않습니다.
# 비디오 추적(Tracking)용 독립된 YOLO 객체를 새로 생성합니다.
tracker = YOLO(str(BASE / "data/demo.pt"))

# 대상 비디오 파일(demo.avi)을 OpenCV의 VideoCapture로 열어 정보를 확인합니다.
cap = cv2.VideoCapture(str(BASE / "data/demo.avi"))
fps = cap.get(cv2.CAP_PROP_FPS)  # 영상의 초당 프레임 수(FPS)를 가져옵니다.
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 영상의 가로, 세로 크기를 가져옵니다.
cap.release()  # 비디오 정보 획득이 끝났으므로 닫아줍니다.

# 추적 결과 박스가 그려진 새로운 비디오 파일을 저장하기 위한 배출구(VideoWriter)를 생성합니다.
writer = cv2.VideoWriter(str(BASE / "p07_tracked.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
tracks = []; frame_rows = []

# >>> B08
# 요구: demo.avi를 ByteTrack으로 읽는 stream=True 결과 generator를 만드세요.
# persist=False는 새로운 파일 스트림의 시작입니다. 같은 파일 내부 프레임은 이어집니다.
# pass  # TODO B08: 위 요구사항의 핵심 코드를 작성하세요.

# [정답] tracker.track() 함수를 호출하여 비디오 경로, 추적 알고리즘(bytetrack), 스트림 처리 및 요구 사항 인자들을 지정하고, 아래 루프에서 순회할 'results' 변수에 할당합니다.
results = tracker.track(source=str(BASE / "data/demo.avi"), conf=0.25, imgsz=160, device="cpu", tracker="bytetrack.yaml", stream=True, persist=False, verbose=False)
# results를 지정했으면 진행 되었을 것 힌트로 results에 대한 값이 지정되지 않았다는 오류문구가 떠있었기 때문에.
# <<< B08

# 제너레이터(results)로부터 영상의 한 프레임씩 결과를 받아서 처리합니다. (enumerate로 프레임 번호 자동 부여)
for frame, result in enumerate(results):
    # 각 프레임에 바운딩 박스와 추적 ID가 시각화된 이미지를 비디오 파일로 저장합니다.
    writer.write(result.plot())
    # 현재 프레임 번호와 해당 프레임에서 탐지된 물체의 총 개수를 보관합니다.
    frame_rows.append([frame, len(result.boxes)])
    
    # 만약 현재 프레임에 추적 중인 객체의 ID(ID가 부여된 상자)가 존재한다면 실행합니다.
    if result.boxes.id is not None:
        # 좌표, 클래스, 신뢰도 점수와 함께 부여된 객체 추적 ID(tid)를 함께 매핑하여 가져옵니다.
        for box, cls, score, tid in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.id.cpu().tolist()):
            # [프레임 번호, 추적 ID, 클래스 ID, 신뢰도, x1, y1, x2, y2] 순으로 리스트에 누적합니다.
            tracks.append([frame, int(tid), int(cls), score, *box])

# 영상 쓰기 작업이 모두 완료되었으므로 파일을 안전하게 닫아줍니다.
writer.release()

# >>> B09
# 요구: tracks를 지정한 열 이름의 CSV로 저장하세요. index=False를 사용합니다.
# pass  # TODO B09: 위 요구사항의 핵심 코드를 작성하세요.

# [정답] 수집한 비디오 추적 결과 데이터(tracks)를 하단에 지정된 순서대로 컬럼 이름을 매겨 데이터프레임으로 만들고, 'p07_tracks.csv' 파일로 내보냅니다.
pd.DataFrame(tracks, columns=["frame", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]).to_csv(BASE / "p07_tracks.csv", index=False)
# <<< B09

# 프레임별 탐지 건수 기록 요약본을 'p07_frames.csv' 파일로 저장합니다.
pd.DataFrame(frame_rows, columns=["frame", "detections"]).to_csv(BASE/"p07_frames.csv", index=False)

# 최종 처리 결과를 모니터에 출력합니다.
print("입력 fps/size:", fps, size, "처리 Frame:", len(frame_rows))
print("실습 08: 설정을 바꾸지 말고 다시 실행해 Frame 수·CSV·영상을 확인하세요.")


""" 출력 결과 값
입력 fps/size: 10.0 (160, 128) 처리 Frame: 30
실습 08: 설정을 바꾸지 말고 다시 실행해 Frame 수·CSV·영상을 확인하세요. """