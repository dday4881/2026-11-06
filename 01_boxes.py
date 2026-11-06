"""실습 01. 정답 박스 3개를 그립니다. 원본 scene.png는 바꾸지 않습니다."""
from pathlib import Path  # 파일이나 폴더의 경로를 다루기 위한 라이브러리를 가져옵니다.
import cv2                # 이미지 처리(OpenCV)를 위한 라이브러리를 가져옵니다.
import pandas as pd       # 표 형태의 데이터(CSV 파일 등)를 다루기 위한 라이브러리를 가져옵니다.

# 현재 실행 중인 파이썬 파일(__file__)의 실제(absolute) 부모 폴더 경로를 기준 경로(BASE)로 설정합니다.
BASE = Path(__file__).resolve().parent

# 기준 경로 아래 'data' 폴더 안의 'scene.png' 이미지를 읽어옵니다. (OpenCV는 경로를 문자열(str)로 받아야 합니다)
image = cv2.imread(str(BASE / "data/scene.png"))

# 'data' 폴더 안의 정답 박스 정보가 담긴 'boxes_gt.csv' 파일을 읽어와 표(DataFrame) 형태로 저장합니다.
boxes = pd.read_csv(BASE / "data/boxes_gt.csv")

result = image.copy()  # 그릴 배열을 복사하여 원본을 보존합니다.

# boxes 데이터프레임의 행(row)을 하나씩 가져와서 반복 루프를 돕니다. (itertuples는 각 행을 객체 형태로 반환합니다)
for row in boxes.itertuples():
    # >>> B01
    # 요구: result에 (x1,y1)~(x2,y2) 박스를 두께 2로 그리세요.
    # cv2.rectangle의 이미지·두 점·색은 필수, thickness는 선택입니다.
    # 해설: 파일의 좌표는 연속 경계이고 화면 표시 때만 정수로 바꿉니다.
    # cv2.rectangle(): 이미지 위에 Bounding Box(사각형)를 그리는 함수
    #
    # result : 사각형을 그릴 대상 이미지 (원본 배열이 직접 수정됨)
    #
    # (int(row.x1), int(row.y1))
    #   → Bounding Box의 왼쪽 위 좌표 (x1, y1)
    #
    # (int(row.x2), int(row.y2))
    #   → Bounding Box의 오른쪽 아래 좌표 (x2, y2)
    #
    # int(): 픽셀 좌표는 정수로 전달해야 하므로 실수 좌표를 정수로 변환
    #
    # (255, 0, 0): 사각형 테두리 색상
    #   → OpenCV는 BGR 순서를 사용하므로 파란색(Blue)
    #
    # 2: 사각형 테두리 두께 (2픽셀)
    #
    # 목적: 객체 탐지 결과의 위치를 이미지에 표시하여
    #       모델이 어떤 영역을 객체로 인식했는지 시각적으로 확인

    cv2.rectangle(
        result,
        (int(row.x1), int(row.y1)),
        (int(row.x2), int(row.y2)),
        (255, 0, 0),
        2
    )
    # <<< B01
    
    # 현재 가져온 행의 클래스 아이디(종류)와 네 군데의 좌표를 콘솔 창에 출력합니다.
    print("class:", row.class_id, "xyxy:", row.x1, row.y1, row.x2, row.y2, "GT confidence: N/A")

# 사각형 박스가 그려진 결과 이미지(result)를 'p01_boxes.png'라는 이름의 파일로 저장합니다.
cv2.imwrite(str(BASE / "p01_boxes.png"), result)

# 원본 이미지의 크기(세로, 가로, 채널 수)를 출력하고, 저장이 완료되었다는 메시지를 출력합니다.
print("shape:", image.shape, "저장: p01_boxes.png")

""" 출력 결과 값
class: 0 xyxy: 12 18 52 58 GT confidence: N/A
class: 1 xyxy: 70 60 108 98 GT confidence: N/A
class: 2 xyxy: 116 18 150 62 GT confidence: N/A
shape: (128, 160, 3) 저장: p01_boxes.png    """

# 20261106, subject 01  TST-001


# 1. 주요 결과 

#본 실험에서는 객체 탐지 데이터셋의 실제 정답(Ground Truth) 좌표 정보를 원본 이미지 위에 사각형(Bounding Box) 형태로 드로잉하여
# 시각적으로 검증하고, 사본 데이터를 안정적으로 렌더링 및 저장하는 작업을 수행하였습니다.

# 2. 결론 
# 본 실습은 객체 탐지 모델 학습을 위한 기초 데이터 준비 과정으로, 판다스(pandas)를 이용한 좌표 메타데이터 파싱과 
# OpenCV(cv2)를 이용한 시각화 연동 프로세스를 성공적으로 구축하였습니다.