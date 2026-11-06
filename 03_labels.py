"""실습 03. 정상 이미지 6장과 YOLO TXT 라벨을 읽습니다."""
from pathlib import Path  # 파일 및 폴더 경로를 안전하게 다루기 위해 가져옵니다.
import cv2                # 이미지를 읽어오기 위해 OpenCV 라이브러리를 가져옵니다.

# 현재 실행 중인 파일의 부모 폴더 경로를 기준 경로(BASE)로 설정합니다.
BASE = Path(__file__).resolve().parent

# 클래스 ID(0, 1, 2)에 대응하는 실제 도형 이름을 리스트로 정의합니다.
names = ["square", "circle", "triangle"]

# train 폴더 안의 모든 .png 이미지를 정렬한 뒤, 앞의 6장([:6])만 가져와 반복문을 돌립니다.
for image_path in sorted((BASE / "data/images/train").glob("*.png"))[:6]:
    # OpenCV를 이용해 이미지 파일을 읽어옵니다.
    image = cv2.imread(str(image_path))
    
    # 이미지 파일명과 똑같은 이름의 텍스트(.txt) 라벨 파일 경로를 매칭하여 생성합니다.
    label_path = BASE / "data/labels/train" / (image_path.stem + ".txt")
    
    # >>> B04
    # 요구: UTF-8 라벨 파일의 문자열을 읽고 줄 목록으로 나누세요.
    # pass  # TODO B04: 위 요구사항의 핵심 코드를 작성하세요.
    
    # 텍스트 파일의 전체 내용을 UTF-8로 읽어온 뒤, 엔터(줄바꿈) 기준으로 쪼개어 리스트로 만듭니다.
    lines = label_path.read_text(encoding="utf-8").splitlines()
    # <<< B04
    
    # 이미지 이름, 크기(세로, 가로, 채널), 그리고 해당 이미지에 포함된 박스(물체) 개수를 출력합니다.
    print(image_path.name, "shape:", image.shape, "boxes:", len(lines))
    
    # 라벨 파일에서 읽어온 한 줄(물체 1개 정보)씩 꺼내어 처리합니다.
    for line in lines:
        # 공백을 기준으로 문자열을 자른 뒤, 각각 소수점(float) 숫자로 변환하여 변수에 할당합니다.
        cls, cx, cy, width, height = map(float, line.split())
        
        # 클래스 번호(int)를 이름으로 치환하고, 정규화된 4개의 좌표 값을 콘솔에 출력합니다.
        print(" ", names[int(cls)], "normalized:", cx, cy, width, height)


""" 출력 결과 값
basic_train_000.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.796875 0.33984375 0.19375 0.2421875
  circle normalized: 0.1875 0.25 0.1625 0.203125
  triangle normalized: 0.48125 0.640625 0.2125 0.265625
basic_train_001.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.23125 0.234375 0.225 0.28125
  circle normalized: 0.50625 0.6875 0.2 0.25
  triangle normalized: 0.834375 0.27734375 0.23125 0.2890625
basic_train_002.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.165625 0.32421875 0.21875 0.2734375
  circle normalized: 0.79375 0.3046875 0.2 0.25
  triangle normalized: 0.55 0.65625 0.1875 0.234375
basic_train_003.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.83125 0.25 0.175 0.21875
  circle normalized: 0.178125 0.29296875 0.19375 0.2421875
  triangle normalized: 0.534375 0.64453125 0.18125 0.2265625
basic_train_004.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.853125 0.26953125 0.18125 0.2265625
  circle normalized: 0.453125 0.76171875 0.23125 0.2890625
  triangle normalized: 0.215625 0.33984375 0.23125 0.2890625
basic_train_005.png shape: (128, 160, 3) boxes: 3
  square normalized: 0.1625 0.328125 0.1875 0.234375
  circle normalized: 0.846875 0.28515625 0.16875 0.2109375
  triangle normalized: 0.54375 0.734375 0.2 0.25    """

# 20261106, subject 03  TST-003

# 1. 주요 결과

# 본 실험에서는 학습용 정상 이미지 데이터 6장과 이에 매칭되는 YOLO 포맷의 텍스트(.txt) 라벨 파일을 상호 참조하여 
# 데이터의 해상도 정보 및 객체(도형) 파싱 현황을 검증하였습니다.
# 검증 이미지 수: 총 6장 (basic_train_000.png ~ basic_train_005.png)
# 이미지 입력 해상도: 160 × 128 (가로 × 세로), 채널 수 3 (Color 이미지)
# 이미지별 탐지 객체 수: 모든 이미지에 일괄적으로 3개(boxes: 3)의 객체가 포함됨을 확인
# 클래스 정의 (Class Names): 총 3개 클래스 (0: square, 1: circle, 2: triangle)

# 2. 결론
# 본 실습을 통해 이미지 파일과 텍스트 라벨 파일이 누락 없이 1:1로 안전하게 동기화되어 있음을 확인하였으며, 
# AI 모델이 정상적으로 학습을 수행하기 위한 데이터셋의 기초 무결성이 확보되었음을 검증하였습니다.
# 정규화된 소수점 좌표계를 활용함으로써 향후 입력 이미지 해상도가 변경되더라도, 
# 박스 좌표 재작업 없이 데이터셋을 유연하게 재사용할 수 있는 데이터 정형화의 이점을 파악하였습니다.
