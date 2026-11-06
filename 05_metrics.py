# """실습 05. 가상 기록 읽기. 이 표는 실제 모델 성능이 아닙니다."""
# from pathlib import Path
# import pandas as pd
# BASE = Path(__file__).resolve().parent
# history = pd.read_csv(BASE / "data/metrics_example.csv")
# # >>> B06
# # 요구: map50_95가 가장 큰 행을 선택하세요. idxmax는 첫 최대값의 index를 반환합니다.
# #pass  # TODO B06: 위 요구사항의 핵심 코드를 작성하세요.
# # <<< B06
# print("자료 종류: 읽기 연습용 가상 기록 — 실제 YOLO 성능 아님")
# print("선택 epoch:", int(best["epoch"]))
# print("mAP50:", best["map50"], "mAP50-95:", best["map50_95"])
# print("mAP는 accuracy가 아닙니다. train이 아니라 valid의 위치·분류 평가 기록입니다.")

"""실습 05. 가상 기록 읽기. 이 표는 실제 모델 성능이 아닙니다."""
from pathlib import Path  # 파일 경로를 조작하기 위한 라이브러리를 가져옵니다.
import pandas as pd       # CSV 데이터를 표(DataFrame) 형태로 다루기 위해 가져옵니다.

# 현재 실행 중인 파일의 부모 폴더 경로를 기준 경로로 잡습니다.
BASE = Path(__file__).resolve().parent

# 데이터 폴더 안의 'metrics_example.csv' 파일을 읽어와 학습 기록 데이터프레임으로 만듭니다.
history = pd.read_csv(BASE / "data/metrics_example.csv")

# >>> B06
# 요구: map50_95가 가장 큰 행을 선택하세요. idxmax는 첫 최대값의 index를 반환합니다.
#pass  # TODO B06: 위 요구사항의 핵심 코드를 작성하세요.

# 1. history['map50_95'].idxmax()로 'map50_95' 컬럼의 값이 가장 큰 행의 번호(인덱스)를 찾습니다.
# 2. .loc[...]를 사용해 해당 인덱스의 전체 행 데이터를 선택하여 best 변수에 저장합니다.
best = history.loc[history["map50_95"].idxmax()]
# <<< B06

# 콘솔에 주의사항 및 결과를 보기 좋게 출력하는 부분입니다.
print("자료 종류: 읽기 연습용 가상 기록 — 실제 YOLO 성능 아님")

# 최고 성능을 낸 에포크(학습 회차)를 정수형으로 변환하여 출력합니다.
print("선택 epoch:", int(best["epoch"]))

# 해당 에포크의 mAP50 성능과 mAP50-95 성능을 각각 출력합니다.
print("mAP50:", best["map50"], "mAP50-95:", best["map50_95"])

# mAP 지표에 대한 올바른 개념 가이드를 출력합니다.
print("mAP는 accuracy가 아닙니다. train이 아니라 valid의 위치·분류 평가 기록입니다.")

""" 출력 결과 값
자료 종류: 읽기 연습용 가상 기록 — 실제 YOLO 성능 아님
선택 epoch: 4
mAP50: 0.55 mAP50-95: 0.27
mAP는 accuracy가 아닙니다. train이 아니라 valid의 위치·분류 평가 기록입니다. """

""" 출력 결과 값(metrics_example 파일 값 변경)
자료 종류: 읽기 연습용 가상 기록 — 실제 YOLO 성능 아님
선택 epoch: 4
mAP50: 0.855 mAP50-95: 0.427
mAP는 accuracy가 아닙니다. train이 아니라 valid의 위치·분류 평가 기록입니다. """


# 20261106, subject 05  TST-005

# 결론: 실제 YOLO를 이용한 값이 아닌 가상 환경에 구축해 실험 하는 단계였습니다.
# mAP50이 mAP50-95보다 값이 낮게 나와 랜덤하게 값을 바꿔서 반대 단계에서도 실험을 해 보았습니다.
# 그럼에도 불구하고 예측값이 50%만 넘으면 통과되어 평균으로 잡히는 mAP50과 
# 50부터 95까지 0.5 단으로 검수하는 mAP50-95의 차이에 구축 실험이니 당연히 값을 바꿔도 평균 값이 낮을 수 밖에 없다는 걸 각인시켰습니다.
# 추후 실제 YOLO를 사용한 향후에 있을 실험에서는 단순히 예측값 뿐 아니라 위치 예측 성능과 클래스별 탐지 성능을 추가 분석할 필요가 있다 생각합니다.
