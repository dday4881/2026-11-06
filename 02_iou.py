"""실습 02. 손으로 정한 예측 4개를 정답 3개와 비교합니다. YOLO 결과가 아닙니다."""
from pathlib import Path  # 파일 경로 처리를 위한 라이브러리입니다.
import numpy as np        # 수학 연산 및 배열 처리를 위한 라이브러리입니다.
import pandas as pd       # 표 데이터를 다루기 위한 라이브러리입니다.

# 현재 파일의 부모 폴더 경로를 기준 경로로 설정합니다.
BASE = Path(__file__).resolve().parent

# 모델이 예측한 박스 정보(상대적 신뢰도 순으로 정렬됨)를 읽어옵니다.
pred = pd.read_csv(BASE / "data/boxes_fixture.csv")  # confidence 내림차순 예시

# 데이터셋의 실제 정답(Ground Truth) 박스 정보를 읽어옵니다.
truth = pd.read_csv(BASE / "data/boxes_gt.csv")

used = set()  # 하나의 정답은 한 번만 TP로 연결합니다. (중복 매칭을 방지하는 집합 변수)
rows = []     # 매칭 결과를 한 행씩 저장해둘 빈 리스트입니다.

# 1. 예측한 박스(p)를 하나씩 순회합니다.
for p in pred.itertuples():
    overlaps = []  # 현재 예측 박스와 모든 정답 박스 간의 IoU들을 저장할 리스트입니다.
    
    # 2. 모든 실제 정답 박스(g)를 하나씩 꺼내어 현재 예측 박스(p)와 비교합니다.
    for g in truth.itertuples():
        # 두 박스가 겹치는 영역(교집합)의 가로 길이를 구합니다. (겹치지 않으면 음수가 되므로 max(0, ...) 처리)
        width = max(0, min(p.x2, g.x2) - max(p.x1, g.x1))
        # 두 박스가 겹치는 영역(교집합)의 세로 길이를 구합니다.
        height = max(0, min(p.y2, g.y2) - max(p.y1, g.y1))
        
        # 교집합(Intersection)의 면적을 계산합니다.
        intersection = width * height
        # 합집합(Union)의 면적을 계산합니다. (박스A 면적 + 박스B 면적 - 교집합 면적)
        union = (p.x2-p.x1)*(p.y2-p.y1) + (g.x2-g.x1)*(g.y2-g.y1) - intersection
        
        # >>> B02
        # 요구: intersection을 union으로 나누어 iou를 계산하세요.
        # 이 실습의 데이터는 모두 양의 면적이므로 union은 0이 아닙니다.
        #pass  # TODO B02: 위 요구사항의 핵심 코드를 작성하세요.
        iou = intersection / union # '/' = 나누기 표현
        # <<< B02
        
        overlaps.append(iou)  # 계산된 IoU 값을 리스트에 누적합니다.
        
    matched = False  # 정답 박스와 매칭 성공 여부를 저장하는 플래그입니다.
    
    # np.argsort(-...)를 사용해 IoU가 가장 높은 정답 박스의 인덱스(j)부터 내림차순으로 확인합니다.
    for j in np.argsort(-np.asarray(overlaps)):
        # 예측한 종류(class_id)가 정답과 다르거나, 이미 다른 예측에 선점당한(used) 정답이라면 건너뜁니다.
        if truth.iloc[j].class_id != p.class_id or j in used:
            continue
            
        # >>> B03
        # 요구: 같은 클래스의 미사용 정답에 IoU>=0.5이면 연결하고 used에 기록하세요.
        #pass  # TODO B03: 위 요구사항의 핵심 코드를 작성하세요.
        if iou >= 0.5:
            matched = True  # 매칭 성공 표시
            used.add(j)     # 이 정답 박스는 사용되었다고 기록 (중복 사용 차단)
        # <<< B03
        
    # 결과를 표에 채우기 위해 [클래스, 신뢰도, 최대 IoU, 매칭 결과(TP/FP)] 형태로 저장합니다.
    rows.append([p.class_id, p.confidence, round(max(overlaps), 4), "TP" if matched else "FP"])

# 수집된 결과를 바탕으로 데이터프레임(표)을 생성합니다.
table = pd.DataFrame(rows, columns=["class", "confidence", "max_iou", "result"])
# 인덱스 열을 제외하고 보기 좋게 콘솔에 출력합니다.
print(table.to_string(index=False))

# 평가 지표 계산: TP(진짜 양성), FP(가짜 양성 = 예측 수 - TP), FN(미검출 정답 = 전체 정답 수 - TP)
tp = len(used); fp = len(pred)-tp; fn = len(truth)-tp
print("TP/FP/FN:", tp, fp, fn)

# 정밀도(Precision = 맞춘 예측 / 전체 예측)와 재현율(Recall = 맞춘 정답 / 전체 정답)을 소수점 4자리까지 구합니다.
print("Precision/Recall:", round(tp/len(pred), 4), round(tp/len(truth), 4))

# 결과를 한글 깨짐 방지(utf-8-sig) 처리를 하여 'p02_matching.csv' 파일로 보관합니다.
table.to_csv(BASE / "p02_matching.csv", index=False, encoding="utf-8-sig")

""" 출력 결과 값
 class  confidence  max_iou result
     0        0.95   0.9025     FP
     0        0.85   0.8162     FP
     1        0.80   1.0000     FP
     2        0.40   0.0000     FP
TP/FP/FN: 0 4 3
Precision/Recall: 0.0 0.0   """

# 20261106, subject 02  TST-002

# 1. 주요 결과
# - FN 3건
# - FP 4건
# - Precision 00.00%
# - Recall 00.00%

# 2. 결과 해석
# IoU가 1.0000로 가장 높은 IoU 값이 나왔고, 클래스가 1임에도 FP가 나왔습니다.
# 잘못된 부분이 있어서 IoU 값과, confidence 값이 있음에도 FP만 주구장창 나오는데에 문제가 있음을 확인하였습니다.

# 3. 결론
# 본 실습에서 객체의 위치가 정확하더라도 중복 탐지나 미탐지가 발생하면 전체 탐지 성능이 저하될 수 있음을 확인하였습니다.
# Confidence Threshold 조정 등을 통해 오탐과 미탐을 관리할 필요를 알았습니다.
# 코딩이나, 실험 중 잘못 설정한 부분이 있어 FP 외에 다른 값을 출력받지 못한점을 확인. 재 검사를 통해 제대로 된 결과값을 뽑아야한단 필요를 확인했습니다.
# 추후 실험에서 문제 재발시 새로운 작업 환경에서 재 진행 할 예정입니다.