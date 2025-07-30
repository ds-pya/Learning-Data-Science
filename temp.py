from sklearn.metrics import precision_recall_curve, classification_report
import numpy as np
import pandas as pd

# 입력
# a: List[List[float]] → 예측 확률값 (sigmoid 통과된 값)
# b: List[List[bool]] → 정답 (멀티라벨 ground truth)
# c: List[str]         → 라벨 이름 (길이 = n)

a = np.array(a)
b = np.array(b).astype(int)
n_labels = len(c)

thresholds_dict = {}
binary_preds = np.zeros_like(a)

# 1. 라벨별 threshold 계산 및 적용
for i in range(n_labels):
    precision, recall, thresholds = precision_recall_curve(b[:, i], a[:, i])
    
    # precision은 len(thresholds)+1 이므로 슬라이싱 필요
    above_idx = np.where(precision[:-1] >= 0.9)[0]
    
    if len(above_idx) > 0:
        t = thresholds[above_idx[0]]  # 최소 threshold
    else:
        t = 0.5  # fallback (또는 None으로 두고 나중에 제외 가능)
    
    thresholds_dict[c[i]] = t
    binary_preds[:, i] = (a[:, i] >= t).astype(int)

# 2. classification_report 출력
report = classification_report(b, binary_preds, target_names=c)
print(report)

# 3. threshold 정리 출력
threshold_df = pd.DataFrame({
    "label": c,
    "threshold@precision>0.9": [thresholds_dict[label] for label in c]
})
print(threshold_df)