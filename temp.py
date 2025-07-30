from sklearn.metrics import classification_report, precision_recall_curve
import numpy as np
import pandas as pd

# a: 예측 확률값, b: 정답 (bool), c: 라벨 이름
a = np.array(a)  # shape = (num_samples, n_labels)
b = np.array(b)  # same shape

# 1. classification report (threshold = 0.5 기준 이진화)
pred_binary = (a >= 0.5).astype(int)
true_binary = b.astype(int)

print(classification_report(true_binary, pred_binary, target_names=c))

# 2. precision > 0.9를 만족하는 최소 threshold 계산
thresholds_dict = {}

for i, label in enumerate(c):
    precision, recall, thresholds = precision_recall_curve(true_binary[:, i], a[:, i])
    # precision 배열은 길이가 thresholds+1 이므로 trimming 필요
    thresholds = thresholds  # shape = len(thresholds)

    # precision > 0.9 조건 만족하는 최소 threshold
    above_thresh = np.where(precision[:-1] >= 0.9)[0]  # 마지막 precision은 무시
    if len(above_thresh) > 0:
        min_thresh = thresholds[above_thresh[0]]
    else:
        min_thresh = None  # precision이 0.9 넘는 지점 없음
    thresholds_dict[label] = min_thresh

# 보기 좋게 출력
df_thresholds = pd.DataFrame({
    'label': c,
    'min_threshold_for_precision>0.9': [thresholds_dict[l] for l in c]
})
print(df_thresholds)