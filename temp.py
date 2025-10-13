import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 예시 데이터
df = pd.DataFrame({
    'a': ['item1', 'item2', 'item3'],
    's1': [70, 80, 90],
    's2': [65, 85, 88],
    's3': [0.3, 0.6, 0.9]
})

# y 위치
y = np.arange(len(df))

# Figure와 두 개의 x축 생성
fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twiny()   # 보조 x축 생성 (위쪽)

# ---- ① 기본축(s1, s2) ----
bar_height = 0.25
ax1.barh(y - bar_height/2, df['s1'], height=bar_height, color='steelblue', label='s1')
ax1.barh(y + bar_height/2, df['s2'], height=bar_height, color='skyblue', label='s2')

ax1.set_xlabel("Score (s1, s2)")
ax1.set_ylabel("항목")
ax1.set_yticks(y)
ax1.set_yticklabels(df['a'])
ax1.set_xlim(0, 100)

# ---- ② 보조축(s3) ----
ax2.barh(y, df['s3'], height=bar_height/2, color='orange', alpha=0.7, label='s3')
ax2.set_xlim(0, 1.0)
ax2.set_xlabel("Score (s3)")

# ---- ③ 레이아웃 조정 ----
ax1.legend(loc='lower right')
ax2.legend(loc='upper right')
plt.title("두 개의 X축을 가진 가로형 바 차트 (Matplotlib)")
plt.tight_layout()
plt.show()