import torch
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# 1. 임베딩 로드 (float16 -> float32 캐스팅 추천)
basic_emb = torch.load("basic_emb.pt").to(torch.float32)   # (N, 384)
after_emb = torch.load("after_emb.pt").to(torch.float32)   # (N, 384)

assert basic_emb.shape == after_emb.shape
N, D = basic_emb.shape

# 2. 라벨 로드 (순서가 임베딩과 동일하다고 가정)
df_label = pd.read_csv("emb_label.csv")  # columns: title, subtopic, topic
assert len(df_label) == N

# 3. drift / norm 특징 계산
basic_np = basic_emb.numpy()
after_np = after_emb.numpy()

drift = np.linalg.norm(after_np - basic_np, axis=1)      # (N,)
basic_norm = np.linalg.norm(basic_np, axis=1)
after_norm = np.linalg.norm(after_np, axis=1)

# 4. 클러스터링 (예: K=128, 필요시 조정)
K = 128
base_kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, random_state=42)
after_kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, random_state=42)

base_cluster_id = base_kmeans.fit_predict(basic_np)   # (N,)
after_cluster_id = after_kmeans.fit_predict(after_np)

# 중심까지 거리 (군집 밀도/중심부 vs 바깥쪽 판별용)
base_center = base_kmeans.cluster_centers_[base_cluster_id]
after_center = after_kmeans.cluster_centers_[after_cluster_id]

base_dist2center = np.linalg.norm(basic_np - base_center, axis=1)
after_dist2center = np.linalg.norm(after_np - after_center, axis=1)

# 5. feature 테이블 구성
df_feat = df_label.copy()
df_feat["drift_norm"] = drift
df_feat["basic_norm"] = basic_norm
df_feat["after_norm"] = after_norm
df_feat["base_cluster_id"] = base_cluster_id
df_feat["after_cluster_id"] = after_cluster_id
df_feat["base_dist2center"] = base_dist2center
df_feat["after_dist2center"] = after_dist2center

# 저장
df_feat.to_csv("emb_features_basic_after.csv", index=False)