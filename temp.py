import torch
import torch.nn.functional as F

class TextClassificationModel(torch.nn.Module):
    def __init__(self, base_model, num_categories=5, num_leafs_per_category=5, embedding_dim=128, min_samples_leaf=10):
        super().__init__()
        self.base_model = base_model  # Pretrained 모델
        self.num_categories = num_categories
        self.num_leafs_per_category = num_leafs_per_category
        self.embedding_dim = embedding_dim
        self.min_samples_leaf = min_samples_leaf  # 최소 샘플 횟수 기준

        # 학습 가능한 카테고리 분점 (leaf들)
        self.category_embeddings = torch.nn.Parameter(torch.randn(num_categories, num_leafs_per_category, embedding_dim))

        # 각 분점이 positive로 선택된 횟수를 저장하는 카운터 (초기값 0)
        self.register_buffer("leaf_usage_counts", torch.zeros(num_categories, num_leafs_per_category))

    def forward(self, x, is_training=True):
        x_expanded = x[:, None, None, :]  # (64, 1, 1, 128)
        category_expanded = self.category_embeddings[None, :, :, :]  # (1, 5, 5, 128)

        # 코사인 유사도 계산
        cosine_sim = F.cosine_similarity(x_expanded, category_expanded, dim=-1)  # (64, 5, 5)

        # **학습 중**에는 선택된 분점(leaf) 카운트 업데이트
        if is_training:
            with torch.no_grad():
                _, max_indices = torch.max(cosine_sim, dim=2)  # (64, 5) 각 카테고리에서 가장 높은 유사도를 가진 leaf index
                for category_idx in range(self.num_categories):
                    leaf_counts = torch.bincount(max_indices[:, category_idx], minlength=self.num_leafs_per_category)
                    self.leaf_usage_counts[category_idx] += leaf_counts.float()

        # **평가 시**에는 min_samples_leaf 기준으로 필터링
        if not is_training:
            valid_leaves = self.leaf_usage_counts >= self.min_samples_leaf  # (5,5) Boolean mask
            valid_cosine_sim = cosine_sim.clone()
            valid_cosine_sim[~valid_leaves[None, :, :]] = -float("inf")  # 무효 leaf는 제거
            cosine_sim_max, _ = torch.max(valid_cosine_sim, dim=2)  # (64, 5)
        else:
            cosine_sim_max, _ = torch.max(cosine_sim, dim=2)  # (64, 5)

        return cosine_sim_max