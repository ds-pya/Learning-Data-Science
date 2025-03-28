import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes, num_prototypes_per_class, margin=-0.1, loss_weight=0.1):
        super().__init__()
        self.encoder = encoder  # 사전학습 텍스트 인코더 (LoRA 적용된 상태로 전달)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = num_classes * num_prototypes_per_class
        self.margin = margin
        self.loss_weight = loss_weight

        # (num_classes * num_prototypes_per_class, hidden_dim)
        self.prototype_embeddings = nn.Parameter(
            torch.randn(self.total_prototypes, hidden_dim) * 0.02
        )

        # 프로토타입 사용 기록용 변수
        self.register_buffer('positive_usage', torch.zeros(self.total_prototypes))
        self.register_buffer('negative_usage', torch.zeros(self.total_prototypes))

    def forward(self, input_ids, attention_mask, labels=None):
        # [B, hidden_dim]
        with torch.no_grad():  # LoRA 적용된 인코더의 경우 fine-tune 시 여기 해제
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

        # 거리 계산: [B, total_prototypes]
        dists = torch.cdist(hidden, self.prototype_embeddings)  # Euclidean
        sims = -dists  # 유사도는 마이너스 거리

        # 카테고리별로 가장 가까운 프로토타입만 선택
        sims_per_class = sims.view(-1, self.num_classes, self.num_prototypes_per_class)
        topk_sims, topk_indices = sims_per_class.max(dim=2)  # [B, num_classes]

        output = topk_sims  # [B, num_classes]

        if labels is None:
            return output  # inference

        # margin filtering (contrastive learning)
        batch_size = labels.size(0)
        correct_class_scores = output[torch.arange(batch_size), labels]
        predicted_labels = output.argmax(dim=1)
        predicted_scores = output[torch.arange(batch_size), predicted_labels]

        margins = correct_class_scores - predicted_scores
        mask = margins > self.margin  # True: 학습 대상

        if mask.sum() == 0:
            return output, torch.tensor(0.0, requires_grad=True, device=output.device)

        filtered_output = output[mask]
        filtered_labels = labels[mask]

        # cross-entropy loss
        ce_loss = F.cross_entropy(filtered_output, filtered_labels)

        # prototype dispersion loss
        proto_norm = F.normalize(self.prototype_embeddings, p=2, dim=1)
        proto_distances = 1 - torch.matmul(proto_norm, proto_norm.T)
        dispersion_loss = -proto_distances.mean()

        total_loss = ce_loss + self.loss_weight * dispersion_loss

        # 사용 기록 업데이트
        with torch.no_grad():
            flat_indices = topk_indices.view(-1)  # B * num_classes
            for i in range(batch_size):
                for c in range(self.num_classes):
                    proto_id = c * self.num_prototypes_per_class + topk_indices[i, c].item()
                    if c == labels[i].item():
                        self.positive_usage[proto_id] += 1
                    elif c == predicted_labels[i].item() and predicted_labels[i] != labels[i]:
                        self.negative_usage[proto_id] += 1

        return output, total_loss