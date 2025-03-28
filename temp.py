import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PrototypeGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_layers=2):
        super().__init__()
        self.node_features = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.02)
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, edge_index):
        x = self.node_features
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x  # (num_nodes, hidden_dim)


class HierarchicalPrototypeClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes, num_prototypes_per_class,
                 edge_index, margin=-0.1, loss_weight=0.1, gcn_layers=2):
        super().__init__()
        self.encoder = encoder  # 사전학습 인코더 (LoRA 포함)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = num_classes * num_prototypes_per_class
        self.margin = margin
        self.loss_weight = loss_weight
        self.edge_index = edge_index  # (2, num_edges), torch.long

        # GCN으로 프로토타입 생성
        self.prototype_gcn = PrototypeGCN(self.total_prototypes, hidden_dim, gcn_layers)

        # 사용 통계 기록
        self.register_buffer('positive_usage', torch.zeros(self.total_prototypes))
        self.register_buffer('negative_usage', torch.zeros(self.total_prototypes))

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.size(0)

        # 인코더 출력: [B, hidden_dim]
        with torch.no_grad():  # LoRA 적용했으면 열어도 됨
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

        # GCN을 통해 프로토타입 임베딩 생성
        proto_embeddings = self.prototype_gcn(self.edge_index)  # [P, hidden_dim]

        # Cosine 유사도 계산: [B, P]
        hidden_norm = F.normalize(hidden, p=2, dim=1)
        proto_norm = F.normalize(proto_embeddings, p=2, dim=1)
        cos_sim = torch.matmul(hidden_norm, proto_norm.T)  # [-1, 1]
        sim_scores = (1 + cos_sim) / 2  # [0, 1]

        # 카테고리별 최상 유사도 프로토타입 선택
        sim_per_class = sim_scores.view(batch_size, self.num_classes, self.num_prototypes_per_class)
        topk_sim, topk_idx = sim_per_class.max(dim=2)  # [B, num_classes]

        # 예측 score: [B, num_classes]
        output = topk_sim

        if labels is None:
            return output

        # Margin filtering
        true_scores = output[torch.arange(batch_size), labels]
        pred_labels = output.argmax(dim=1)
        pred_scores = output[torch.arange(batch_size), pred_labels]
        margin_values = true_scores - pred_scores
        mask = margin_values > self.margin

        if mask.sum() == 0:
            return output, torch.tensor(0.0, requires_grad=True, device=output.device)

        filtered_output = output[mask]
        filtered_labels = labels[mask]

        # Cross-entropy loss
        ce_loss = F.cross_entropy(filtered_output, filtered_labels)

        # Dispersion loss: prototype 임베딩 간 퍼짐 유도
        pairwise_sim = torch.matmul(proto_norm, proto_norm.T)
        dispersion_loss = -pairwise_sim.mean()

        total_loss = ce_loss + self.loss_weight * dispersion_loss

        # Positive / negative 사용 기록 업데이트
        with torch.no_grad():
            flat_indices = topk_idx.view(-1)
            for i in range(batch_size):
                for c in range(self.num_classes):
                    proto_id = c * self.num_prototypes_per_class + topk_idx[i, c].item()
                    if c == labels[i].item():
                        self.positive_usage[proto_id] += 1
                    elif c == pred_labels[i].item() and pred_labels[i] != labels[i]:
                        self.negative_usage[proto_id] += 1

        return output, total_loss