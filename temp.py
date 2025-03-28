import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, num_heads)

    def forward(self, hidden_states):  # [B, L, H]
        attn_scores = self.attn(hidden_states)  # [B, L, K]
        attn_scores = F.softmax(attn_scores, dim=1)
        out = torch.einsum('blh,blk->bkh', hidden_states, attn_scores)  # [B, K, H]
        return out

class TAEModel(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes, num_prototypes_per_class,
                 margin=-0.1, disp_weight=0.1, align_weight=0.1, num_heads=3,
                 prune_threshold=3, prune_ratio=5.0, taxonomy_distance_matrix=None):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = num_classes * num_prototypes_per_class

        self.prototype_embeddings = nn.Parameter(torch.randn(self.total_prototypes, hidden_dim) * 0.02)
        self.pooler = MultiHeadPooling(hidden_dim, num_heads)

        self.margin = margin
        self.disp_weight = disp_weight
        self.align_weight = align_weight
        self.num_heads = num_heads
        self.prune_threshold = prune_threshold
        self.prune_ratio = prune_ratio

        self.register_buffer('positive_usage', torch.zeros(self.total_prototypes))
        self.register_buffer('negative_usage', torch.zeros(self.total_prototypes))
        self.register_buffer('prototype_mask', torch.ones(self.total_prototypes).bool())

        if taxonomy_distance_matrix is None:
            self.register_buffer('taxonomy_dist', torch.ones(self.total_prototypes, self.total_prototypes))
        else:
            self.register_buffer('taxonomy_dist', taxonomy_distance_matrix)