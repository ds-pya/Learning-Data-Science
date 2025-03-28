class TAEModel(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes, num_prototypes_per_class,
                 taxonomy_size, taxonomy_dim, margin=-0.1,
                 disp_weight=0.1, align_weight=0.1, prune_threshold=3,
                 prune_ratio=5.0, num_heads=3):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = num_classes * num_prototypes_per_class
        self.prototype_embeddings = nn.Parameter(torch.randn(self.total_prototypes, hidden_dim) * 0.02)

        self.pooler = MultiHeadPooling(hidden_dim, num_heads)
        self.taxonomy = TaxonomyEmbedding(taxonomy_size, taxonomy_dim)
        self.margin = margin
        self.disp_weight = disp_weight
        self.align_weight = align_weight
        self.prune_threshold = prune_threshold
        self.prune_ratio = prune_ratio
        self.num_heads = num_heads

        self.register_buffer('positive_usage', torch.zeros(self.total_prototypes))
        self.register_buffer('negative_usage', torch.zeros(self.total_prototypes))

    def forward(self, input_ids, attention_mask, labels=None, taxonomy_labels=None):
        # 1. 인코더 출력
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, L, H]
        semantic_vecs = self.pooler(hidden_states)  # [B, K, H]

        # 2. cosine similarity 계산
        proto_norm = F.normalize(self.prototype_embeddings, dim=1)  # [P, H]
        sem_norm = F.normalize(semantic_vecs, dim=2)  # [B, K, H]
        sims = torch.matmul(sem_norm, proto_norm.T)  # [B, K, P]
        sims = (1 + sims) / 2  # 정규화된 cosine similarity [0, 1]

        # 3. 카테고리별 최고 유사도 선택
        sims = sims.view(-1, self.num_heads, self.num_classes, self.num_prototypes_per_class)
        max_sims, _ = sims.max(dim=3)  # [B, K, C]
        final_sims, _ = max_sims.max(dim=1)  # [B, C]

        if labels is None:
            return final_sims  # inference

        # 4. margin filtering
        batch_size = labels.size(0)
        correct_scores = final_sims[torch.arange(batch_size), labels]
        predicted = final_sims.argmax(dim=1)
        pred_scores = final_sims[torch.arange(batch_size), predicted]
        margins = correct_scores - pred_scores
        mask = margins > self.margin

        if mask.sum() == 0:
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=final_sims.device)
            return final_sims, dummy_loss

        filtered_output = final_sims[mask]
        filtered_labels = labels[mask]

        # 5. Cross-entropy loss
        ce_loss = F.cross_entropy(filtered_output, filtered_labels)

        # 6. Prototype dispersion loss
        cosine_matrix = torch.matmul(proto_norm, proto_norm.T)
        dispersion_loss = -cosine_matrix.mean()

        # 7. Taxonomy alignment loss (optional)
        taxo_embed = self.taxonomy()  # [T, D]
        if taxonomy_labels is not None:
            label_embed = taxo_embed[taxonomy_labels]  # [B, D]
            pooled = semantic_vecs.mean(dim=1)  # [B, H]
            pooled = F.normalize(pooled, dim=1)
            label_embed = F.normalize(label_embed, dim=1)
            align_loss = 1 - (pooled * label_embed).sum(dim=1).mean()
        else:
            align_loss = torch.tensor(0.0, device=final_sims.device)

        total_loss = ce_loss + self.disp_weight * dispersion_loss + self.align_weight * align_loss

        # 8. 사용 통계 업데이트
        with torch.no_grad():
            sims_reshaped = sims.view(batch_size, self.num_heads, self.num_classes, self.num_prototypes_per_class)
            best_proto_ids = sims_reshaped.argmax(dim=3)  # [B, K, C]
            for i in range(batch_size):
                for c in range(self.num_classes):
                    for k in range(self.num_heads):
                        proto_idx = c * self.num_prototypes_per_class + best_proto_ids[i, k, c].item()
                        if c == labels[i].item():
                            self.positive_usage[proto_idx] += 1
                        elif c == predicted[i].item() and predicted[i] != labels[i]:
                            self.negative_usage[proto_idx] += 1

        return final_sims, total_loss