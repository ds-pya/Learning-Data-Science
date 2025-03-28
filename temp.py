def forward(self, input_ids, attention_mask, labels=None):
    B = input_ids.size(0)
    P = self.total_prototypes
    C = self.num_classes
    K = self.num_heads

    # 1. 텍스트 인코딩
    hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, L, H]
    semantic_vecs = self.pooler(hidden_states)  # [B, K, H]
    sem_norm = F.normalize(semantic_vecs, dim=2)

    # 2. cosine 유사도 계산
    proto_norm = F.normalize(self.prototype_embeddings, dim=1)  # [P, H]
    sims = torch.matmul(sem_norm, proto_norm.T)  # [B, K, P]
    sims = (1 + sims) / 2  # normalize [0, 1]

    # 3. mask 적용
    sims[:, :, ~self.prototype_mask] = -1e9

    # 4. 카테고리별 최상 유사도 추출
    sims = sims.view(B, K, C, -1)
    max_sims, _ = sims.max(dim=3)  # [B, K, C]
    final_sims, _ = max_sims.max(dim=1)  # [B, C]

    if labels is None:
        return final_sims  # inference mode

    # 5. margin filtering
    correct_scores = final_sims[torch.arange(B), labels]
    pred_labels = final_sims.argmax(dim=1)
    pred_scores = final_sims[torch.arange(B), pred_labels]
    margin_mask = (correct_scores - pred_scores) > self.margin

    if margin_mask.sum() == 0:
        return final_sims, torch.tensor(0.0, requires_grad=True, device=final_sims.device)

    # 6. Cross-Entropy
    filtered_output = final_sims[margin_mask]
    filtered_labels = labels[margin_mask]
    ce_loss = F.cross_entropy(filtered_output, filtered_labels)

    # 7. Dispersion Loss (taxonomy-weighted)
    cosine_sim = torch.matmul(proto_norm, proto_norm.T)  # [P, P]
    disp_loss = -(self.taxonomy_dist * cosine_sim).mean()

    # 8. Taxonomy Alignment (프로토타입 평균 기반)
    aligned_vecs = semantic_vecs.mean(dim=1)  # [B, H]
    aligned_vecs = F.normalize(aligned_vecs, dim=1)

    target_protos = []
    for c in labels:
        start = c * self.num_prototypes_per_class
        end = start + self.num_prototypes_per_class
        class_protos = proto_norm[start:end]
        class_mean = class_protos.mean(dim=0)
        target_protos.append(class_mean)
    target_protos = torch.stack(target_protos, dim=0)  # [B, H]
    target_protos = F.normalize(target_protos, dim=1)

    align_loss = 1 - (aligned_vecs * target_protos).sum(dim=1).mean()

    # 9. 총 Loss
    total_loss = ce_loss + self.disp_weight * disp_loss + self.align_weight * align_loss

    # 10. 사용 통계 업데이트
    with torch.no_grad():
        sims_reshaped = sims.view(B, K, C, -1)
        best_proto_ids = sims_reshaped.argmax(dim=3)  # [B, K, C]
        for i in range(B):
            for c in range(C):
                for k in range(K):
                    proto_idx = c * self.num_prototypes_per_class + best_proto_ids[i, k, c].item()
                    if c == labels[i].item():
                        self.positive_usage[proto_idx] += 1
                    elif c == pred_labels[i].item() and pred_labels[i] != labels[i]:
                        self.negative_usage[proto_idx] += 1

    return final_sims, total_loss