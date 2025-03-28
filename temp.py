def prune_prototypes(model: TAEModel):
    """
    모델 내부의 positive/negative 사용 통계를 기반으로
    프로토타입을 pruning합니다.
    """
    pos = model.positive_usage
    neg = model.negative_usage
    keep_mask = torch.ones_like(pos).bool()

    for i in range(model.total_prototypes):
        pos_count = pos[i].item()
        neg_count = neg[i].item()
        # 조건 1: 긍정 사용이 너무 적음
        if pos_count <= model.prune_threshold:
            keep_mask[i] = False
        # 조건 2: 부정 사용이 과도함 (비율 기반)
        elif pos_count > 0 and (neg_count / pos_count) >= model.prune_ratio:
            keep_mask[i] = False

    # 실제 pruning: 학습 가능한 파라미터로 다시 생성
    keep_indices = torch.where(keep_mask)[0]
    pruned_embeddings = model.prototype_embeddings.data[keep_indices]

    # 프로토타입 수 업데이트
    model.prototype_embeddings = nn.Parameter(pruned_embeddings)
    model.total_prototypes = len(keep_indices)
    model.positive_usage = model.positive_usage[keep_indices]
    model.negative_usage = model.negative_usage[keep_indices]

    print(f"Pruned to {len(keep_indices)} prototypes.")