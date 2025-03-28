def prune_prototypes(model: TAEModel):
    """
    positive/negative 사용 통계를 바탕으로 마스킹을 적용하는 pruning 함수
    """
    pos = model.positive_usage
    neg = model.negative_usage
    keep = torch.ones_like(pos).bool()

    for i in range(model.total_prototypes):
        pos_count = pos[i].item()
        neg_count = neg[i].item()
        if pos_count <= model.prune_threshold:
            keep[i] = False
        elif pos_count > 0 and (neg_count / pos_count) >= model.prune_ratio:
            keep[i] = False

    model.prototype_mask = keep  # [P] boolean mask
    print(f"[Pruning] Kept {keep.sum().item()} / {len(keep)} prototypes")

def reset_prototype_usage(model: TAEModel):
    """
    사용 통계 초기화
    """
    model.positive_usage.zero_()
    model.negative_usage.zero_()