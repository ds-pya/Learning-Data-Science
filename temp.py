def forward(self, emissions, tags, mask):
    if not self.batch_first:
        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        mask = mask.transpose(0, 1)

    B, L, C = emissions.shape
    transitions = self.transitions
    if self.transition_mask is not None:
        transitions = transitions + self.transition_mask

    # validate tags
    if (tags >= C).any() or (tags < 0).any():
        raise ValueError("Label index out of bounds in CRF.")

    score = torch.zeros(B, device=emissions.device)

    for t in range(L - 1):
        valid = mask[:, t] & mask[:, t + 1]
        valid_idx = valid.nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0:
            continue

        curr_tag = tags[valid_idx, t]
        next_tag = tags[valid_idx, t + 1]
        emit = emissions[valid_idx, t, :].gather(1, curr_tag.unsqueeze(1)).squeeze(1)
        trans = transitions[curr_tag, next_tag]
        score[valid_idx] += emit + trans

    # 마지막 토큰 emission 추가
    last_tag_idx = (mask.sum(1) - 1).clamp(min=0)
    batch_idx = torch.arange(B, device=emissions.device)
    last_tag = tags[batch_idx, last_tag_idx]
    last_emit = emissions[batch_idx, last_tag_idx, last_tag]
    score += last_emit

    log_Z = self._compute_log_partition(emissions, transitions, mask)
    return (log_Z - score).mean()