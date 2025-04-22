import torch
import torch.nn as nn

class NeuralCRF(nn.Module):
    def __init__(self, num_labels, batch_first=True):
        super().__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))  # from_i → to_j
        self.transition_mask = None  # optional: (C, C) mask

    def set_transition_mask(self, mask: torch.Tensor):
        """mask: (num_labels, num_labels) where invalid transitions are -inf"""
        self.transition_mask = mask

    def forward(self, emissions, tags, mask):
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        B, L, C = emissions.shape
        transitions = self.transitions
        if self.transition_mask is not None:
            transitions = transitions + self.transition_mask

        # Score of the real path
        score = torch.zeros(B, device=emissions.device)
        for t in range(L - 1):
            curr_tag = tags[:, t]
            next_tag = tags[:, t + 1]
            emit = emissions[:, t, :].gather(1, curr_tag.unsqueeze(1)).squeeze(1)
            trans = transitions[curr_tag, next_tag]
            score += emit * mask[:, t] + trans * mask[:, t + 1]

        # Add final token's emission
        last_tag = tags.gather(1, (mask.sum(1) - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        last_emission = emissions[torch.arange(B), mask.sum(1).clamp(min=1) - 1, last_tag]
        score += last_emission

        log_Z = self._compute_log_partition(emissions, transitions, mask)
        return (log_Z - score).mean()

    def _compute_log_partition(self, emissions, transitions, mask):
        B, L, C = emissions.shape
        alpha = emissions[:, 0]  # (B, C)
        for t in range(1, L):
            emit = emissions[:, t].unsqueeze(2)        # (B, C, 1)
            trans = transitions.unsqueeze(0)           # (1, C, C)
            alpha_t = alpha.unsqueeze(1) + trans + emit  # (B, C, C)
            alpha = torch.logsumexp(alpha_t, dim=2)
            alpha = alpha * mask[:, t].unsqueeze(1) + alpha * (1 - mask[:, t].unsqueeze(1))
        return torch.logsumexp(alpha, dim=1)