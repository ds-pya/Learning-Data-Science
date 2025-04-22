import numpy as np

def viterbi_decode_np(emissions: np.ndarray, transitions: np.ndarray, mask: np.ndarray):
    """
    Args:
        emissions: (L, C) np.ndarray - emission scores
        transitions: (C, C) np.ndarray - transition scores (from_i → to_j)
        mask: (L,) np.ndarray - 1 for valid token, 0 for padding

    Returns:
        List[int] - predicted label indices
    """
    L, C = emissions.shape
    score = emissions[0]  # (C,)
    backpointers = []

    for t in range(1, L):
        broadcast_score = score[:, np.newaxis] + transitions  # (C, C)
        best_prev = np.argmax(broadcast_score, axis=0)        # (C,)
        score = np.max(broadcast_score, axis=0) + emissions[t]
        backpointers.append(best_prev)

    best_last = np.argmax(score)
    best_path = [best_last]

    for t in reversed(range(L - 1)):
        best_last = backpointers[t][best_last]
        best_path.append(best_last)

    best_path.reverse()
    return best_path[:int(mask.sum())]