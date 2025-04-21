import numpy as np

def viterbi_decode_np(emissions: np.ndarray, transition_matrix: np.ndarray, mask: np.ndarray = None):
    """
    Viterbi decoding for a single sequence.

    Args:
        emissions: np.ndarray of shape (L, C) - emission scores
        transition_matrix: np.ndarray of shape (C, C) - transition[i][j] = score from i → j
        mask: np.ndarray of shape (L,), optional - 1 for valid positions, 0 for padding

    Returns:
        best_path: list of predicted label indices
    """
    seq_len, num_labels = emissions.shape

    # Initialize the score table and backpointers
    score = np.full((seq_len, num_labels), -np.inf, dtype=np.float32)
    backpointers = np.zeros((seq_len, num_labels), dtype=np.int32)

    # Step 1: Initialize first step
    score[0] = emissions[0]

    # Step 2: Dynamic programming
    for t in range(1, seq_len):
        for curr_label in range(num_labels):
            transition_scores = score[t - 1] + transition_matrix[:, curr_label]
            best_prev_label = np.argmax(transition_scores)
            score[t, curr_label] = transition_scores[best_prev_label] + emissions[t, curr_label]
            backpointers[t, curr_label] = best_prev_label

    # Step 3: Backtrace
    best_last_label = np.argmax(score[-1])
    best_path = [best_last_label]

    for t in range(seq_len - 1, 0, -1):
        best_last_label = backpointers[t, best_last_label]
        best_path.append(best_last_label)

    best_path.reverse()

    # If mask is provided, cut to valid length
    if mask is not None:
        valid_len = int(mask.sum())
        best_path = best_path[:valid_len]

    return best_path