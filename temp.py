def viterbi_decode_np(emissions, transition_matrix, mask=None):
    # emissions: (seq_len, num_labels)
    # transition_matrix: (num_labels, num_labels)
    # mask: (seq_len,), optional

    dp = np.full((seq_len, num_labels), -inf)
    backpointer = np.zeros((seq_len, num_labels), dtype=int)

    for t in 1..T:
        for curr_label in 0..C:
            dp[t][curr_label] = max over all prev_label (
                dp[t-1][prev] + transition[prev][curr] + emission[t][curr]
            )

    backtrack from best at dp[-1] → recover best_path