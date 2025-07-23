import numpy as np
import heapq

def topk_viterbi_decode_np(emissions, transitions, k=10, mask=None):
    """
    emissions: (seq_len, num_tags)
    transitions: (num_tags, num_tags)
    mask: (seq_len,) or None
    """
    seq_len, num_tags = emissions.shape

    # heap: [(-score, path)]  # use negative for max heap
    dp = [ [(-emissions[0][tag], [tag])] for tag in range(num_tags) ]

    for t in range(1, seq_len):
        new_dp = [[] for _ in range(num_tags)]
        for curr_tag in range(num_tags):
            candidates = []
            for prev_tag in range(num_tags):
                for prev_score, prev_path in dp[prev_tag]:
                    score = prev_score - transitions[prev_tag][curr_tag] - emissions[t][curr_tag]
                    path = prev_path + [curr_tag]
                    candidates.append((score, path))
            # 상위 k개 유지
            new_dp[curr_tag] = heapq.nsmallest(k, candidates)
        dp = new_dp

    # 종료점에서 상위 k개 선택
    all_paths = []
    for tag in range(num_tags):
        all_paths.extend(dp[tag])
    topk = heapq.nsmallest(k, all_paths)

    # 우도는 음수로 저장되어 있으므로 다시 반대로 변환
    topk = [(-score, path) for score, path in topk]
    return topk  # list of (score, path)