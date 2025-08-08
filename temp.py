# 코드 실행 상태 초기화로 인해 필요한 함수 및 맵 재정의
from typing import List, Tuple
from collections import Counter

# BIO 인덱스 매핑
label_map = {
    0: 'O',
    1: 'B-PER', 2: 'I-PER',
    3: 'B-LOC', 4: 'I-LOC',
    5: 'B-ORG', 6: 'I-ORG',
    7: 'B-MISC', 8: 'I-MISC',
    9: 'O'  # 예외 처리용
}

def decode_entities(labels: List[int]) -> List[Tuple[int, int, str]]:
    entities = []
    start = None
    entity_type = None
    for i, label in enumerate(labels):
        tag = label_map[label]
        if tag == 'O':
            if start is not None:
                entities.append((start, i - 1, entity_type))
                start = None
                entity_type = None
        elif tag.startswith('B-'):
            if start is not None:
                entities.append((start, i - 1, entity_type))
            start = i
            entity_type = tag[2:]
        elif tag.startswith('I-'):
            if start is None or tag[2:] != entity_type:
                continue
        if tag.startswith('B-') or (tag.startswith('I-') and start is not None and entity_type is None):
            entity_type = tag[2:]
    if start is not None:
        entities.append((start, len(labels) - 1, entity_type))
    return entities

def analyze_case_2_exist_only(
    token_lists: List[List[str]],
    true_labels: List[List[int]],
    pred_labels: List[List[int]]
):
    true_only_counter = Counter()
    pred_only_counter = Counter()

    for tokens, t_labels, p_labels in zip(token_lists, true_labels, pred_labels):
        true_spans = decode_entities(t_labels)
        pred_spans = decode_entities(p_labels)

        true_span_set = {(s, e, t) for s, e, t in true_spans}
        pred_span_set = {(s, e, t) for s, e, t in pred_spans}

        only_true = true_span_set - pred_span_set
        only_pred = pred_span_set - true_span_set

        for _, _, t in only_true:
            true_only_counter[t] += 1
        for _, _, p in only_pred:
            pred_only_counter[p] += 1

    return {
        "2a_true_only_total": sum(true_only_counter.values()),
        "2b_pred_only_total": sum(pred_only_counter.values()),
        "2a_true_only_by_type": true_only_counter,
        "2b_pred_only_by_type": pred_only_counter
    }