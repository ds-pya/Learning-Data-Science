def relabel_with_offset_matching(tokens, ner_tags, sentence, tokenizer, label_map):
    # 1. 문장 단위 BIO 라벨 시퀀스 만들기
    tag_seq = [label_map(tag) for tag in ner_tags]
    char_labels = [0] * len(sentence)
    pos = 0
    for token, tag in zip(tokens, tag_seq):
        for _ in token:
            if pos < len(char_labels):
                char_labels[pos] = tag
            pos += 1

    # 2. 모델 토크나이저로 문장 토크나이징 + offset mapping
    tokenized = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True
    )
    offsets = tokenized["offset_mapping"]

    # 3. offset 기준으로 서브토큰 단위 라벨 정렬
    aligned_labels = []
    for (start, end) in offsets:
        if end == 0:
            aligned_labels.append(-100)
            continue

        window = char_labels[start:end]
        unique = list(set(window))

        if all(v == 0 for v in unique):
            aligned_labels.append(0)  # 전부 O
        elif len(set([v // 2 for v in unique if v != 0])) == 1:  # 동일 entity
            aligned_labels.append(min(unique))  # B 우선
        else:
            aligned_labels.append(0)  # 애매한 경우 O

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": aligned_labels
    }