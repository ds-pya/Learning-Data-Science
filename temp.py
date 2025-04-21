def tokenize_and_align_labels_conll_char_based(examples):
    all_tokens = examples["tokens"]       # 단어 단위 토큰 리스트
    all_tags = examples["ner_tags"]       # 단어 단위 라벨 인덱스

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for tokens, tags in zip(all_tokens, all_tags):
        sentence = " ".join(tokens)
        char_labels = [0] * len(sentence)

        # 각 토큰을 문자 단위로 라벨 확장
        pos = 0
        for token, tag_idx in zip(tokens, tags):
            tag = conll_idx_to_our_tag[tag_idx]     # 예: B-LOC
            label_id = our_tag_to_id[tag]           # 예: 3

            for i, ch in enumerate(token):
                if pos + i < len(char_labels):
                    if i == 0 and tag.startswith("B-"):
                        char_labels[pos + i] = label_id
                    elif tag.startswith("I-") or (tag.startswith("B-") and i > 0):
                        char_labels[pos + i] = label_id + 1  # I-tag = B+1
            pos += len(token) + 1  # 단어 사이의 공백 포함

        # 모델 tokenizer 적용
        tokenized = tokenizer(
            sentence,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=True
        )
        offsets = tokenized["offset_mapping"]

        # offset 기반 라벨 재정렬
        aligned_labels = []
        for start, end in offsets:
            if end == 0:
                aligned_labels.append(-100)
                continue

            window = char_labels[start:end]
            unique = list(set(window))

            if all(v == 0 for v in unique):
                aligned_labels.append(0)
            elif len(set([v // 2 for v in unique if v != 0])) == 1:
                aligned_labels.append(min(unique))  # B 우선
            else:
                aligned_labels.append(0)  # 불일치는 O로 처리

        batch_input_ids.append(tokenized["input_ids"])
        batch_attention_mask.append(tokenized["attention_mask"])
        batch_labels.append(aligned_labels)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }