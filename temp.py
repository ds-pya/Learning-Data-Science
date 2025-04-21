def relabel_with_offset_matching_batch(examples):
    sentences = examples["sentence"]
    tokens_list = examples["tokens"]
    tags_list = examples["ner_tags"]

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for tokens, tags, sentence in zip(tokens_list, tags_list, sentences):
        # 1. char-level BIO 라벨 시퀀스 구성
        tag_names = [klue_idx_to_our_tag[t] for t in tags]
        tag_ids = [our_tag_to_id[t] for t in tag_names]
        char_labels = [0] * len(sentence)

        pos = 0
        for token, tag in zip(tokens, tag_ids):
            for _ in token:
                if pos < len(char_labels):
                    char_labels[pos] = tag
                pos += 1

        # 2. 토크나이징 + offset
        tokenized = tokenizer(
            sentence,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=True
        )
        offsets = tokenized["offset_mapping"]

        # 3. offset -> sub-token 단위 라벨 재정렬
        aligned = []
        for start, end in offsets:
            if end == 0:
                aligned.append(-100)
                continue

            window = char_labels[start:end]
            unique = list(set(window))

            if all(v == 0 for v in unique):
                aligned.append(0)
            elif len(set([v // 2 for v in unique if v != 0])) == 1:
                aligned.append(min(unique))
            else:
                aligned.append(0)  # 애매하면 O

        batch_input_ids.append(tokenized["input_ids"])
        batch_attention_mask.append(tokenized["attention_mask"])
        batch_labels.append(aligned)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }