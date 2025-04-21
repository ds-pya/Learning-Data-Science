def tokenize_and_align_labels_from_sentence(examples):
    sentences = examples["sentence"]
    all_tokens = examples["tokens"]
    all_tags = examples["ner_tags"]

    batch_input_ids, batch_attention_mask, batch_labels = [], [], []

    for tokens, tags, sent in zip(all_tokens, all_tags, sentences):
        # 1. BIO span 복원 (char-level)
        spans = []
        offset = 0
        current_tag, start_idx = None, None
        for i, (char, tag_idx) in enumerate(zip(tokens, tags)):
            tag = klue_idx_to_our_tag[tag_idx]
            if tag == "O":
                if current_tag:
                    spans.append((start_idx, offset, current_tag))
                    current_tag, start_idx = None, None
            elif tag.startswith("B-"):
                if current_tag:
                    spans.append((start_idx, offset, current_tag))
                current_tag = tag[2:]
                start_idx = offset
            elif tag.startswith("I-") and current_tag:
                pass  # continue span
            offset += len(char)
        if current_tag:
            spans.append((start_idx, offset, current_tag))

        # 2. 모델 기준 토크나이징 + offset 추출
        tokenized = tokenizer(sent, return_offsets_mapping=True, truncation=True)
        offsets = tokenized.pop("offset_mapping")

        # 3. subword 단위로 BIO 라벨 재정렬
        aligned = []
        for start, end in offsets:
            if end == 0:
                aligned.append(-100)  # special token
                continue
            matched = None
            for span_start, span_end, ent_type in spans:
                if start >= span_start and end <= span_end:
                    prefix = "B-" if start == span_start else "I-"
                    matched = f"{prefix}{ent_type}"
                    break
            aligned.append(our_tag_to_id.get(matched, 0))  # default to "O"

        batch_input_ids.append(tokenized["input_ids"])
        batch_attention_mask.append(tokenized["attention_mask"])
        batch_labels.append(aligned)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }