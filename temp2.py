from datasets import load_dataset
from transformers import AutoTokenizer

# tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 우리의 라벨 체계
our_labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
our_tag_to_id = {tag: i for i, tag in enumerate(our_labels)}

# conll 라벨 → our 라벨로 매핑
def conll_tag_to_our_tag(tag):
    if tag == "O":
        return "O"
    prefix, ent_type = tag.split("-")
    ent_type = ent_type if ent_type in ["PER", "LOC", "ORG"] else "MISC"
    return f"{prefix}-{ent_type}"

# 전처리 함수
def tokenize_and_align_labels_conll(examples):
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding=False)
    all_labels = []

    for i, word_ids in enumerate(tokenized.word_ids(batch_index=i) for i in range(len(examples["tokens"]))):
        labels = examples["ner_tags"][i]
        aligned = []
        prev = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != prev:
                tag = conll_tag_to_our_tag(conll_label_list[labels[word_id]])
                aligned.append(our_tag_to_id[tag])
            else:
                tag = conll_tag_to_our_tag(conll_label_list[labels[word_id]]).replace("B-", "I-")
                aligned.append(our_tag_to_id[tag])
            prev = word_id
        all_labels.append(aligned)

    tokenized["labels"] = all_labels
    return tokenized

# 로딩 및 처리
conll_dataset = load_dataset("conll2003")
conll_label_list = conll_dataset["train"].features["ner_tags"].feature.names
tokenized_conll = conll_dataset.map(tokenize_and_align_labels_conll, batched=True, remove_columns=conll_dataset["train"].column_names)