from datasets import concatenate_datasets, DatasetDict

# 기존 KLUE-NER, CoNLL2003 토큰화 결과
# tokenized_klue = ...
# tokenized_conll = ...

# 각 split을 합치자 (train + validation)
merged_dataset = DatasetDict({
    "train": concatenate_datasets([tokenized_klue["train"], tokenized_conll["train"]]),
    "validation": concatenate_datasets([tokenized_klue["validation"], tokenized_conll["validation"]])
})

# DataLoader 재정의
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

collator = DataCollatorForTokenClassification(tokenizer)

train_loader = DataLoader(merged_dataset["train"], batch_size=32, shuffle=True, collate_fn=collator)
eval_loader = DataLoader(merged_dataset["validation"], batch_size=32, shuffle=False, collate_fn=collator)