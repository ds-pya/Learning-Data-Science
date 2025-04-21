NER Model (LoRA + CRF) 학습 전체 코드

import torch import torch.nn as nn from torch.utils.data import DataLoader from torch.optim import AdamW from datasets import load_dataset from transformers import AutoTokenizer, AutoModel, DataCollatorForTokenClassification from peft import get_peft_model, LoraConfig, TaskType from torchcrf import CRF from seqeval.metrics import classification_report from tqdm import tqdm

설정

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

라벨 정의 및 맵핑

our_labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"] our_tag_to_id = {tag: i for i, tag in enumerate(our_labels)}

klue_short_to_our = {"PS": "PER", "LC": "LOC", "OG": "ORG"}

def klue_tag_to_our_tag(tag): if tag == "O": return "O" prefix, ent_type = tag.split("-") return f"{prefix}-{klue_short_to_our.get(ent_type, 'MISC')}"

토크나이저 로딩

tokenizer = AutoTokenizer.from_pretrained(model_name)

데이터 로딩

dataset = load_dataset("klue", "ner", cache_dir="./klue_cache") klue_label_list = dataset["train"].features["ner_tags"].feature.names klue_idx_to_our_tag = {i: klue_tag_to_our_tag(t) for i, t in enumerate(klue_label_list)}

전처리 함수 (batched + 단일샘플 대응)

def tokenize_and_align_labels(examples): is_batched = isinstance(examples["tokens"][0], list) if not is_batched: examples = {k: [v] for k, v in examples.items()}

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
            tag = klue_idx_to_our_tag[labels[word_id]]
            aligned.append(our_tag_to_id[tag])
        else:
            tag = klue_idx_to_our_tag[labels[word_id]].replace("B-", "I-")
            aligned.append(our_tag_to_id[tag])
        prev = word_id
    all_labels.append(aligned)

tokenized["labels"] = all_labels
return tokenized if is_batched else {k: v[0] for k, v in tokenized.items()}

전체 전처리

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset["train"].column_names) data_collator = DataCollatorForTokenClassification(tokenizer) train_loader = DataLoader(tokenized_dataset["train"], batch_size=32, shuffle=True, collate_fn=data_collator) eval_loader = DataLoader(tokenized_dataset["validation"], batch_size=32, shuffle=False, collate_fn=data_collator)

모델 정의

class NERModelWithCRF(nn.Module): def init(self, model_name, num_labels): super().init() self.num_labels = num_labels base_model = AutoModel.from_pretrained(model_name) lora_config = LoraConfig( task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, ) self.encoder = get_peft_model(base_model, lora_config) self.dropout = nn.Dropout(0.1) self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels) self.crf = CRF(num_labels, batch_first=True)

def forward(self, input_ids, attention_mask, labels=None):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = self.dropout(outputs.last_hidden_state)
    emissions = self.classifier(sequence_output)

    if labels is not None:
        labels = labels.clone()
        labels[labels == -100] = 0  # dummy O tag
        loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
        return loss
    else:
        return self.crf.decode(emissions, mask=attention_mask.bool())

학습 및 평가 루프

def train(model, dataloader): model.train() total_loss = 0 for batch in tqdm(dataloader, desc="Training"): input_ids = batch["input_ids"].to(device) attention_mask = batch["attention_mask"].to(device) labels = batch["labels"].to(device)

loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    total_loss += loss.item()
print(f"\n📘 Avg Training Loss: {total_loss / len(dataloader):.4f}")

def evaluate(model, dataloader): model.eval() all_preds, all_labels = [], [] with torch.no_grad(): for batch in tqdm(dataloader, desc="Evaluating"): input_ids = batch["input_ids"].to(device) attention_mask = batch["attention_mask"].to(device) labels = batch["labels"].to(device)

preds = model(input_ids=input_ids, attention_mask=attention_mask)
        for pred, label, mask in zip(preds, labels, attention_mask):
            true_labels, pred_labels = [], []
            for p, l, m in zip(pred, label, mask):
                if m.item() == 1 and l.item() != -100:
                    true_labels.append(our_labels[l.item()])
                    pred_labels.append(our_labels[p])
            all_labels.append(true_labels)
            all_preds.append(pred_labels)
print("\n📊 Evaluation Result:")
print(classification_report(all_labels, all_preds, mode="strict"))

실행

model = NERModelWithCRF(model_name, num_labels=len(our_labels)).to(device) optimizer = AdamW(model.parameters(), lr=3e-5)

EPOCHS = 5 for epoch in range(EPOCHS): print(f"\n======= Epoch {epoch+1} =======") train(model, train_loader) evaluate(model, eval_loader)

