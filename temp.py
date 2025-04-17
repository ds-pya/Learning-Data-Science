import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import numpy as np

# 1. Dataset 정의
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_pad_token_id=-100, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, label_seq = self.data[idx]
        encoding = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            is_split_into_words=False
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 라벨 padding (길이가 tokenized input보다 짧으면 -100으로 패딩)
        label_ids = label_seq[:self.max_length] + [self.label_pad_token_id] * (self.max_length - len(label_seq))
        label_ids = torch.tensor(label_ids[:self.max_length])
        return input_ids, attention_mask, label_ids


# 2. 모델 정의
class TokenClassifierWithLoRA(nn.Module):
    def __init__(self, base_model_name, num_labels, lora_r=8):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model_name)
        base_model = AutoModel.from_pretrained(base_model_name, config=config)

        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            inference_mode=False
        )
        self.encoder = get_peft_model(base_model, lora_config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        logits = self.classifier(self.dropout(token_embeddings))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss


# 3. 학습 루프
def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        logits, loss = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# 4. 실행 예시
if __name__ == "__main__":
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    num_labels = 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 예시 데이터
    data = [
        ("손흥민은 토트넘 소속이다.", [1, 0, 0, 2, 0, 0]),  # 라벨: [B-PER, O, O, B-ORG, O, O]
        ("이강인은 파리 생제르맹 소속이다.", [1, 0, 2, 2, 2, 0, 0])
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = NERDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = TokenClassifierWithLoRA(model_name, num_labels)
    model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 3)

    for epoch in range(3):
        loss = train(model, dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")