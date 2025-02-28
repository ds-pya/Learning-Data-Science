import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np

class HierarchicalMultiLabelDataset(Dataset):
    def __init__(self, examples, tokenizer, num_coarse_classes, num_fine_classes, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.num_coarse_classes = num_coarse_classes
        self.num_fine_classes = num_fine_classes
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        text = item["text"]
        # multi‑hot 벡터 생성 (해당하지 않으면 0)
        coarse_vec = torch.zeros(self.num_coarse_classes)
        for cl in item.get("coarse_labels", []):
            coarse_vec[cl] = 1.0
        fine_vec = torch.zeros(self.num_fine_classes)
        for fl in item.get("fine_labels", []):
            fine_vec[fl] = 1.0
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # 불필요한 배치 차원 제거
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding, coarse_vec, fine_vec

class HierarchicalMultiLabelModel(nn.Module):
    def __init__(self, model_name, num_coarse_classes, num_fine_classes, pooling="cls"):
        """
        Args:
            model_name: Hugging Face 모델
            num_coarse_classes: 상위 클래스 수
            num_fine_classes: 하위 클래스 수
            pooling: "cls" 또는 "mean" (문장 임베딩 선택 방식)
        """
        super(HierarchicalMultiLabelModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.base_model.config.hidden_size
        self.pooling = pooling.lower()  # "cls" 또는 "mean"
        # 두 개의 분류 헤드: 상위, 하위
        self.coarse_classifier = nn.Linear(self.embedding_dim, num_coarse_classes)
        self.fine_classifier = nn.Linear(self.embedding_dim, num_fine_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        if self.pooling == "mean":
            # Mean Pooling: attention_mask를 고려하여 평균을 계산
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            cls_embedding = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        else:
            # 기본적으로 [CLS] 토큰 (첫 번째 토큰) 사용
            cls_embedding = last_hidden_state[:, 0, :]
        coarse_logits = self.coarse_classifier(cls_embedding)
        fine_logits = self.fine_classifier(cls_embedding)
        return cls_embedding, coarse_logits, fine_logits

# 각 분류 헤드에 대해 BCEWithLogitsLoss를 적용합니다.
class HierarchicalMultiLabelLoss(nn.Module):
    def __init__(self, coarse_weight=1.0, fine_weight=1.0, consistency_weight=0.0):
        super(HierarchicalMultiLabelLoss, self).__init__()
        self.coarse_loss_fn = nn.BCEWithLogitsLoss()
        self.fine_loss_fn = nn.BCEWithLogitsLoss()
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
        self.consistency_weight = consistency_weight

    def forward(self, coarse_logits, fine_logits, coarse_labels, fine_labels):
        loss_coarse = self.coarse_loss_fn(coarse_logits, coarse_labels)
        loss_fine = self.fine_loss_fn(fine_logits, fine_labels)
        loss = self.coarse_weight * loss_coarse + self.fine_weight * loss_fine
        # 계층적 일관성 손실
        if self.consistency_weight > 0:
            consistency_loss = self.compute_consistency_loss(coarse_logits, fine_logits)
            loss += self.consistency_weight * consistency_loss
        return loss

    def compute_consistency_loss(self, coarse_logits, fine_logits):
        # 계층적 일관성 손실 정의
        return torch.tensor(0.0, device=coarse_logits.device)

def predict(model, sentence, device, threshold=0.5, max_length=128):
    """
    Args:
        model: 학습된 모델
        sentence: 입력 문장 (str)
        device: torch.device
        threshold: 이진화 임계값 (default 0.5)
        max_length: 토크나이징 시 최대 길이
    Returns:
        coarse_preds: 상위 클래스 예측 (0/1 numpy array)
        fine_preds: 하위 클래스 예측 (0/1 numpy array)
    """
    model.eval()
    encoding = model.tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        _, coarse_logits, fine_logits = model(input_ids, attention_mask)
    coarse_probs = torch.sigmoid(coarse_logits).squeeze(0).cpu().numpy()
    fine_probs = torch.sigmoid(fine_logits).squeeze(0).cpu().numpy()
    coarse_preds = (coarse_probs >= threshold).astype(int)
    fine_preds = (fine_probs >= threshold).astype(int)
    return coarse_preds, fine_preds

train_examples = [
    {
        "text": "This text does not belong to any predefined category.",
        "coarse_labels": [],
        "fine_labels": []
    }
]

num_coarse_classes = 3
num_fine_classes = 9
model_name = "all-MiniLM-L6-v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalMultiLabelModel(model_name, num_coarse_classes, num_fine_classes, pooling="cls")
model.to(device)

loss_fn = HierarchicalMultiLabelLoss(coarse_weight=1.0, fine_weight=1.0, consistency_weight=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

dataset = HierarchicalMultiLabelDataset(train_examples, model.tokenizer, num_coarse_classes, num_fine_classes, max_length=128)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs, coarse_labels, fine_labels = batch
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        coarse_labels = coarse_labels.to(device)
        fine_labels = fine_labels.to(device)
        
        optimizer.zero_grad()
        _, coarse_logits, fine_logits = model(input_ids, attention_mask)
        loss = loss_fn(coarse_logits, fine_logits, coarse_labels, fine_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "hierarchical_overlapping_multilabel_sentence_classifier.pt")

test_sentence = "Test"
coarse_preds, fine_preds = predict(model, test_sentence, device, threshold=0.5, max_length=128)
print("Predicted Coarse Labels:", coarse_preds)
print("Predicted Fine Labels:", fine_preds)
