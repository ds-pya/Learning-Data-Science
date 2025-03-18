import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import onnx
import onnxruntime

# 모델 및 토크나이저 로드
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 간단한 커스텀 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# 샘플 데이터 (실제 데이터로 대체 필요)
texts = ["이 문장은 예제입니다.", "이 모델은 다국어 지원을 합니다."]
labels = [0, 1]  # 9개의 클래스 중 일부
dataset = CustomDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 모델 정의
class SentenceClassifier(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super(SentenceClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        for param in self.base_model.parameters():
            param.requires_grad = False  # Base 모델 동결
        
        hidden_size = self.base_model.config.hidden_size
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS 토큰 출력 사용
        return self.fc_layers(pooled_output)

# 모델 초기화
num_classes = 9
model = SentenceClassifier(MODEL_NAME, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.fc_layers.parameters(), lr=1e-4)

# 학습 루프
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# 모델을 ONNX로 저장
dummy_input = {
    "input_ids": torch.randint(0, tokenizer.vocab_size, (1, 128)).to(device),
    "attention_mask": torch.ones((1, 128)).to(device),
}

onnx_path = "sentence_classifier.onnx"
torch.onnx.export(
    model, 
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
    opset_version=11
)

print(f"Model saved as {onnx_path}")