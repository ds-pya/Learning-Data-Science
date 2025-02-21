import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 모델 및 데이터 로드
model = SentenceTransformer('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 하이퍼파라미터 설정
learning_rate = 3e-5
weight_decay = 0.01
num_epochs = 30
warmup_ratio = 0.1
total_steps = len(train_dataloader) * num_epochs  # 총 학습 스텝 수

# 옵티마이저 및 LR 스케줄러 설정
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def lr_lambda(current_step):
    if current_step < warmup_ratio * total_steps:
        return current_step / (warmup_ratio * total_steps)
    else:
        return max(0.0, (total_steps - current_step) / (total_steps * (1 - warmup_ratio)))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

early_stopping = EarlyStopping(patience=5)

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    print(f"Epoch {epoch + 1}/{num_epochs}")

    for batch in tqdm(train_dataloader):
        anchor, positive, negative, margin = batch
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        optimizer.zero_grad()

        # 임베딩 생성
        anchor_emb = model.encode(anchor, convert_to_tensor=True)
        positive_emb = model.encode(positive, convert_to_tensor=True)
        negative_emb = model.encode(negative, convert_to_tensor=True)

        # 손실 계산
        loss = train_loss(anchor_emb, positive_emb, negative_emb, margin)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_train_loss += loss.item()

    print(f"Training Loss: {total_train_loss / len(train_dataloader)}")

    # 평가 및 Early Stopping 체크
    model.eval()
    with torch.no_grad():
        val_loss = custom_evaluator(model, None, epoch, 0)
    print(f"Validation Loss: {val_loss}")
    
    if early_stopping(val_loss):
        print("Early stopping activated")
        break