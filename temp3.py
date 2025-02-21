import torch
from transformers import Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer

# 모델 및 데이터 로드
model = SentenceTransformer("bert-base-uncased")

# 학습 파라미터 설정
training_args = TrainingArguments(
    output_dir="./best_model",
    evaluation_strategy="epoch",  # 매 epoch마다 평가
    save_strategy="epoch",  # 매 epoch마다 체크포인트 저장
    save_total_limit=3,  # 최근 3개의 체크포인트만 저장
    load_best_model_at_end=True,  # Early Stopping이 발생하면 최적 모델 로드
    metric_for_best_model="loss",  # Early Stopping 기준
    greater_is_better=False,  # Loss는 낮을수록 좋음
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=30,  # Early Stopping이 없으면 최대 30 epochs 학습
    warmup_ratio=0.06,  # 6% warmup
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",  # 일정 step마다 평가
    eval_steps=144,  # 한 epoch마다 평가
    save_steps=144,  # 모델 저장 주기
    load_best_model_at_end=True
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 학습 시작
trainer.train()