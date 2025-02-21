from torch.optim import AdamW
from transformers import get_scheduler
from sentence_transformers import losses, evaluation

# 학습 파라미터 설정
epochs = 30  # Early Stopping을 고려한 넉넉한 값
learning_rate = 3e-5
weight_decay = 0.01
warmup_ratio = 0.06  # 전체 스텝의 6%를 warmup으로 설정
total_training_steps = epochs * 144  # steps_per_epoch = 144
warmup_steps = int(total_training_steps * warmup_ratio)

# Optimizer 설정
optimizer = AdamW(emb_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning Rate Scheduler 설정
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps
)

# Early Stopping을 위한 Evaluator 설정
evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1=val_sentences1,  # 평가 데이터 (문장 1 리스트)
    sentences2=val_sentences2,  # 평가 데이터 (문장 2 리스트)
    scores=val_labels,  # 유사도 레이블 (0~1 사이 값)
    batch_size=128
)

# SentenceTransformer 모델 학습
emb_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,  # Early Stopping을 위해 필요
    epochs=epochs,
    optimizer_class=AdamW,
    optimizer_params={'lr': learning_rate, 'eps': 1e-8, 'weight_decay': weight_decay},
    scheduler=scheduler,
    warmup_steps=warmup_steps,
    evaluation_steps=144,  # 한 epoch마다 평가
    early_stopping_patience=5,  # 5번 연속 개선되지 않으면 중단
    output_path="./best_model"  # 최적 모델 저장 경로
)