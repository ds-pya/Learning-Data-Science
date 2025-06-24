import torch
import torch.nn as nn
import math

# LoRA adapter 정의
class LoRAAdapter(nn.Module):
    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.lora_down = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, hidden_dim, bias=False)

        # 초기화: LoRA 논문 방식
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return x + self.lora_up(self.lora_down(x))


# 전체 모델 병렬 구조 정의
class ParallelLoRAHeads(nn.Module):
    def __init__(self, base_model, hidden_dim, rank, mlp1, mlp2):
        super().__init__()
        self.base_model = base_model  # shared encoder

        self.lora1 = LoRAAdapter(hidden_dim, rank)
        self.lora2 = LoRAAdapter(hidden_dim, rank)

        self.mlp1 = mlp1
        self.mlp2 = mlp2

    def forward(self, input_ids, attention_mask):
        # 공유 encoder 실행
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 각 LoRA + MLP 적용
        out1 = self.mlp1(self.lora1(base_output))
        out2 = self.mlp2(self.lora2(base_output))

        return out1, out2