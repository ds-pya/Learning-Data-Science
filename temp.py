import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig

class SentenceClassifier(nn.Module):
    def __init__(self, base_model_name, num_classes, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super(SentenceClassifier, self).__init__()
        
        # Pretrained transformer model 로드
        self.base_model = AutoModel.from_pretrained(base_model_name)
        
        # RoLA (Low-Rank Adaptation) 적용
        lora_config = LoraConfig(
            r=lora_r,  # Low-rank matrix 차원
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],  # Self-attention에 적용
            bias="none"
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        
        hidden_size = self.base_model.config.hidden_size

        # Fully Connected Layer 추가
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