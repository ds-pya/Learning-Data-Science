import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

class SentenceCLSModel(nn.Module):
    def __init__(self, base_model_name, num_classes=3, lora_r=8, lora_alpha=32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)

        # Apply LoRA to attention layers
        self.encoder = get_peft_model(self.encoder, LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        ))

        self.hidden_size = self.encoder.config.hidden_size

        # 3-layer classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(cls_emb)