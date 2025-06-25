import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM

class QwenSentenceClassifier(nn.Module):
    def __init__(self, model_name, num_labels, lora_config: LoraConfig):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = get_peft_model(self.base_model, lora_config)
        self.hidden_size = self.base_model.config.hidden_size

        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # shape: (B, T, H)

        # Use mean pooling (ignoring padding)
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        summed = (hidden * mask).sum(1)
        count = mask.sum(1).clamp(min=1e-6)
        pooled = summed / count  # (B, H)

        logits = self.mlp_head(pooled)  # (B, num_labels)
        return logits


# 모델 초기화
model_name = "Qwen/Qwen2.5-0.5B"
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "q_proj", "v_proj"],  # depends on Qwen internals
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = QwenSentenceClassifier(model_name, num_labels=3, lora_config=lora_config)

# dummy input
dummy_input = {
    "input_ids": torch.randint(0, 100, (1, 32)),
    "attention_mask": torch.ones(1, 32, dtype=torch.int64),
}

# Export
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "qwen_cls.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {1: "seq"}, "attention_mask": {1: "seq"}},
    opset_version=17
)