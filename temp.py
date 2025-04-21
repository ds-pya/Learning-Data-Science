import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from torchcrf import CRF

class NERModelWithCRF(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()

        self.num_labels = num_labels

        # Base encoder (no labels passed!)
        base_model = AutoModel.from_pretrained(model_name)

        # LoRA config (safe defaults, and no init args passed)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # NOT TOKEN_CLS!
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        self.encoder = get_peft_model(base_model, lora_config)

        # Token classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

        # CRF Layer
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # No labels passed to encoder!
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = 0  # mask용 dummy "O" 라벨
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction="mean")
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())