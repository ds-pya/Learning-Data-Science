import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel
from torchcrf import CRF
from peft import LoraConfig, get_peft_model

class XLMRobertaForNERWithCRFLoRA(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.backbone = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        targets = ["q_proj", "k_proj", "v_proj", "out_proj"]
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=targets, task_type="TOKEN_CLS")
        self.backbone = get_peft_model(self.backbone, cfg)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(out.last_hidden_state)
        emissions = self.classifier(x)
        mask = attention_mask.bool()
        mask[:, 0] = True
        result = {"logits": emissions}
        if labels is not None:
            tags = labels.clone().long()
            tags[~mask] = 0
            loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
            result["loss"] = loss
        preds = self.crf.decode(emissions, mask=mask)
        result["preds"] = preds
        return result