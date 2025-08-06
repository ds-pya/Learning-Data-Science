import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

class MiniLM_LoRA_SpanNER_ONNX(nn.Module):
    def __init__(
        self,
        encoder_name="paraphrase-multilingual-MiniLM-L12-v2",
        num_entity_types=7,
        entity_token_len=16,
        n_max=10,
        start_threshold=0.5
    ):
        super().__init__()
        # 1. LoRA 적용된 encoder
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "value"],
            lora_dropout=0.1, bias="none", task_type=TaskType.TOKEN_CLS
        )
        base_encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder = get_peft_model(base_encoder, lora_config)

        self.hidden_size = base_encoder.config.hidden_size  # 384
        self.entity_token_len = entity_token_len
        self.n_max = n_max
        self.start_threshold = start_threshold

        # 2. Token binary classifier
        self.entity_start_head = nn.Linear(self.hidden_size, 1)

        # 3. Entity type classifier
        self.entity_type_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_entity_types)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        start_labels=None,
        entity_labels=None
    ):
        """
        input_ids: [B, T]
        attention_mask: [B, T]
        start_labels: [B, T] (optional, for training)
        entity_labels: [B, N] (optional, for training)
        """
        B, T = input_ids.shape

        # ===== 1. Encode tokens =====
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = encoder_out.last_hidden_state  # [B, T, H]

        # ===== 2. Token binary classification =====
        start_logits = self.entity_start_head(token_embeddings).squeeze(-1)  # [B, T]

        loss = torch.tensor(0.0, device=input_ids.device)
        if start_labels is not None:
            bce_loss = nn.BCEWithLogitsLoss()
            loss = loss + bce_loss(start_logits, start_labels.float())

        # ===== 3. Top-k entity start positions =====
        topk_vals, topk_indices = torch.topk(start_logits, k=self.n_max, dim=1)  # [B, N]
        entity_mask = (topk_vals.sigmoid() > self.start_threshold).float()       # [B, N]

        # ===== 4. Gather entity token spans =====
        # batch index for gather
        batch_idx = torch.arange(B, device=input_ids.device).unsqueeze(1).expand(B, self.n_max)  # [B, N]

        # shape: [B, N, entity_token_len, H]
        # offset broadcast
        offsets = torch.arange(self.entity_token_len, device=input_ids.device).view(1, 1, -1)  # [1, 1, L]
        span_positions = topk_indices.unsqueeze(-1) + offsets  # [B, N, L]
        span_positions = torch.clamp(span_positions, max=T - 1)

        # gather embeddings
        batch_idx_exp = batch_idx.unsqueeze(-1).expand(B, self.n_max, self.entity_token_len)  # [B, N, L]
        span_embeddings = token_embeddings[batch_idx_exp, span_positions]  # [B, N, L, H]

        # ===== 5. Mean pooling for entity representation =====
        entity_repr = span_embeddings.mean(dim=2)  # [B, N, H]

        # ===== 6. Entity type classification =====
        type_logits = self.entity_type_head(entity_repr)  # [B, N, C]

        if entity_labels is not None:
            ce_loss = nn.CrossEntropyLoss(reduction="none")  # no reduction
            raw_loss = ce_loss(type_logits.view(-1, type_logits.size(-1)), entity_labels.view(-1))  # [B*N]
            masked_loss = raw_loss * entity_mask.view(-1)
            loss = loss + masked_loss.sum() / (entity_mask.sum() + 1e-5)

        return {
            "start_logits": start_logits,  # [B, T]
            "type_logits": type_logits,    # [B, N, C]
            "entity_mask": entity_mask,    # [B, N]
            "loss": loss if (start_labels is not None or entity_labels is not None) else None
        }