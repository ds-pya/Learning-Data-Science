import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

class LoRAMultiHeadClassifier(nn.Module):
    def __init__(self, model_name, hidden_dim, num_labels_per_level):
        super().__init__()
        
        # Pre-trained embedding model
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        # LoRA 적용
        self.lora_config = {
            'r': 8,  # LoRA rank
            'lora_alpha': 32,
            'target_modules': ['query', 'key', 'value'],
            'lora_dropout': 0.1,
            'bias': 'none',
        }
        self.model = self._apply_lora(self.model, self.lora_config)

        embedding_dim = self.model.config.hidden_size

        # Multi-head linear classifier 정의
        self.heads = nn.ModuleDict({
            'level_1': nn.Linear(embedding_dim, num_classes_level_1),
            'level_2': nn.Linear(embedding_dim, num_classes_level_2),
            # 추가 레벨이 있다면 추가
        })

    def _apply_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)
        return embeddings

    def _apply_lora(self, model, lora_config):
        lora_model = model
        from peft import get_peft_model, LoraConfig, TaskType
        config = LoraConfig(
            task_type='FEATURE_EXTRACTION',
            inference_mode=False,
            r=lora_config['r'],
            lora_alpha=32,
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            target_modules=['query', 'key', 'value']
        )
        model = get_peft_model(model, config=config)
        return model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self._apply_pooling(outputs, attention_mask)

        logits = {level: head(embeddings) for level, head in self.heads.items()}
        return logits


# Example usage
num_classes_level_1 = 10  # 각 계층별 클래스 개수 정의
num_classes_level_2 = 30

model = YourLoRAMultiHeadModel(num_classes_level_1, num_classes_level_2)

# loss example (cross-entropy)
criterion = nn.CrossEntropyLoss()

embeddings = model(input_ids, attention_mask)
loss_level_1 = criterion(model.heads['level_1'](embeddings), labels_level_1)
loss_level_2 = criterion(model.heads['level_2'](embeddings), labels_level_2)

loss = loss_level_1 + loss_level_2  # weighted sum도 가능
