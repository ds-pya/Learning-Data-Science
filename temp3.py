import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig

class LoRACategoryEmbeddingClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_categories, embedding_per_category, embedding_dim, lora_r=8):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(pretrained_model_name)
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=['query', 'key', 'value'],
            task_type="FEATURE_EXTRACTION"
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        
        # Category Embedding Layer
        self.category_embeddings = nn.ParameterDict({
            'level_1': nn.Parameter(torch.randn(num_classes_level_1, embedding_dim)),
            'level_2': nn.Parameter(torch.randn(num_classes_level_2, embedding_dim))
            # Add more levels if necessary
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Simple pooling, adjust if needed

        # Calculate similarity to category embeddings
        logits = {}
        for level, category_embeddings in self.category_embeddings.items():
            # Compute cosine similarity
            similarity = torch.matmul(
                outputs.unsqueeze(1), 
                category_embeddings.T.unsqueeze(0)
            ).squeeze(1)
            logits[level] = similarity

        return logits

# Instantiate model
model = LoRAMultiHeadClassifier(
    pretrained_model_name='sentence-transformers/all-MiniLM-L6-v2',
    num_classes_level_1=10,
    num_classes_level_2=30
)

# Example loss computation
criterion = nn.CrossEntropyLoss()
logits = model(input_ids, attention_mask)

loss_level_1 = criterion(logits['level_1'], labels_level_1)
loss_level_2 = criterion(logits['level_2'], labels_level_2)

loss = loss_level_1 + 0.5 * loss_level_2  # Example of weighted hierarchical loss
