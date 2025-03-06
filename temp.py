import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultiHeadLinearClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes_level_1, num_classes_level_2):
        super(MultiHeadLinearClassifier, self).__init__()
        
        # Load pretrained embedding model
        self.model = AutoModel.from_pretrained(pretrained_model_name)

        embedding_dim = self.model.config.hidden_size

        # Multi-head linear classifier
        self.heads = nn.ModuleDict({
            'level_1': nn.Linear(embedding_dim, num_classes_level_1),
            'level_2': nn.Linear(embedding_dim, num_classes_level_2),
            # 추가 레벨이 있으면 더 추가
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Simple mean pooling

        logits = {level: head(embeddings) for level, head in self.heads.items()}
        return logits

# Example usage
pretrained_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
num_classes_level_1 = 10
num_classes_level_2 = 30

model = MultiHeadLinearClassifier(pretrained_model_name, num_classes_level_1, num_classes_level_2)

# Example loss
criterion = nn.CrossEntropyLoss()
logits = model(input_ids, attention_mask)
loss_level_1 = criterion(logits['level_1'], labels_level_1)
loss_level_2 = criterion(logits['level_2'], labels_level_2)
loss = loss_level_1 + 0.5 * loss_level_2  # example weighted hierarchical loss
