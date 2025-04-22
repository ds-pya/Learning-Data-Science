from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

class NERModelWithCustomCRF(nn.Module):
    def __init__(self, model_name: str, num_labels: int, transition_mask: torch.Tensor = None):
        super().__init__()
        self.num_labels = num_labels

        # Load encoder with LoRA
        base_model = AutoModel.from_pretrained(model_name)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.encoder = get_peft_model(base_model, lora_config)
        hidden_size = self.encoder.config.hidden_size

        # Classifier
        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )

        # CRF
        self.crf = NeuralCRF(num_labels, batch_first=True)
        if transition_mask is not None:
            self.crf.set_transition_mask(transition_mask)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        logits = self.mlp(x)
        if labels is not None:
            return self.crf(logits, labels, attention_mask.bool())
        return logits  # for inference