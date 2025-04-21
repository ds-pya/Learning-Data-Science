import torch
import torch.nn as nn
import os

class InferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.dropout = model.dropout
        self.classifier = model.mlp if hasattr(model, "mlp") else model.classifier

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

def export_ner_model_and_crf(model: nn.Module, export_dir: str, seq_len: int = 128):
    os.makedirs(export_dir, exist_ok=True)

    wrapper = InferenceWrapper(model).cpu().eval()

    dummy_input_ids = torch.ones(1, seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        os.path.join(export_dir, "ner_model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"}
        },
        opset_version=13
    )

    torch.save(model.crf.transitions.detach().cpu(), os.path.join(export_dir, "crf_transition_matrix.pt"))