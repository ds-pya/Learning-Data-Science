import torch

# ONNX용 dummy input
dummy_input_ids = torch.ones(1, 128, dtype=torch.long)
dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)

# 모델 평가 모드
model.eval()

# ONNX export
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    "ner_model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],  # 또는 decoded result
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"},
    },
    opset_version=13
)

import onnxruntime
from transformers import AutoTokenizer

# 로드
session = onnxruntime.InferenceSession("ner_model.onnx")

# 예시 문장
text = "홍길동은 서울에 살고 있습니다."
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
inputs = tokenizer(text, return_tensors="np", padding="max_length", max_length=128, truncation=True)

# ONNX 추론
ort_inputs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
}
logits = session.run(["logits"], ort_inputs)[0]  # shape: (1, 128, num_labels)

# 가장 높은 값 → 예측 라벨
predicted_ids = logits.argmax(-1)[0]  # shape: (128,)

# 정리해서 토큰과 같이 출력
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
for token, pred_id in zip(tokens, predicted_ids):
    print(f"{token}\t→\t{our_labels[pred_id]}")