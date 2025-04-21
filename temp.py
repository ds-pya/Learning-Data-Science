def run_inference_onnx_with_crf(
    text: str,
    onnx_path: str,
    crf_path: str,
    model_name: str,
    label_list: list,
    max_length: int = 128
):
    import torch
    import onnxruntime
    import numpy as np
    from transformers import AutoTokenizer

    # Load tokenizer and CRF matrix
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transitions = torch.load(crf_path).numpy()

    # Tokenize input
    enc = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Run ONNX
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    emissions = ort_session.run(None, ort_inputs)[0]  # shape: (1, L, C)
    emissions = emissions[0]  # (L, C)
    mask = attention_mask[0]

    # Decode using Viterbi
    decoded = viterbi_decode_np(emissions, transitions, mask)

    # Get tokens and predicted labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    results = [(tok, label_list[tag]) for tok, tag, m in zip(tokens, decoded, mask) if m == 1]
    return results