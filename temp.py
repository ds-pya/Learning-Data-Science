from olive.workflows import run as olive_run
config = {
  "input_model": {
    "type": "HfInferenceModel",
    "model_path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "io_config": {
      "input_names": ["input_ids", "attention_mask"],
      "input_shapes": [[1, 128], [1, 128]],
      "output_names": ["last_hidden_state"],
      "dynamic_axes": {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"}
      }
    }
  },
  "use_ort_genai": True,
  "optimization": [...],
  "output_path": "base_model_dir"
}
olive_run(config)