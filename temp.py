import torch import torch.nn.functional as F from transformers import AutoTokenizer, AutoModel from typing import List, Tuple from tqdm import tqdm import os

=== CONFIG ===

TARGET_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" BLEND_MODEL_NAMES = [ "bge-m3-korean", "xlm-roberta-base", "bert-base-multilingual-uncased" ] BATCH_SIZE = 4 LEARNING_RATE = 3e-5 SAVE_PATH = "blended_model.bin" DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

=== DATA ===

sentences: List[str] = [...]  # 문장 5만 개 리스트를 여기에 삽입하거나 로딩하세요

=== HELPER FUNCTIONS ===

def get_word_spans(text: str) -> List[Tuple[int, int]]: spans = [] start = 0 for word in text.split(): end = start + len(word) spans.append((start, end)) start = end + 1 return spans

def pool_by_span(token_embs, token_spans, word_spans): pooled_embs = [] for word_start, word_end in word_spans: matched = [ i for i, (tok_start, tok_end) in enumerate(token_spans) if not (tok_end <= word_start or tok_start >= word_end) ] if matched: span_emb = token_embs[matched].mean(dim=0) else: span_emb = torch.zeros(token_embs.size(1), device=token_embs.device) pooled_embs.append(span_emb) return torch.stack(pooled_embs, dim=0)

def compute_similarity_matrix(x): x = F.normalize(x, dim=-1) return torch.matmul(x, x.transpose(-1, -2))

def aligned_embedding(text: str, tokenizer, model, word_spans, device="cpu"): encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True).to(device) offset_mapping = encoded.pop('offset_mapping')[0].tolist() with torch.no_grad(): outputs = model(**encoded).last_hidden_state[0] return pool_by_span(outputs, offset_mapping, word_spans)

=== LOAD MODELS ===

tokenizers = {name: AutoTokenizer.from_pretrained(name) for name in [TARGET_MODEL_NAME] + BLEND_MODEL_NAMES} blend_models = {name: AutoModel.from_pretrained(name).eval().cpu() for name in BLEND_MODEL_NAMES} target_model = AutoModel.from_pretrained(TARGET_MODEL_NAME).train().to(DEVICE)

optimizer = torch.optim.AdamW(target_model.parameters(), lr=LEARNING_RATE)



