from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# ====== 서버/모델 초기화 ======
app = FastAPI(title="ONNX NER + CRF API")

# 경로는 환경에 맞게 바꾸세요
MODEL_PATH = "app/model/model.onnx"
TRANS_PATH = "app/model/transitions.npy"   # shape: [num_labels, num_labels]
LABEL_PATH = "app/model/labels.txt"        # e.g., lines: "O", "B-PER", "I-PER", ...

# 토크나이저 (HF fast tokenizer 권장: word_ids() 제공)
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # MiniLM 계열 예시. 실제껄로 교체.

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

# ONNX Runtime 세션
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

# 라벨/전이행렬 로드
labels: List[str] = [l.strip() for l in open(LABEL_PATH, "r", encoding="utf-8")]
label2id = {l: i for i, l in enumerate(labels)}
num_labels = len(labels)
transitions: np.ndarray = np.load(TRANS_PATH).astype(np.float32)  # [L, L]

# ====== 입출력 모델 ======
class InferIn(BaseModel):
    text: str

class InferOut(BaseModel):
    result: Dict[str, List[str]]  # {type: [word, ...]}

# ====== 유틸 ======
def bio_decode_to_spans(tag_ids: List[int], tokens: List[str], word_ids: List[int]) -> Dict[str, List[str]]:
    """
    토큰 단위 BIO 태그 시퀀스를 단어 단위 엔티티로 변환.
    1) word_ids로 서브워드를 단어에 매핑
    2) 각 단어의 대표 태그 선택(가장 앞 서브워드의 태그를 사용)
    3) BIO 규칙으로 span 추출 후 {TYPE: [word]} 생성
    """
    # 1) 단어단위로 태그 구성 (대표 서브워드의 태그 사용)
    max_word_id = max([wid for wid in word_ids if wid is not None]) if any(w is not None for w in word_ids) else -1
    word_tags: List[str] = []
    word_texts: List[str] = []
    current_word_id = -1
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != current_word_id:
            # 새 단어 시작
            current_word_id = wid
            # 대표 서브워드의 태그/텍스트
            word_tags.append(labels[tag_ids[i]])
            word_texts.append(tokens[i])

    # 2) BIO 스팬을 타입별 리스트로
    entities: Dict[str, List[str]] = {}
    cur_type = None
    cur_words: List[str] = []

    def flush():
        nonlocal cur_type, cur_words
        if cur_type and cur_words:
            entities.setdefault(cur_type, []).append(" ".join(cur_words))
        cur_type, cur_words = None, []

    for tag, word in zip(word_tags, word_texts):
        if tag == "O":
            flush()
            continue
        # B-XXX / I-XXX
        if "-" in tag:
            bio, typ = tag.split("-", 1)
        else:
            bio, typ = "I", tag  # 비정형 안전처리

        if bio == "B":
            flush()
            cur_type = typ
            cur_words = [word]
        elif bio == "I":
            if cur_type == typ:  # 같은 엔티티 지속
                cur_words.append(word)
            else:
                # 잘못 붙은 I(타입 불일치) -> 새 엔티티로 취급
                flush()
                cur_type = typ
                cur_words = [word]
        else:
            # 알 수 없는 태그 -> 끊기
            flush()

    flush()
    return entities

def viterbi_decode(emissions: np.ndarray, transitions: np.ndarray, mask: np.ndarray) -> List[int]:
    """
    emissions: [T, L] (float32)
    transitions: [L, L]  (from -> to)
    mask: [T]  (1 valid, 0 pad)
    반환: best_path(List[int], 길이 T)
    """
    T, L = emissions.shape
    # 초기화
    dp = np.full((T, L), -1e9, dtype=np.float32)
    bp = np.zeros((T, L), dtype=np.int32)
    # t=0
    dp[0] = emissions[0]
    # 동적계획
    for t in range(1, T):
        if mask[t] == 0:
            dp[t] = dp[t-1]  # 패딩 구간
            bp[t] = np.arange(L)
            continue
        prev = dp[t-1].reshape(-1, 1) + transitions  # [L,1] + [L,L] => [L,L] (prev_label -> cur_label)
        bp[t] = np.argmax(prev, axis=0)
        dp[t] = np.max(prev, axis=0) + emissions[t]

    # 백트랙
    last_t = int(mask.sum() - 1)
    last_label = int(np.argmax(dp[last_t]))
    best_path = [0] * T
    best_path[last_t] = last_label
    for t in range(last_t, 0, -1):
        best_path[t-1] = int(bp[t, best_path[t]])
    return best_path

def run_inference(text: str) -> Dict[str, List[str]]:
    # 토크나이즈 (단, word_ids를 위해 fast tokenizer 필요)
    enc = tokenizer(
        text,
        return_tensors="np",          # onnxruntime 입력용 (numpy)
        return_attention_mask=True,
        return_offsets_mapping=False,
        return_token_type_ids=False,
        truncation=True,
        max_length=512,
        padding=False
    )
    input_ids = enc["input_ids"]        # [1, T]
    attn_mask = enc["attention_mask"]   # [1, T]
    T = input_ids.shape[1]

    # ONNX 추론
    # 모델의 입력/출력 이름은 export 시점에 따라 다를 수 있습니다.
    # 아래는 흔한 예시: "input_ids", "attention_mask" -> "logits"
    inputs = {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": attn_mask.astype(np.int64),
    }
    ort_out = session.run(None, inputs)
    # 토큰 분류 로짓: [1, T, L]
    logits = ort_out[0].astype(np.float32)
    emissions = logits[0]  # [T, L]

    # 마스크
    mask = attn_mask[0].astype(np.int32)  # [T]

    # Viterbi (CRF)
    tag_ids = viterbi_decode(emissions, transitions, mask)  # 길이 T

    # 단어단위 변환 (특수토큰/패딩 제외)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    word_ids = enc.word_ids()  # 길이 T, fast tokenizer만 지원 (None 포함)
    # [CLS], [SEP] 등의 None/특수토큰은 bio_decode 내부에서 word_ids(None)로 스킵됨
    entities = bio_decode_to_spans(tag_ids, tokens, word_ids)

    return entities

# ====== API ======
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/infer", response_model=InferOut)
def infer(req: InferIn):
    ents = run_inference(req.text)
    # { "type": [ "word", ... ] } 형태로 반환
    return {"result": ents}