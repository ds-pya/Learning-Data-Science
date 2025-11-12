Span-First Lightweight NER 설계 개요 (XLM-RoBERTa + Start/LenBin/Type)

0) 문제 스펙 & 목표

문장 길이: p50=25, p90=45, max=97 (짧은 문장 위주, 오른쪽 꼬리 존재)

문장당 엔티티 수: mean≈2, p90=4, max=18 (대부분 0~5개)

엔티티 길이(토큰): p50=3, p90=6, max=26 (롱테일 존재)

목표: CRF보다 가볍고, RPN보다 단순하면서 중첩/겹침에 유연한 스팬-퍼스트 NER

데이터 포맷:

{"title": "가수 김흥국이 왕십리에서 콘서트를 열었다.",
 "label": {"person": ["김흥국"], "location": ["왕십리"]}}



---

1) 백본 & 토크나이저

Backbone: xlm-roberta-base (한글 포함 멀티링구얼), 출력 last_hidden_state ∈ R^{B×T×H}

LoRA(옵션): q/k/v 모듈 대상, r=8, α=16, dropout=0.05 (리소스/데이터 적을 때 유리)

토크나이즈: HF offset mapping으로 char span → token span 정밀 매핑

특수토큰 제외(보통 첫/끝 한두 토큰)하고 실제 텍스트 토큰 인덱스로 변환

동일 표면형 다중 등장 시 모두 라벨링




---

2) 모델 헤드 (라이트 설계)

2.1 Start Head (token-wise 시작점)

Linear(H→1) → start_logits ∈ R^{B×T}

의미: 각 토큰이 엔티티 시작일 확률


2.2 Length Head (token-wise 길이 bin)

길이 bin 4개: [1], [2], [3–4], [5–6]

Linear(H→4) → len_logits ∈ R^{B×T×4}

학습 타깃: 시작 토큰에만 bin 라벨, 그 외 토큰은 ignore(−100)

설계 이유: p90=6, max=26 → 회귀 대신 다중 bin으로 경계 안정화, 최대 6으로 clip


2.3 Type Head (span-wise 타입)

(start,end) 구간의 mean pooling(경량/안정) → Linear(H→K)

type_logits ∈ R^{(#spans)×K}, K=엔티티 타입 수

샘플 구성: pos(정답 span) + neg(하드 네거티브; start=0 위치 주변의 대표길이 스팬 1–2개)



---

3) 후보 스팬 생성 (학습/추론 공통)

동적 Top-k 시작점: k = min(12, max(4, ceil(0.12·T)+2))

예) T=25→k=5, T=45→k=8, T=97→k=12


각 시작점 i, 각 길이 bin b의 대표 길이로 end 산출: repLen = [1,2,3,6]
→ 후보 (i, i+repLen[b]−1) (경계는 문장 길이에 맞춰 clip)



---

4) 손실 함수 (기본)

L = L_{\text{start}} + L_{\text{len}} + L_{\text{type}}

Len: CE + class-weight(롱테일 bin 보정; 예: w=1/log(1.1+freq) )

Type: CE + class-weight(타입 불균형 시)

(옵션) Start Soft Label: GT 시작=1.0, 이웃 토큰(±1)=0.3 → 경계 흔들림 완화

(옵션) Label smoothing: Type에 ε=0.05



---

5) “겹침 부분 점수(Partial Credit)” 확장 (선택)

단일 GT 대비 복수 예측 스팬이 있을 때, IoU 기반 가중치로 부분 보상:

1. GT span 와 예측 span 의 1D IoU 계산


2. 


3. 가중 CE:



L_{\text{type}}^{\text{partial}}=\sum_k w_k\cdot CE(\text{type}_k, y_{\text{type}})

4. 중복 억제(옵션): 스팬 간 overlap penalty 또는 토큰 합 Σ_k mask_k[t]에 대한 제약 추가



> 효과: 경계 일부만 맞아도 부분 크레딧을 부여 → recall/정밀 균형 개선
구현 난이도: 기존 Type loss 부근에 15~20줄 추가




---

6) 추론 & 후처리

스코어: 

1D NMS(IoU_thr=0.5, 동점 시 짧은 스팬 우선), max_out=10

중첩 허용 정책은 태스크별로 선택(허용/부분 제한/금지)



---

7) 학습 레시피(권장)

Max length: 128

Optimizer: AdamW — lr(backbone)=2e-5, lr(head)=1e-4, wd=0.01

Scheduler: Linear warmup 10%

Batch: 16–32, Epoch: 5–8 (early stopping by strict span-F1)

LoRA: 처음엔 백본 고정+헤드 학습 → 안정화 후 LoRA만 풀어 미세조정

네거티브 샘플링: 문장당 1–2개 hard negative span



---

8) 평가 지표

Strict span-F1: (start,end,type) 완전 일치

Lenient span-F1(±1): 경계 ±1 허용 (p90=6/롱테일 안정성 점검)

길이 bin 혼동 행렬: B2↔B3, B3↔B4(삭제 전 설계) 경계 오류 파악



---

9) 실무 체크리스트

토큰 오프셋 매핑 검증(특수토큰 제외, 중복 표면형 처리)

길이 bin 클래스 가중치 업데이트(배치/에폭 단위 재추정 가능)

동적 Top-k 모니터링(평균 후보 수, 리콜 영향)

NMS 결과 평균 IoU/겹침율 로깅

부분 점수 적용 시 stop-gradient로 안정성 확보



---

10) 간단 의사코드 스니펫

# 후보 생성
k = min(12, max(4, math.ceil(0.12*T)+2))
start_prob = sigmoid(start_logits)             # (B,T)
topk_idx = start_prob.topk(k, dim=-1).indices  # per batch

# 길이 bin 확장
cands = []
for i in topk_idx[b]:
    for b in range(4):                        # bins: [1],[2],[3-4],[5-6]
        L = [1,2,3,6][b]
        s, e = i, min(i+L-1, T-1)
        span_feat = H[b, s:e+1].mean(0)
        type_prob = softmax(type_head(span_feat), -1).max()
        score = start_prob[b, i] * softmax(len_logits[b, i], -1)[b] * type_prob
        cands.append((s, e, score))

# NMS
keep = nms_1d(cands, iou_thr=0.5, prefer_short=True, max_out=10)


---

11) 왜 이 설계가 “이 데이터”에 맞는가

bin4 삭제 & 6 clip: p90=6을 정확히 커버, 롱테일은 짧은 대표길이로 근사 → 과설계 방지

동적 Top-k: 평균 2개, p90 4개인 분포에 맞춰 후보 폭발 억제

Focal start + class-weighted len/type: 시작 희소성 & 길이/타입 불균형 동시 대응

Partial credit(옵션): 경계 일부만 맞는 사례에서 학습 신호 손실 최소화



---

필요하면 위 설계를 바로 쓸 수 있는 미니 학습 루프/데이터셋 코드도 파트별로 잘라서 드릴게요.