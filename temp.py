import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- CRF-RNN (dense mean-field) ----------
class CRFRNN(nn.Module):
    def __init__(self, num_labels, hidden_dim, T=5, use_pos_kernel=True, use_emb_kernel=True):
        super().__init__()
        self.num_labels = num_labels
        self.T = T
        # compatibility μ (학습 가능). 기본은 Potts(동라벨 선호)
        self.mu = nn.Parameter(torch.ones(num_labels, num_labels) - torch.eye(num_labels))

        # 커널 가중치/스케일
        self.w_pos = nn.Parameter(torch.tensor(1.0)) if use_pos_kernel else None
        self.w_emb = nn.Parameter(torch.tensor(1.0)) if use_emb_kernel else None
        self.log_sigma_pos = nn.Parameter(torch.log(torch.tensor(2.0)))
        self.log_sigma_emb = nn.Parameter(torch.log(torch.tensor(1.0)))

        # 토큰 임베딩을 커널용 저차 사영
        self.emb_proj = nn.Linear(hidden_dim, 64, bias=False)

    @torch.no_grad()
    def init_bio_constraints(self, id2label):
        """
        선택: BIO 합법성 반영 초기화.
        label 예시: B-PER, I-PER, O ...
        """
        L = self.num_labels
        mu = torch.ones(L, L)  # 디폴트: 서로 다른 라벨에 패널티(>0)
        for i in range(L):
            for j in range(L):
                a, b = id2label[i], id2label[j]
                if a == b:
                    mu[i, j] = 0.0  # 동일 라벨 선호
                # I-X는 B-X 또는 I-X 다음만 허용
                if a.startswith("I-"):
                    typ = a[2:]
                    if not (b == f"B-{typ}" or b == f"I-{typ}"):
                        mu[i, j] = 2.0  # 큰 패널티
        with torch.no_grad():
            self.mu.copy_(mu)

    def _pairwise_kernel(self, pos, emb, mask):
        """
        pos: (B,N) float positions
        emb: (B,N,D') projected embeddings
        mask: (B,N) 1/0
        -> K: (B,N,N) row-normalized affinity
        """
        m = (mask.unsqueeze(1) * mask.unsqueeze(2)).bool()  # (B,N,N)
        K = 0.0
        if self.w_pos is not None:
            sigma2 = torch.exp(self.log_sigma_pos) ** 2
            dist2 = (pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2  # (B,N,N)
            K_pos = torch.exp(-dist2 / (2.0 * sigma2))
            K = K + self.w_pos * K_pos
        if self.w_emb is not None:
            sigma2 = torch.exp(self.log_sigma_emb) ** 2
            e = emb
            e2 = (e**2).sum(-1, keepdim=True)
            dist2 = e2 + e2.transpose(1,2) - 2.0 * (e @ e.transpose(1,2))
            K_emb = torch.exp(-dist2 / (2.0 * sigma2))
            K = K + self.w_emb * K_emb

        K = K.masked_fill(~m, float("-inf"))
        K = F.softmax(K, dim=-1)
        return torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, unary_logits, token_reps, mask):
        """
        unary_logits: (B,N,L)
        token_reps:   (B,N,H)  — (NN/BiLSTM 결과)
        mask:         (B,N)    — 1=valid, 0=pad
        return: log_probs (B,N,L)
        """
        B, N, L = unary_logits.shape
        pos = torch.arange(N, device=unary_logits.device).float().unsqueeze(0).expand(B, N)
        emb = self.emb_proj(token_reps)
        K = self._pairwise_kernel(pos, emb, mask)

        Q = F.softmax(unary_logits, dim=-1)  # 초기 분포
        pad_mask = (mask == 0).unsqueeze(-1)

        for _ in range(self.T):
            M = K @ Q                  # (B,N,L)
            S = M @ self.mu.T          # (B,N,L)
            updated = unary_logits - S # 부호 일관만 유지
            updated = updated.masked_fill(pad_mask, -1e9)
            Q = F.softmax(updated, dim=-1)

        return torch.log(Q.clamp_min(1e-8))


# ---------- 상단 인코더 + (NN/비LSTM) + Linear head + CRF-RNN ----------
class NERWithCRFRNN(nn.Module):
    def __init__(self, encoder, hidden_dim, num_labels, use_bilstm=True, lstm_dim=256, T=5):
        """
        encoder: HF RoBERTa 등 (LoRA로 이미 튜닝된 인스턴스 주입)
                 forward(...) -> last_hidden_state (B,N,D_enc)
        hidden_dim: encoder 출력 차원 (D_enc)
        """
        super().__init__()
        self.encoder = encoder
        self.use_bilstm = use_bilstm

        if use_bilstm:
            self.ctx = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            feat_dim = 2 * lstm_dim
        else:
            self.ctx = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            feat_dim = hidden_dim

        self.head = nn.Linear(feat_dim, num_labels)
        self.crfrnn = CRFRNN(num_labels=num_labels, hidden_dim=feat_dim, T=T)

    @torch.no_grad()
    def init_bio_constraints(self, id2label):
        self.crfrnn.init_bio_constraints(id2label)

    def forward(self, input_ids, attention_mask, labels=None):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = enc.last_hidden_state  # (B,N,D_enc)

        if self.use_bilstm:
            H, _ = self.ctx(H)  # (B,N,2*lstm_dim)
        else:
            H = self.ctx(H)     # (B,N,D_enc)

        unary = self.head(H)                       # (B,N,L)
        log_probs = self.crfrnn(unary, H, attention_mask)  # (B,N,L)

        if labels is not None:
            loss = F.nll_loss(
                log_probs.view(-1, log_probs.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
            return {"loss": loss, "logits": log_probs}
        return {"logits": log_probs}

    @torch.no_grad()
    def decode(self, input_ids, attention_mask):
        out = self.forward(input_ids, attention_mask)
        return out["logits"].argmax(-1)  # (B,N)