# pip install transformers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# ----------------------------
# 1) 1D Positional (필요시)
# ----------------------------
class PosEnc1D(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [L, D]
    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)

# ----------------------------
# 2) 얕은 경계(Edge) 보조 헤드
#    p_start, p_end  (focal BCE 권장)
# ----------------------------
class EdgeHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))
    def forward(self, h, mask):  # h:[B,T,D], mask:[B,T]
        logits = self.proj(h)                # [B,T,2]
        # padding 위치는 아주 작은 값으로 덮어서 topk에 안걸리게
        logits = logits.masked_fill((mask == 0).unsqueeze(-1), -1e9)
        p = torch.sigmoid(logits)            # [B,T,2]
        return p, logits

# ----------------------------
# 3) 1D FPN (멀티스케일 융합; 중·상위 4개 층 결합)
# ----------------------------
class FPN1D(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.pj = nn.ModuleList([nn.Linear(d, d) for _ in range(4)])
        self.gate = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    def forward(self, hs):  # hs: list of 4 tensors [B,T,D]
        zs = [pj(h) for pj, h in zip(self.pj, hs)]     # 투영
        g = torch.softmax(self.gate, dim=0)
        z = g[0]*zs[0] + g[1]*zs[1] + g[2]*zs[2] + g[3]*zs[3]  # [B,T,D]
        return z

# ----------------------------
# 4) 간결 디코더(2층 권장)
#    Self-Attn(1층만) + Cross-Attn + FFN
# ----------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d=384, nhead=6, ffd=1024):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ffd), nn.ReLU(), nn.Linear(ffd, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.n3 = nn.LayerNorm(d)
    def forward(self, q, mem, key_padding_mask=None, use_self=True):
        if use_self:
            q2,_ = self.self_attn(q, q, q, need_weights=False)
            q = self.n1(q + q2)
        q2,_ = self.cross_attn(q, mem, mem, key_padding_mask=key_padding_mask, need_weights=False)
        q = self.n2(q + q2)
        q2 = self.ff(q)
        q = self.n3(q + q2)
        return q

# ----------------------------
# 5) DETR-style NER Head (start/length 분해)
#    - class: focal CE (num_types + 1 for no_obj)
#    - start: CE over [0..T-1]
#    - length: CE over [1..Lmax]
# ----------------------------
class DetrSpanHead(nn.Module):
    def __init__(self, d, num_types, K=16, n_dec_layers=2, nhead=6, ffd=1024, Lmax=32):
        super().__init__()
        self.query_learn = nn.Embedding(K//2, d)   # learnable 절반
        self.n_dec = n_dec_layers
        self.decs = nn.ModuleList([DecoderLayer(d, nhead, ffd) for _ in range(n_dec_layers)])
        self.class_head  = nn.Linear(d, num_types + 1)  # + no_obj
        self.start_head  = nn.Linear(d, 1)              # start 점수 (나중에 T로 투사)
        self.length_cls  = nn.Linear(d, Lmax)           # length 분류 (1..Lmax 중)
        self.Lmax = Lmax
    def _init_queries(self, edge_p, h, mask):
        # 하이브리드 쿼리: learnable (K/2) + edge top-k (K/2)
        B, T, D = h.shape
        K2 = self.query_learn.num_embeddings
        # edge score: start/end 합산으로 topk 위치 선택
        edge_score = (edge_p[...,0] + edge_p[...,1])  # [B,T]
        k = K2
        topk = torch.topk(edge_score, k=min(k, T), dim=1).indices  # [B, k]
        # gather token-anchored queries
        idx = topk.unsqueeze(-1).expand(-1, -1, D)                 # [B,k,D]
        q_anchor = torch.gather(h, 1, idx)                         # [B,k,D]
        # concat with learnable
        q_learn = self.query_learn.weight.unsqueeze(0).expand(B, -1, -1)  # [B,K2,D]
        q = torch.cat([q_learn, q_anchor], dim=1)                  # [B, K, D]
        return q
    def forward(self, mem, mask, pos=None, edge_p=None):
        # mem: [B,T,D], mask:[B,T], pos(optional): [B,T,D]
        if pos is not None: mem = mem + pos
        q = self._init_queries(edge_p, mem, mask)                  # [B,K,D]
        key_pad = (mask == 0)                                      # True=pad
        for i,dec in enumerate(self.decs):
            q = dec(q, mem, key_padding_mask=key_pad, use_self=(i==0))
        # heads
        class_logits = self.class_head(q)                          # [B,K,C]
        # start: 각 query가 start 위치 점수를 갖도록 token dim으로 투사
        # 간단한 방법: q->[B,K,1]로 만든 뒤 mem와의 dot로 [B,K,T] 생성
        start_query = self.start_head(q)                           # [B,K,1]
        start_scores = torch.bmm(start_query.transpose(1,2), mem.transpose(1,2)).squeeze(1)  # [B,T]
        start_scores = start_scores.masked_fill(mask==0, -1e9)     # [B,T]
        length_logits = self.length_cls(q)                         # [B,K,Lmax]
        return {"class_logits": class_logits,
                "start_scores": start_scores,   # per-batch 공통 start heatmap
                "length_logits": length_logits, # per-query length
                "queries": q}

# ----------------------------
# 6) 전체 모델
# ----------------------------
class MiniLM_DETR_SingleNER(nn.Module):
    def __init__(self, num_types, K=16, Lmax=32):
        super().__init__()
        cfg = AutoConfig.from_pretrained("sentence-transformers/paraphrase-MiniLM-L12-v2")
        cfg.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L12-v2", config=cfg)
        # 임베딩만 freeze
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        d = self.backbone.config.hidden_size  # 384
        self.pos = PosEnc1D(d)
        self.edge = EdgeHead(d)
        self.fpn = FPN1D(d)
        self.head = DetrSpanHead(d, num_types, K=K, n_dec_layers=2, nhead=6, ffd=1024, Lmax=Lmax)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.hidden_states  # tuple(len=13): emb + 12 layers
        # 중/상위 4개 층 선택 (6,8,10,12층) — 필요시 조정
        f = self.fpn([hs[6], hs[8], hs[10], hs[12]])               # [B,T,D]
        pos = self.pos(f)
        edge_p, edge_logits = self.edge(f, attention_mask)         # [B,T,2], [B,T,2]
        head_out = self.head(f, attention_mask, pos=pos, edge_p=edge_p)
        return {**head_out, "edge_p": edge_p, "edge_logits": edge_logits}

# ----------------------------
# 7) 간단 매칭/로스 (스켈레톤)
#    - 실제론 Hungarian 권장(여긴 짧은 문장이라 greedy 매칭 예시)
# ----------------------------
def focal_ce(logits, targets, alpha=0.25, gamma=2.0):
    # logits:[N,C], targets:[N] (int, 0..C-1)
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    loss = alpha * (1-pt)**gamma * ce
    return loss.mean()

def build_targets(batch_gt, lengths, Lmax):
    """
    batch_gt: list of list of (label_id, start_idx, end_idx)
    lengths: [B] 유효 길이
    return per-batch tensors for matching에 활용
    """
    # 스켈레톤: 이미 매칭된 인덱스가 있다고 가정하거나,
    # 추후 Hungarian으로 교체.
    return batch_gt

def simple_greedy_match(start_scores, class_logits, length_logits, targets, lengths, Lmax):
    """
    매우 단순한 매칭: start_scores top-N, 길이/클래스와 근접한 GT로 Greedy.
    실제 프로덕션은 Hungarian으로 교체하세요.
    """
    B, T = start_scores.shape
    K = class_logits.shape[1]
    matches = []
    for b in range(B):
        # 후보 start top-K
        topk = torch.topk(start_scores[b], k=min(K, lengths[b].item())).indices
        # 쿼리 순회
        qb = []
        for qi in range(min(K, topk.numel())):
            qb.append((qi, topk[qi].item()))
        # 타겟 순회 (여기선 1:1로 index대응 흉내)
        gts = targets[b]
        m = []
        for i in range(min(len(qb), len(gts))):
            m.append((qb[i][0], i))  # (query_idx, gt_idx)
        matches.append(m)
    return matches

def loss_fn(outputs, batch_gt, attention_mask, num_types, alpha=0.25, gamma=2.0):
    """
    outputs: model forward 결과
    batch_gt: list[ list[(label, start, end)] ]
    """
    class_logits = outputs["class_logits"]     # [B,K,C]
    start_scores = outputs["start_scores"]     # [B,T]
    length_logits = outputs["length_logits"]   # [B,K,Lmax]
    edge_logits = outputs["edge_logits"]       # [B,T,2]
    mask = attention_mask                      # [B,T]
    B, K, C = class_logits.shape
    Lmax = length_logits.size(-1)

    # length = (end - start + 1)로 변환, Lmax 클램프
    targets = []
    lengths = mask.sum(dim=1)
    for gts in batch_gt:
        fixed = []
        for (lab, s, e) in gts:
            l = max(1, min(Lmax, e - s + 1))
            fixed.append((lab, s, l))
        targets.append(fixed)

    # (아래 매칭은 데모용 greedy; 실제는 Hungarian으로 교체)
    matches = simple_greedy_match(start_scores, class_logits, length_logits, targets, lengths, Lmax)

    # 로스 초기화
    loss_class = []
    loss_start = []
    loss_length = []

    # 1) match 된 것에 대해 positive loss
    for b in range(B):
        for (q_idx, gt_idx) in matches[b]:
            lab, s, l = targets[b][gt_idx]
            # class
            lc = F.cross_entropy(class_logits[b, q_idx:q_idx+1, :], torch.tensor([lab], device=class_logits.device))
            loss_class.append(lc)
            # start (token CE): start_scores[b]: [T] → CE 위해 [1,T]로 확장
            ls = F.cross_entropy(start_scores[b].unsqueeze(0), torch.tensor([s], device=class_logits.device))
            loss_start.append(ls)
            # length CE
            ll = F.cross_entropy(length_logits[b, q_idx], torch.tensor([l-1], device=class_logits.device))  # 1..Lmax → 0..Lmax-1
            loss_length.append(ll)

    # 2) no-object (미매칭 query) focal CE
    #    no_obj 클래스 id = num_types
    no_obj_id = num_types
    targets_noobj = []
    logits_noobj = []
    for b in range(B):
        matched_q = {m[0] for m in matches[b]}
        for q in range(K):
            if q not in matched_q:
                logits_noobj.append(class_logits[b, q])
                targets_noobj.append(no_obj_id)
    if logits_noobj:
        logits_noobj = torch.stack(logits_noobj, dim=0)   # [Nno,C]
        targets_noobj = torch.tensor(targets_noobj, device=class_logits.device)
        l_no = focal_ce(logits_noobj, targets_noobj, alpha=alpha, gamma=gamma)
    else:
        l_no = torch.tensor(0.0, device=class_logits.device)

    # Edge 보조 (focal BCE; 여기선 일반 BCE 예시)
    edge_target = torch.zeros_like(edge_logits)
    for b in range(B):
        for (lab, s0, e0) in batch_gt[b]:
            edge_target[b, s0, 0] = 1.0
            edge_target[b, e0, 1] = 1.0
    edge_loss = F.binary_cross_entropy_with_logits(outputs["edge_logits"], edge_target, reduction='mean')

    # 합산
    lc = (torch.stack(loss_class).mean() if loss_class else torch.tensor(0.0, device=class_logits.device))
    ls = (torch.stack(loss_start).mean() if loss_start else torch.tensor(0.0, device=class_logits.device))
    ll = (torch.stack(loss_length).mean() if loss_length else torch.tensor(0.0, device=class_logits.device))

    loss = 1.0*lc + 2.0*ls + 1.5*ll + 0.5*l_no + 0.5*edge_loss
    return {"loss": loss, "loss_class": lc, "loss_start": ls, "loss_len": ll, "loss_noobj": l_no, "loss_edge": edge_loss}