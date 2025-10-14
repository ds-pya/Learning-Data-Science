# pip install transformers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from # pip install transformers
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# -----------------------------
# Utils: 1D IoU / NMS / Deltas
# -----------------------------
def iou1d(a, b):
    # a: [N,2] (s,e), b: [M,2]
    N, M = a.size(0), b.size(0)
    s1, e1 = a[:,0].unsqueeze(1), a[:,1].unsqueeze(1)
    s2, e2 = b[:,0].unsqueeze(0), b[:,1].unsqueeze(0)
    inter = (torch.minimum(e1, e2) - torch.maximum(s1, s2) + 1).clamp(min=0)
    len1 = (e1 - s1 + 1).clamp(min=0); len2 = (e2 - s2 + 1).clamp(min=0)
    union = len1 + len2 - inter + 1e-6
    return inter / union  # [N,M]

def nms1d(spans, scores, iou_thr=0.6):
    # spans: [K,2], scores: [K]
    order = scores.sort(descending=True).indices
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1: break
        rest = order[1:]
        ious = iou1d(spans[i:i+1], spans[rest]).squeeze(0)
        order = rest[ious <= iou_thr]
    return torch.tensor(keep, device=spans.device, dtype=torch.long)

def apply_deltas_1d(anchors, deltas):
    # anchors:[K,2](s,e) -> center/len; deltas:[K,2](d_center, d_loglen)
    c = (anchors[:,0] + anchors[:,1]) * 0.5
    l = (anchors[:,1] - anchors[:,0] + 1).clamp(min=1.)
    dc, dl = deltas[:,0], deltas[:,1]
    c2 = c + dc * l
    l2 = l * torch.exp(dl)
    s = (c2 - 0.5*l2).round(); e = (c2 + 0.5*l2).round()
    return torch.stack([s, e], dim=-1).long()

# -----------------------------
# Anchors (1D): at each token with preset lengths
# -----------------------------
def generate_anchors_1d(T, lengths=(1,2,3,4,6,8,12), stride=1, device="cpu"):
    # returns [T*A, 2] spans (s,e)
    spans = []
    for t in range(0, T, stride):
        for L in lengths:
            s = t; e = t + L - 1
            spans.append((s, e))
    spans = torch.tensor(spans, device=device, dtype=torch.long)
    return spans  # NOTE: 이후 valid length로 clamp 필요

# -----------------------------
# FPN(간단 게이트 합): 중·상위 레이어 융합
# -----------------------------
class FPN1D(nn.Module):
    def __init__(self, d, n=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, d) for _ in range(n)])
        self.gate = nn.Parameter(torch.ones(n)/n)
    def forward(self, hs):  # list of [B,T,D]
        zs = [p(h) for p,h in zip(self.proj, hs)]
        g = torch.softmax(self.gate, dim=0)
        z = sum(g[i]*zs[i] for i in range(len(zs)))
        return z  # [B,T,D]

# -----------------------------
# SPN (RPN in 1D): objectness + deltas
# -----------------------------
class SpanProposal1D(nn.Module):
    def __init__(self, d, A):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.obj = nn.Linear(d, A)      # per-token A anchors
        self.reg = nn.Linear(d, A*2)    # (dc, dlogl)
    def forward(self, feat, mask):
        # feat:[B,T,D], mask:[B,T]
        x = self.shared(feat)                # [B,T,D]
        obj = self.obj(x)                    # [B,T,A]
        reg = self.reg(x).view(x.size(0), x.size(1), -1, 2)  # [B,T,A,2]
        obj = obj.masked_fill(mask.unsqueeze(-1)==0, -1e9)
        return obj, reg

# -----------------------------
# RoIAlign1D: span -> fixed P bins
# -----------------------------
class RoIAlign1D(nn.Module):
    def __init__(self, P=16, mode="avg"):
        super().__init__()
        self.P = P; self.mode = mode
    def forward(self, feat, spans, lengths):
        # feat:[B,T,D], spans:[B,K,2] (clamped to [0, L-1]), lengths:[B]
        B,T,D = feat.shape; K = spans.size(1); P = self.P
        out = []
        for b in range(B):
            L = lengths[b].item()
            fb = feat[b, :L]  # [L,D]
            sb = spans[b].clamp(min=0, max=L-1)  # [K,2]
            pooled = []
            for k in range(K):
                s,e = sb[k].tolist()
                if e < s: s,e = e,s
                if e >= L: e = L-1
                if s < 0: s = 0
                seg = fb[s:e+1]  # [len,D] (len>=1)
                # linear bins to P
                idx = torch.linspace(0, seg.size(0)-1, steps=P, device=seg.device)
                idx0 = idx.floor().long().clamp(max=seg.size(0)-1)
                idx1 = (idx0+1).clamp(max=seg.size(0)-1)
                w = idx - idx0.float()
                samp = (1-w).unsqueeze(-1)*seg[idx0] + w.unsqueeze(-1)*seg[idx1]  # [P,D]
                pooled.append(samp)
            pooled = torch.stack(pooled, dim=0) if K>0 else torch.zeros(0,P,D, device=feat.device)
            out.append(pooled)
        # [B,K,P,D]
        maxK = spans.size(1)
        for b in range(B):
            if out[b].size(0) == 0:
                out[b] = torch.zeros(maxK, self.P, D, device=feat.device)
        return torch.stack(out, dim=0)

# -----------------------------
# Heads: Box / Mask
# -----------------------------
class BoxHead1D(nn.Module):
    def __init__(self, d, P, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # over P
        self.fc1 = nn.Linear(d, d); self.fc2 = nn.Linear(d, d)
        self.cls = nn.Linear(d, num_classes)       # +1 for background 포함
        self.reg = nn.Linear(d, 2)                 # (dc, dlogl)
    def forward(self, roi_feat):
        # roi_feat: [B,K,P,D]
        x = roi_feat.transpose(2,1)                # [B,K,D,P]
        x = self.pool(x).squeeze(-1)               # [B,K,D]
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.cls(x), self.reg(x)            # [B,K,C], [B,K,2]

class MaskHead1D(nn.Module):
    def __init__(self, d, P):
        super().__init__()
        self.conv1 = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.out   = nn.Conv1d(d, 1, kernel_size=1) # class-agnostic
    def forward(self, roi_feat):
        # [B,K,P,D] -> [B,K,P,1]
        x = roi_feat.permute(0,1,3,2)              # [B,K,D,P]
        B,K,D,P = x.shape
        x = x.reshape(B*K, D, P)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        m = torch.sigmoid(self.out(x))             # [B*K,1,P]
        m = m.reshape(B, K, 1, P).permute(0,1,3,2) # [B,K,P,1]
        return m

# -----------------------------
# 전체 모델
# -----------------------------
class BertMaskRCNN1D(nn.Module):
    def __init__(self, num_classes, anchors=(1,2,3,4,6,8,12), P=16):
        super().__init__()
        name = "sentence-transformers/paraphrase-MiniLM-L12-v2"
        cfg = AutoConfig.from_pretrained(name); cfg.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(name, config=cfg)
        for p in self.backbone.embeddings.parameters(): p.requires_grad = False
        d = self.backbone.config.hidden_size  # 384
        self.fpn = FPN1D(d, n=4)
        self.spn = SpanProposal1D(d, A=len(anchors))
        self.roi = RoIAlign1D(P=P)
        self.box_head = BoxHead1D(d, P, num_classes=num_classes)  # num_classes = (labels + 1 bg)
        self.mask_head = MaskHead1D(d, P)
        self.anchors_set = anchors
        self.P = P

    def forward(self, input_ids, attention_mask, targets=None, topk=64, nms_thr=0.6, keep=16):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.hidden_states  # (emb + 12 layers)
        feat = self.fpn([hs[6], hs[8], hs[10], hs[12]])            # [B,T,D]
        B,T,D = feat.shape; device = feat.device

        # SPN
        obj_logit, deltas = self.spn(feat, attention_mask)         # [B,T,A], [B,T,A,2]
        proposals_batch, scores_batch = [], []
        for b in range(B):
            L = int(attention_mask[b].sum().item())
            anchors = generate_anchors_1d(L, self.anchors_set, device=device)   # [L*A,2]
            obj = obj_logit[b,:L].reshape(-1)                                   # [L*A]
            reg = deltas[b,:L].reshape(-1,2)                                    # [L*A,2]
            props = apply_deltas_1d(anchors, reg).clamp(min=0, max=L-1)         # [L*A,2]
            # valid (len>=1)
            valid = (props[:,1] >= props[:,0])
            props, scr = props[valid], obj[valid]
            # top-k & NMS
            k_idx = torch.topk(scr, k=min(topk, scr.numel())).indices
            props, scr = props[k_idx], scr[k_idx]
            keep_idx = nms1d(props, scr, iou_thr=nms_thr)[:keep]
            proposals_batch.append(props[keep_idx])
            scores_batch.append(scr[keep_idx])
        # pad to K
        K = max(p.size(0) for p in proposals_batch) if proposals_batch else 0
        for b in range(B):
            pad_n = K - proposals_batch[b].size(0)
            if pad_n > 0:
                pad_span = torch.zeros(pad_n, 2, device=device, dtype=torch.long)
                pad_scr  = torch.zeros(pad_n, device=device)
                proposals_batch[b] = torch.cat([proposals_batch[b], pad_span], dim=0)
                scores_batch[b]    = torch.cat([scores_batch[b], pad_scr], dim=0)
        proposals = torch.stack(proposals_batch, dim=0) if K>0 else torch.zeros(B,0,2, device=device, dtype=torch.long)
        scores    = torch.stack(scores_batch, dim=0)  if K>0 else torch.zeros(B,0, device=device)

        # RoIAlign1D
        lengths = attention_mask.sum(dim=1)
        roi_feat = self.roi(feat, proposals, lengths)             # [B,K,P,D]
        cls_logits, bbox_deltas = self.box_head(roi_feat)         # [B,K,C], [B,K,2]
        masks = self.mask_head(roi_feat)                          # [B,K,P,1]

        outputs = {"proposals": proposals, "scores": scores,
                   "cls_logits": cls_logits, "bbox_deltas": bbox_deltas,
                   "masks": masks, "lengths": lengths}
        # 학습 시 targets로 loss 계산 (아래 참조)
        return outputs

# -----------------------------
# Loss (스켈레톤, 핵심 개념만)
# -----------------------------
def loss_maskrcnn_1d(outputs, targets, num_classes, P=16):
    """
    targets: list of dict per batch:
      {"labels":[N], "spans":[N,2], "mask":[N,T](optional)}
    매칭: IoU 기반 positive/negative, RoI마다 가장 IoU 높은 GT 선택.
    """
    cls_logits = outputs["cls_logits"]   # [B,K,C]
    bbox_deltas= outputs["bbox_deltas"]  # [B,K,2]
    masks      = outputs["masks"]        # [B,K,P,1]
    proposals  = outputs["proposals"]    # [B,K,2]
    lengths    = outputs["lengths"]      # [B]

    B,K,C = cls_logits.shape
    device = cls_logits.device

    loss_cls_all, loss_reg_all, loss_mask_all = [], [], []
    for b in range(B):
        L = int(lengths[b].item())
        props = proposals[b].clamp(0, L-1)         # [K,2]
        t = targets[b]
        gt_spans = t["spans"].to(device)           # [N,2]
        gt_labels= t["labels"].to(device)          # [N]
        if gt_spans.numel()==0:
            # all background
            loss_cls_all.append(F.cross_entropy(cls_logits[b].reshape(-1,C),
                                               torch.full((K,), num_classes-1, device=device))) # bg id = C-1 가정
            continue
        ious = iou1d(props, gt_spans)              # [K,N]
        max_iou, gt_idx = ious.max(dim=1)          # [K]
        pos = max_iou >= 0.5
        neg = (max_iou < 0.1)

        # cls
        cls_target = torch.full((K,), num_classes-1, device=device, dtype=torch.long)  # bg
        cls_target[pos] = gt_labels[gt_idx[pos]]
        loss_cls = F.cross_entropy(cls_logits[b], cls_target)
        loss_cls_all.append(loss_cls)

        # box reg (pos only): smooth L1 on deltas vs GT-deltas
        if pos.any():
            anchors = props[pos].float()
            gts = gt_spans[gt_idx[pos]].float()
            # compute target deltas
            ac = (anchors[:,0]+anchors[:,1])/2; al = (anchors[:,1]-anchors[:,0]+1).clamp(min=1.)
            gc = (gts[:,0]+gts[:,1])/2; gl = (gts[:,1]-gts[:,0]+1).clamp(min=1.)
            dc = (gc - ac)/al; dl = torch.log(gl/al)
            pred = bbox_deltas[b, pos]  # [P,2]
            loss_reg = F.smooth_l1_loss(pred, torch.stack([dc, dl], dim=-1), reduction='mean')
            loss_reg_all.append(loss_reg)
        else:
            loss_reg_all.append(torch.tensor(0.0, device=device))

        # mask (pos only): GT span을 P 길이 binary로 리스케일 → BCE/Dice
        if pos.any():
            # build GT masks at length P
            Pbins = torch.linspace(0, 1, steps=P, device=device)
            gt_bin = []
            sel_props = props[pos]; sel_gts = gt_spans[gt_idx[pos]]
            for (s,e),(gs,ge) in zip(sel_props, sel_gts):
                # normalize proposal to [0,1], make gt mask inside proposal
                Lp = (e - s + 1).clamp(min=1)
                rel_s = (gs - s).clamp(min=0); rel_e = (ge - s).clamp(min=0)
                rel_e = torch.clamp(rel_e, max=Lp-1)
                # continuous mask → sample to P
                gt_vec = torch.zeros(Lp, device=device)
                gt_vec[int(rel_s):int(rel_e)+1] = 1.0
                idx = torch.linspace(0, Lp-1, steps=P, device=device)
                i0 = idx.floor().long().clamp(max=Lp-1)
                i1 = (i0+1).clamp(max=Lp-1)
                w = idx - i0.float()
                samp = (1-w)*gt_vec[i0] + w*gt_vec[i1]  # [P]
                gt_bin.append(samp)
            gt_bin = torch.stack(gt_bin, dim=0)[:, :, None]  # [P_pos, P, 1]
            pred_mask = masks[b, pos]                         # [P_pos, P, 1]
            loss_mask = F.binary_cross_entropy(pred_mask, gt_bin)
            loss_mask_all.append(loss_mask)
        else:
            loss_mask_all.append(torch.tensor(0.0, device=device))

    loss_cls = torch.stack(loss_cls_all).mean()
    loss_reg = torch.stack(loss_reg_all).mean()
    loss_mask= torch.stack(loss_mask_all).mean()
    loss = loss_cls + 2.0*loss_reg + 1.5*loss_mask
    return {"loss": loss, "loss_cls": loss_cls, "loss_reg": loss_reg, "loss_mask": loss_mask} import AutoModel, AutoConfig

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