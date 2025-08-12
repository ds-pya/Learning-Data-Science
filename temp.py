import os, torch, torch.distributed as dist
from seqeval.metrics import f1_score, classification_report
from seqeval.scheme import IOB2  # BIO면 IOB2, BIOES면 IOBES

@torch.no_grad()
def evaluate_ner(model, eval_loader, idx2label, scheme=IOB2, return_report=False):
    """
    model: forward(labels=None) → CRF.decode(emissions, mask=...) 경로가 동작해야 함
    batch: {
      "input_ids": Long[B,T], "attention_mask": Long/Bool[B,T],
      "special_token_mask": Long/Bool[B,T], "labels": Long[B,T]  # 유효 아님: -100
    }
    """
    model.eval()
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank  = dist.get_rank() if dist.is_initialized() else 0

    all_true_local, all_pred_local = [], []

    for batch in eval_loader:
        input_ids = batch["input_ids"].cuda(non_blocking=True).long()
        attn_full = batch["attention_mask"].cuda(non_blocking=True).bool()           # encoder용(특수토큰 포함)
        spmask    = batch["special_token_mask"].cuda(non_blocking=True).bool()
        labels    = batch["labels"].cuda(non_blocking=True).long()

        # 평가/CRF용 유효 마스크(패딩/특수토큰 제외)
        valid_mask = attn_full & (~spmask)                                          # [B,T] bool

        # --- 모델 추론 ---
        # encoder에는 attn_full(특수토큰 포함), CRF decode엔 valid_mask를 쓰도록
        # model.forward 내부가 labels=None이면 decode 경로로 valid_mask를 attention_mask로 사용하도록 구현되어 있어야 함.
        paths = model(input_ids=input_ids, attention_mask=valid_mask, labels=None)  # list[list[int]]

        # --- 수집 ---
        B, T = input_ids.size()
        for b in range(B):
            idx = valid_mask[b].nonzero(as_tuple=False).squeeze(1)                  # [L]
            # labels는 유효 아님 위치가 -100이므로 valid만 취하면 -100 없음
            true_ids = labels[b, idx].tolist()
            # decode 길이가 유효 길이와 동일해야 정상. 안전하게 min으로 맞춤.
            L = min(len(true_ids), len(paths[b]))
            all_true_local.append([idx2label[i] for i in true_ids[:L]])
            all_pred_local.append([idx2label[i] for i in paths[b][:L]])

    # --- 분산 수집 ---
    if world > 1:
        gather_true = [None] * world
        gather_pred = [None] * world
        dist.all_gather_object(gather_true, all_true_local)
        dist.all_gather_object(gather_pred, all_pred_local)
        if rank == 0:
            all_true = sum(gather_true, [])
            all_pred = sum(gather_pred, [])
        else:
            all_true, all_pred = None, None
    else:
        all_true, all_pred = all_true_local, all_pred_local

    # --- F1 (rank0만 계산/리턴) ---
    if (world == 1) or (rank == 0):
        f1 = f1_score(all_true, all_pred, scheme=scheme)
        if return_report:
            report = classification_report(all_true, all_pred, scheme=scheme, digits=4)
            return f1, report
        return f1
    else:
        return None