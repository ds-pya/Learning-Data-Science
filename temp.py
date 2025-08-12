import torch.distributed as dist
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

@torch.no_grad()
def evaluate_ner_distributed(model, eval_loader, idx2label, o_id=0):
    model.eval()
    rank = int(os.environ.get("RANK", "0"))
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    all_true_local, all_pred_local = [], []

    for batch in eval_loader:
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        attn      = batch["attention_mask"].cuda(non_blocking=True).bool()
        labels    = batch["labels"].cuda(non_blocking=True).long()

        labels = torch.where(labels.eq(-100), torch.full_like(labels, o_id), labels)
        paths = model(input_ids=input_ids, attention_mask=attn, labels=None)  # list[list[int]]

        for b in range(input_ids.size(0)):
            valid_idx = attn[b].nonzero(as_tuple=False).squeeze(1)          # [L]
            true_ids  = labels[b, valid_idx].tolist()
            pred_ids  = paths[b][:len(true_ids)]
            all_true_local.append([idx2label[i] for i in true_ids])
            all_pred_local.append([idx2label[i] for i in pred_ids])

    # --- 모든 rank의 리스트를 rank0로 모으기 ---
    if world_size > 1:
        gather_true = [None for _ in range(world_size)]
        gather_pred = [None for _ in range(world_size)]
        dist.all_gather_object(gather_true, all_true_local)
        dist.all_gather_object(gather_pred, all_pred_local)

        if rank == 0:
            all_true = sum(gather_true, [])   # 리스트 concat
            all_pred = sum(gather_pred, [])
        else:
            all_true, all_pred = None, None
    else:
        all_true, all_pred = all_true_local, all_pred_local

    # --- rank0에서만 F1 계산/리턴 ---
    if (world_size == 1) or (rank == 0):
        f1 = f1_score(all_true, all_pred, scheme=IOB2)
        return f1
    else:
        return None