import torch
import torch.distributed as dist
from seqeval.metrics import f1_score

def ids_to_labels(row_1d: torch.Tensor, id2label):
    # row_1d: [512], -1 패딩
    valid = row_1d != -1
    ids = row_1d[valid].tolist()
    if isinstance(id2label, dict):
        return [id2label[int(i)] for i in ids]
    else:  # list
        return [id2label[int(i)] for i in ids]

device = true_mat_local.device
world = dist.get_world_size() if dist.is_initialized() else 1
rank  = dist.get_rank() if dist.is_initialized() else 0

# 1) 랭크 간 gather (모두 같은 [N_local, 512] shape 가정)
g_true = [torch.empty_like(true_mat_local) for _ in range(world)]
g_pred = [torch.empty_like(pred_mat_local) for _ in range(world)]
dist.all_gather(g_true, true_mat_local)
dist.all_gather(g_pred, pred_mat_local)

# 2) rank0에서 concat → List[List[str]] 복원
if rank == 0:
    true_all = torch.cat(g_true, dim=0)  # [sum_N, 512]
    pred_all = torch.cat(g_pred, dim=0)  # [sum_N, 512]

    y_true, y_pred = [], []
    # 각 문장(row) 단위로 -1 패딩 제거 후 라벨 이름으로 변환
    for t_row, p_row in zip(true_all, pred_all):
        seq_true = ids_to_labels(t_row, id2label)
        seq_pred = ids_to_labels(p_row, id2label)
        if len(seq_true) == 0:
            continue
        # 길이 안 맞을 가능성이 거의 없지만 혹시 모를 경우 맞춰 자름
        L = min(len(seq_true), len(seq_pred))
        y_true.append(seq_true[:L])
        y_pred.append(seq_pred[:L])

    # 3) F1 계산
    f1 = f1_score(y_true, y_pred)
    print(f"val F1: {f1:.4f}")