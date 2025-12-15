import torch
from torch.utils.data import Dataset, DataLoader

class TitleTopicDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_length=64, title_col="title", label_col="topic"):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.title_col = title_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        title = str(self.df.at[i, self.title_col])
        label = self.df.at[i, self.label_col]
        y = self.label2id[label] if not isinstance(label, int) else int(label)

        enc = self.tok(
            title,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(y, dtype=torch.long),
            "row_idx": torch.tensor(i, dtype=torch.long),  # 나중에 점수 DF로 다시 붙일 때 유용
        }
        return item

def make_label_maps(df, label_col="topic"):
    # topic이 문자열이면 label2id 만들기
    if df[label_col].dtype == "int64" or df[label_col].dtype == "int32":
        num_labels = int(df[label_col].max()) + 1
        label2id = {i: i for i in range(num_labels)}
        id2label = {i: i for i in range(num_labels)}
        return label2id, id2label, num_labels

    labels = sorted(df[label_col].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label, len(labels)



import torch.nn as nn

class TopicClassifier(nn.Module):
    def __init__(self, encoder, num_labels, hidden_size=384, mlp_hidden=384, dropout_p=0.1, use_dropout=True):
        super().__init__()
        self.encoder = encoder

        layers = []
        if use_dropout:
            layers.append(nn.Dropout(dropout_p))
        layers += [
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout_p))
        layers += [
            nn.Linear(mlp_hidden, num_labels),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [B, 384]
        logits = self.mlp(cls)

        if labels is None:
            return logits
        loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits



import torch
from torch.optim import AdamW

def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))

@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn)
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += labels.numel()
    return correct / max(1, total)


import torch.nn.functional as F

def _extract_mlp_layers(mlp: nn.Sequential):
    # mlp가 [Dropout?, Linear, ReLU, Dropout?, Linear] 형태라고 가정하고 찾아냄
    lin = [m for m in mlp if isinstance(m, nn.Linear)]
    relu = next((m for m in mlp if isinstance(m, nn.ReLU)), None)
    assert len(lin) == 2 and relu is not None, "MLP 구조가 예상과 다릅니다."
    lin1, lin2 = lin[0], lin[1]
    return lin1, lin2

@torch.no_grad()
def batch_distill_scores(model, input_ids, attention_mask, labels):
    model.eval()

    out = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    x = out.last_hidden_state[:, 0, :]  # [B, D]

    lin1, lin2 = _extract_mlp_layers(model.mlp)

    # dropout은 eval에서 꺼지므로 그냥 mlp를 통과시켜도 되지만,
    # per-sample grad proxy를 위해 z1/a1/logits를 직접 계산
    z1 = F.linear(x, lin1.weight, lin1.bias)   # [B, H]
    a1 = F.relu(z1)                            # [B, H]
    logits = F.linear(a1, lin2.weight, lin2.bias)  # [B, C]

    loss = F.cross_entropy(logits, labels, reduction="none")  # [B]

    p = logits.softmax(dim=-1)
    y = F.one_hot(labels, num_classes=logits.size(-1)).float()
    delta2 = (p - y)  # [B, C]

    delta2_norm2 = (delta2 ** 2).sum(dim=-1)  # [B]
    a1_norm2 = (a1 ** 2).sum(dim=-1)          # [B]
    x_norm2  = (x ** 2).sum(dim=-1)           # [B]

    # layer2
    gradW2_norm2 = delta2_norm2 * a1_norm2
    gradb2_norm2 = delta2_norm2

    # layer1 (ReLU mask)
    mask = (z1 > 0).to(delta2.dtype)
    delta1 = (delta2 @ lin2.weight) * mask
    delta1_norm2 = (delta1 ** 2).sum(dim=-1)

    gradW1_norm2 = delta1_norm2 * x_norm2
    gradb1_norm2 = delta1_norm2

    grad_score = gradW1_norm2 + gradb1_norm2 + gradW2_norm2 + gradb2_norm2
    score = loss + 0.1 * torch.sqrt(grad_score + 1e-12)

    return {
        "loss": loss.cpu(),
        "grad_score": grad_score.cpu(),
        "score": score.cpu(),
        "pred": logits.argmax(dim=-1).cpu(),
        "conf": p.max(dim=-1).values.cpu(),
        "cls_emb": x.cpu(),  # (선택) 클러스터링/대표성에 쓰려면 저장
    }



from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

def run_train_then_score(df_train, df_val, base_model_name, batch_size=64, max_len=64, epochs=1, lr=2e-4,
                         use_dropout=True, device="cuda"):
    label2id, id2label, num_labels = make_label_maps(df_train, "topic")

    tok = AutoTokenizer.from_pretrained(base_model_name)

    base = AutoModel.from_pretrained(base_model_name)
    # LoRA 설정 예시 (타깃 모듈은 모델에 따라 조정 필요)
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,          # LoRA 자체 dropout은 일단 0 추천(분석 안정)
        target_modules=["q_proj","k_proj","v_proj","o_proj"],  # 모델에 따라 다를 수 있음
    )
    encoder = get_peft_model(base, lora_cfg)

    model = TopicClassifier(
        encoder=encoder,
        num_labels=num_labels,
        hidden_size=base.config.hidden_size,  # MiniLM이면 384
        mlp_hidden=base.config.hidden_size,   # 당신 케이스(384->384->25)
        dropout_p=0.1,
        use_dropout=use_dropout,
    ).to(device)

    ds_tr = TitleTopicDataset(df_train, tok, label2id, max_length=max_len)
    ds_va = TitleTopicDataset(df_val, tok, label2id, max_length=max_len)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    opt = AdamW(model.parameters(), lr=lr)

    for ep in range(epochs):
        tr_loss = train_one_epoch(model, dl_tr, opt, device)
        va_acc = eval_accuracy(model, dl_va, device)
        print(f"epoch {ep+1}/{epochs} | train_loss={tr_loss:.4f} | val_acc={va_acc:.4f}")

    # ---- 점수 산출 (train df 전체에 대해) ----
    dl_score = DataLoader(ds_tr, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    all_rows = []
    for batch in dl_score:
        scores = batch_distill_scores(
            model,
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].to(device),
        )
        row_idx = batch["row_idx"].cpu()
        for i in range(len(row_idx)):
            all_rows.append((
                int(row_idx[i]),
                float(scores["loss"][i]),
                float(scores["grad_score"][i]),
                float(scores["score"][i]),
                int(scores["pred"][i]),
                float(scores["conf"][i]),
            ))

    return model, all_rows, label2id, id2label