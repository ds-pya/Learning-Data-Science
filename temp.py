import os, math, argparse, torch, torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchcrf import CRF  # pip install torchcrf

# ---------------- Model ----------------
class TokenCRF(torch.nn.Module):
    def __init__(self, base_model_name: str, num_labels: int, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name, output_hidden_states=False)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)
        # LoRA 적용 (Q/K/V/Output proj에 주로 걸립니다; 미세조정 범위는 필요시 수정)
        lconf = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           target_modules=["query","key","value","dense","out_proj","o_proj","q_proj","k_proj","v_proj"],
                           task_type="TOKEN_CLS")
        self.encoder = get_peft_model(self.encoder, lconf)

        hidden = self.encoder.config.hidden_size
        self.classifier = torch.nn.Linear(hidden, num_labels, bias=True)  # emissions
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # encoder
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                                # [B, T, H]
        emissions = self.classifier(hidden)                            # [B, T, C]

        # torchcrf expects mask bool, labels Long
        mask = attention_mask.bool()
        if labels is not None:
            # labels: [B, T], -100는 무시 → mask에서 빼고, CRF에는 유효 영역만 전달
            if labels.dtype != torch.long:
                labels = labels.long()
            eff_mask = mask & (labels != -100)
            # CRF negative log-likelihood (평균)
            nll = -self.crf(emissions, labels, mask=eff_mask, reduction='mean')
            return nll
        else:
            # decode (list of list[int]) 길이 가변
            paths = self.crf.decode(emissions, mask=mask)
            return paths

# ---------------- Setup ----------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
def cleanup_ddp():
    dist.destroy_process_group()
def is_main():
    return int(os.environ.get("RANK", "0")) == 0

# ---------------- Train ----------------
def train_one_epoch(model, loader, optimizer, scaler_or_none, scheduler, device, accum=1, max_grad=1.0):
    model.train()
    total_loss, step = 0.0, 0
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids, attn, labels)
            loss = loss / accum

        loss.backward()
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler: scheduler.step()
        total_loss += loss.item()
        step += 1
    return total_loss / max(step, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    # 예: 토큰 정확도/시퀀스 F1 등 원하는대로 구현
    # 여기서는 단순 loss만 측정 (라벨 필요)
    total, n = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids, attn, labels)
        total += loss.item()
        n += 1
    return total / max(n, 1)

def save_ckpt_ddp_peft(ddp_model, optimizer, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if is_main():
        # LoRA 가중치만 저장 (베이스는 재로딩 시 자동)
        ddp_model.module.encoder.save_pretrained(os.path.join(out_dir, f"epoch{epoch}_lora"))
        torch.save({"epoch": epoch,
                    "cls_head": ddp_model.module.classifier.state_dict(),
                    "opt": optimizer.state_dict()}, os.path.join(out_dir, f"epoch{epoch}_extra.pt"))

def load_ckpt_ddp_peft(ddp_model, optimizer, path_dir):
    # 필요시 구현: ddp_model.module.encoder.from_pretrained(path_dir_lora), classifier/opt 로드 등
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_labels", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bsz", type=int, default=16)   # GPU당 batch
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="./runs/minilm-crf")
    args = parser.parse_args()

    setup_ddp()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

    # ----- Dataset 준비 -----
    # train_dataset / val_dataset 은 사용자 제공.
    # 각 item: {"input_ids": Long[B,T], "attention_mask": Long[B,T], "labels": Long[B,T] (pad: -100)}
    from your_project.datasets import train_dataset, val_dataset  # 사용자의 구현부

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_size=args.bsz, sampler=train_sampler,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.bsz, sampler=val_sampler,
                              num_workers=8, pin_memory=True, persistent_workers=True)

    # ----- Model/Optim/Sched -----
    torch.set_float32_matmul_precision("high")  # TF32 on Ampere
    model = TokenCRF(args.model_name, args.num_labels)
    model.to(device)
    model = DDP(model, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)

    # AdamW (LoRA + head만 학습: encoder.parameters()는 LoRA만 실제 grad)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optim_groups = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    # 스케줄러 (linear)
    total_steps = math.ceil(len(train_loader) / args.accum) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

    # ----- Train Loop -----
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        tr_loss = train_one_epoch(model, train_loader, optimizer, None, scheduler, device,
                                  accum=args.accum, max_grad=1.0)
        # 평가는 각 rank가 해도 되지만, 로깅은 main만
        val_loss = evaluate(model, val_loader, device)
        if is_main():
            print(f"[epoch {epoch}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")
        save_ckpt_ddp_peft(model, optimizer, epoch, args.out_dir)

    cleanup_ddp()

if __name__ == "__main__":
    main()