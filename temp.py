from torch.optim.lr_scheduler import ReduceLROnPlateau

# 옵티마이저 & 스케쥴러
optimizer = AdamW(model.parameters(), lr=3e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

best_f1 = 0
patience_counter = 0
early_stop_patience = 5

for epoch in range(1, 50):  # 에폭 크게 잡고
    train(model, train_loader)
    f1 = evaluate(model, eval_loader)  # F1 반환하도록 구성해야 함

    scheduler.step(f1)

    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1

    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch}")
        break