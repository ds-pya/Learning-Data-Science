def train(model, dataloader, optimizer, device, num_epochs=10, prune_every=3):
    model.to(device)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            scores, loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = scores.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += input_ids.size(0)

        avg_loss = total_loss / total_count
        acc = total_correct / total_count

        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

        # pruning 주기마다 실행
        if epoch % prune_every == 0:
            prune_prototypes(model)
            reset_prototype_usage(model)

# 예: BERT tokenizer와 함께 사용할 수 있는 dataloader
# dataloader는 {'input_ids', 'attention_mask', 'labels'} 포함된 batch 반환 가정

from transformers import AdamW

model = TAEModel(encoder=your_bert_encoder,
                 hidden_dim=384,
                 num_classes=10,
                 num_prototypes_per_class=5,
                 taxonomy_distance_matrix=your_taxonomy_weight_matrix)

optimizer = AdamW(model.parameters(), lr=2e-5)

train(model, dataloader=train_dataloader, optimizer=optimizer, device='cuda', num_epochs=10)