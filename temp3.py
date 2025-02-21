class CustomEvaluator(evaluation.SentenceEvaluator):
    def __init__(self, model, dataloader, loss_fn):
        super(CustomEvaluator, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn

    def __call__(self, model, output_path, epoch, steps):
        model.eval()  # 평가 모드
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.dataloader:
                anchor, positive, negative, margin = batch
                anchor_emb = model.encode(anchor, convert_to_tensor=True)
                positive_emb = model.encode(positive, convert_to_tensor=True)
                negative_emb = model.encode(negative, convert_to_tensor=True)
                loss = self.loss_fn(anchor_emb, positive_emb, negative_emb, margin)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Evaluation Loss at Epoch {epoch}, Step {steps}: {avg_loss}")

        model.train()  # 평가 종료 후 다시 훈련 모드 설정 (★ 추가)

        return -avg_loss