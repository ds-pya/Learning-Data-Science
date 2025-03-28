class MultiHeadPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, num_heads)

    def forward(self, hidden_states):  # [B, L, H]
        attn_weights = self.attn(hidden_states)  # [B, L, K]
        attn_weights = F.softmax(attn_weights, dim=1)  # softmax over tokens

        # weighted sum over sequence for each head
        out = torch.einsum("blh,blk->bkh", hidden_states, attn_weights)  # [B, K, H]
        return out