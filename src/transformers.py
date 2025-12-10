import torch
import torch.nn as nn
    
class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, L, _ = x.size() # [64,50, 128]
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()  # Causal mask #[50,50]
        x2, _ = self.attn(x, x, x, attn_mask=mask) # [64,50, 128]
        return self.norm(x + self.dropout(x2))


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.ff = FeedForward(hidden_dim, dim_feedforward, dropout)

    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)

    def forward(self, x):
        return self.encoder(x)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)

    def forward(self, x):
        return self.block(x)

class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):  # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.proj(x)      # [B, D_out, T/2] 
        return x.transpose(1, 2)  # [B, T/2, D_out]

class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):  # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.proj(x)      # [B, D_out, T*2]
        return x.transpose(1, 2)  # [B, T*2, D_out]