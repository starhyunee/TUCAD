import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformers import CausalSelfAttention,FeedForward,TransformerBlock,TransformerEncoderBlock,TransformerDecoderBlock,Downsample,Upsample

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# === Sinusoidal Time Embedding === #
def get_timestep_embedding(timesteps, embedding_dim):
    if timesteps.dim() == 2:
        timesteps = timesteps.squeeze(1)
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb


class CNN_Denoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, time_emb_dim=32, num_layers=4):
        super().__init__()
        self.name = 'CNN_Denoiser'
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU()
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
            ) for _ in range(num_layers)
        ])
        self.out_conv = nn.Conv1d(hidden_dim, input_dim, kernel_size, padding=kernel_size//2)

    def forward(self, x, t):
        B, L, D = x.size()  # [64, 100, 25]
        x = x.permute(0, 2, 1) # [64, 25, 100]
        h = self.input_conv(x) #(64, hidden_dim, 100)
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features) #(64, time_emb_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(2)   #(64, hidden_dim, 1)
        h = h + t_emb # broadcast
        for layer in self.layers:
            h = h + layer(h)
        h = self.out_conv(h)
        return h.permute(0, 2, 1)
    


class Denoiser_Transformer_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim=32, num_layers=4, num_heads=4, dim_feedforward=256):
        super().__init__()
        self.name = 'Denoiser_Transformer_Decoder'
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim * 2),
            nn.ReLU(),
        ) 
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dim_feedforward) for _ in range(num_layers)
        ])
        self.pos_emb = PositionalEncoding(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        # x: (B, L, D)
        h = self.input_proj(x)  # (B, L, H)
        h = self.pos_emb(h)    
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)  # (B, time_emb_dim)
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)  # (B, H), (B, H)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # FiLM
        for block in self.transformer_blocks:
            h = block(h)
        # project back to input dim (predict noise)
        noise_pred = self.output_proj(h)  # (B, L, D)
        return noise_pred


class Denoiser_Transformer_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim=32, num_layers=4, num_heads=4, dim_feedforward=256):
        super().__init__()
        self.name = 'Denoiser_Transformer_Encoder'
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim * 2),
            nn.ReLU(),
        )
        self.pos_emb = PositionalEncoding(hidden_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        h = self.input_proj(x)  # (B, L, hidden_dim)
        h = self.pos_emb(h)

        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)  # (B, time_emb_dim)
        film_params = self.time_mlp(t_emb)  # (B, 2 * hidden_dim)
        scale, shift = film_params.chunk(2, dim=1)  # each (B, hidden_dim)

        for layer in self.transformer_layers:
            h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            h = layer(h)
        noise_pred = self.output_proj(h)  # (B, L, D)
        return noise_pred

class TransformerUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim=32, num_layers=4, num_heads=4, dim_feedforward=256):
        super().__init__()
        self.name = 'TransformerUNet'
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim * 2),
            nn.ReLU(),
        )
        self.pos_emb = PositionalEncoding(hidden_dim)

        # Encoder blocks
        self.encoder1 = TransformerEncoderBlock(hidden_dim, num_heads, dim_feedforward)
        self.down1 = Downsample(hidden_dim, hidden_dim * 2)
        self.encoder2 = TransformerEncoderBlock(hidden_dim * 2, num_heads, dim_feedforward * 2)
        self.down2 = Downsample(hidden_dim * 2, hidden_dim * 4)
        # Bottleneck
        self.bottleneck = TransformerEncoderBlock(hidden_dim * 4, num_heads, dim_feedforward * 4)

        # Decoder blocks
        self.up1 = Upsample(hidden_dim * 4, hidden_dim*2)
        self.decoder1 = TransformerDecoderBlock(hidden_dim*2, num_heads, dim_feedforward*2)

        self.up2 = Upsample(hidden_dim*2, hidden_dim)
        self.decoder2 = TransformerDecoderBlock(hidden_dim, num_heads, dim_feedforward)

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x,t):
        x = self.input_proj(x)
        h = self.pos_emb(x)
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        film_params = self.time_mlp(t_emb)  # (B, 2*hidden_dim)
        scale, shift = film_params.chunk(2, dim=1)  # each (B, hidden_dim)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        # Encoder
        e1 = self.encoder1(h)           
        x = self.down1(e1)             
        e2 = self.encoder2(x)         
        x = self.down2(e2) 

        x = self.bottleneck(x)        

        # Decoder
        x = self.up1(x)                
        x = x + e2                    
        x = self.decoder1(x)          
        x = self.up2(x) 
        x = x + e1                
        x = self.decoder2(x)          

        noise_pred = self.output_proj(x)

        return noise_pred

class AttentionPool(nn.Module):
    def __init__(self, input_dim, time_emb_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 1)
        )
        self.timelinear = nn.Linear(input_dim, time_emb_dim)

    def forward(self, x):  # x: [B, L, D]
        scores = self.attn(x).squeeze(-1)            # [B, L]
        weights = torch.softmax(scores, dim=1)       # [B, L]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, D]
        pooled = self.timelinear(pooled)    # [B, 1, time_emb_dim]
        return pooled
    
class Denoiser_Encoder_condition(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim=32, num_layers=4, num_heads=4, dim_feedforward=256):
        super().__init__()
        self.name = 'Denoiser_Encoder_condition'
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention_pool = AttentionPool(input_dim, time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim * 2),
            nn.ReLU(),
        )
        self.pos_emb = PositionalEncoding(hidden_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        h = self.input_proj(x)  # (B, L, hidden_dim)
        condition = self.attention_pool(x).squeeze(1)
        h = self.pos_emb(h)

        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)  # (B, time_emb_dim)
        final_t_emb = t_emb + condition
        film_params = self.time_mlp(final_t_emb)  # (B, 2 * hidden_dim)
        scale, shift = film_params.chunk(2, dim=1)  # each (B, hidden_dim)

        for layer in self.transformer_layers:
            h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            h = layer(h)
        noise_pred = self.output_proj(h)  # (B, L, D)
        return noise_pred
    

