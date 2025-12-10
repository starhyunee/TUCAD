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
    


class TransformerUNet_condition(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim=32, num_layers=4, num_heads=4, dim_feedforward=256):
        super().__init__()
        self.name = 'TransformerUNet_condition'
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

    def forward(self, x, t, condition):
        x = self.input_proj(x)
        condition = condition
        h = self.pos_emb(x)
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)

        final_t_emb = t_emb + condition
        film_params = self.time_mlp(final_t_emb)  # (B, 2*hidden_dim)
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
    
class TransformerUNet_No_condition(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim=32, num_layers=4, num_heads=4, dim_feedforward=256):
        super().__init__()
        self.name = 'TransformerUNet_No_condition'
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

    def forward(self, x, t, condition):
        x = self.input_proj(x)
        condition = 0
        h = self.pos_emb(x)
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)

        final_t_emb = t_emb
        film_params = self.time_mlp(final_t_emb)  # (B, 2*hidden_dim)
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
    

class AttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.attn_weights_list = []

    def forward(self, x, encoder_feats):
        # x:  feature [B, L, D] from decoder
        # skip: encoder correspond feature [B, L, D]
        attn_output, attn_weights  = self.attn(query=x, key=encoder_feats, value=encoder_feats, need_weights=True)
        fused = self.norm(x + attn_output)  # residual connection + normalization
        self.attn_weights_list.append(attn_weights.detach().cpu())
        return attn_output

    def clear_attention(self):
        self.attn_weights_list = []
    

class TransformerUNet_crossattention(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim=32, num_layers=4, num_heads=4, dim_feedforward=256):
        super().__init__()
        self.name = 'TransformerUNet_crossattention'
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
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
        self.fusion1 = AttentionFusion(hidden_dim*2)
        self.up2 = Upsample(hidden_dim*2, hidden_dim)
        self.decoder2 = TransformerDecoderBlock(hidden_dim, num_heads, dim_feedforward)
        self.fusion2 = AttentionFusion(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, condition):
        self.fusion1.clear_attention()
        self.fusion2.clear_attention()
        x = self.input_proj(x)
        condition = 0
        h = self.pos_emb(x)
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)

        final_t_emb = t_emb + condition
        film_params = self.time_mlp(final_t_emb)  # (B, 2*hidden_dim)
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
        #x = x + e2
        x = self.fusion1(x, e2)                    
        x = self.decoder1(x)          
        x = self.up2(x) 
        #x = x + e1                
        x = self.fusion2(x, e1)      
        x = self.decoder2(x)          

        noise_pred = self.output_proj(x)

        return noise_pred, self.fusion1.attn_weights_list, self.fusion2.attn_weights_list
    
class TransformerUNet_crossattention_condition(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim=32, num_layers=4, num_heads=4, dim_feedforward=256):
        super().__init__()
        self.name = 'TransformerUNet_crossattention'
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
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
        self.fusion1 = AttentionFusion(hidden_dim*2)
        self.up2 = Upsample(hidden_dim*2, hidden_dim)
        self.decoder2 = TransformerDecoderBlock(hidden_dim, num_heads, dim_feedforward)
        self.fusion2 = AttentionFusion(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, condition):
        x = self.input_proj(x)
        condition = condition
        h = self.pos_emb(x)
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)

        final_t_emb = t_emb + condition
        film_params = self.time_mlp(final_t_emb)  # (B, 2*hidden_dim)
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
        #x = x + e2
        x = self.fusion1(x, e2)                    
        x = self.decoder1(x)          
        x = self.up2(x) 
        #x = x + e1                
        x = self.fusion2(x, e1)      
        x = self.decoder2(x)          

        noise_pred = self.output_proj(x)

        return noise_pred

class mean_std(nn.Module):
    def __init__(self, input_dim, time_emb_dim):
        super().__init__()
        self.linear = nn.Linear(2*input_dim, input_dim)
        self.timelinear = nn.Linear(input_dim, time_emb_dim)

    def forward(self, x):  # x: [B, L, D]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        represent = torch.cat([mean, std], dim =1) #  [B, 2, D]
        represent = represent.reshape(x.size(0),-1)
        represent = self.linear(represent) # [B, D]
        pooled = self.timelinear(represent)    # [B, 1, time_emb_dim]
        return pooled
    

