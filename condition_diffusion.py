import torch
import torch.nn as nn
from tqdm import tqdm
from src.loss_functions import *
#import src.condition_denoiser_models as denoiser_models
import src.denoiser_models as denoiser_models

from src.condition_denoiser_models import AttentionPool,mean_std



# === Diffusion Model === #
class DiffusionTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, time_emb_dim=32,
                 num_layers=4, num_heads=4, dim_feedforward=256, T=100, denoiser_name="TimeSeriesDenoiser"):
        super().__init__()
        self.criterion = CharbonnierLoss(eps=1e-4)
        self.input_dim = input_dim
        self.T = T
        if isinstance(denoiser_name, str):
            try:
                TimeSeriesDenoiser = getattr(denoiser_models, denoiser_name)
            except AttributeError:
                raise ValueError(f"`denoiser_models` module `{denoiser_name}` no class.")
        elif isinstance(denoiser_name, type):
            TimeSeriesDenoiser = denoiser_name
        else:
            raise ValueError("`denoiser_name` should be class name (str) class object.")
        
        self.denoiser = TimeSeriesDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward
        )
        # diffusion buffers
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, T))
        alphas = 1 - self.betas
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", torch.cumprod(alphas, dim=0))

        self.attention_pool = AttentionPool(input_dim, time_emb_dim)
        self.mean_std = mean_std(input_dim, time_emb_dim)
        
    def forward_diffusion_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.alphas_bar[t].sqrt().view(-1, 1, 1)
        sqrt_omb = (1 - self.alphas_bar[t]).sqrt().view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise

    def forward(self, x0):
        B = x0.size(0)
        condition = self.mean_std(x0).squeeze(1)
        t  = torch.randint(0, self.T, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        x_t = self.forward_diffusion_sample(x0, t, eps)

        eps_pred, _, _ = self.denoiser(x_t, t, condition)         

        loss_eps = F.mse_loss(eps_pred, eps)
        return loss_eps 

    @torch.no_grad()
    def reconstruct(self, x0):
        self.denoiser.fusion1.clear_attention()
        self.denoiser.fusion2.clear_attention()
        B, L, D = x0.size()
        condition = self.mean_std(x0).squeeze(1)
        device = x0.device
        # start from x_T
        TT = int((self.T - 1)/10)
        t_T = torch.full((B,), TT, dtype=torch.long, device=device)
        x_t = self.forward_diffusion_sample(x0, t_T)
        for t in reversed(range(1, TT)):
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)
            noise_pred, maps1, maps2 = self.denoiser(x_t, t_batch, condition)
            beta = self.betas[t]
            alpha = self.alphas[t]
            alpha_bar = self.alphas_bar[t]
            sigma = beta.sqrt()
            # reverse diffusion step
            x_t = (x_t - (beta / (1 - alpha_bar).sqrt()) * noise_pred) / alpha.sqrt()
            if t > 1:
                x_t = x_t + sigma * torch.randn_like(x_t)

        attn1_all  = torch.cat(self.denoiser.fusion1.attn_weights_list, dim=0)
        attn2_all  = torch.cat(self.denoiser.fusion2.attn_weights_list, dim=0)

        return x_t, attn1_all, attn2_all

        

    @torch.no_grad()
    def compute_anomaly_score(self, x0):
        # x0: [B, W, D]
        recon, maps1, maps2 = self.reconstruct(x0)                 # [B, W, D]
        error = ((x0 - recon) ** 2).mean(dim=2)      # [B, W] → timestep-wise error
        return error, x0, recon, maps1, maps2                      # [B, W], [B, W, D], [B, W, D]

    @torch.no_grad()
    def compute_anomaly_scores_batched(self, data_tensor, batch_size=128, overlap=True):
        attn1_list, attn2_list = [], []
        if overlap:
            N, W, D = data_tensor.shape
            T = N + W - 1
            device = next(self.parameters()).device

            score_sum = torch.zeros(T, device=device)
            score_count = torch.zeros(T, device=device)
            full_input = torch.zeros((T, D), device=device)
            full_recon = torch.zeros((T, D), device=device)

            for i in tqdm(range(0, N, batch_size)):
                batch = data_tensor[i:i+batch_size].to(device)       # [B, W, D]
                B = batch.shape[0]

                error, x0, recon, maps1, maps2 = self.compute_anomaly_score(batch)  # [B, W], [B, W, D], [B, W, D]
                attn1_list.append(maps1)   # maps1: (B, T_dec, T_enc)
                attn2_list.append(maps2)

                for b in range(B):
                    start = i + b
                    end = start + W
                    score_sum[start:end] += error[b]
                    score_count[start:end] += 1
                    full_input[start:end] += x0[b]
                    full_recon[start:end] += recon[b]

            final_score = score_sum / torch.clamp(score_count, min=1)
            final_input = full_input / torch.clamp(score_count.unsqueeze(1), min=1)
            final_recon = full_recon / torch.clamp(score_count.unsqueeze(1), min=1)
            attn1_all = torch.cat(attn1_list, dim=0)
            attn2_all = torch.cat(attn2_list, dim=0)

            return final_score.cpu(), final_input.cpu(), final_recon.cpu(), attn1_all, attn2_all
        else:
            scores = []
            input = []
            reconstruction =[]
            attn1_list, attn2_list = [], []
            N = data_tensor.shape[0]
            device = next(self.parameters()).device
            for i in tqdm(range(0, N, batch_size)):
                batch = data_tensor[i:i+batch_size].to(device)
                batch_scores, x0, recon, maps1, maps2  = self.compute_anomaly_score(batch)
                scores.append(batch_scores.cpu())
                input.append(x0.cpu())
                reconstruction.append(recon.cpu())
                attn1_list.append(maps1)
                attn2_list.append(maps2)

            return torch.cat(scores, dim=0).cpu(), torch.cat(input, dim=0).cpu(), torch.cat(reconstruction, dim=0).cpu(),  torch.cat(attn1_list, dim=0), torch.cat(attn2_list, dim=0)






