import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from diffusion import DiffusionTimeSeriesModel
#from condition_diffusion import DiffusionTimeSeriesModel
from src.dataset_utils import TimeSeriesWindowDataset, create_sliding_windows
from tqdm import tqdm
import argparse



def train(args):
    print(f"Training with model: {args.model_name}")
    traindata = np.load(f"./data/{args.dataset}_sub/minmax/train 1-1.npy")
    scaler = StandardScaler()
    traindata_scaled = scaler.fit_transform(traindata) 
    train_tensor = torch.tensor(traindata_scaled, dtype=torch.float32)
    train_tensor = torch.tensor(traindata, dtype=torch.float32)

    dataset = args.dataset
    window_size = args.window_size
    stride = args.stride
    batch_size=args.batch_size
    epoch = args.epochs
    T = args.T
    denoiser_name =  args.model_name
    
    train_windows = create_sliding_windows(train_tensor, window_size, stride)

    train_dataset = TimeSeriesWindowDataset(train_windows)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = DiffusionTimeSeriesModel(
        input_dim=train_tensor.shape[1],
        hidden_dim=128,
        time_emb_dim=32,
        num_layers=4,
        T=T,
        denoiser_name=denoiser_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    print(f"{dataset} Train Start!!!")
    for e in tqdm(range(epoch)):
        epoch_loss = 0
        for batch in tqdm(train_loader):
            batch = batch.to(device)  
            #print("batch.shape", batch.shape)   ## [batch, win, dim]
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"[Epoch {e+1}] Loss: {epoch_loss:.6f}")

    print(f"{args.dataset} Train Finish!!!")
    save_path = f"./checkpoints/{dataset}/model_w{window_size}_b{batch_size}_e{epoch}_t{T}_{denoiser_name}_minmax.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion model for time series")

    parser.add_argument('--model_name', type=str, default='Denoiser_Transformer_Encoder', help='Name of the denoiser model')
    parser.add_argument('--dataset', type=str, default='PSM', help='Dataset name (e.g., MetroPT)')
    parser.add_argument('--window_size', type=int, default=100, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=1, help='Stride for sliding window')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--T', type=int, default=500, help='Diffusion timestep T')
    args = parser.parse_args()
    train(args)
    