import torch
import numpy as np
from torch.utils.data import DataLoader
from src.denoiser_models import *
from diffusion import DiffusionTimeSeriesModel
#from condition_diffusion import DiffusionTimeSeriesModel
#from diffusion import DiffusionTimeSeriesModel
from sklearn.preprocessing import StandardScaler
from src.dataset_utils import create_sliding_windows

from tqdm import tqdm
from src.utils_eval import compute_metrics, apply_point_adjustment, eval_func
import argparse



def test(args):

    train_data = np.load(f"./data/{args.dataset}_sub/minmax/train 1-1.npy")
    test_data = np.load(f"./data/{args.dataset}_sub/minmax/test 1-1.npy")
    scaler = StandardScaler()
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    test_scaled  = scaler.transform(test_data)

    train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
    test_tensor  = torch.tensor(test_scaled,  dtype=torch.float32)

    dataset = args.dataset
    window_size = args.window_size
    stride= 0
    overlap = args.overlap
    print(overlap)
    if overlap == True:
        stride = 1
    else:
        stride = window_size
    batch_size=args.batch_size
    epoch = args.epochs
    T = args.T
    denoiser_name =  args.model_name

    train_windows = create_sliding_windows(train_tensor, window_size, stride=stride)

    test_windows = create_sliding_windows(test_tensor, window_size, stride=stride)

    model = DiffusionTimeSeriesModel(
        input_dim=test_tensor.shape[1],
        hidden_dim=128,
        time_emb_dim=32,
        num_layers=4,
        T=T,
        denoiser_name = denoiser_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(f"./checkpoints/{dataset}/model_w{window_size}_b{batch_size}_e{epoch}_t{T}_{denoiser_name}.pth", map_location=device))
    model.to(device)
    model.eval()
    print("model ready")
    print(f"{dataset} Calculating Train scores!!!")

    with torch.no_grad():
        train_scores, train_input, train_reconstruction,_,_ = model.compute_anomaly_scores_batched(train_windows,batch_size, overlap =overlap)
        train_scores = train_scores.numpy()
        np.save(f"./results/ablation/{dataset}/train_scores_w{window_size}_b{batch_size}_e{epoch}_t{T}_{denoiser_name}.npy", train_scores)
        # threshold1 = torch.quantile(train_scores, 0.999).item()
        # threshold2 = torch.quantile(train_scores, 0.995).item()
        # threshold3 = torch.quantile(train_scores, 0.99).item()
        # print(f"Threshold (99.9th percentile): {threshold1:.3f}")
        # print(f"Threshold (99.5th percentile): {threshold2:.3f}")
        # print(f"Threshold (99th percentile): {threshold3:.3f}")
        

        # test score
        print(f"{dataset} Calculating Anomaly Score!!!")
        test_scores, input, reconstruction, test_attn1_all, test_attn2_all = model.compute_anomaly_scores_batched(test_windows,batch_size, overlap =overlap)
        test_scores = test_scores.numpy()
        print("test_scores.shape", test_scores.shape)
        input = input.numpy()
        reconstruction = reconstruction.numpy()
        test_attn1_all  = test_attn1_all.numpy()     # shape: (N_windows, T_dec, T_enc)
        test_attn2_all  = test_attn2_all.numpy()

        print(f"{dataset} Test Finish!!!")
        np.save(f"./results/{dataset}/test_scores_w{window_size}_b{batch_size}_e{epoch}_t{T}_{denoiser_name}.npy", train_scores)
        np.save(f"./results/{dataset}/test_scores_w{window_size}_b{batch_size}_e{epoch}_t{T}_{denoiser_name}.npy", test_scores)
        np.save(f"./results/{dataset}/input_w{window_size}_b{batch_size}_e{epoch}_t{T}_{denoiser_name}.npy", input)

        print(f"Test anomaly scores saved.")
        eval_func(dataset, window_size, batch_size, epoch, T, denoiser_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Training script")
    parser.add_argument('--model_name', type=str, default='Denoiser_Transformer_Encoder', help='Name of the denoiser model')
    parser.add_argument('--dataset', type=str, default='PSM', help='Dataset name (e.g., MetroPT)')
    parser.add_argument('--window_size', type=int, default=100, help='Sliding window size')
    parser.add_argument('--overlap', type=lambda x: x.lower() == 'true', default=True, help='overlap vs nonoverlap on test')
    parser.add_argument('--batch_size', type=int, default=64, help='Testing batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (for loading correct checkpoint)')
    parser.add_argument('--T', type=int, default=500, help='Diffusion timestep T')
    args = parser.parse_args()
    #dataset = ["SMAP","SWaT","SMD", "PSM", "MSL"]
    test(args)
