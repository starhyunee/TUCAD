import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
import os

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(gt, pred, prefix="PW"):
    acc = accuracy_score(gt, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(gt, pred, average='binary')
    print(f"{prefix}-Accuracy : {acc:.4f}, Precision : {prec:.4f}, Recall : {rec:.4f}, F1 : {f1:.4f}")
    return acc, prec, rec, f1

def apply_point_adjustment(gt, pred):
    pred_pa = pred.copy()
    anom_indices = np.where(gt == 1)[0]
    if anom_indices.size == 0:
        return pred_pa

    splits = np.where(np.diff(anom_indices) != 1)[0] + 1
    segments = np.split(anom_indices, splits)
    for seg in segments:
        if np.any(pred[seg] == 1):
            pred_pa[seg] = 1

    return pred_pa

def eval_func(dataset, window_size, batch_size, epoch, T, denoiser_name):

    anomaly_ratio_01 = 0.1
    #anomaly_ratio_05 = 0.5
    anomaly_ratio_1 = 1

    print(f"Dataset: {dataset}") 
    method = f'w{window_size}_b{batch_size}_e{epoch}_t{T}_{denoiser_name}'
    print(f"method: {method}")
    
    test_score= np.load(f'./{dataset}/test_scores_w{window_size}_b{batch_size}_e{epoch}_t{T}_{denoiser_name}.npy')
    test_score = test_score.reshape(-1)
    train_score= np.load(f'./{dataset}/train_scores_w{window_size}_b{batch_size}_e{epoch}_t{T}_{denoiser_name}.npy')
    train_score= train_score.reshape(-1)
    label= np.load(f'./{dataset}/label.npy')
    label = label[:len(test_score)]

    print(test_score.shape)
    print(label.shape)
    if label.ndim > 1:
        label = label[:, 0]
    # anom_ratio_01 = anomaly_rate_01[dataset]
    # anom_ratio_05 = anomaly_rate_05[dataset]
    # anom_ratio_1 = anomaly_rate_1[dataset]
    thresh_01 = np.percentile(train_score, 100 - anomaly_ratio_01)
    #thresh_05 = np.percentile(train_score, 100 - anomaly_ratio_05)
    thresh_1 = np.percentile(train_score, 100 - anomaly_ratio_1)
    print(f"Threshold_01: {thresh_01:.4f}")
    #print(f"Threshold_05: {thresh_05:.4f}")
    print(f"Threshold_1: {thresh_1:.4f}")

    thresholds = {
    '01': thresh_01, 
    #'05': thresh_05, 
    '1' : thresh_1
    }

    for name, threshold in thresholds.items():

        pred_pw = (test_score > threshold).astype(int)
        pw_acc, pw_prec, pw_rec, pw_f1 = compute_metrics(label, pred_pw, prefix="PW")

        pred_pa = apply_point_adjustment(label, pred_pw)
        pa_acc, pa_prec, pa_rec, pa_f1 = compute_metrics(label, pred_pa, prefix="PA")

        os.makedirs('results_full/zscaling', exist_ok=True)
        csv_path = f'results_full/zscaling/{dataset}_additional.csv'
        header = [
            'method',f'threshold{name}','F-beta',
            'PW-Accuracy','PW-Precision','PW-Recall','PW-F1',
            'PA-Accuracy','PA-Precision','PA-Recall','PA-F1'
        ]
        row = {
            'method':method, f'threshold{name}':threshold, #'F-beta':f_beta,
            'PW-Accuracy':pw_acc,'PW-Precision':pw_prec,'PW-Recall':pw_rec,'PW-F1':pw_f1,
            'PA-Accuracy':pa_acc,'PA-Precision':pa_prec,'PA-Recall':pa_rec,'PA-F1':pa_f1
        }
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
   
        fig, axes = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
        axes[0].plot(test_score); axes[0].set_title('Anomaly Score')
        axes[1].plot(label);      axes[1].set_title('Ground Truth')
        axes[2].plot(pred_pw);    axes[2].set_title('Prediction (PW)')
        axes[3].plot(pred_pa);    axes[3].set_title('Prediction (PA)')
        plt.tight_layout() 

        out_dir = f'fig_full/{dataset}'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/{method}_thr{name}.png')
        plt.close()

    print("Done.\n")




def precision_recall_balance_score(precision, recall, beta=3):
    if precision + recall == 0:
        return 0.0
    
    beta_squared = beta ** 2
    prb_score = (1 + beta_squared) * (precision * recall) / (beta_squared *precision +recall)
    return prb_score