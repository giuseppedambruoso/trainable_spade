import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from src.physics import compute_U_matrix
from src.data import get_dataloaders
from src.model import QuantumOpticsClassifier
from src.train import train_and_evaluate

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Retrieve the specific output directory for this parallel Hydra job
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    
    # Extract the job number to position the tqdm bar correctly in multirun
    job_num = hydra_cfg.job.get("num", 0) 
    
    # 1. Physics Setup
    U_matrix = compute_U_matrix(sigma=cfg.sigma, N=28)
    
    # 2. Data Setup (passing cfg.num_classes and fixed cfg.data_seed)
    # Go up to the original cwd to find the common data directory across parallel runs
    original_cwd = hydra.utils.get_original_cwd()
    data_path = os.path.join(original_cwd, cfg.data_dir)
    train_loader, test_loader = get_dataloaders(data_path, cfg.batch_size, cfg.num_classes, cfg.data_seed)
    
    # 3. Model Setup (Sweepable model_seed ensures deterministic initialization)
    torch.manual_seed(cfg.model_seed)
    
    # Pass unitary parameter to the model
    model = QuantumOpticsClassifier(U_matrix=U_matrix, num_classes=cfg.num_classes, unitary=cfg.unitary)
    
    # 4. Train (passing cfg.unitary for logging)
    metrics_df, test_acc, unitary_matrix = train_and_evaluate(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        epochs=cfg.epochs, 
        lr=cfg.learning_rate,
        sigma=cfg.sigma,
        unitary=cfg.unitary,
        job_num=job_num
    )
    
    # 5. Save Artifacts directly to the Hydra-managed run directory
    
    # Save training metrics CSV
    metrics_path = os.path.join(output_dir, "training_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save training loss profile PDF 
    # (Note the 'r' before the f-string to prevent \s regex warnings)
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_df["epoch"], metrics_df["loss"], marker='o', color='blue')
    plt.title(rf"Training Loss Profile ($\sigma={cfg.sigma}$, U={cfg.unitary}, Seed={cfg.model_seed})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True)
    plot_path = os.path.join(output_dir, "loss_profile.pdf")
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    # Track the sweep parameters alongside the accuracy in the output CSV
    acc_path = os.path.join(output_dir, "test_accuracy.csv")
    pd.DataFrame([{
        "sigma": cfg.sigma,
        "unitary": cfg.unitary,
        "model_seed": cfg.model_seed,
        "final_test_accuracy": test_acc
    }]).to_csv(acc_path, index=False)
    
    # Save the complex Unitary Matrix in Native NumPy format
    # (If unitary=False, this will just save the Identity matrix)
    matrix_path = os.path.join(output_dir, "unitary_matrix.npy")
    np.save(matrix_path, unitary_matrix)

if __name__ == "__main__":
    main()
