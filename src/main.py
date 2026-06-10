import hydra
from omegaconf import DictConfig
import logging
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.physics import compute_U_matrix
from src.data import get_dataloaders
from src.model import QuantumOpticsClassifier
from src.train import train_and_evaluate

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Initializing run with sigma: {cfg.sigma}, lr: {cfg.learning_rate}, batch_size: {cfg.batch_size}")
    
    # 1. Physics Setup
    U_matrix = compute_U_matrix(sigma=cfg.sigma, N=28)
    
    # 2. Data Setup
    # Go up one level to access common data dir across multi-runs
    original_cwd = hydra.utils.get_original_cwd()
    data_path = os.path.join(original_cwd, cfg.data_dir)
    train_loader, test_loader = get_dataloaders(data_path, cfg.batch_size)
    
    # 3. Model Setup
    model = QuantumOpticsClassifier(U_matrix=U_matrix, num_classes=10)
    
    # 4. Train
    metrics_df, test_acc = train_and_evaluate(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        epochs=cfg.epochs, 
        lr=cfg.learning_rate
    )
    
    # 5. Save Artifacts (Hydra automatically handles isolated directories per run)
    logger.info(f"Saving artifacts to {os.getcwd()}")
    
    # Save training metrics CSV
    metrics_df.to_csv("training_metrics.csv", index=False)
    
    # Save training loss profile PDF
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_df["epoch"], metrics_df["loss"], marker='o', color='blue')
    plt.title(f"Training Loss Profile ($\sigma={cfg.sigma}$)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True)
    plt.savefig("loss_profile.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    
    # Save final test accuracy CSV
    pd.DataFrame([{"final_test_accuracy": test_acc}]).to_csv("test_accuracy.csv", index=False)

if __name__ == "__main__":
    main()
