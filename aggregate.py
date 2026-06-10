import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def setup_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "Computer Modern Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 1.0,
    })

def main():
    if len(sys.argv) < 2:
        return
    
    target_dir = Path(sys.argv[1])
    
    csv_files = list(target_dir.rglob("test_accuracy.csv"))
    if not csv_files:
        return

    # 1. Load and aggregate data
    dfs = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    combined_df = combined_df.sort_values(by=["sigma", "unitary", "model_seed"])
    combined_df.to_csv(target_dir / "all_results.csv", index=False)

    stats_df = combined_df.groupby(["sigma", "unitary"])["final_test_accuracy"].agg(["mean", "std"]).reset_index()
    stats_df.rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"}, inplace=True)
    stats_df["accuracy_std"] = stats_df["accuracy_std"].fillna(0.0)
    
    stats_df.to_csv(target_dir / "averaged_results.csv", index=False)
    
    # 2. Generate PRL-style plot
    setup_prl_style()
    
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    
    df_true = stats_df[stats_df["unitary"] == True].sort_values("sigma")
    df_false = stats_df[stats_df["unitary"] == False].sort_values("sigma")
    
    if not df_true.empty:
        ax.errorbar(
            df_true["sigma"], 
            df_true["accuracy_mean"], 
            yerr=df_true["accuracy_std"],
            fmt='-o', 
            color='#1f77b4', 
            label='with optimized unitary transformation',
            capsize=3
        )
        
    if not df_false.empty:
        ax.errorbar(
            df_false["sigma"], 
            df_false["accuracy_mean"], 
            yerr=df_false["accuracy_std"],
            fmt='--s', 
            color='#d62728', 
            label='without unitary transformation',
            capsize=3
        )
        
    ax.set_xlabel(r"Point-spread function variance, $\sigma$")
    ax.set_ylabel("Test Accuracy")
    
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    
    plot_path = target_dir / "plot.pdf"
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
