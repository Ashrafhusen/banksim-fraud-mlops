import matplotlib.pyplot as plt
import pandas as pd
from joblib import load
import os
import numpy as np


def plot_feature_importance(model_path: str, output_dir: str, top_n: int = 20):
    model, feature_names = load(model_path)

    importances = model.feature_importances_

    features_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    features_df = features_df.sort_values("importance", ascending=False).head(top_n)


    plt.figure(figsize=(10, 6))
    plt.barh(features_df['feature'], features_df['importance'], color='skyblue')
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.gca().invert_yaxis() 
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    print(f"Feature importance plot saved to {output_dir}/feature_importance.png")


if __name__ == "__main__":
    plot_feature_importance("models/model.joblib", "reports", top_n=20)
