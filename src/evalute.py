import pandas as pd 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt
import os 


def evalute_model(data_path : str, model_path : str, output_dir: str):
    df = pd.read_parquet(data_path)
    
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            df[col] = df[col].astype("category").cat.codes


    

    X = df.drop(columns=['fraud'])
    y = df['fraud']

    X.rename(columns={"hours_of_day": "hour_of_day"}, inplace=True)

    model, feature_names = load(model_path)

    y_pred = model.predict(X)

    print("Classification Report:")
    print(classification_report(y, y_pred))

    print("ROC AUC:", roc_auc_score(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")


    os.makedirs(output_dir, exist_ok = True)
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix1.png")


if __name__ == "__main__":
    evalute_model("data/processed/processed.parquet", "models/model.joblib", "reports/1")




