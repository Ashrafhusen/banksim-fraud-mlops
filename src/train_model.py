import pandas as pd
import mlflow 
import mlflow.sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump 
import os

def train_model(data_path : str, model_path: str):

    df = pd.read_parquet(data_path)
    df.rename(columns={"hours_of_day": "hour_of_day"}, inplace=True)


    target = 'fraud'
    X = df.drop(columns = [target])
    y = df[target]

    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except:
                X[col] = X[col].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )


    model = xgb.XGBClassifier(
        n_estimators = 100,
        max_depth = 6,
        learning_rate = 0.1,
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder = False,
        enable_categorical=True,
        eval_metric = 'logloss'
    )


    mlflow.set_experiment("Fraud Detection")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)

        mlflow.log_param("model_type", "XGBoost Classifier")
        mlflow.log_param("features", ",".join(X.columns.tolist()))
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "XGboost_model")


        os.makedirs(model_path, exist_ok = True)
        dump((model, X_train.columns.tolist()), f"{model_path}/model.joblib")

        
        print("Model trained and logged. AUC:", auc)
        print(f"Model saved to: {model_path}/model.joblib")

if __name__ == "__main__":
    train_model("data/processed/processed.parquet", "models")


