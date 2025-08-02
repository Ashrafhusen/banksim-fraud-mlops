import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from joblib import dump
import mlflow
import mlflow.sklearn
import os


def tune_model(data_path: str, model_path: str):
    df = pd.read_parquet(data_path)
    df.rename(columns={"hours_of_day": "hour_of_day"}, inplace=True)

    target = 'fraud'
    X = df.drop(columns=[target])
    y = df[target]

    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except:
                X[col] = X[col].astype('category')


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)       

    model = xgb.XGBClassifier(
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            use_label_encoder=False,
            enable_categorical=True,
            eval_metric='logloss'
        )


    param_grid = {
        'n_estimators' : [50, 100, 200],
        'max_depth' : [3, 6, 10],
        'learning_rate' : [0.01, 0.1, 0.2]
    }

    grid = GridSearchCV(model, param_grid, scoring = 'roc_auc', cv = 3, verbose = 1, n_jobs= 1)

    mlflow.set_experiment("Fine Tuning HyperParameter Tuning")
    with mlflow.start_run():

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        auc = roc_auc_score(y_test,best_model.predict(X_test))



        for param, value in grid.best_params_.items():
            mlflow.log_param(param, value)

            mlflow.log_param("model_type" , "XGBoost + GridSearchCV")
            mlflow.log_param("features" , ",".join(X.columns.tolist()))
            mlflow.log_metric("roc_auc", auc)
            mlflow.sklearn.log_model(best_model, "Tuned_XGBoost_Model")



        

        os.makedirs(model_path, exist_ok=True)
        dump((best_model, X_train.columns.tolist()), f"{model_path}/model.joblib")

        print(f"Best Params: {grid.best_params_}")
        print(f"Tuned Model AUC: {auc}")
        print(f"Model saved to: {model_path}/model.joblib")

if __name__ == "__main__":
    tune_model("data/processed/processed.parquet", "models")







