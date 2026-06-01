import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging
from data_processing import FullDataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. Load raw data
    logging.info("Loading data from data/xente_data.csv")
    df = pd.read_csv("data/xente_data.csv")
    
    # 2. Process data and create proxy target
    logging.info("Creating RFM-based proxy target and feature engineering")
    processor = FullDataProcessor(random_state=42)
    model_df = processor.fit_transform(df)
    
    # 3. Separate features and target
    X = model_df.drop('is_high_risk', axis=1)
    y = model_df['is_high_risk']
    
    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # 5. Define models and hyperparameter grids
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    }
    
    param_grids = {
        'LogisticRegression': {'C': [0.01, 0.1, 1, 10]},
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
        'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
    }
    
    # 6. Experiment tracking with MLflow
    mlflow.set_experiment("CreditRiskExperiment")
    best_score = 0.0
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            logging.info(f"Training {name} with GridSearchCV")
            gs = GridSearchCV(model, param_grids[name], cv=3, scoring='roc_auc', n_jobs=-1)
            gs.fit(X_train, y_train)
            
            best_est = gs.best_estimator_
            y_pred = best_est.predict(X_test)
            y_proba = best_est.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            
            # Log to MLflow
            mlflow.log_params(gs.best_params_)
            mlflow.log_metrics(metrics)
            
            # Log model artifact
            if name == 'XGBoost':
                mlflow.xgboost.log_model(best_est, artifact_path="model")
            else:
                mlflow.sklearn.log_model(best_est, artifact_path="model")
            
            logging.info(f"{name} - ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
            
            # Track best model
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = best_est
                best_model_name = name
    
    # 7. Register the best model in MLflow Model Registry
    logging.info(f"Best model: {best_model_name} with ROC AUC = {best_score:.4f}")
    if best_model_name == 'XGBoost':
        mlflow.xgboost.log_model(best_model, artifact_path="best_model", registered_model_name="CreditRiskXGB")
    else:
        mlflow.sklearn.log_model(best_model, artifact_path="best_model", registered_model_name="CreditRiskLR")
    
    logging.info("Model registration complete. Training finished.")

if __name__ == "main":
    main()
