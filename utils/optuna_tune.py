import optuna
import mlflow
from pathlib import Path
from foetal_health_predictor import FoetalHealthModel
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score

def objective(trial, df):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=10),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }

    model_wrapper = FoetalHealthModel(**params)
    X, y = model_wrapper.preprocess(df)

    y_pred = cross_val_predict(model_wrapper.model, X, y, cv=3)
    y_prob = cross_val_predict(model_wrapper.model, X, y, cv=3, method='predict_proba')

    acc = accuracy_score(y, y_pred)
    f1 = cross_val_score(model_wrapper.model, X, y, cv=3, scoring='f1_weighted').mean()
    roc_auc = roc_auc_score(y, y_prob, multi_class='ovr')

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy_cv", acc)
        mlflow.log_metric("f1_weighted_cv", f1)
        mlflow.log_metric("roc_auc_cv", roc_auc)

    return acc

def get_study(df, n_trials=5):
    project_root = Path(__file__).resolve().parents[1]
    tracking_dir = project_root / "mlflow_runs"
    mlflow.set_tracking_uri(tracking_dir.as_uri())

    experiment_name = "Foetal_Health_Training"
    mlflow.set_experiment(experiment_name)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=tracking_dir.as_uri())

    study = optuna.create_study(
        direction="maximize",
        study_name="Foetal_Health_Training",
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)

    print("\n✅ Best Hyperparameters:", study.best_params)
    print(f'🔍 View MLflow UI with:\n mlflow ui --backend-store-uri "{tracking_dir.as_uri()}"')
    return study
