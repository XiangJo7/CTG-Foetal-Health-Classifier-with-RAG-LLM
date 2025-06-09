import yaml
import pandas as pd
from pathlib import Path
from utils.optuna_tune import get_study
import joblib
from foetal_health_predictor import FoetalHealthModel

# === Define Paths ===
project_root = Path(__file__).resolve().parents[1]
train_data_path = project_root / "data" / "processed" / "train.csv"
config_path = project_root / "configs" / "selected_columns.yaml"

# === Load Data & Features ===
df = pd.read_csv(train_data_path)
with open(config_path, "r") as f:
    selected_features = yaml.safe_load(f)["selected_columns"]
df_selected = df[selected_features]

# === Run Optuna Study + Final Model Training ===
study = get_study(df_selected, n_trials=100)

# === Train Final Model on Full Data ===
best_params = study.best_params
final_model_wrapper = FoetalHealthModel(**best_params)
X_final, y_final = final_model_wrapper.preprocess(df_selected)
final_model_wrapper.model.fit(X_final, y_final)

# === Save Model and Scaler ===
artifacts_path = project_root / "train_eval_scripts" / "artifacts"
artifacts_path.mkdir(parents=True, exist_ok=True)

joblib.dump(final_model_wrapper.model, artifacts_path / "best_random_forest.pkl")

print(f"\n💾 Model and scaler saved to: {artifacts_path}")
