import mlflow
import dagshub
import os
import joblib
import pandas as pd
from sklearn.metrics import r2_score

# ✅ Initialize DagsHub MLflow Tracking
dagshub.init(repo_owner='gbhasin0828', repo_name='MMX_MLFlow', mlflow=True)

# ✅ Set MLflow Tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/gbhasin0828/MMX_MLFlow.mlflow")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ Define Experiment Name
EXPERIMENT_NAME = "Marketing_Mix_Model_Tracking"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"🚀 MLflow is now tracking at {MLFLOW_TRACKING_URI} in experiment: {EXPERIMENT_NAME}")

# ✅ Define Model Name
MODEL_NAME = "Best_Marketing_Model"

# ✅ Load the Trained Model from `/models/`
model_path = "models/model_bundle.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}. Train the model first!")

model_bundle = joblib.load(model_path)
rf_model = model_bundle["rf_model"]
y_scaler = model_bundle["y_scaler"]

# ✅ Load `data_copy.csv` for Model Testing
file_path = "data/data_copy.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Data file not found at {file_path}. Ensure the correct dataset is available!")

df = pd.read_csv(file_path)

# ✅ Define Features and Target
feature_columns = ['mdsp_dm', 'mdsp_inst', 'mdsp_nsp', 'mdsp_auddig',
                   'mdsp_audtr', 'mdsp_vidtr', 'mdsp_viddig', 'mdsp_so',
                   'mdsp_on', 'mdsp_sem']
target_column = "sales"

# ✅ Extract Data for Testing
X_test = df[feature_columns].to_numpy()
y_true = df[target_column]

# ✅ Run Predictions & Inverse Transform
y_pred_scaled = rf_model.predict(X_test)
y_pred_scaled = y_pred_scaled.reshape(-1, 1)  # Fix potential shape issue
y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

# ✅ Compute R² Score
r2_new = r2_score(y_true, y_pred)
print(f"🚀 Computed R² Score: {r2_new:.4f}")

# ✅ Start MLflow Run and Log Model
with mlflow.start_run(run_name="RandomForest_Model_Test"):
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("r2_score", r2_new)

    # ✅ Log Model to MLflow
    mlflow.sklearn.log_model(
        rf_model,
        "random_forest_model",
        registered_model_name=MODEL_NAME
    )

print(f"🚀 First Model Registered in MLflow: {MODEL_NAME} | R² Score: {r2_new:.4f}")
