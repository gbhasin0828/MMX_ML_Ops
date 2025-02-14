import mlflow
import joblib
import os
import pandas as pd
from sklearn.metrics import r2_score

# ✅ Set MLflow Tracking URI (Remote MLflow Server)
mlflow.set_tracking_uri("https://1b2f-34-150-254-72.ngrok-free.app")
mlflow.set_experiment("Marketing_Mix_Model_Tracking")

# ✅ Fix MLflow Artifact Path for Local Execution
artifact_location = os.path.abspath("mlruns/")  # ✅ Ensure writable path
mlflow.set_registry_uri(f"file://{artifact_location}")  # ✅ Enforce local artifact storage
os.makedirs(artifact_location, exist_ok=True)  # ✅ Ensure directory exists

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
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# ✅ Compute R² Score
r2_new = r2_score(y_true, y_pred)
print(f"🚀 Computed R² Score: {r2_new:.4f}")

# ✅ Start MLflow Run and Log Model
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("r2_score", r2_new)

    # ✅ Fix: Ensure Correct Model Logging
    mlflow.sklearn.log_model(
        rf_model,
        "random_forest_model",
        registered_model_name=MODEL_NAME  # ✅ Ensures correct registration
    )

print(f"🚀 First Model Registered in MLflow: {MODEL_NAME} | R² Score: {r2_new:.4f}")
