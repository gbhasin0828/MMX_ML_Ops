
import mlflow
import dagshub
import os
import joblib
import pandas as pd
from sklearn.metrics import r2_score
from mlflow.models.signature import infer_signature

import numpy as np
import pandas as pd

# ✅ Set your DagsHub Token manually
os.environ["DAGSHUB_TOKEN"] = "5426eef888cf0b4b5b1fee3d6b1e77182263e144"

# ✅ Set your DagsHub Username (optional)
os.environ["DAGSHUB_USER"] = "gbhasin0828"

# ✅ Read DagsHub Credentials from Environment Variables
DAGSHUB_USER = os.getenv("DAGSHUB_USER", "gbhasin0828")  # Default to your username
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")  # Should be set in GitHub Secrets

if not DAGSHUB_TOKEN:
    raise ValueError("❌ DAGSHUB_TOKEN is missing. Add it as a GitHub Secret!")

# ✅ Authenticate DagsHub with Token
dagshub.auth.add_app_token(DAGSHUB_TOKEN)

# ✅ Set MLflow Tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"https://dagshub.com/{DAGSHUB_USER}/MMX_MLFlow.mlflow")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ Initialize DagsHub MLflow Tracking
dagshub.init(repo_owner=DAGSHUB_USER, repo_name="MMX_MLFlow", mlflow=True)

# ✅ Define Experiment Name
EXPERIMENT_NAME = "Marketing_Mix_Model_Tracking"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"🚀 MLflow is now tracking at {MLFLOW_TRACKING_URI} in experiment: {EXPERIMENT_NAME}")

# ✅ Define Model Name
MODEL_NAME = "Best_Marketing_Model_New"

# ✅ Load the Trained Model from `/models/`
model_path = "models/model_bundle.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}. Train the model first!")

model_bundle = joblib.load(model_path)
print(model_bundle.keys())

model_bundle = joblib.load(model_path)
rf_model = model_bundle["rf_model"]
media_scaler = model_bundle["media_scaler"]
y_scaler = model_bundle["y_scaler"]
best_alphas = model_bundle["best_alphas"]
best_ec_values = model_bundle["best_ec_values"]
best_slope_values = model_bundle["best_slope_values"]

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

"""# **Define Transformations**"""

# ✅ Define Adstock Transformation
def adstock_transform(X, alpha):
    """Apply adstock transformation recursively."""
    X_adstocked = np.zeros_like(X, dtype=float)
    X_adstocked[0] = X[0]  # Initialize the first value

    for t in range(1, len(X)):
        X_adstocked[t] = X[t] + alpha * X_adstocked[t - 1]  # Carryover effect

    return X_adstocked

# ✅ Define Hill Transformation
def hill_transform(media_slice, ec, slope):
    """Apply the Hill function transformation."""
    return (media_slice ** slope) / (ec ** slope + media_slice ** slope)

"""# **Store Artefacts**"""

model_directory = "models/best_model"
os.makedirs(model_directory, exist_ok=True)

extra_objects = {
    "media_scaler": media_scaler,
    "y_scaler": y_scaler,
    "best_alphas": best_alphas,
    "best_ec_values": best_ec_values,
    "best_slope_values": best_slope_values
}
joblib.dump(extra_objects, os.path.join(model_directory, "extra_objects.pkl"))

media_columns = feature_columns  # Assuming media variables are all features

# ✅ Apply Adstock Transformation
X_Media = df[media_columns].to_numpy()
X_Media = media_scaler.transform(X_Media)  # ✅ Apply scaling here!
X_adstocked = np.zeros_like(X_Media, dtype=float)

for i in range(X_Media.shape[1]):  # Loop over each media column
    X_adstocked[:, i] = adstock_transform(X_Media[:, i], best_alphas[i])

# ✅ Apply Hill Transformation
X_adstocked_hill = np.zeros_like(X_adstocked, dtype=float)
for i in range(X_adstocked.shape[1]):
    X_adstocked_hill[:, i] = hill_transform(X_adstocked[:, i], best_ec_values[i], best_slope_values[i])

# ✅ Run Model Prediction
y_pred_scaled = rf_model.predict(X_adstocked_hill)
y_pred_scaled = y_pred_scaled.reshape(-1, 1)
y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

# ✅ Compute R² Score
r2_new = r2_score(df['sales'], y_pred)
print(f"🚀 Computed R² Score: {r2_new:.4f}")

try:
  model_versions = dagshub.get_latest_version(repo_owner=DAGSHUB_USER, repo_name="MMX_MLFlow", model_name=MODEL_NAME)
  if model_versions:
    latest_version = model_versions[-1]
    run_id = latest_version.run_id
    metrics = mlflow.get_run(run_id).data.metrics
    previous_r2 = metrics.get("r2_score", -np.inf)
    print(f"ℹ️ Latest Registered Model R²: {previous_r2:.4f}")

  else:
    previous_r2 = -np.inf
    print("ℹ️ No model currently registered in DagsHub.")

except Exception:
    previous_r2 = -np.inf
    print("ℹ️ No model currently registered in DagsHub.")

class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
      self.model = joblib.load(context.artifacts["rf_model"])
      self.extra_objects = joblib.load(context.artifacts["extra_objects"])

      def predict(self, context, X):
        best_alphas = self.extra_objects["best_alphas"]
        best_ec_values = self.extra_objects["best_ec_values"]
        best_slope_values = self.extra_objects["best_slope_values"]
        media_scaler = self.extra_objects["media_scaler"]
        y_scaler = self.extra_objects["y_scaler"]

        X_transformed = media_scaler.transform(X)

        X_Media = X_transformed.to_numpy()
        X_adstocked = np.zeros_like(X_Media, dtype=float)

        for i in range(X_Media.shape[1]):
            X_adstocked[:, i] = adstock_transform(X_Media[:, i], best_alphas[i])

        X_adstocked_hill = np.zeros_like(X_adstocked, dtype=float)
        for i in range(X_adstocked.shape[1]):
            X_adstocked_hill[:, i] = hill_transform(X_adstocked[:, i], best_ec_values[i], best_slope_values[i])

        # ✅ Make Predictions
        y_pred_scaled = self.model.predict(X_adstocked_hill)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return y_pred

if r2_new > previous_r2:

    with mlflow.start_run(run_name = "Best_Model_Registration)"):
      mlflow.log_param("model_type", "Random Forest")
      mlflow.log_metric("r2_score", r2_new)

      mlflow.pyfunc.log_model(
      artifact_path="best_model",
      python_model=CustomModelWrapper(),
      artifacts={"model_bundle": model_path},  # ✅ Single storage now
      registered_model_name=MODEL_NAME)

      print("🚀 Model and all objects successfully logged in MLflow!")

else:
    print(f"⚠️ New model R² ({r2_new:.4f}) is NOT better than the registered model R² ({previous_r2:.4f}). Model NOT registered.")

