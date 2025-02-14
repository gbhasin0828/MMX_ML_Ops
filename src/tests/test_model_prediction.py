import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import mlflow
import joblib
from mlflow.tracking import MlflowClient

# ✅ Ensure `src/` is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# ✅ Import only what is needed for Random Forest
from utils import adstock_transform, hill_transform, find_optimal_alpha_per_channel

# ✅ Get absolute path of the model file
model_path = os.path.abspath("models/model_bundle.pkl")

# ✅ Load the model using the correct path
random_forest_model = joblib.load(model_path)

# ✅ Sample Test Data
test_data = pd.DataFrame({
    'mdsp_dm': [100, 200],
    'mdsp_inst': [150, 250],
    'mdsp_nsp': [80, 120],
    'mdsp_auddig': [130, 180],
    'mdsp_audtr': [170, 210],
    'mdsp_vidtr': [200, 300],
    'mdsp_viddig': [90, 150],
    'mdsp_so': [50, 70],
    'mdsp_on': [120, 160],
    'mdsp_sem': [140, 190],
    'sales': [300, 400]  # ✅ Adding Fake Sales Data
})

@pytest.mark.parametrize("model, model_type", [(random_forest_model, "RandomForest")])
def test_model_prediction(model, model_type):
    """Test model prediction pipeline for Random Forest."""

    # ✅ Load Model Components
    main_model = model["rf_model"]
    media_scaler = model["media_scaler"]
    y_scaler = model["y_scaler"]
    best_alphas = model["best_alphas"]
    best_ec_values = model["best_ec_values"]
    best_slope_values = model["best_slope_values"]

    # ✅ Define X_test (Feature Variables Only) 
    X_test = test_data.drop(columns=["sales"])  # ✅ Remove target column before scaling

    # ✅ Define y_true (Actual Sales)
    y_true = test_data["sales"].values  # ✅ Extract actual sales for R² calculation

    # ❌ Incorrect: media_scaler.transform(test_data) → ✅ Correct: Use X_test
    X_scaled = media_scaler.transform(X_test)  # ✅ Use X_test, not test_data

    # ✅ Apply Adstock Transformation
    X_adstocked = np.zeros_like(X_scaled)
    for i, alpha in enumerate(best_alphas):
        X_adstocked[:, i] = adstock_transform(X_scaled[:, i], alpha)

    # ✅ Apply Hill Transformation
    X_hill_transformed = np.zeros_like(X_adstocked)
    for i in range(len(best_alphas)):
        X_hill_transformed[:, i] = hill_transform(X_adstocked[:, i], best_ec_values[i], best_slope_values[i])

    # ✅ Model Prediction
    predictions_scaled = main_model.predict(X_hill_transformed)
    predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # ✅ Compute R² Score
    r2 = r2_score(y_true, predictions)

    # ✅ Assertions
    assert predictions is not None, f"{model_type} model returned no predictions!"
    assert predictions.shape[0] == test_data.shape[0], f"{model_type} model output shape mismatch!"
    print(f"✅ {model_type} Model Passed All Tests!")

    # ✅ Log Model in MLflow
    mlflow.set_experiment("Marketing_Mix_Model_Tracking")
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(main_model, "random_forest_model")

        print(f"✅ Logged Random Forest Model with R²: {r2:.4f}")
