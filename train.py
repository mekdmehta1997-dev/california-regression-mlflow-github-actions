import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# Ensure mlruns directory
os.makedirs("/app/mlruns", exist_ok=True)

mlflow.set_tracking_uri("file:///app/mlruns")
mlflow.set_experiment("california-rf")

mlflow.autolog()

# Load data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

print("Training completed successfully")
