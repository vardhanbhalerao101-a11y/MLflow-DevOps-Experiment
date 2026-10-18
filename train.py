import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Connect to the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MLflow_DevOps_Experiment")

# Load a sample dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train and track Model 1
with mlflow.start_run(run_name="Random_Forest"):
    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    print("Random Forest Accuracy:", accuracy)

# Train and track Model 2
with mlflow.start_run(run_name="Logistic_Regression"):
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 5000)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    print("Logistic Regression Accuracy:", accuracy)

print("\nBoth models trained and logged successfully!")