import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "MLflow_DevOps_Experiment"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)

best_run = runs[0]

print("Best model run:", best_run.info.run_name)
print("Best accuracy:", best_run.data.metrics["accuracy"])

model_uri = f"runs:/{best_run.info.run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="Best_DevOps_Model"
)

print("Model registered successfully!")
print("Registered model name:", registered_model.name)
print("Model version:", registered_model.version)