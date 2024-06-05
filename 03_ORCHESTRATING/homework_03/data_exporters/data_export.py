import mlflow
import mlflow.sklearn
import pickle 
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(model, dv, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Specify your data exporting logic here
    with mlflow.start_run():
        # log the model file
        mlflow.sklearn.log_model(model, artifact_path="linear_regression_model")
        # save dictvectorizer
        dv_path = "dict_vectorizer.pkl"
        with open(dv_path, "wb") as f:
            pickle.dump(dv, f)
        
        mlflow.log_artifact(dv_path, artifact_path="dict_vectorizer")