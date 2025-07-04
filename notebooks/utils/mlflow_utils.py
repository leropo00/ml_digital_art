from fastai.learner import load_learner
from fastai.callback.core import Callback
from google.colab import userdata
import mlflow
from mlflow import MlflowClient
from pyngrok import ngrok
import os
from typing import List


def create_mlflow_client(local_registry):
    mlfclient = mlflow.tracking.MlflowClient(tracking_uri=local_registry)
    mlflow.set_tracking_uri(local_registry)
    return mlfclient


def get_mlflow_experiment(mlfclient, experiment_name, experiment_tags={}) -> str:
    active_experiment = mlfclient.get_experiment_by_name(experiment_name)
    if active_experiment is None:
        mlfclient.create_experiment(name=experiment_name, tags=experiment_tags)

    active_experiment = mlflow.set_experiment(experiment_name)
    return active_experiment.experiment_id


def log_mlflow_params(mlfclient, run, params):
    for k, v in params.items():
        mlfclient.log_param(run_id=run.info.run_id, key=k, value=v)


def get_last_run_id(mlfclient, active_experiment_id):
    runs = mlfclient.search_runs(
        experiment_ids=[active_experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def save_fastai_model_as_artifact(
    mlfclient, run_id, learner, exported_model_filename, artifact_path="fastai_model"
) -> str:
    learner.export(exported_model_filename)
    mlfclient.log_artifact(
        run_id,
        local_path=exported_model_filename,
        artifact_path=artifact_path,
    )
    artifact_uri = f"runs:/{run_id}/{artifact_path}/{exported_model_filename}"

    print("artifact_uri saved as model")
    print(artifact_uri)

    # clear the exported fastai model
    os.remove(exported_model_filename)

    return artifact_uri


def fastai_model_from_artifact(artifact_uri):
    local_download_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    return load_learner(local_download_path)


def start_mlflow_server_in_collab(local_registry, mlflow_port) -> None:
    # get_ipython is part of IPython.core.interactiveshell
    get_ipython().system_raw(  # noqa: F821
        f"mlflow ui --backend-store-uri {local_registry}  --port {mlflow_port} &"
    )  # run tracking UI in the background


def ngrok_access_to_mlflow_in_collab(mlflow_port) -> str:
    # if you get error that session already runs and you are only limited to 1 session,
    # stop the session here:
    # https://dashboard.ngrok.com/agents

    # Terminate open tunnels if exist
    ngrok.kill()

    # Setting the authtoken (optional)
    # Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
    ngrok.set_auth_token(userdata.get("NGROK_AUTH_TOKEN"))

    public_url = ngrok.connect(mlflow_port).public_url
    print("MLflow Tracking UI:", public_url)
    return public_url


class MLFlowTracking(Callback):
    "A `LearnerCallback` that tracks the loss and other metrics into MLFlow"

    def __init__(self, metric_names: List[str], client: MlflowClient, run_id: str):
        self.client = client
        self.run_id = run_id
        self.metric_names = metric_names

    def after_epoch(self):
        "Compare the last value to the best up to now"
        for metric_name in self.metric_names:
            m_idx = list(self.recorder.metric_names[1:]).index(metric_name)
            if len(self.recorder.values) > 0:
                val = self.recorder.values[-1][m_idx]
                self.client.log_metric(
                    self.run_id, metric_name, float(val), step=self.learn.epoch
                )
