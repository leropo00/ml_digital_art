import re
from typing import Optional

import mlflow
from google.colab import userdata  # type: ignore
from pyngrok import ngrok

"""
Utils for specific frameworks like fastai,
should be in other files
"""


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


def find_run_id_by_code(
    mlfclient, active_experiment_id: str, run_code: str
) -> str | None:
    regex = r"^\d+_?$"
    if not re.match(regex, run_code):
        print("error not a valid run code", run_code)
        return None
    run_code_search = run_code if run_code.endswith("_") else run_code + "_"

    results = mlfclient.search_runs(
        experiment_ids=[active_experiment_id],
        filter_string=f"attributes.run_name LIKE '{run_code_search}%'",
    )
    if len(results) == 0:
        print(f"error, run with name starting with {run_code_search} not found")
        return None
    if len(results) > 1:
        print(
            f"warning, multiple runs with name starting with {run_code_search}, returning None, found ids were:"
        )
        print(list(map(lambda x: x.info.run_id, results.to_list())))
        return None
    return results[0].info.run_id


def get_run_id_from_name(
    mlfclient, active_experiment_id: str, run_name: str
) -> str | None:
    results = mlfclient.search_runs(
        experiment_ids=[active_experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
    )
    if len(results) == 0:
        print(f"error, run with name {run_name} not found")
        return None
    if len(results) > 1:
        print("warning, multiple runs with name, returning None, found ids were ")
        print(list(map(lambda x: x.info.run_id, results.to_list())))
        return None
    return results[0].info.run_id


def start_mlflow_server_in_collab(local_registry, mlflow_port) -> None:
    # get_ipython is part of IPython.core.interactiveshell
    get_ipython().system_raw(  # noqa: F821 # pyright: ignore[reportUndefinedVariable]
        f"mlflow ui --backend-store-uri {local_registry}  --port {mlflow_port}  --allowed-hosts '*' &"
    )  # run tracking UI in the background


def ngrok_access_to_mlflow_in_collab(mlflow_port) -> Optional[str]:
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
