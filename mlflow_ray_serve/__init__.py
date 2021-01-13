import logging

import ray
from ray import serve
from ray.serve.exceptions import RayServeException
from mlflow.deployments import BaseDeploymentClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

import mlflow.pyfunc

logger = logging.getLogger(__name__)


def target_help():
    # TODO: Improve
    help_string = (
        "The mlflow-ray-serve plugin integrates Ray Serve "
        "with the MLFlow deployments API.\n\n"
        "Before using this plugin, you must set up a "
        "detached Ray Serve instance running on "
        "a long-running Ray cluster. "
        "One way to do this is to run `ray start --head` followed by "
        "`serve start`.\n\n"
        "Basic usage:\n\n"
        "    mlflow deployments <command> -t ray-serve\n\n"
        "For more details and examples, see the README at "
        "https://github.com/ray-project/mlflow-ray-serve"
        "/blob/master/README.md")
    return help_string


def run_local(name, model_uri, flavor=None, config=None):
    # TODO: implement
    raise MlflowException("mlflow-ray-serve does not currently "
                          "support run_local.")


# TODO: All models appear in Ray Dashboard as "MLflowBackend".  Improve this.
class MLflowBackend:
    def __init__(self, model_uri):
        self.model = mlflow.pyfunc.load_model(model_uri=model_uri)

    async def __call__(self, request):
        df = await request.body()
        return self.model.predict(df)


class RayServePlugin(BaseDeploymentClient):
    def __init__(self, uri):
        super().__init__(uri)
        try:
            # TODO: support URI and redis password (ray-serve:/192.168....)?
            ray.init(address="auto")
        except ConnectionError:
            raise MlflowException("Could not find a running Ray instance.")
        try:
            self.client = serve.connect()
        except RayServeException:
            raise MlflowException(
                "Could not find a running Ray Serve instance on this Ray "
                "cluster.")

    def help(self):
        return target_help()

    def create_deployment(self, name, model_uri, flavor=None, config=None):
        if flavor is not None and flavor != "python_function":
            raise MlflowException(
                message=(
                    f"Flavor {flavor} specified, but only the python_function "
                    f"flavor is supported by mlflow-ray-serve."),
                error_code=INVALID_PARAMETER_VALUE)
        self.client.create_backend(
            name, MLflowBackend, model_uri, config=config)
        self.client.create_endpoint(name, backend=name)
        return {"name": name, "config": config, "flavor": "python_function"}

    def delete_deployment(self, name):
        self.client.delete_endpoint(name)
        self.client.delete_backend(name)
        logger.info("Deleted model with name: {}".format(name))

    def update_deployment(self, name, model_uri=None, flavor=None,
                          config=None):
        if model_uri is None:
            self.client.update_backend_config(name, config)
        else:
            self.delete_deployment(name)
            self.create_deployment(name, model_uri, flavor, config)
        return {"name": name, "config": config, "flavor": "python_function"}

    def list_deployments(self, **kwargs):
        return [{
            "name": name,
            "config": config
        } for (name, config) in self.client.list_backends().items()]

    def get_deployment(self, name):
        try:
            return {"name": name, "config": self.client.list_backends()[name]}
        except KeyError:
            raise MlflowException(f"No deployment with name {name} found")

    def predict(self, deployment_name, df):
        return ray.get(self.client.get_handle(deployment_name).remote(df))
