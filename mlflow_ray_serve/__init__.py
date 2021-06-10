import logging
import urllib.parse
from dataclasses import dataclass
from typing import Optional, Tuple

import mlflow.pyfunc
import pandas as pd
import ray
from mlflow.deployments import BaseDeploymentClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from ray import serve
from ray.serve.exceptions import RayServeException
from starlette.requests import Request

try:
    import ujson as json
except ModuleNotFoundError:
    import json

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
        "/blob/master/README.md"
    )
    return help_string


def run_local(name, model_uri, flavor=None, config=None):
    # TODO: implement
    raise MlflowException("mlflow-ray-serve does not currently " "support run_local.")


# TODO: All models appear in Ray Dashboard as "MLflowBackend".  Improve this.
class MLflowBackend:
    def __init__(self, model_uri):
        self.model = mlflow.pyfunc.load_model(model_uri=model_uri)

    async def _process_request_data(self, request: Request) -> pd.DataFrame:
        body = await request.body()
        if isinstance(body, pd.DataFrame):
            return body
        return pd.read_json(json.loads(body))

    async def __call__(self, request: Request):
        df = await self._process_request_data(request)
        return self.model.predict(df).to_json(orient="records")


@dataclass
class DeploymentConfig:
    """
    Deployment configuration data class

    :arg config:
        model backend configuration
    :arg ray_actor_options:
        the ray actor options
    :arg model_traffic:
        traffic portion assigned to deployed model
    """

    config: Optional[Dict] = None
    ray_actor_options: Optional[Dict] = None
    model_traffic: float = 0.5


class RayServePlugin(BaseDeploymentClient):
    def __init__(self, uri):
        super().__init__(uri)
        try:
            address = self._parse_ray_server_uri(uri)
            if address is not None and address.strip():
                ray.util.connect(address)  # client connection
            else:
                ray.init(address="auto")
        except ConnectionError:
            raise MlflowException("Could not find a running Ray instance.")
        try:
            self.client = serve.connect()
        except RayServeException:
            raise MlflowException(
                "Could not find a running Ray Serve instance on this Ray " "cluster."
            )

    def help(self):
        return target_help()

    def create_deployment(self, name, model_uri, flavor=None, config=None):
        if flavor is not None and flavor != "python_function":
            raise MlflowException(
                message=(
                    f"Flavor {flavor} specified, but only the python_function "
                    f"flavor is supported by mlflow-ray-serve."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        deployment_config = self._parse_deployment_config(config)
        try:
            self.client.create_backend(
                model_uri,
                MLflowBackend,
                model_uri,
                ray_actor_options=deployment_config.ray_actor_options,
                config=deployment_config.config,
            )
        except ValueError as ex:
            logger.error("Cannot deploy model. Model {} already exits.".format(model_uri))
            raise ex

        self._update_endpoint(
            endpoint=name,
            new_backend=model_uri,
            backend_traffic=deployment_config.model_traffic,
        )
        return {"name": name, "config": config, "flavor": "python_function"}

    @staticmethod
    def _update_endpoint(endpoint: str, new_backend: str, backend_traffic: float = 0.1):
        """
        Updates or creates an endpoint by setting backend traffic

        :param endpoint:
            The target endpoint
        :param new_backend:
            New model backend to include under endpoint
        :param backend_traffic:
            Backend traffic portion to assign to model backend
        """
        endpoints = serve.list_endpoints()
        endpoint_cfg = endpoints.get(endpoint)
        if endpoint_cfg is None:
            serve.create_endpoint(endpoint, backend=new_backend, route=f"/{endpoint}")
            return

        traffic = endpoint_cfg.get("traffic", {})
        new_endpoint_traffic = {new_backend: backend_traffic}

        backends_to_remove = []
        if backend_traffic >= 1.0:
            backends_to_remove = [backend for backend in traffic]
        else:
            traffic_per_backend = (1 - backend_traffic) / len(traffic)
            new_endpoint_traffic.update({k: traffic_per_backend for k in traffic})

        serve.set_traffic(endpoint, traffic_policy_dictionary=new_endpoint_traffic)
        for backend in backends_to_remove:
            serve.delete_backend(backend)

    @staticmethod
    def _parse_deployment_config(config):
        def unflatten(dictionary):
            resultDict = dict()
            for key, value in dictionary.items():
                parts = key.split(".")
                d = resultDict
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = dict()
                    d = d[part]
                d[parts[-1]] = value
            return resultDict
        config = unflatten(config or {})
        deployment_config = DeploymentConfig(**config)
        return deployment_config

    def delete_deployment(self, name):
        deployment_endpoint = self.client.list_endpoints().get(name)
        if deployment_endpoint:
            self.client.delete_endpoint(name)
            backends = [backend for backend in deployment_endpoint.get("traffic", [])]
            backends.extend(
                [backend for backend in deployment_endpoint.get("shadow", [])]
            )
            for backend in backends:
                self.client.delete_backend(backend)
            logger.info(
                "Deleted deployment endpoint with name: {} and all related backend models [{}]".format(
                    name, ",".join(backends)
                )
            )

    def update_deployment(self, name, model_uri=None, flavor=None, config=None):
        deployment_config = self._parse_deployment_config(config)
        if model_uri is None:
            deployment_endpoint = self.client.list_endpoints().get(name)
            if deployment_endpoint:
                for backend in deployment_endpoint.get("traffic", []):
                    self.client.update_backend_config(backend, deployment_config.config)
        else:
            self.create_deployment(name, model_uri, flavor, config)
        return {"name": name, "config": config, "flavor": "python_function"}

    def list_deployments(self, **kwargs):
        endpoints = self.client.list_endpoints()
        return [
            {
                "name": name,
                "config": {
                    "traffic": endpoint_config.get("traffic"),
                    "backends": {
                        k: self.client.get_backend_config(k)
                        for k in endpoint_config.get("traffic")
                    },
                },
            }
            for (name, endpoint_config) in endpoints.items()

    def get_deployment(self, name):
        try:
            endpoints = self.client.list_endpoints()
            endpoint_config = endpoints[name]
            return {
                "name": name,
                "config": {
                    "traffic": endpoint_config.get("traffic"),
                    "backends": {
                        k: self.client.get_backend_config(k)
                        for k in endpoint_config.get("traffic")
                    },
                },
            }
        except KeyError:
            raise MlflowException(f"No deployment with name {name} found")

    def predict(self, deployment_name, df):
        predictions_json = ray.get(self.client.get_handle(deployment_name).remote(df))
        return pd.read_json(predictions_json)

    @staticmethod
    def _parse_ray_server_uri(uri: str) -> Optional[str]:
        """
        Uri accepts password and host/port

        Examples:
        >> ray-serve://my-host:10001
        """

        if not uri.startswith("ray-serve://"):
            return None

        parsed_url = urllib.parse.urlparse(uri)
        address = parsed_url.hostname
        return address
