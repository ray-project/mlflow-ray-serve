# MLflow-Ray-Serve

Am experimental plugin that integrates [Ray Serve](https://docs.ray.io/en/master/serve/) with the MLflow pipeline.
``mlflow-ray-serve`` enables MLflow users to deploy MLflow models at scale on Ray Serve.

This plugin implements the [Python API](https://www.mlflow.org/docs/latest/python_api/mlflow.deployments.html)
and [command-line interface](https://www.mlflow.org/docs/latest/cli.html#mlflow-deployments) for MLflow deployment plugins.

## Installation

```bash
pip install mlflow-ray-serve
```

The following packages are required and will be installed along with the plugin:

1. `"ray[serve]"`
2. `"mlflow>=1.12.0"`

Currently, this plugin only works with the Ray [nightly build](https://docs.ray.io/en/master/installation.html#daily-releases-nightlies).
It will be supported in the stable Ray releases starting with Ray 1.2.0.

To install the Ray nightly build:
```bash
pip install -U ray
ray install-nightly
pip install "ray[serve]"
```

## Usage
This plugin must be used with a detached Ray Serve instance running on a Ray cluster.  An easy way to set this up is by running the following two commands:

```bash
ray start --head # Start a single-node Ray cluster locally.
serve start # Start a detached Ray Serve instance.
```

The API is summarized below. For full details see the MLflow deployment plugin [Python API](https://www.mlflow.org/docs/latest/python_api/mlflow.deployments.html)
and [command-line interface](https://www.mlflow.org/docs/latest/cli.html#mlflow-deployments) documentation.

### Create deployment
Deploy a model built with MLflow using Ray Serve with the desired [configuration parameters](https://docs.ray.io/en/master/serve/package-ref.html#backend-configuration); for example, `num_replicas`.  Currently this plugin only supports the `python_function` flavor of MLflow models, and this is the default flavor.

##### CLI
```bash
mlflow deployments create -t ray-serve -m <model uri> --name <deployment name> -C num_replicas=<number of replicas>
```

##### Python API
```python
from mlflow.deployments import get_deploy_client
target_uri = 'ray-serve'
plugin = get_deploy_client(target_uri)
plugin.create_deployment(
    name=<deployment name>,
    model_uri=<model uri>,
    config={"num_replicas": 4})
```

### Update deployment
Modify the configuration of a deployed model and/or replace the deployment with a new model URI.

##### CLI
```bash
mlflow deployments update -t ray-serve --name <deployment name> -C num_replicas=<new number of replicas>
```

##### Python API
```python
plugin.update_deployment(name=<deployment name>, config={"num_replicas": <new number of replicas>})
```

### Delete deployment
Delete an existing deployment.

##### CLI
```bash
mlflow deployments delete -t ray-serve --name <deployment name>
```

##### Python API
```python
plugin.delete_deployment(name=<deployment name>)
```

### List deployments
List the names of all the models deployed on Ray Serve.  Includes models not deployed via this plugin.

##### CLI
```bash
mlflow deployments list -t ray-serve
```

##### Python API
```python
plugin.list_deployments()
```

### Get deployment details

##### CLI
```bash
mlflow deployments get -t ray-serve --name <deployment name>
```

##### Python API
```python
plugin.get_deployment(name=<deployment name>)
```

### Run prediction on deployed model
For the prediction inputs, DataFrame, Tensor and JSON formats are supported by the Python API.  To invoke via the command line, pass in the path to a JSON file containing the input.

##### CLI
```bash
mlflow deployments predict -t ray-serve --name <deployment name> --input-path <input file path> --output-path <output file path>
```

`output-path` is an optional parameter. Without it, the result will be printed in the terminal.

##### Python API
```python
plugin.predict(name=<deployment name>, df=<prediction input>)
```

### Plugin help
Prints the plugin help string.

##### CLI
```bash
mlflow deployments help -t ray-serve
```

