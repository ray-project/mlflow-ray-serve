import os
import shutil

import mlflow.pyfunc
import pandas as pd
import requests
from mlflow.deployments import get_deploy_client


# Define the model class
from mlflow.exceptions import MlflowException


class AddN(mlflow.pyfunc.PythonModel):
    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):
        return model_input.apply(lambda column: column + self.n)


# Construct and save the model
model_path = os.path.abspath("models/add_n_model")

try:
    mlflow.pyfunc.save_model(path=model_path, python_model=AddN(n=5))
except MlflowException as e:
    pass
# Evaluate the model
client = get_deploy_client("ray-serve")

client.delete_deployment("add5")
client.create_deployment("add5", model_uri=model_path)
print(client.list_deployments())
print(client.get_deployment("add5"))

model_input = pd.DataFrame([range(10)])
expected_output = pd.DataFrame([range(5, 15)])
model_output = client.predict("add5", model_input)
assert model_output.equals(expected_output)

response = requests.post("http://localhost:8000/add5", json=model_input.to_json())
assert response.status_code == 200
assert pd.read_json(response.content).equals(expected_output)

client.delete_deployment("add5")
shutil.rmtree(model_path)
