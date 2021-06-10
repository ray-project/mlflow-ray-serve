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

model_5_path = os.path.abspath("models/add_5_model")
try:
    mlflow.pyfunc.save_model(path=model_5_path, python_model=AddN(n=5))
except MlflowException:
    pass

model_6_path = os.path.abspath("models/add_6_model")
try:
    mlflow.pyfunc.save_model(path=model_6_path, python_model=AddN(n=6))
except MlflowException:
    pass

# Evaluate the model
client = get_deploy_client("ray-serve")


try:
    client.delete_deployment("addN")
    client.create_deployment("addN", model_5_path)
    print(client.list_deployments())
    print(client.get_deployment("addN"))

    model_input = pd.DataFrame([range(10)])
    model_output = client.predict("addN", model_input)
    assert model_output.equals(pd.DataFrame([range(5, 15)]))
    
    client.update_deployment("addN", model_uri=model_6_path, config={"model_traffic": 1.0})
    print(client.get_deployment("addN"))

    model_input = pd.DataFrame([range(10)])
    expected_output = pd.DataFrame([range(6, 16)])
    model_output = client.predict("addN", model_input)
    assert model_output.equals(expected_output)
    
    response = requests.post("http://localhost:8000/addN", json=model_input.to_json())
    assert response.status_code == 200
    assert pd.read_json(response.content).equals(expected_output)
finally:
    client.delete_deployment("addN")
    shutil.rmtree(os.path.dirname(model_5_path))
