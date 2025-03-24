# azure_ml_deploy.py (Azure ML Python SDK v2 style)

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (Environment, AciWebserviceDeploymentConfiguration, Model, ManagedOnlineEndpoint, ManagedOnlineDeployment)
from azure.identity import DefaultAzureCredential
import os
import joblib

def main():
    # 1. Authenticate & initialize MLClient (assumes you are using DefaultAzureCredential)
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_ML_WORKSPACE_NAME")

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name
    )

    # 2. Register model
    model_path = "./model/model.joblib"
    model = Model(
        path=model_path,
        name="real-estate-model",
        description="Linear regression model for real estate pricing",
    )
    registered_model = ml_client.models.create_or_update(model)

    # 3. Create environment (Docker environment with your dependencies)
    # This is a minimal example; you can also point to a curated environment or environment.yml
    environment = Environment(
        name="real-estate-env",
        description="Environment for real estate model",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    created_environment = ml_client.environments.create_or_update(environment)

    # 4. Create an endpoint (unique name needed, e.g. real-estate-endpoint-123)
    endpoint_name = "real-estate-endpoint"
    endpoint = ManagedOnlineEndpoint(name=endpoint_name)
    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        print(f"Endpoint {endpoint_name} already exists.")
    except Exception:
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"Endpoint {endpoint_name} created.")

    # 5. Create a deployment for that endpoint
    deployment_name = "real-estate-deployment"
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=registered_model,
        environment=created_environment,
        instance_type="Standard_DS2_v2",
        instance_count=1,
        # scoring script if you want a custom entry script for inference
        # or an environment variable specifying your python inference function
        code_path="./",  # folder that contains your scoring script
        entry_script="inference.py", 
    )
    # We update or create the deployment
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # 6. Set traffic to the new deployment (100% traffic)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Deployed model to endpoint {endpoint_name} with deployment {deployment_name}")

if __name__ == "__main__":
    main()
