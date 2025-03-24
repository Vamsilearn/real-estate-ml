# azure_ml_deploy.py (Azure ML Python SDK v2 style)

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Environment,
    Model,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration  # NEW: Import CodeConfiguration
)
from azure.identity import DefaultAzureCredential
import os

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
    environment = Environment(
        name="real-estate-env",
        description="Environment for real estate model",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    created_environment = ml_client.environments.create_or_update(environment)

    # 4. Create or get the endpoint
    endpoint_name = "real-estate-endpoint"
    endpoint = ManagedOnlineEndpoint(name=endpoint_name)
    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        print(f"Endpoint '{endpoint_name}' already exists.")
    except Exception:
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"Endpoint '{endpoint_name}' created.")

    # 5. Create a deployment for that endpoint using CodeConfiguration
    deployment_name = "real-estate-deployment"
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=registered_model,
        environment=created_environment,
        instance_type="Standard_DS3_v2",  # Consider using a larger SKU
        instance_count=1,
        code_configuration=CodeConfiguration(
            code="./",                # Folder containing your inference script
            scoring_script="inference.py"
        )
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # 6. Direct all traffic to the new deployment
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Deployed model to endpoint '{endpoint_name}' with deployment '{deployment_name}'.")

if __name__ == "__main__":
    main()
