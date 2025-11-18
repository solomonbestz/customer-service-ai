# This will create both resource group and workspace
# az ml workspace create \
#   --name "ai-customer-service" \
#   --resource-group "AI-Group" \
#   --location "eastus" \
#   --create-resource-group

from azure.ai.ml import MLClient
from azure.core.exceptions import HttpResponseError
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment, CodeConfiguration
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import uuid
import time
import os

load_dotenv()


SUBSCRIPTION = os.getenv("YOUR_SUBSCRIPTION_ID")
RESOURCE_GROUP = "AI-Group"
WORKSPACE = "ai-customer-service"

try:
    # --- Initialize ML Client ---
    ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    print("Connected to Azure ML workspace")

    # --- Use existing model and environment ---
    model = ml_client.models.get(name="bank-intent-model", version="4")
    env = Environment(
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    
    print(f"Using model: {model.name} v{model.version}")
    # print(f"Using environment: {environment.name} v{environment.version}")

    # --- Create endpoint ---
    endpoint_name = f"bank-live-{uuid.uuid4().hex[:6]}"
    print(f"Creating endpoint: {endpoint_name}")
    
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name, 
        auth_mode="key",
        description="Bank intent classification endpoint"
    )

    # --- Create the endpoint ---
    print("Creating endpoint...")
    endpoint_poller = ml_client.begin_create_or_update(endpoint, local=False)
    endpoint_result = endpoint_poller.result()
    print(f"Endpoint created: {endpoint_name}")

    # Wait for endpoint to be ready
    print("Waiting for endpoint to be ready...")
    time.sleep(20)


    instance_types_to_try = [
        "Standard_DS2_v2",
        "Standard_F2s_v2", 
        "Standard_E2s_v3",
        "Standard_D2as_v4"
    ]

    successful_deployment = False

    for instance_type in instance_types_to_try:
        try:
            print(f"Trying instance type: {instance_type}")
            
            deployment = ManagedOnlineDeployment(
                name="blue",
                endpoint_name=endpoint_name,
                model=model.id,
                environment=env,
                code_configuration=CodeConfiguration(
                code=".", scoring_script="score.py"
                ),
                instance_type=instance_type,
                instance_count=1,
            )

            deployment_poller = ml_client.online_deployments.begin_create_or_update(deployment, local=False, timeout=3600)
            deployment_result = deployment_poller.result()
            print(f"Deployment successful with {instance_type}!")
            successful_deployment = True
            break
            
        except HttpResponseError as e:
            if "quota" in str(e).lower() or "outofquota" in str(e).lower():
                print(f"Quota exceeded for {instance_type}, trying next...")
                continue
            else:
                print(e)
                continue

    if not successful_deployment:
        print("All deployment attempts failed")
        exit(1)


    # --- Set traffic ---
    print("Setting traffic to 100%...")
    ml_client.online_endpoints.update(
        name=endpoint_name,
        traffic={"blue": 100}
        
    ).result()

    print(f"\nDEPLOYMENT SUCCESSFUL!")
    print(f"Endpoint: {endpoint_name}")
    print(f"Scoring URI: {endpoint_result.scoring_uri}")
    print(f"To test: curl -X POST {endpoint_result.scoring_uri} -H 'Authorization: Bearer YOUR_KEY' -H 'Content-Type: application/json' -d 'YOUR_DATA'")

except HttpResponseError as e:
    print(f"Azure Error: {e.message}")
    if "SubscriptionNotRegistered" in str(e):
        print("\nSOLUTION: Resource providers not registered yet.")
        print("Run these commands and wait 10-15 minutes:")
        print("az provider register --namespace Microsoft.Network")
        print("az provider register --namespace Microsoft.Compute") 
        print("az provider register --namespace Microsoft.ContainerInstance")
        print("\nThen run this script again.")
    elif "QuotaExceeded" in str(e):
        print("\nTry a smaller instance: Standard_B2s or Standard_B1s")
        
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

