# from azure.ai.ml import MLClient
# from azure.ai.ml.entities import Model, Environment, ManagedOnlineEndpoint, ManagedOnlineDeployment
# from azure.identity import DefaultAzureCredential
# from dotenv import load_dotenv
# import time
# import os



# load_dotenv()

# SUBSCRIPTION = os.getenv("YOUR_SUBSCRIPTION_ID")

# RESOURCE_GROUP = "AI-Group"
# WORKSPACE = "ai-customer-service"
# ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)


# model = Model(
#     path="bank_model.pkl", 
#     name="bank-intent-model",
#     description="Bank intent model (vectorizer + logistic regression)"
# )
# registered = ml_client.models.create_or_update(model)
# print("Registered model id:", registered.id)


# env = Environment(
#     name="bank-intent-env",
#     description="Environment for intent model",
#     conda_file="environment.yml",
#     image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
# )

# env = ml_client.environments.create_or_update(env)
# print("Environment created:", env.id)

# endpoint_name = f"bank-intent-endpoint-{uuid.uuid4().hex[:6]}"

# endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")

# ml_client.begin_create_or_update(endpoint).result()
# print("Endpoint created:", endpoint_name)


# deployment = ManagedOnlineDeployment(
#     name="blue",
#     endpoint_name=endpoint_name,
#     model=registered.id,
#     environment=env.id,
#     code_path=".",
#     instance_type="Standard_DS2_v2",  
#     instance_count=1,
# )

# ml_client.online_deployments.begin_create_or_update(deployment).wait()
# print("Deployment finished.")


# ml_client.online_endpoints.begin_update(
#     endpoint_name,
#     traffic={"blue": 100}
# ).wait()

# print("Deployment ACTIVE.")

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, Environment, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os
import uuid
import time

load_dotenv()

# --- Azure subscription details ---
SUBSCRIPTION = os.getenv("YOUR_SUBSCRIPTION_ID")
RESOURCE_GROUP = "AI-Group"
WORKSPACE = "ai-customer-service"

# --- Initialize ML Client ---
ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)

# --- Register the model ---
model = Model(
    path="bank_model.pkl", 
    name="bank-intent-model",
    description="Bank intent model (vectorizer + logistic regression)"
)
registered = ml_client.models.create_or_update(model)
print("Registered model id:", registered.id)

# --- Create / update environment ---
env = Environment(
    name="bank-intent-env",
    description="Environment for intent model",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)
env = ml_client.environments.create_or_update(env)
print("Environment created:", env.id)

# --- Generate a unique endpoint name ---
endpoint_name = f"bank-intent-endpoint-{uuid.uuid4().hex[:6]}"
endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")

# --- Create the endpoint ---
print(f"Creating endpoint: {endpoint_name} ...")
ml_client.begin_create_or_update(endpoint).result()
print(f"Endpoint created: {endpoint_name}")

# --- Create deployment ---
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=registered.id,
    environment=env.id,
    code_path=".",          # Your scoring scripts are in this folder
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

print("Deploying model ...")
ml_client.online_deployments.begin_create_or_update(deployment).wait()
print("Deployment finished.")

# --- Set traffic to deployment ---
ml_client.online_endpoints.begin_update(
    endpoint_name,
    traffic={"blue": 100}
).wait()
print("Deployment ACTIVE and receiving 100% traffic.")
