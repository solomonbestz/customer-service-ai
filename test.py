import requests
import json

# Your endpoint details from the output
scoring_uri = "https://bank-live-78bb0e.eastus.inference.ml.azure.com/score"
key = "your-primary-key-here"

# Test request
data = {"text": "I want to check my balance"}
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}

response = requests.post(scoring_uri, json=data, headers=headers)
print(f"Response: {response.status_code}")
print(f"Prediction: {response.text}")