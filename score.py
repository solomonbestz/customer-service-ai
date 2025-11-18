# import json
# import pickle
# import numpy
# import joblib
# import logging
# import os

# def init():
#     global model, vectorizer

#     model_path = os.path.join(
#         os.getenv("AZUREML_MODEL_DIR"), "model/bank_model.pkl"
#     )
#     # deserialize the model file back into a sklearn model
#     model = joblib.load(model_path)
#     logging.info("Init complete")


# def run(raw_data):
#     """
#     This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
#     In the example we extract the data from the json input and call the scikit-learn model's predict()
#     method and return the result back
#     """
#     logging.info("model 1: request received")
#     data = json.loads(raw_data)["data"]
#     data = numpy.array(data)
#     result = model.predict(data)
#     logging.info("Request processed")
#     return result.tolist()

# # def run(raw_data):
# #     try:
# #         body = json.loads(raw_data)

# #         text = body.get("text", "")

# #         X= vectorizer.transform([text])
# #         pred = model.predict(X)

# #         probs = []

# #         if hasattr(model, "predict_proba"):
# #             probs = model.predict_proba(X).tolist()[0]
# #         return {"prediction": str(pred[0]), "probabilities": probs}
    
# #     except Exception as e:
# #         return {"error": str(e)}
    
    

import os
import logging
import json
import numpy as np
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)

def init():
    """
    Initialize the model when the container starts
    """
    global model, vectorizer
    
    try:
        # AZUREML_MODEL_DIR is provided by Azure - it's where your model files are stored
        model_dir = os.getenv("AZUREML_MODEL_DIR")
        logging.info(f"Model directory: {model_dir}")
        
        # List files to see what's available
        if model_dir and os.path.exists(model_dir):
            logging.info(f"Files in model dir: {os.listdir(model_dir)}")
        
        # Load your model - adjust path based on what you see above
        model_path = os.path.join(model_dir, "bank_model.pkl")
        
        if not os.path.exists(model_path):
            # Fallback: try current directory
            model_path = "bank_model.pkl"
            logging.info("Trying current directory for model file")
        
        logging.info(f"Loading model from: {model_path}")
        
        # Load model using joblib (more reliable than pickle)
        with open(model_path, "rb") as f:
            # Your model is stored as (vectorizer, model) tuple
            vectorizer, model = joblib.load(f)
        
        logging.info("✅ Model loaded successfully")
        logging.info(f"Vectorizer type: {type(vectorizer)}")
        logging.info(f"Model type: {type(model)}")
        
    except Exception as e:
        logging.error(f"❌ Model loading failed: {str(e)}")
        raise

def run(raw_data):
    """
    Handle prediction requests
    """
    try:
        logging.info("📥 Received prediction request")
        
        # Parse input data
        data = json.loads(raw_data)
        
        # Get text from request - support different input formats
        text = data.get("text", "")
        if not text:
            # Alternative: check if data is directly the text
            if isinstance(data, str):
                text = data
            else:
                return {"error": "No text provided in 'text' field"}
        
        logging.info(f"Processing text: {text}")
        
        # Transform and predict
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        
        # Get probabilities if available
        probabilities = []
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X).tolist()[0]
        
        # Return result
        result = {
            "prediction": str(prediction),
            "probabilities": probabilities
        }
        
        logging.info(f"✅ Prediction: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logging.error(f"❌ {error_msg}")
        return {"error": error_msg}