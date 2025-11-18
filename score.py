import json
import pickle
import numpy as np

def init():
    global model, vectorizer

    with open("bank_model.pkl", "rb") as f:
        vecorizer, model = pickle.load(f)


def run(raw_data):
    try:
        body = json.loads(raw_data)

        text = body.get("text", "")

        X= vectorizer.transform([text])
        pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X).toList()[0]
            return {"prediction": pred[0], "probabilities": probs}
        return {"prediction": pred[0], "probabilities": probs}
    
    except Exception as e:
        return {"error": str(e)}
    
    