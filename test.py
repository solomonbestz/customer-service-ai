import sys
import os

# Add current directory to path
sys.path.append('.')

try:
    # Test imports
    import json
    import pickle
    import numpy as np
    print("✅ All imports successful")
    
    # Test model loading
    if os.path.exists("bank_model.pkl"):
        with open("bank_model.pkl", "rb") as f:
            vectorizer, model = pickle.load(f)
        print("✅ Model loaded successfully")
        
        # Test prediction
        test_text = "check my balance"
        X = vectorizer.transform([test_text])
        pred = model.predict(X)
        print(f"✅ Test prediction: {pred[0]}")
    else:
        print("❌ Model file not found!")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()