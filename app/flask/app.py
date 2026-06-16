from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime

app = Flask(__name__)
limiter = Limiter(app, key_func=get_remote_address)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and preprocessor
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/best_model.pkl')
PREPROCESSOR_PATH = os.environ.get('PREPROCESSOR_PATH', 'models/preprocessor.pkl')

model = None
preprocessor = None

def load_model():
    global model, preprocessor
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info("Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

load_model()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
@limiter.limit("100/minute")
def predict():
    try:
        data = request.get_json()
        features = data.get('features', {})

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Preprocess
        if preprocessor:
            X = preprocessor.transform(df)
        else:
            X = df.values

        # Predict
        probability = model.predict_proba(X)[0, 1]
        prediction = int(probability > 0.5)

        # Risk category
        if probability > 0.7:
            risk = "Very High"
        elif probability > 0.5:
            risk = "High"
        elif probability > 0.3:
            risk = "Medium"
        else:
            risk = "Low"

        return jsonify({
            'success': True,
            'prediction': prediction,
            'transaction_probability': float(probability),
            'risk_category': risk,
            'will_transact': bool(prediction == 1),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
@limiter.limit("10/minute")
def predict_batch():
    try:
        data = request.get_json()
        records = data.get('records', [])

        df = pd.DataFrame(records)

        if preprocessor:
            X = preprocessor.transform(df)
        else:
            X = df.values

        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities > 0.5).astype(int)

        results = []
        for prob, pred in zip(probabilities, predictions):
            if prob > 0.7:
                risk = "Very High"
            elif prob > 0.5:
                risk = "High"
            elif prob > 0.3:
                risk = "Medium"
            else:
                risk = "Low"

            results.append({
                'prediction': int(pred),
                'transaction_probability': float(prob),
                'risk_category': risk
            })

        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': type(model).__name__ if model else 'Not loaded',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
