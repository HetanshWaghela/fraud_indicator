from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model at startup
try:
    model = joblib.load('fraud_detection_model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

# Expected feature names
REQUIRED_FEATURES = ['vendor_score', 'image_qty', 'site_age', 'delivery_period', 
                     'typo_count', 'payment_options', 'cost_usd']

def validate_input(data):
    """Validate input data"""
    if not isinstance(data, dict):
        return False, "Input must be a JSON object"
    
    # Check if all required features are present
    missing_features = [feature for feature in REQUIRED_FEATURES if feature not in data]
    if missing_features:
        return False, f"Missing required features: {missing_features}"
    
    # Check data types and ranges
    try:
        # Vendor score should be between 1-5
        if not (1.0 <= float(data['vendor_score']) <= 5.0):
            return False, "vendor_score must be between 1.0 and 5.0"
        
        # Image quantity should be positive integer
        if not (1 <= int(data['image_qty']) <= 20):
            return False, "image_qty must be between 1 and 20"
        
        # Site age should be positive
        if float(data['site_age']) < 0:
            return False, "site_age must be non-negative"
        
        # Delivery period should be positive
        if not (1 <= int(data['delivery_period']) <= 50):
            return False, "delivery_period must be between 1 and 50"
        
        # Typo count should be non-negative
        if int(data['typo_count']) < 0:
            return False, "typo_count must be non-negative"
        
        # Payment options should be between 1-5
        if not (1 <= int(data['payment_options']) <= 5):
            return False, "payment_options must be between 1 and 5"
        
        # Cost should be positive
        if float(data['cost_usd']) <= 0:
            return False, "cost_usd must be positive"
            
    except (ValueError, TypeError):
        return False, "Invalid data types in input"
    
    return True, "Valid input"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Create DataFrame with correct feature order
        df = pd.DataFrame([data], columns=REQUIRED_FEATURES)
        
        # Make prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        # Get confidence score
        confidence = max(probabilities)
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'prediction_label': 'Fraud' if prediction == 1 else 'Not Fraud',
            'confidence': float(confidence),
            'probability_fraud': float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0]),
            'probability_not_fraud': float(probabilities[0]) if len(probabilities) > 1 else float(1 - probabilities[0]),
            'input_data': data
        }
        
        logger.info(f"Prediction made: {response['prediction_label']} with confidence {confidence:.3f}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Handle batch predictions"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided. Use format: {"samples": [...]'}), 400
        
        samples = data['samples']
        if not isinstance(samples, list):
            return jsonify({'error': 'Samples must be a list'}), 400
        
        results = []
        for i, sample in enumerate(samples):
            is_valid, message = validate_input(sample)
            if not is_valid:
                results.append({
                    'sample_index': i,
                    'error': message
                })
                continue
            
            df = pd.DataFrame([sample], columns=REQUIRED_FEATURES)
            prediction = model.predict(df)[0]
            probabilities = model.predict_proba(df)[0]
            
            results.append({
                'sample_index': i,
                'prediction': int(prediction),
                'prediction_label': 'Fraud' if prediction == 1 else 'Not Fraud',
                'confidence': float(max(probabilities)),
                'probability_fraud': float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
            })
        
        return jsonify({'predictions': results}), 200
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        'status': 'API is running',
        'model_loaded': model is not None,
        'required_features': REQUIRED_FEATURES
    }
    return jsonify(status), 200

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = {
            'model_type': str(type(model).__name__),
            'features': REQUIRED_FEATURES,
            'n_features': len(REQUIRED_FEATURES)
        }
        
        # Try to get additional model info if available
        if hasattr(model, 'feature_names_in_'):
            info['model_features'] = model.feature_names_in_.tolist()
        
        return jsonify(info), 200
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    docs = {
        'message': 'Fraud Detection API',
        'endpoints': {
            '/predict': 'POST - Make single prediction',
            '/predict/batch': 'POST - Make batch predictions',
            '/health': 'GET - Health check',
            '/model/info': 'GET - Model information',
            '/': 'GET - API documentation'
        },
        'required_features': REQUIRED_FEATURES,
        'example_request': {
            'vendor_score': 4.5,
            'image_qty': 5,
            'site_age': 1000.0,
            'delivery_period': 7,
            'typo_count': 1,
            'payment_options': 3,
            'cost_usd': 150.50
        }
    }
    return jsonify(docs), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)