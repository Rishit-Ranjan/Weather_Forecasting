
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('rainfall_predictor_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)
        
        # Create a numpy array from the input data
        final_features = np.array([data['features']])
        
        # Scale the features
        scaled_features = scaler.transform(final_features)
        
        # Make a prediction
        prediction = model.predict(scaled_features)
        
        # Prepare the response
        # The model is LinearRegression, so it predicts rainfall amount in mm, not Yes/No
        output = f"{prediction[0]:.2f} mm"
        
        return jsonify({'prediction': output})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
