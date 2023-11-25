from flask import Flask, request, jsonify
from flasgger import Swagger
import numpy as np
import joblib

app = Flask(__name__)

swagger_config = {
                    "headers": [],
                    "specs": [{
                                "endpoint": 'phishing_detection',
                                "route": '/phishing_detection.json',
                                "rule_filter": lambda rule: True,
                                "model_filter": lambda tag: True,
                             }],
                    "static_url_path": "/flasgger_static",
                    "swagger_ui": True,
                    "specs_route": "/"
                }

swagger = Swagger(app, config=swagger_config)

knn_model = joblib.load('models/knn_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')


model_dict = {
    'knn': knn_model,
    'rf': rf_model,
    'xgb': xgb_model,
    'svm': svm_model
}

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions.

    ---
    parameters:
      - name: model
        in: query
        type: string
        required: true
        description: The model to use for prediction
        enum: ['knn', 'rf', 'xgb', 'svm']
      - name: data
        in: body
        required: true
        schema:
          type: array
          items:
            type: number
    responses:
      200:
        description: OK
        schema:
          type: object
          properties:
            prediction:
              type: array
              items:
                type: number
    """
    try:
        model_name = request.args.get('model')
        if model_name not in model_dict:
            return jsonify({'error': f'Model {model_name} not found'})

        selected_model = model_dict[model_name]
        data = request.json
        input_data = np.array(data).reshape(1, -1)
        prediction = selected_model.predict(input_data)
        prediction_list = prediction.tolist()

        return jsonify({'prediction': prediction_list})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
