from flask import Flask, request, jsonify
from flasgger import Swagger
import numpy as np
import joblib

app = Flask(__name__)

swagger_config = {
                    "headers": [],
                    "specs": [{
                                "rule_filter": lambda rule: True,
                                "model_filter": lambda tag: True,
                             }],
                    "static_url_path": "/flasgger_static",
                    "swagger_ui": True,
                    "specs_route": "/"
                }

swagger = Swagger(app, config=swagger_config)

knn_model = joblib.load('knn_model.pkl')
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions.

    ---
    parameters:
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
        data = request.json

        input_data = np.array(data).reshape(1, -1)

        prediction = knn_model.predict(input_data)

        prediction_list = prediction.tolist()

        return jsonify({'prediction': prediction_list})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
