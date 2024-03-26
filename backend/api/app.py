from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sentence_transformers import SentenceTransformer
from SentenceLibrary import get_model_predictions

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

app = Flask(__name__)

# Enable CORS for all domains on all routes
CORS(app)

@app.route('/capitalize', methods=['POST'])
@cross_origin(origins='*')  # This allows all origins
def capitalize():
    data = request.get_json()
    
    # Your logic remains unchanged
    def capitalize_strings(value):
        if isinstance(value, str):
            return value.upper()
        elif isinstance(value, dict):
            return {k: capitalize_strings(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [capitalize_strings(i) for i in value]
        else:
            return value
    
    capitalized_data = capitalize_strings(data)
    
    return jsonify(capitalized_data)


@app.route('/predict_classes', methods=['POST'])
@cross_origin(origins='*')
def predict_classes():
  data = request.get_json()
  text = data['text']
  return jsonify(get_model_predictions(text))

# @app.route('/euclidian', methods=['POST'])
# @cross_origin(origins='*')  # This allows all origins
# def get_euclidian_class():
#     data = request.get_json()
#     print(data)
#     sentence = data['text']
#     classes= get_euclidian_predictions(sentence)
#     return jsonify({"data": classes})

if __name__ == '__main__':
    app.run(debug=True)
