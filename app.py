from flask import Flask, request, jsonify
from ml_model import setup_ml_model

app = Flask(__name__)
ml_model = setup_ml_model()

@app.route('/ml_model', methods=['POST'])
def ml_model_response():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'response': 'No message provided'}), 400

    response = ml_model['user_input'](user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
