from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import traceback
from ml_model import setup_ml_model

app = Flask(__name__)
CORS(app)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Setup ML model
ml_functions = setup_ml_model()

@app.route('/ml_model', methods=['POST'])
def ml_model():
    try:
        data = request.get_json()
        user_message = data['message']
        
        # Call the user_input function from ml_model.py
        response_message = ml_functions['user_input'](user_message)
        
        return jsonify({"response": response_message})
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
