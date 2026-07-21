from utils import db_connect
engine = db_connect()

# your code here
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image

app = Flask(__name__, template_folder='../templates', static_folder='../static')

MODEL_PATH = 'models/mnist_model.h5'
model = None

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo cargado")
else:
    print("No se encontró el archivo")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'error'}), 500
    try:
        file = request.files['image']
        img = Image.open(file).convert('L').resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = 1.0 - img_array
        img_array = img_array.reshape(1, 28, 28)

        prediction = model.predict(img_array)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        return jsonify({
            'digit': predicted_digit,
            'confidence': f"{confidence * 100:.2f}%"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
