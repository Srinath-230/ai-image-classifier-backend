import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, request, jsonify
from io import BytesIO
from flask_cors import CORS 

model = tf.keras.applications.MobileNetV2(weights='imagenet')

app = Flask(__name__)
CORS(app)  

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    file = request.files['image']

    img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    
    result = {
        'class_name': decoded_predictions[0][1],
        'confidence': float(decoded_predictions[0][2])
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)