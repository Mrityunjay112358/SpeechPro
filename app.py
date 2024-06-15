from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os

app = Flask(__name__)
model = tf.keras.models.load_model('models/sign_language_model.h5')

def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

def convert_gesture_to_text(gesture):
    if gesture < 10:
        return str(gesture)
    elif 10 <= gesture < 36:
        return chr(gesture - 10 + ord('A'))
    else:
        return '_'

def text_to_speech(text, language):
    lang_code = 'hi' if language == 'hi' else 'en'
    print(f"Generating speech in language: {lang_code}")
    tts = gTTS(text=text, lang=lang_code)
    tts.save("output.mp3")
    os.system("mpg321 output.mp3 & wait")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request")
        file = request.files['file']
        language = request.form.get('language', 'en')
        print(f"Language: {language}")

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        print(f"Image shape: {img.shape}")

        processed_img = preprocess_image(img)
        print(f"Processed image shape: {processed_img.shape}")

        prediction = model.predict(processed_img)
        gesture = np.argmax(prediction)
        gesture_text = convert_gesture_to_text(gesture)
        print(f"Predicted gesture: {gesture_text}")

        text_to_speech(f'{gesture_text}', language)
        print("Speech generated and played successfully")

        return jsonify({'gesture': gesture_text})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
