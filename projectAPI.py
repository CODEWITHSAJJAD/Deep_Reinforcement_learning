import librosa
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os,io

model = tf.keras.models.load_model('audiomodel.h5')
app = Flask(__name__)

# Load class labels
dataset_path = 'ds_audio'
class_labels = sorted(os.listdir(dataset_path))
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram
# Define a prediction function
def predict_audio(audio):
    # Preprocess the image
    waveform, sample_rate = librosa.load(audio, sr=16000)

    # Ensure the waveform is 16000 samples long (pad or truncate if needed)
    if len(waveform) > 16000:
        waveform = waveform[:16000]
    elif len(waveform) < 16000:
        waveform = np.pad(waveform, (0, max(0, 16000 - len(waveform))), "constant")

    # Convert to tensor and process as in the original code
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    spectrogram = get_spectrogram(waveform)
    spectrogram = spectrogram[tf.newaxis, ...]

    # Make prediction
    prediction = model(spectrogram)
    confidence = np.max(prediction)
    predicted_class = class_labels[np.argmax(prediction)]

    # Check confidence and adjust predicted class if necessary
    if confidence < 0.6:  # 60% confidence threshold
        predicted_class = "NOT training audios"

    return predicted_class, confidence
@app.route('/', methods=['GET'])
def welcome():
    return jsonify({
        "message": "Welcome to Audio Classification API",
        "usage": "Send a POST request to /predict with an audio file",
        "expected_classes": class_labels
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        audio_bytes = file.read()
        audio_io = io.BytesIO(audio_bytes)
        predicted_class, confidence = predict_audio(audio_io)
        return jsonify({
            'class': predicted_class,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# import pathlib
# import io, os
# import librosa
#
# app = Flask(__name__)
#
# # Configuration
# DATASET_PATH = 'ds_audio'
# class_labels = sorted(os.listdir(DATASET_PATH))
# data_dir = pathlib.Path(DATASET_PATH)
# model = tf.keras.models.load_model("audiomodel.h5")
# CONFIDENCE_THRESHOLD = 0.8
#
# ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
#
#
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# def preprocess_audio(waveform, target_length=16000):
#     """Preprocess audio to match training conditions"""
#     # Normalize volume
#     waveform = librosa.util.normalize(waveform)
#
#     # Ensure correct length
#     if len(waveform) > target_length:
#         waveform = waveform[:target_length]
#     elif len(waveform) < target_length:
#         padding = target_length - len(waveform)
#         waveform = np.pad(waveform, (0, padding), 'constant')
#
#     return waveform
#
#
# def get_spectrogram(waveform):
#     """Create spectrogram matching training parameters"""
#     spectrogram = tf.signal.stft(waveform, frame_length=512, frame_step=256, fft_length=512)
#     spectrogram = tf.abs(spectrogram)
#     spectrogram = tf.math.log(spectrogram + 1e-6)
#     spectrogram = spectrogram[..., tf.newaxis]
#     return spectrogram
#
#
# @app.route('/', methods=['GET'])
# def welcome():
#     return jsonify({
#         "message": "Welcome to Audio Classification API",
#         "usage": "Send a POST request to /predict with an audio file",
#         "expected_classes": class_labels,
#         "confidence_threshold": CONFIDENCE_THRESHOLD
#     })
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#
#     file = request.files['file']
#
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#     if not allowed_file(file.filename):
#         return jsonify({"error": "File type not allowed"}), 400
#
#     try:
#         # Read and preprocess audio
#         audio_bytes = file.read()
#         audio_io = io.BytesIO(audio_bytes)
#
#         # Load with librosa (mono, 16kHz)
#         waveform, sample_rate = librosa.load(audio_io, sr=16000, mono=True)
#         waveform = preprocess_audio(waveform)
#
#         # Convert to spectrogram
#         waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
#         spectrogram = get_spectrogram(waveform)
#         spectrogram = spectrogram[tf.newaxis, ...]
#
#         # Make prediction
#         probabilities = model.predict(spectrogram, verbose=0)[0]
#         predicted_index = np.argmax(probabilities)
#         confidence = float(probabilities[predicted_index])
#
#         # Check confidence threshold
#         if confidence < CONFIDENCE_THRESHOLD:
#             return jsonify({
#                 "error": "Low confidence prediction",
#                 "predicted_class": class_labels[predicted_index],
#                 "confidence": confidence,
#                 "message": f"Confidence below threshold of {CONFIDENCE_THRESHOLD}"
#             }), 400
#
#         # Prepare successful response
#         response = {
#             "predicted_class": class_labels[predicted_index],
#             "confidence": confidence,
#             "all_predictions": {label: float(prob) for label, prob in zip(class_labels, probabilities)}
#         }
#
#         return jsonify(response)
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)