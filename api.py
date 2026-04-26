from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io, os
import librosa
from flask_cors import CORS
DATASET_PATH = 'ds_audio'
class_labels = sorted(os.listdir(DATASET_PATH))
model = tf.keras.models.load_model("audiomodel.h5")
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

app = Flask(__name__)
CORS(app)
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
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
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        audio_bytes = file.read()
        audio_io = io.BytesIO(audio_bytes)
        waveform, sample_rate = librosa.load(audio_io, sr=16000)
        if len(waveform) > 16000:
            waveform = waveform[:16000]
        elif len(waveform) < 16000:
            waveform = np.pad(waveform, (0, max(0, 16000 - len(waveform))), "constant")
        waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
        spectrogram = get_spectrogram(waveform)
        spectrogram = spectrogram[tf.newaxis, ...]
        prediction = model.predict(spectrogram)[0]
        predicted = tf.nn.softmax(prediction).numpy()
        confidence = float(np.max(predicted))
        predicted_class = class_labels[np.argmax(predicted)]
        if confidence<0.8:
            return jsonify({
                "error": f"Audio is not belongs to class{class_labels}",
                "message": f"Confidence below threshold of {0.8}"
            }), 400
        response = {
            "predicted_class": predicted_class,
            "confidence": confidence,

        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)