import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
import gdown
import os
from tensorflow.keras.models import load_model

# Define the Google Drive file ID
MODEL_FILE_ID = '1GsplInchAd4d6hQMciDTCz8pxHCTEOLz'

# Define the local path to save the model
MODEL_PATH = 'cnn_model.h5'

# Check if the model is already downloaded
if not os.path.exists(MODEL_PATH):
    # Construct the download URL
    download_url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
    # Download the model file
    gdown.download(download_url, MODEL_PATH, quiet=False)

# Load CNN model
model = load_model("cnn_model.h5")

# Emotion mapping (update according to your model)
emotion_map = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

# Preprocess audio
def preprocess_audio(file_path, sr=16000, max_len=128):
    signal, sr = librosa.load(file_path, sr=sr, mono=True)
    signal, _ = librosa.effects.trim(signal)
    signal = signal / np.max(np.abs(signal))
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    n_mels, t = mel_spec.shape
    if t < max_len:
        mel_spec = np.pad(mel_spec, pad_width=((0,0),(0,max_len-t)), mode='constant')
    else:
        mel_spec = mel_spec[:, :max_len]
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    mel_spec = np.expand_dims(mel_spec, axis=0)
    return mel_spec

# Predict emotion
def predict_emotion(file_path):
    X = preprocess_audio(file_path)
    pred = model.predict(X)
    emotion = emotion_map[np.argmax(pred)]
    return emotion

# Streamlit UI
st.title("🎤 Voice Emotion Recognition")
st.write("Upload a voice file (wav/mp3), and the model will predict the emotion.")

uploaded_file = st.file_uploader("Choose a voice clip", type=["wav","mp3"])
if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    emotion = predict_emotion("temp.wav")
    st.success(f"Predicted Emotion: **{emotion}**")

    # Optional: Show waveform
    signal, sr = librosa.load("temp.wav", sr=16000)
    st.line_chart(signal)
