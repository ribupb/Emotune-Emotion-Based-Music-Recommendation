from tensorflow.keras.models import load_model
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import time

# -----------------------------
# LOAD MODEL
# -----------------------------

model = load_model("final_voice_model.keras")
emotion_labels = np.load("emotion_labels.npy")

print(model.input_shape)

# -----------------------------
# RECORD AUDIO
# -----------------------------

def record_audio(
    filename="recording.wav",
    duration=2,
    fs=22050
):

    print("\n🎤 Voice Emotion Detection")
    print("🎙️ Recording...")

    print("🎙️ SPEAK NOW...")

    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1
    )

    sd.wait()

    write(
        filename,
        fs,
        recording.squeeze()
    )

    print("✅ Recording Complete")

# -----------------------------
# EXTRACT FEATURES + PREDICT
# -----------------------------

def predict_emotion(
    audio_file="recording.wav"
):

    audio, sr = librosa.load(
        audio_file,
        sr=22050,
        duration=3
    )

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfccs_scaled = np.mean(
        mfccs.T,
        axis=0
    ).reshape(
        1,
        40,
        1
    )

    prediction = model.predict(
        mfccs_scaled,
        verbose=0
    )

    emotion_index = np.argmax(
        prediction
    )

    print("\n📊 Confidence Scores")

    for i, emotion in enumerate(
        emotion_labels
    ):

        score = prediction[0][i] * 100

        print(
            f"{emotion}: {score:.2f}%"
        )

    predicted_emotion = emotion_labels[
        emotion_index
    ]

    print(
        f"\n🎯 Predicted Emotion: {predicted_emotion}"
    )

    return predicted_emotion

# -----------------------------
# MAP TO MUSIC MOOD
# -----------------------------

def map_emotion(emotion):

    mapping = {

        "happy": "happy",

        "sad": "sad",

        "angry": "angry",

        "fearful": "calm",

        "neutral": "calm",

        "disgust": "sad",

        "surprised": "happy"

    }

    return mapping.get(
        emotion.lower(),
        "calm"
    )

# -----------------------------
# MAIN FUNCTION
# -----------------------------

def detect_emotion_from_voice():

    try:

        record_audio()

        emotion = predict_emotion()

        mapped_mood = map_emotion(
            emotion
        )

        print(
            f"🎵 Music Mood: {mapped_mood}"
        )

        return mapped_mood

    except Exception as e:

        print(
            "❌ Voice Detection Error:",
            e
        )

        return "calm"