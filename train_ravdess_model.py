import os
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# DATASET PATH (IMPORTANT)
# -----------------------------
DATASET_PATH = r"D:\Project trial\Emotion based project\audio_dataset"

# -----------------------------
# EMOTION MAP
# -----------------------------
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_emotion(filename):
    try:
        return emotion_map.get(filename.split("-")[2], None)
    except:
        return None

# -----------------------------
# IMPROVED FEATURE EXTRACTION
# -----------------------------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)

        # Pad short audio
        if len(audio) < 22050 * 3:
            audio = np.pad(audio, (0, 22050*3 - len(audio)))

        audio = audio[:22050*3]  # trim to 3 sec

        # Extract multiple features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)

        feature = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0)
        ])

        return feature

    except Exception as e:
        print("Error processing:", file_path)
        return None

# -----------------------------
# LOAD DATA
# -----------------------------
features = []
emotions = []

for actor in os.listdir(DATASET_PATH):

    if not actor.startswith("Actor"):
        continue

    actor_path = os.path.join(DATASET_PATH, actor)

    for file in os.listdir(actor_path):

        emotion = extract_emotion(file)
        if emotion is None:
            continue

        file_path = os.path.join(actor_path, file)

        feature = extract_features(file_path)

        if feature is not None:
            features.append(feature)
            emotions.append(emotion)

# 🔥 DEBUG CHECK
print("✅ Total samples loaded:", len(features))

# -----------------------------
# PREPROCESS
# -----------------------------
X = np.array(features)

# reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

le = LabelEncoder()
y_encoded = le.fit_transform(emotions)
y = to_categorical(y_encoded)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL (IMPROVED)
# -----------------------------
model = Sequential([
    Conv1D(128, 5, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(256, 5, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.3),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.4),

    Dense(y.shape[1], activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# TRAIN (WITH EARLY STOPPING)
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -----------------------------
# SAVE (SAFE FORMAT)
# -----------------------------
model.save("emotion_voice_model1.keras")
np.save("emotion_labels1.npy", le.classes_)

print("✅ MODEL SAVED SUCCESSFULLY (NO CORRUPTION)")
