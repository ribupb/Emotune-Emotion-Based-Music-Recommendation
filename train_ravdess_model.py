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
# DATASET PATH (CHANGE IF NEEDED)
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
    return emotion_map.get(filename.split("-")[2], None)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=3)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("❌ Error processing:", file_path)
        return None

# -----------------------------
# LOAD DATA
# -----------------------------
features = []
emotions = []

print("🔄 Loading dataset...")

for actor_folder in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_folder)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        emotion = extract_emotion(file)
        if emotion is None:
            continue

        file_path = os.path.join(actor_path, file)
        feature = extract_features(file_path)

        if feature is not None:
            features.append(feature)
            emotions.append(emotion)

# -----------------------------
# CHECK DATA
# -----------------------------
if len(features) == 0:
    raise ValueError("❌ No features extracted. Check dataset path.")

print(f"✅ Loaded {len(features)} samples")

# -----------------------------
# PREPROCESS
# -----------------------------
X = np.array(features).reshape(-1, 40, 1)

le = LabelEncoder()
y_encoded = le.fit_transform(emotions)
y = to_categorical(y_encoded)

# Save labels
np.save("emotion_labels.npy", le.classes_)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL
# -----------------------------
model = Sequential([
    Conv1D(64, 5, activation='relu', input_shape=(40, 1)),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(y.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# TRAIN
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("🚀 Training started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("final_voice_model.keras")

print("\n✅ Model saved as final_voice_model.keras")
print("✅ Labels saved as emotion_labels.npy")