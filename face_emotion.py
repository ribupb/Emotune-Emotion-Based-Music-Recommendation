from deepface import DeepFace
import cv2
import pandas as pd

def detect_face_emotion():

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    if not ret:
        return "Camera Error", []

    try:

        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend='opencv'
        )

        emotion_scores = result[0]["emotion"]

        emotion = max(
            emotion_scores,
            key=emotion_scores.get
        )
        
        print("\nEmotion Scores:")
        print(emotion_scores)

        print("\nDetected:")
        print(emotion)

    except:

        emotion = "neutral"

    emotion_map = {
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "calm",
        "neutral": "calm",
        "surprise": "happy",
        "disgust": "sad"
    }
    
    print("Raw emotion:", emotion)

    mapped_mood = emotion_map.get(
        emotion.lower(),
        "calm"
    )

    df = pd.read_csv("new_songlist.csv")

    df["mood"] = df["mood"].str.lower().str.strip()

    filtered = df[df["mood"] == mapped_mood]

    if len(filtered) == 0:
        songs = df.sample(min(6, len(df))).to_dict("records")
    else:
        songs = filtered.sample(
            min(6, len(filtered))
        ).to_dict("records")

    return mapped_mood, songs