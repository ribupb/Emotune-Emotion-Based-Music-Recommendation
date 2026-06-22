import os
import pandas as pd
import sqlite3
from flask import Flask, render_template, request
from text_emotion import detect_text_emotion, recommend_text_songs
from face_emotion import detect_face_emotion
from voice_emotion import detect_emotion_from_voice
app = Flask(__name__)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "new_songlist.csv")

df = pd.read_csv(csv_path)
df = df.fillna("")

# Normalize moods
df['mood'] = df['mood'].str.strip().str.lower()


# --------------------------------------------------
# Quotes
# --------------------------------------------------

quotes = {

    "happy":
    "Happiness depends upon ourselves.",

    "sad":
    "Every storm runs out of rain.",

    "calm":
    "Peace begins with a smile.",

    "energetic":
    "The future depends on what you do today.",

    "angry":
    "Speak when you are calm, not when you are angry."
}

# --------------------------------------------------
# Mood Insights
# --------------------------------------------------

insights = {

    "happy":
    "You seem to be feeling positive today. Enjoy the moment and keep the good energy going.",

    "sad":
    "You seem a little low today. Music and small positive activities may help improve your mood.",

    "calm":
    "You appear relaxed and peaceful. This is a great time to focus and recharge.",

    "energetic":
    "You seem motivated and energetic. This is a good time to be productive and active.",

    "angry":
    "You may be feeling frustrated. Taking a short break and listening to music can help."
}

def save_emotion(emotion, method):

    conn = sqlite3.connect("emotion_history.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO history(emotion, method)
        VALUES(?, ?)
        """,
        (emotion, method)
    )

    conn.commit()
    conn.close()

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route('/')
def index():

    conn = sqlite3.connect("emotion_history.db")

    history = pd.read_sql_query(
        "SELECT * FROM history",
        conn
    )

    conn.close()

    total = len(history)

    if total > 0:

        emotion_counts = (
            history["emotion"]
            .value_counts()
            .to_dict()
        )

        method_counts = (
            history["method"]
            .value_counts()
            .to_dict()
        )
        
        max_mood = max(
            emotion_counts,
            key=emotion_counts.get
        ).title()

        top_method = max(
            method_counts,
            key=method_counts.get
        ).title()

        recent = history.tail(10)

        positive = [
            "happy",
            "calm",
            "energetic"
        ]

        positive_count = 0

        for mood in recent["emotion"]:

            if str(mood).lower() in positive:
                positive_count += 1

        if positive_count >= 7:

            trend_message = (
                "😊 Your recent moods have been mostly positive."
            )

        elif positive_count >= 4:

            trend_message = (
                "😌 Your mood has been fairly balanced recently."
            )

        else:

            trend_message = (
                "💙 You may be going through a difficult period recently."
            )

        records = (
            history
            .sort_values("id", ascending=False)
            .head(10)
            .to_dict("records")
        )

    else:

        emotion_counts = {}
        method_counts = {}
        records = []

        max_mood = "None"
        top_method = "None"

        trend_message = "No mood history available yet."

    return render_template(
        "index.html",
        total=total,
        emotion_counts=emotion_counts,
        method_counts=method_counts,
        trend_message=trend_message,
        records=records,
        max_mood=max_mood,
        top_method=top_method
    )

# ---------- TEXT ----------
@app.route('/detect_text', methods=['POST'])
def detect_text():
    user_text = request.form['text_input']
    emotion = detect_text_emotion(user_text)
    save_emotion(emotion, "text")

    songs = recommend_text_songs(emotion, df, n=6)

    return render_template(
        'result.html',
        emotion=emotion,
        method="text",
        songs=songs,
        quote=quotes.get(emotion, ""),
        insight=insights.get(emotion, "")
    )


# ---------- FACE ----------
@app.route('/detect_face', methods=['POST'])
def detect_face():
    emotion, songs = detect_face_emotion()
    
    save_emotion(emotion, "face")

    return render_template(
        'result.html',
        emotion=emotion,
        method="face",
        songs=songs,
        quote=quotes.get(emotion, ""),
        insight=insights.get(emotion, "")
    )


# ---------- VOICE (TEMP) ----------
@app.route('/detect_voice', methods=['POST'])
def detect_voice():
    emotion = detect_emotion_from_voice()
    save_emotion(emotion, "voice")

    # Get songs based on emotion
    filtered = df[df["mood"].str.lower() == emotion.lower()]

    songs = filtered.to_dict("records")

    return render_template(
        'result.html',
        emotion=emotion,
        method="voice",
        songs=songs,
        quote=quotes.get(emotion, ""),
        insight=insights.get(emotion, "")
    )

# ---------- TEST ----------
@app.route('/test_happy')
def test_happy():
    filtered = df[df['mood'] == "happy"]
    songs = filtered.sample(min(6, len(filtered))).to_dict('records')

    return render_template(
        'result.html',
        emotion='happy',
        method='test',
        songs=songs
    )
    

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)