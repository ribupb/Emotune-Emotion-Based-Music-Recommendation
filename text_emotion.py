from textblob import TextBlob
import pandas as pd

# -------------------------
# DETECT EMOTION
# -------------------------
def detect_text_emotion(user_text):

    text = user_text.lower()

    # Energetic keywords
    energetic_words = [
        "energetic",
        "energy",
        "excited",
        "pumped",
        "motivated",
        "powerful",
        "active",
        "workout",
        "gym",
        "dance",
        "party",
        "rock",
        "adventure",
        "sports",
        "exercise",
        "enthusiastic"
    ]

    # Sad keywords
    sad_words = [
        "sad",
        "depressed",
        "cry",
        "lonely",
        "heartbroken",
        "upset",
        "miserable",
        "hurt",
        "pain",
        "sorrow"
    ]

    # Calm keywords
    calm_words = [
        "calm",
        "relaxed",
        "peaceful",
        "quiet",
        "serene",
        "meditation",
        "rest",
        "sleep",
        "comfortable",
        "gentle"
    ]

    # Happy keywords
    happy_words = [
        "happy",
        "joy",
        "great",
        "awesome",
        "wonderful",
        "smile",
        "cheerful",
        "delighted",
        "pleased",
        "fantastic"
    ]

    # Priority-based keyword matching
    if any(word in text for word in energetic_words):
        return "energetic"

    if any(word in text for word in sad_words):
        return "sad"

    if any(word in text for word in calm_words):
        return "calm"

    if any(word in text for word in happy_words):
        return "happy"

    # -------------------------
    # Fallback to TextBlob
    # -------------------------
    analysis = TextBlob(user_text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.5:
        return "happy"

    elif 0 < polarity <= 0.5:
        return "calm"

    elif polarity < -0.3:
        return "sad"

    else:
        # No neutral mood in dataset
        return "calm"


# -------------------------
# RECOMMEND SONGS
# -------------------------
def recommend_text_songs(emotion, df, n=6):

    filtered = df[
        df["mood"].str.lower() == emotion.lower()
    ]

    if len(filtered) == 0:
        return []

    return filtered.sample(
        min(n, len(filtered))
    ).to_dict("records")