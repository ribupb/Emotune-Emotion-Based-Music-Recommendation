import os
import pandas as pd

songs_folder = r"D:\Project trial\Emotion based project\static\songs"

rows = []

for mood in os.listdir(songs_folder):

    mood_path = os.path.join(songs_folder, mood)

    if not os.path.isdir(mood_path):
        continue

    for file in os.listdir(mood_path):

        # accept common audio formats
        if file.lower().endswith((".mp3", ".m4a", ".webm", ".wav")):

            rows.append({
                "name": os.path.splitext(file)[0],
                "artist": "Unknown",
                "mood": mood.lower(),
                "mp3_path": f"songs/{mood}/{file}"
            })

df = pd.DataFrame(rows)

output_file = r"D:\Project trial\Emotion based project\new_songlist.csv"

df.to_csv(output_file, index=False)

print("✅ Dataset updated successfully")
print(f"Total songs: {len(df)}")
print(df.head(10))
