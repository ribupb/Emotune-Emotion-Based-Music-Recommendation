!pip install yt-dlp

import pandas as pd
import os
import yt_dlp

# CSV created in previous step
csv_file = r"D:\Project trial\Emotion based project\songs_to_download.csv"

# Base songs folder
base_folder = r"D:\Project trial\Emotion based project\static\songs"

df = pd.read_csv(csv_file)

for _, row in df.iterrows():

    song_name = row["Name"]
    artist = row["Artist"]
    mood = row["Mood"]

    print(f"\nDownloading: {song_name}")

    search_query = f"ytsearch1:{song_name} {artist}"

    save_folder = os.path.join(base_folder, mood)

    os.makedirs(save_folder, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(save_folder, f"{song_name}.%(ext)s"),
        "noplaylist": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([search_query])

        print(f"✓ Downloaded: {song_name}")

    except Exception as e:
        print(f"✗ Failed: {song_name}")
        print(e)

print("\nAll downloads completed.")
