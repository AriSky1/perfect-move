from pytube import YouTube
import os

# YouTube video URL
video_url = 'https://www.youtube.com/watch?v=CBkVrEF8HmQ'

# Download YouTube video
yt = YouTube(video_url)
video = yt.streams.filter(file_extension='mp4').first()
video.download()

# Rename the downloaded file to video.mp4
default_filename = video.default_filename
new_filename = 'video2.mp4'

# Check if file exists and rename
if os.path.exists(default_filename):
    os.rename(default_filename, new_filename)
    print(f"Successfully downloaded and renamed the video to {new_filename}")
else:
    print("File download failed or file not found.")
