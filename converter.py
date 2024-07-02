from pytube import YouTube
import os

# YouTube video URL
#video_url = 'https://www.youtube.com/watch?v=QTuCHmq1Vm4' #money lalisa girl1
#video_url = 'https://www.youtube.com/watch?v=TJyRtucJGuQ' #money lalisa girl2
video_url = "https://www.youtube.com/watch?v=6cS7ks0rG8A" #serafim tuto
# Download YouTube video
yt = YouTube(video_url)
video = yt.streams.filter(file_extension='mp4').first()
video.download()

# Rename the downloaded file to video.mp4
default_filename = video.default_filename
new_filename = 'video3.mp4'

# Check if file exists and rename
if os.path.exists(default_filename):
    os.rename(default_filename, new_filename)
    print(f"Successfully downloaded and renamed the video to {new_filename}")
else:
    print("File download failed or file not found.")