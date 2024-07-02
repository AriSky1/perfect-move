from moviepy.editor import VideoFileClip

def extract_audio(video_file, audio_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(audio_file)

# Example usage:
extract_audio("video.mp4", "audio.mp3")
