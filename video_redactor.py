import subprocess

def crop_video(input_file, output_file, n_seconds):
    # Command to crop video using FFmpeg
    command = [
        'ffmpeg',           # FFmpeg executable
        '-ss', str(n_seconds),     # Start trimming from `n_seconds` seconds
        '-i', input_file,   # Input video file
        '-c', 'copy',       # Copy the video codec
        '-avoid_negative_ts', '1',  # Avoid negative timestamps
        '-movflags', '+faststart',  # Fast start for better streaming
        '-y',               # Overwrite output files without asking
        output_file        # Output filename (different from input)
    ]

    # Run FFmpeg command
    subprocess.run(command)

# Example usage: crop 10 seconds from the beginning of video2.mp4 and save as cropped_video.mp4
crop_video('video2.mp4', 'cropped_video2.mp4', 5)
