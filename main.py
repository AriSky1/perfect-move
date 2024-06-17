# Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the video
video_path = 'video.mp4'
cap_video = cv2.VideoCapture(video_path)

# Initialize webcam capture
cap_webcam = cv2.VideoCapture(0)  # Use 0 for the first webcam, 1 for the second, etc.

# Use ThreadPoolExecutor for concurrent processing
executor = ThreadPoolExecutor(max_workers=2)

# Function to read and process frames asynchronously
def read_and_process_frames(video_capture):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # Resize frame to improve processing speed
        frame_resized = cv2.resize(frame, (640, 480))
        # Submit pose detection task to executor
        future = executor.submit(detect_pose, frame_resized.copy())
        yield future.result()

# Function to detect poses in a frame
def detect_pose(frame):
    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make detection
    results = pose.process(image)

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            # Draw only for visible landmarks
            if landmark.visibility > 0.5:
                cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    return frame

@app.route('/')
def index():
    return render_template('index.html')

def generate_feed(video_capture):
    return Response(generate_frames(video_capture), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames(video_capture):
    for frame in read_and_process_frames(video_capture):
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return generate_feed(cap_video)

@app.route('/webcam_feed')
def webcam_feed():
    return generate_feed(cap_webcam)

if __name__ == '__main__':
    app.run(debug=True)
