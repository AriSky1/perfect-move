

# Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Initialize MediaPipe Pose models
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose()
pose_webcam = mp_pose.Pose()

# Load the video
video_path = 'video.mp4'
cap_video = cv2.VideoCapture(video_path)

# Initialize webcam capture
cap_webcam = cv2.VideoCapture(0)  # Use 0 for the first webcam, 1 for the second, etc.

# Use ThreadPoolExecutor for concurrent processing
executor_video = ThreadPoolExecutor(max_workers=2)
executor_webcam = ThreadPoolExecutor(max_workers=2)

# Function to read and process frames asynchronously
def read_and_process_frames(video_capture, executor, pose_model):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # Resize frame to improve processing speed
        frame_resized = cv2.resize(frame, (640, 480))
        # Submit pose detection task to executor
        future = executor.submit(detect_pose, frame_resized.copy(), pose_model)
        yield future.result()

# Function to detect poses in a frame
# Function to detect poses in a frame
# Function to detect poses in a frame
# Function to detect poses in a frame
def detect_pose(frame, pose_model):
    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make detection
    results = pose_model.process(image)

    # Draw pose connections (lines) on the frame
    if results.pose_landmarks:
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # Define connections between landmarks to draw lines
        connections = [
                       (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                       (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                       (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                       (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                       (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                       (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                       (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                       (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                       (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                       (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                       (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                       (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                       (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST),
                       (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_WRIST)
                       ]

        # Draw each connection
        for connection in connections:
            joint1 = connection[0]
            joint2 = connection[1]

            # Check if both landmarks are visible
            if landmarks[joint1].visibility > 0.5 and landmarks[joint2].visibility > 0.5:
                joint1_x = int(landmarks[joint1].x * frame.shape[1])
                joint1_y = int(landmarks[joint1].y * frame.shape[0])
                joint2_x = int(landmarks[joint2].x * frame.shape[1])
                joint2_y = int(landmarks[joint2].y * frame.shape[0])

                # Draw the line between joint1 and joint2
                cv2.line(frame, (joint1_x, joint1_y), (joint2_x, joint2_y), (0, 255, 0),
                         3)  # Adjust color and thickness as needed

    return frame


# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(cap_video, executor_video, pose_video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for webcam feed
@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_feed(cap_webcam, executor_webcam, pose_webcam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Generator function to yield frames
def generate_feed(video_capture, executor, pose_model):
    for frame in read_and_process_frames(video_capture, executor, pose_model):
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/compare_pose')
def compare_pose():
    frame_webcam, landmarks_webcam = read_and_process_frames(cap_webcam, pose_webcam)
    frame_video, landmarks_video = read_and_process_frames(cap_video, pose_video)

    if landmarks_webcam is not None and landmarks_video is not None:

        return render_template('index.html')
    else:
        return "Pose not detected in one of the streams."


# Index route
@app.route('/')
def index():
    return render_template('index.html')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)





