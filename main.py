import pygame
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose()
pose_webcam = mp_pose.Pose()

video_path = 'video_x4_1.mp4'
cap_video = cv2.VideoCapture(video_path)
cap_webcam = cv2.VideoCapture(0)  # Use 0 for the first webcam, 1 for the second, etc.

# Use ThreadPoolExecutor for concurrent processing
executor_video = ThreadPoolExecutor(max_workers=2)
executor_webcam = ThreadPoolExecutor(max_workers=2)

# Store angles for comparison
angle_comparisons = {'video': {}, 'webcam': {}}


# Function to read and process frames asynchronously
def read_and_process_frames(video_capture, executor, pose_model, stream_type, slow_down=False):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        # Horizontally flip the frame
        if stream_type == 'webcam':
            frame = cv2.flip(frame, 1)  # 1 for horizontal flip

        # Resize frame to improve processing speed
        frame_resized = cv2.resize(frame, (640, 480))

        # Submit pose detection task to executor
        future = executor.submit(detect_pose, frame_resized.copy(), pose_model, stream_type)

        # Slow down the frame processing for video stream
        if slow_down:
            time.sleep(0.10)  # Adding a delay of 40ms to slow down by 50%

        yield future.result()


def detect_pose(frame, pose_model, stream_type):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose_model.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        connections = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]

        angles = {}

        for connection in connections:
            joint1, joint2, joint3 = connection

            if landmarks[joint1].visibility > 0.5 and landmarks[joint2].visibility > 0.5 and landmarks[
                joint3].visibility > 0.5:
                # Calculate the vectors
                vec1 = np.array([landmarks[joint1].x - landmarks[joint2].x, landmarks[joint1].y - landmarks[joint2].y])
                vec2 = np.array([landmarks[joint3].x - landmarks[joint2].x, landmarks[joint3].y - landmarks[joint2].y])

                # Calculate the angle between vectors
                angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
                angle = np.degrees(angle)

                # Calculate the direction (normalize vectors)
                direction1 = vec1 / np.linalg.norm(vec1)
                direction2 = vec2 / np.linalg.norm(vec2)

                # Store the angle and direction with joint names
                angles[f'{joint1.name}-{joint2.name}-{joint3.name}'] = (angle, direction1, direction2)

        # Store the angles for comparison
        angle_comparisons[stream_type] = angles

        # Compare angles between video and webcam streams
        other_stream_type = 'webcam' if stream_type == 'video' else 'video'
        if angle_comparisons[other_stream_type]:
            for connection in connections:
                joint1, joint2, joint3 = connection
                joint_name = f'{joint1.name}-{joint2.name}-{joint3.name}'

                if joint_name in angles and joint_name in angle_comparisons[other_stream_type]:
                    angle_diff = abs(angles[joint_name][0] - angle_comparisons[other_stream_type][joint_name][0])
                    direction_diff1 = np.linalg.norm(
                        angles[joint_name][1] - angle_comparisons[other_stream_type][joint_name][1])
                    direction_diff2 = np.linalg.norm(
                        angles[joint_name][2] - angle_comparisons[other_stream_type][joint_name][2])

                    if angle_diff <= 10 and direction_diff1 <= 0.1 and direction_diff2 <= 0.1:
                        color = (0, 255, 0)  # Green
                    else:
                        color = (255, 255, 255)  # White
                else:
                    color = (255, 255, 255)  # White

                # Draw connections with the respective color
                joint1_pos = (int(landmarks[joint1].x * frame.shape[1]), int(landmarks[joint1].y * frame.shape[0]))
                joint2_pos = (int(landmarks[joint2].x * frame.shape[1]), int(landmarks[joint2].y * frame.shape[0]))
                joint3_pos = (int(landmarks[joint3].x * frame.shape[1]), int(landmarks[joint3].y * frame.shape[0]))

                cv2.line(frame, joint1_pos, joint2_pos, color, 6)
                cv2.line(frame, joint2_pos, joint3_pos, color, 6)
                cv2.circle(frame, joint1_pos, 10, color, -1)
                cv2.circle(frame, joint2_pos, 8, color, -1)
                cv2.circle(frame, joint3_pos, 6, color, -1)

    return frame


def generate_feed(video_capture, executor, pose_model, stream_type, slow_down=False):
    for frame in read_and_process_frames(video_capture, executor, pose_model, stream_type, slow_down):
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(cap_video, executor_video, pose_video, 'video', slow_down=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_feed(cap_webcam, executor_webcam, pose_webcam, 'webcam'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html', angle_comparisons=angle_comparisons)


if __name__ == '__main__':
    app.run(debug=True)
