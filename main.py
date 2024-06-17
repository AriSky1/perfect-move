from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the video
video_path = 'video.mp4'
cap_video = cv2.VideoCapture(video_path)

# Initialize webcam capture
cap_webcam = cv2.VideoCapture(0)  # Use 0 for the first webcam, 1 for the second, etc.

def detect_pose_video():
    while cap_video.isOpened():
        ret, frame = cap_video.read()
        if not ret:
            break

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

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_pose_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)