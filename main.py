

import cv2
import mediapipe as mp

# Load the video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make detection
    results = pose.process(image)

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        # Example to draw a specific point (e.g., right shoulder)
        # right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        # cv2.circle(frame, (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0])), 5, (255, 0, 0), -1)

        # Loop through all landmarks and draw them
        for landmark in results.pose_landmarks.landmark:
            # Draw only for arms and legs landmarks (adjust according to your needs)
            if landmark.visibility > 0.5:
                cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Display the frame
    cv2.imshow('Dancer Pose Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
