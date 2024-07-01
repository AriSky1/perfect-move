
def read_and_process_frames(video_capture, executor):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # Resize frame to improve processing speed
        frame_resized = cv2.resize(frame, (640, 480))
        # Submit pose detection task to executor
        future = executor.submit(detect_pose, frame_resized.copy())
        yield future.result()


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