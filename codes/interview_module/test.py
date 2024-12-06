import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

# Load Dlib's pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Path to the shape predictor file

def mouth_aspect_ratio(mouth_points):
    # Define the points
    left = np.linalg.norm(mouth_points[0] - mouth_points[6])  # Distance between P49 and P55
    vertical_1 = np.linalg.norm(mouth_points[2] - mouth_points[10])  # Distance between P51 and P59
    vertical_2 = np.linalg.norm(mouth_points[4] - mouth_points[8])  # Distance between P53 and P57
    
    # Compute MAR
    return (vertical_1 + vertical_2) / (2.0 * left)

def process_video_with_mar(video_path, extra_margin=10, yawn_threshold=0.6, consecutive_frames=10):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    mar_values = []  # List to store MAR values
    yawn_count = 0  # Count of yawns detected
    consecutive_yawn_frames = 0  # Track consecutive frames where MAR > threshold

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(gray, face)

            # Get mouth region points (Dlib indices: 48-67)
            mouth_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(48, 68)])

            # Calculate MAR
            mar = mouth_aspect_ratio(mouth_points)
            mar_values.append(mar)  # Save MAR value for later
            print(f"Mouth Aspect Ratio (MAR): {mar:.2f}")

            # Check if MAR exceeds the threshold
            if mar > yawn_threshold:
                consecutive_yawn_frames += 1  # Increment if MAR > threshold
            else:
                consecutive_yawn_frames = 0  # Reset if MAR < threshold

            # If consecutive frames exceed the threshold, count it as a yawn
            if consecutive_yawn_frames >= consecutive_frames:
                yawn_count += 1
                consecutive_yawn_frames = 0  # Reset after counting the yawn
                cv2.putText(frame, "Yawn Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Compute bounding box around the mouth region
            x, y, w, h = cv2.boundingRect(mouth_points)

            # Expand the bounding box by the extra margin
            x = max(0, x - extra_margin)
            y = max(0, y - extra_margin)
            w = min(frame.shape[1] - x, w + 2 * extra_margin)
            h = min(frame.shape[0] - y, h + 2 * extra_margin)

            # Draw the bounding box and landmarks
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.polylines(frame, [mouth_points], True, (0, 0, 255), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Display the processed frame
        cv2.imshow("Processed Frame", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

    # Plot MAR values after video processing
    plt.figure(figsize=(10, 6))
    plt.plot(mar_values, label='MAR Values', color='blue')
    plt.axhline(y=yawn_threshold, color='red', linestyle='--', label='Yawn Threshold')
    plt.title('Mouth Aspect Ratio (MAR) Over Time')
    plt.xlabel('Frame')
    plt.ylabel('MAR')
    plt.legend()
    plt.grid()
    plt.show()

    # Print the number of yawns detected
    print(f"Total Yawns Detected: {yawn_count}")

# Example usage
video_path = 0  # Use 0 for webcam or replace with the path to your video
process_video_with_mar(video_path, extra_margin=20, yawn_threshold=0.6, consecutive_frames=10)
