import cv2
import dlib
import numpy as np

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Ensure the file is downloaded

# Function to calculate MAR
def calculate_mar(landmarks):
    # Define mouth landmarks
    mouth = [60, 61, 62, 63, 64, 65, 66, 67]  # Mouth region (inner)
    P1, P2, P3, P4, P5, P6 = (
        landmarks[mouth[0]],
        landmarks[mouth[3]],
        landmarks[mouth[2]],
        landmarks[mouth[6]],
        landmarks[mouth[1]],
        landmarks[mouth[5]],
    )
    # Calculate distances
    vertical_1 = np.linalg.norm(P2 - P6)
    vertical_2 = np.linalg.norm(P3 - P5)
    horizontal = np.linalg.norm(P1 - P4)
    # MAR formula
    mar = (vertical_1 + vertical_2) / (2 * horizontal)
    return mar

# Threshold for detecting a smile
MAR_THRESHOLD = 0.25

# Initialize video capture
video_path = "sample2.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

smile_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Calculate MAR
        mar = calculate_mar(landmarks)

        # Check if MAR exceeds threshold
        if mar > MAR_THRESHOLD:
            smile_count += 1
            cv2.putText(frame, "Smiling", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'{mar:.2f}', (face.left(), face.top() - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw face rectangle
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

        # Draw landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)  # Yellow dots for landmarks

    # Display the frame (optional)
    cv2.imshow("Smile Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total number of smiles detected: {smile_count}")
