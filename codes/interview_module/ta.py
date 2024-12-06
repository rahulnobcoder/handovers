import cv2
import dlib
import numpy as np

# Load pre-trained shape predictor and DNN model
dnn_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Function to extract face
def extract_face(image, net=dnn_net, predictor=predictor):    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Convert bounding box to dlib rectangle format
            dlib_rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            landmarks = predictor(gray, dlib_rect)  
            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
            return landmarks_np
    return None

# Function to calculate EAR
def calculate_ear(eye_points):
    # Vertical distances
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# Video capture and setup
cap = cv2.VideoCapture('ganesh_sample.mp4')

# Video writer setup
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output_with_ear_ganesh.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Extract face landmarks
    landmarks = extract_face(frame)

    if landmarks is not None:
        # Extract eye landmarks (indexes are specific to 68-point facial landmarks)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # Average EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Display EAR on the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Optionally, draw eye landmarks for visualization
        for point in left_eye:
            cv2.circle(frame, tuple(point), 2, (255, 0, 0), -1)
        for point in right_eye:
            cv2.circle(frame, tuple(point), 2, (0, 0, 255), -1)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Eye Aspect Ratio (EAR)", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture, writer, and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
