# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Define a dictionary to store intensity data for each emotion
# emotion_intensity_data = {
#     'sad': {'eyes': [], 'mouth': []},
#     'happy': {'cheeks': [], 'mouth': []},
#     'angry': {'eyes': [], 'mouth': []},
#     'disgust': {'eyes': [], 'mouth': []},
#     'fear': {'eyes': [], 'mouth': []},
#     'surprise': {'eyes': [], 'mouth': []}
# }

# # Load the video
# cap = cv2.VideoCapture('videos/s1.mp4')

# # Check if video loaded successfully
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Read the first frame
# ret, prev_frame = cap.read()
# if not ret:
#     print("Error: Could not read the video frame.")
#     exit()

# # Convert first frame to grayscale
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# # Define regions of interest based on facial regions
# h, w = prev_gray.shape
# eyes_region = (0, int(h/4), w, int(h/2))
# mouth_region = (0, int(3*h/4), w, h)
# cheeks_region = (0, int(h/2), w, int(3*h/4))

# # Function to calculate region intensity
# def calc_region_intensity(flow, region):
#     x_start, y_start, x_end, y_end = region
#     flow_region = flow[y_start:y_end, x_start:x_end]
#     mag, _ = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
#     return np.mean(mag)

# # Iterate through frames
# while True:
#     # Read next frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Calculate optical flow
#     flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # Calculate and store intensities for each emotion
#     for emotion, regions in emotion_intensity_data.items():
#         if 'eyes' in regions:
#             regions['eyes'].append(calc_region_intensity(flow, eyes_region))
#         if 'mouth' in regions:
#             regions['mouth'].append(calc_region_intensity(flow, mouth_region))
#         if 'cheeks' in regions:
#             regions['cheeks'].append(calc_region_intensity(flow, cheeks_region))

#     # Update previous frame
#     prev_gray = gray

# cap.release()

# # Plotting the intensity data for each emotion
# fig, axs = plt.subplots(3, 2, figsize=(15, 10))
# fig.suptitle('Facial Feature Intensity per Emotion')

# # Emotion to subplot mapping
# emotions = ['sad', 'happy', 'angry', 'disgust', 'fear', 'surprise']
# positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

# for i, emotion in enumerate(emotions):
#     pos = positions[i]
#     ax = axs[pos]
#     ax.set_title(f'{emotion.capitalize()} Expression')
#     for region, intensity in emotion_intensity_data[emotion].items():
#         ax.plot(intensity, label=f'{region.capitalize()} Intensity')
#     ax.set_xlabel('Frame')
#     ax.set_ylabel('Optical Flow Intensity')
#     ax.legend()

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

import cv2
import numpy as np

# Load pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open video capture (0 for webcam, or provide video file path)
cap = cv2.VideoCapture('ganesh_sample.mp4')

# Variable to track previous eye detection state
prev_eyes_detected = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert to grayscale for Haar cascade detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    blink_detected = False

    for (x, y, w, h) in faces:
        # Define the Region of Interest (ROI) for the face
        face_roi = frame[y:y+h, x:x+w]
        face_gray = gray_frame[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(face_gray)

        # Blink logic: Eyes detected in previous frame but not in the current frame
        if len(eyes) == 0 and prev_eyes_detected:
            blink_detected = True

        # Update the previous state of eye detection
        prev_eyes_detected = len(eyes) > 0

        # Apply edge detection for face contour
        edges = cv2.Canny(face_gray, 100, 200)

        # Display the face contour
        cv2.imshow('Face Contour', edges)

    # Display blink detection status on the original frame
    text = "Blink Detected!" if blink_detected else "No Blink"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if blink_detected else (0, 255, 0), 2)

    # Show the original video with status
    cv2.imshow('Blink Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
