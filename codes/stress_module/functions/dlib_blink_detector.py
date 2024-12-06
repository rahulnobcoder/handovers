import math
import dlib
import cv2
import numpy as np
import csv
from tqdm import tqdm

def EuclideanDistance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

class BlinkDetector:
    def __init__(self, consecutive_frames=2, ear_threshold=0.45):
        self.COUNTER = 0
        self.TOTAL_BLINKS = 0
        self.eye_closed_flag = False
        self.closed_eye_counter = 0
        self.consecutive_frames = consecutive_frames
        self.ear_threshold = ear_threshold
        self.frame_count = 0
        self.initialisation_phase = 50

    def EARCalculator(self, eye_coordinates):
        """Calculate the Eye Aspect Ratio (EAR) for blink detection."""
        p2_p6 = EuclideanDistance(eye_coordinates[1], eye_coordinates[5])
        p3_p5 = EuclideanDistance(eye_coordinates[2], eye_coordinates[4])
        p1_p4 = EuclideanDistance(eye_coordinates[0], eye_coordinates[3])
        EAR = (p2_p6 + p3_p5) / (2 * p1_p4)
        return EAR

    def CoordinateExtractor(self, face_landmarks, eye_indices):
        """Extract the eye coordinates from the face landmarks."""
        coordinates = np.array([[face_landmarks.part(i).x, face_landmarks.part(i).y] for i in eye_indices])
        return coordinates

    def BlinkCounter(self, EAR_left, EAR_right):
        """Count blinks based on EAR threshold."""
        if EAR_left < self.ear_threshold or EAR_right < self.ear_threshold:
            self.closed_eye_counter += 1
            if self.closed_eye_counter >= self.consecutive_frames and not self.eye_closed_flag:
                self.TOTAL_BLINKS += 1
                self.eye_closed_flag = True
        else:
            self.closed_eye_counter = 0
            self.eye_closed_flag = False

def InitialiseVariables():
    """Initialize blink detection variables."""
    return BlinkDetector()

class Facetrack(BlinkDetector):
    def __init__(self):
        super().__init__()
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.left_eye_indices = range(36, 42)
        self.right_eye_indices = range(42, 48)

    def predict(self, img):
        """Predict blinks for each frame."""
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(grayscale_img)
        if faces:
            largest_face = max(faces, key=lambda face: (face.right() - face.left()) * (face.bottom() - face.top()))
            face_landmarks = self.landmark_predictor(grayscale_img, largest_face)
            left_eye_coordinates = self.CoordinateExtractor(face_landmarks, self.left_eye_indices)
            right_eye_coordinates = self.CoordinateExtractor(face_landmarks, self.right_eye_indices)
            EAR_left = self.EARCalculator(left_eye_coordinates)
            EAR_right = self.EARCalculator(right_eye_coordinates)
            self.BlinkCounter(EAR_left, EAR_right)

def eye_track_predict(fc, frames):
    """Track blinks for multiple frames."""
    preds = []
    for frame in tqdm(frames):
        if frame is not None:
            fc.predict(frame)
            preds.append(fc.TOTAL_BLINKS)
        else:
            preds.append('frame error')
    return preds