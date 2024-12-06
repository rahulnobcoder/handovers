import math
import mediapipe
import time
import cv2
from tqdm import tqdm
import numpy as np 
def EuclideanDistance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        return distance

class BlinkDetector:
    def __init__(self):
        self.COUNTER = 0
        self.TOTAL_BLINKS = 0   # put 1 when divide 
        self.blink_start_time = 0
        self.blink_durations = []
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    
    def FaceMeshInitialiser(self, 
                            max_num_faces,
                            min_detection_confidence,
                            min_tracking_confidence):
        
        face_mesh = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=max_num_faces,
                                        min_detection_confidence=min_detection_confidence,
                                        min_tracking_confidence=min_tracking_confidence)
        
        return face_mesh
    
    def LandmarksDetector(self, 
                          frame,
                          face_mesh_results,
                          draw: bool=False
                          ):
        
        image_height, image_width = frame.shape[:2]
        mesh_coordinates = [(int(point.x * image_width), int(point.y * image_height)) for point in face_mesh_results.multi_face_landmarks[0].landmark]

        if draw:
            [cv2.circle(frame, i, 2, (0, 255, 0), -1) for i in mesh_coordinates]
        
        return mesh_coordinates
    
    def BlinkRatioCalculator(self, 
                             landmarks):
        
        right_eye_landmark1 = landmarks[self.RIGHT_EYE[0]]
        right_eye_landmark2 = landmarks[self.RIGHT_EYE[8]]
        right_eye_landmark3 = landmarks[self.RIGHT_EYE[12]]
        right_eye_landmark4 = landmarks[self.RIGHT_EYE[4]]

        left_eye_landmark1 = landmarks[self.LEFT_EYE[0]]
        left_eye_landmark2 = landmarks[self.LEFT_EYE[8]]
        left_eye_landmark3 = landmarks[self.LEFT_EYE[12]]
        left_eye_landmark4 = landmarks[self.LEFT_EYE[4]]

        right_eye_horizontal_distance = EuclideanDistance(right_eye_landmark1, right_eye_landmark2)
        right_eye_verticle_distance = EuclideanDistance(right_eye_landmark3, right_eye_landmark4)

        left_eye_horizontal_distance = EuclideanDistance(left_eye_landmark1, left_eye_landmark2)
        left_eye_verticle_distance = EuclideanDistance(left_eye_landmark3, left_eye_landmark4)
        try:
            right_eye_ratio = right_eye_horizontal_distance / right_eye_verticle_distance
        except:
            right_eye_ratio = 0 
        try:
            left_eye_ratio = left_eye_horizontal_distance / left_eye_verticle_distance
        except:
            left_eye_ratio=0
        # eyes_ratio = (right_eye_ratio + left_eye_ratio) / 2

        return [right_eye_ratio, left_eye_ratio]
    
    def BlinkCounter(self,
                     eyes_ratio):
        
        if eyes_ratio[0] > 4 or eyes_ratio[1] > 4:
            if self.COUNTER == 0:
                self.blink_start_time = time.time()
            self.COUNTER += 1
        else:
            if self.COUNTER > 4:
                self.TOTAL_BLINKS += 1
                blink_duration = time.time() - self.blink_start_time
                self.blink_durations.append(blink_duration)
                self.COUNTER = 0
        
        return [self.TOTAL_BLINKS, self.blink_durations]
    

def InitialiseVariables():
    return BlinkDetector()

class Facetrack(BlinkDetector):
    def __init__(self):
        super().__init__()

        # create object for mediapipe face_mesh
        self.mediapipe_face_mesh = self.FaceMeshInitialiser(max_num_faces=1,
                                                            min_detection_confidence=0.6,
                                                            min_tracking_confidence=0.7)
        self.frame = None
        self.avg_blink_duration=0
        self.list_blinks=[]

    def predict(self,img):
        self.rgb_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.results = self.mediapipe_face_mesh.process(self.rgb_frame)
        if self.results.multi_face_landmarks:
            self.mesh_coordinates = self.LandmarksDetector(img, self.results, draw=True)
            self.eyes_ratio = self.BlinkRatioCalculator(self.mesh_coordinates)
            self.list_blinks = self.BlinkCounter(self.eyes_ratio)
            if self.list_blinks[1]:
                try:
                    self.avg_blink_duration = sum(self.list_blinks[1]) / len(self.list_blinks[1])
                except:
                    self.avg_blink_duration = sum(self.list_blinks[1])

                self.blink_durations = self.list_blinks[1]
        if len(self.list_blinks)>0 :

            self.TOTAL_BLINKS = self.list_blinks[0]
        else:
            self.TOTAL_BLINKS = 0
            

def eye_track_predict(fc,frames,fps):
    preds=[]
    for frame in tqdm(frames):
        if frame is not None:
            frame=np.copy(frame)
            fc.predict(frame)
            data=fc.TOTAL_BLINKS
        else:
            data='frame error'
        preds.append(data)
    return preds,fc.blink_durations,fc.TOTAL_BLINKS
