import numpy as np 
from scipy.spatial import distance as dist
from imutils import face_utils
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def eyebrow(landmarks,sizes):
    eyebrow_dist=[]
    for landmark,size in zip(landmarks,sizes):
        if landmark is not None:
            right_eyebrow_inner = landmark[21]
            left_eyebrow_inner = landmark[22]
            eyebrow_distance = euclidean_distance(right_eyebrow_inner, left_eyebrow_inner)
            normalized_eyebrow_distance = eyebrow_distance / size[0]
            
        else:
            normalized_eyebrow_distance=None
        eyebrow_dist.append(normalized_eyebrow_distance)
    return eyebrow_dist

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance 1
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance 2
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)  # EAR formula
    return ear

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Function to detect smiles based on mouth aspect ratio
def detect_smiles(landmarks_list, face_sizes, fps=30, consecutive_frames=2):
    smile_ratios = []  # Store the smile ratios for plotting
    smiles = []
    smile_durations = []  # To store the duration of each smile
    total_smiles = 0
    smile_in_progress = False
    smile_start_frame = None
    avg_dynamic_threshold=[]
    for frame_idx, (landmarks, face_size) in enumerate(zip(landmarks_list, face_sizes)):
        if landmarks is not None:
            # Use NumPy array indices for the relevant mouth landmarks
            left_corner = np.array(landmarks[48])
            right_corner = np.array(landmarks[54])
            top_lip = np.array(landmarks[51])
            bottom_lip = np.array(landmarks[57])
            
            mouth_width = euclidean_distance(left_corner, right_corner)
            mouth_height = euclidean_distance(top_lip, bottom_lip)
            
            face_width, face_height = face_size  # face_size is (width, height)
            
            if face_width > 0 and face_height > 0:
                normalized_mouth_width = mouth_width / face_width
                normalized_mouth_height = mouth_height / face_height
            else:
                normalized_mouth_width = 0
                normalized_mouth_height = 0
            
            smile_ratios.append(normalized_mouth_width)
            dynamic_threshold = 0.2 + (0.05 * face_width / 100)
            avg_dynamic_threshold.append(dynamic_threshold)
            # print(dynamic_threshold)  
            # Check if the smile meets the threshold
            if (normalized_mouth_width > dynamic_threshold) and (normalized_mouth_height > 0.06):
                smiles.append(True)
                if not smile_in_progress:
                    smile_in_progress = True
                    smile_start_frame = frame_idx  # Record the start of the smile
            else:
                smiles.append(False)
                if smile_in_progress and (frame_idx - smile_start_frame >= consecutive_frames):
                    smile_in_progress = False
                    smile_end_frame = frame_idx
                    smile_duration = (smile_end_frame - smile_start_frame) / fps  # Calculate smile duration
                    smile_durations.append(smile_duration)
                    total_smiles += 1  # Increment total smile count
        else:
            smiles.append(None)
    try:
        avg_thr=sum(avg_dynamic_threshold)/len(avg_dynamic_threshold)
    except:
        avg_thr=0
    return smiles, smile_ratios, total_smiles, smile_durations,avg_thr


# Function to detect blinks based on the eye aspect ratio (EAR)
import numpy as np

# Function to detect blinks based on the eye aspect ratio (EAR)
def detect_blinks(landmarks_list, face_sizes, ear_threshold=0.24, consecutive_frames=2):
    ear_ratios = []  # Store EAR for plotting
    blinks = []
    
    # Variables to monitor consecutive low EAR values
    blink_count = 0
    consec_low_ear = 0
    
    for landmarks, face in zip(landmarks_list, face_sizes):
        if landmarks is not None:
            left_eye = landmarks[36:42]  # Points 36-41 (inclusive) for the left eye
            right_eye = landmarks[42:48]
            
            def eye_aspect_ratio(eye):
                A = euclidean_distance(eye[1], eye[5])
                B = euclidean_distance(eye[2], eye[4])
                C = euclidean_distance(eye[0], eye[3])
                ear = (A + B) / (2.0 * C)
                return ear
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            if avg_ear <= ear_threshold:
                blinks.append(True)
            else:
                blinks.append(False)
            ear_ratios.append(avg_ear)
            
            if avg_ear < ear_threshold:
                consec_low_ear += 1
            else:
                # If low EAR is detected for enough consecutive frames, count as a blink
                if consec_low_ear >= consecutive_frames:
                    blink_count += 1
                consec_low_ear = 0  # Reset the consecutive low EAR counter
        else:
            blinks.append(None)
    return blinks,blink_count, ear_ratios

# Function to detect yawns based on the vertical distance between top and bottom lips
# Function to detect yawns based on the vertical distance between top and bottom lips
def detect_yawns(landmarks_list, face_sizes, fps=30, consecutive_frames=3):
    yawn_ratios = []  # Store the yawn ratios for plotting
    yawns = []
    yawn_durations = []  # To store the duration of each yawn
    total_yawns = 0
    yawn_in_progress = False
    yawn_start_frame = None
    
    for frame_idx, (landmarks, face_size) in enumerate(zip(landmarks_list, face_sizes)):
        if landmarks is not None:
            top_lip = np.array(landmarks[51])
            bottom_lip = np.array(landmarks[57])
            
            mouth_height = euclidean_distance(top_lip, bottom_lip)
            face_width, face_height = face_size  # face_size is (width, height)
            
            if face_height > 0:
                normalized_mouth_height = mouth_height / face_height
            else:
                normalized_mouth_height = 0
            
            yawn_ratios.append(normalized_mouth_height)
            
            # Check if the yawn meets the threshold
            if normalized_mouth_height > 0.22:
                yawns.append(True)
                if not yawn_in_progress:
                    yawn_in_progress = True
                    yawn_start_frame = frame_idx  # Record the start of the yawn
            else:
                yawns.append(False)
                if yawn_in_progress and (frame_idx - yawn_start_frame >= consecutive_frames):
                    yawn_in_progress = False
                    yawn_end_frame = frame_idx
                    yawn_duration = (yawn_end_frame - yawn_start_frame) / fps  # Calculate yawn duration
                    yawn_durations.append(yawn_duration)
                    total_yawns += 1  # Increment total yawn count
        else:
            yawns.append(None)
    
    return yawns, yawn_ratios, total_yawns, yawn_durations
