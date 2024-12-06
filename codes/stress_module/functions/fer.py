import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import timm
from tqdm import tqdm
import torch.nn as nn
import os
import matplotlib.pyplot as plt

import torch.nn.functional as F
import dlib
import pandas as pd


# dnn_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")

# # Initialize dlib's facial landmark predictor
# predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


def extract_face(image, net, predictor):
    # Prepare the image for DNN face detection
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
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
            x, y, w, h = cv2.boundingRect(landmarks_np)
            x -= 25
            y -= 25
            w += 50
            h += 50

            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            # Crop and resize the face
            try:
                face_crop = cv2.resize(face_crop, (224, 224))
            except:
                face_crop = cv2.resize(image, (224, 224))
            return face_crop
    return None




class Model:
    def __init__(self,fps,fer_model):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.fermodel= timm.create_model("tf_efficientnet_b0_ns", pretrained=False)
        self.fermodel.classifier = torch.nn.Identity()
        self.fermodel.classifier=nn.Sequential(
        nn.Linear(in_features=1280, out_features=7)
        )
        self.fermodel = torch.load(
        fer_model,
        map_location=self.device)
        self.fermodel.to(self.device)

        self.class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
        self.emotion_reorder = {
        0: 6,
        1: 5,
        2: 4,
        3: 1,
        4: 0,
        5: 2,
        6: 3,
        }
        self.label_dict = {
                            0: "angry",
                            1: "disgust",
                            2: "fear",
                            3: "happy",
                            4: "neutral",
                            5: "sad",
                            6: "surprised",
                        }
        self.class_wise_frame_count=None
        self.emotion_count = [0] * 7
        self.frame_count=0
        self.fps=fps
        self.df=None
        self.faces_=0
    def predict(self,frames):
        emotion_list=[]
        emt=[]
        for frame in tqdm(frames):
            if frame is not None:
                frame=np.copy(frame)
                face_pil = Image.fromarray(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            )
                face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.fermodel(face_tensor)
                    _, predicted = torch.max(output, 1)
                    emotion = self.emotion_reorder[predicted.item()]
                    if isinstance(emotion, np.ndarray):
                        emotion = (
                            emotion.astype(float).item()
                            if emotion.size == 1
                            else emotion.tolist()
                                    )
                    emotion = torch.tensor(
                                    [emotion], dtype=torch.float32
                                )  # Ensures it's a tensor
                    emotion.to(self.device)
                    emt.append(emotion)
                self.emotion_count[predicted.item()] += 1
                label = f"{self.label_dict[predicted.item()]}"
                emotion_list.append(label)
            else:
                emt.append('frame error')
                emotion_list.append('frame error')
        return emotion_list,emt
        
    def get_data(self,emotion_list,emt):
        self.class_wise_frame_count = dict(zip(self.class_labels, self.emotion_count))
        return emotion_list,self.class_wise_frame_count,emt

def fer_predict(video_frames,fps,model):
    emotion_list,emt=model.predict(video_frames)
    return model.get_data(emotion_list,emt)

def filter(list1,list2):
    filtered_list1 = [x for i, x in enumerate(list1) if list2[i]!='fnf']
    filtered_list2 = [x for x in list2 if x!='fnf']
    return [filtered_list1,filtered_list2]

def plot_graph(x,y,var,path):
    y = [value if isinstance(value, (int, float)) else np.nan for value in y]
    print(len(y))
    plt.plot(range(len(x)), y, linestyle='-')
    plt.xlabel('Frame')
    plt.ylabel(var)
    plt.title(f'{var} Values vs Frame')
    plt.savefig(path)
    plt.clf()



# def save_frames(frames,folder_path):
#     for i in tqdm(range(len(frames))):
#         frame_filename = os.path.join(folder_path, f'frame_{i+1:04d}.jpg')
#         # Save the frame as a .jpg file
#         frame=cv2.cvtColor(frames[i],cv2.COLOR_BGR2RGB)
#         cv2.imwrite(frame_filename, frame)
