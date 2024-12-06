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
import matplotlib
matplotlib.use('Agg')
from io import BytesIO

import torch.nn.functional as F
import pandas as pd

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

def plot_graph(x, y_vals, labels, path, calib_vals=None):
    """
    Plots multiple subplots (one for each variable) in one figure.
    
    Parameters:
    - x: List of timestamps or frame numbers.
    - y_vals: List of y-values for valence, arousal, and stress (or other metrics).
    - labels: List of variable names corresponding to y_vals (e.g., ['valence', 'arousal', 'stress']).
    - path: Path to save the combined plot.
    - calib_vals: List of calibration values for each variable (optional).
    """
    buf = BytesIO()
    plt.figure(figsize=(12, 8))  # Create a figure of appropriate size
    
    # Iterate over y-values, labels, and calibration values to create subplots
    for i, (y, label) in enumerate(zip(y_vals, labels)):
        y = [value if isinstance(value, (int, float)) else np.nan for value in y]
        # Create a subplot (n rows, 1 column, and the current subplot index)

        plt.subplot(len(y_vals), 1, i+1)
        plt.plot(range(max(len(x),len(y))), y, linestyle='-')
        
        # Plot calibration line if provided
        if calib_vals and calib_vals[i] is not None:
            plt.axhline(y=calib_vals[i], color='r', linestyle='--', label=f'{label} calibration = {calib_vals[i]}')
            
        plt.xlabel('Frame')
        plt.ylabel(label)
        plt.title(f'{label} By Frames')
        plt.legend()
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(buf, format='png')
    plt.clf()  # Clear the figure after saving
    buf.seek(0)
    return buf

