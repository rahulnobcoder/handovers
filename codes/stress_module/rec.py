import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
from moviepy.editor import VideoFileClip
import pandas as pd
from tqdm import tqdm
import time
import json
import cv2
import dlib
from collections import Counter
import statistics
import shutil
import numpy as np
import asyncio

from functions.valence_arousal import va_predict
from functions.speech import speech_predict
from functions.eye_track import Facetrack, eye_track_predict
from functions.fer import extract_face,fer_predict,plot_graph,filter,save_frames

models_folder='models'    #change this path to folder_path where models are saved (docker: /app/app/models)

speech_model=os.path.join(models_folder,'speech.keras')
fer_model=os.path.join(models_folder,'22.6_AffectNet_10K_part2.pt')
valence_arousal_model=os.path.join(models_folder,'emotion_model.pt')
val_ar_feat_path=os.path.join(models_folder,'resnet_features.pt')
valence_dict_path=os.path.join(models_folder,'valence-NRC-VAD-Lexicon.txt')
arousal_dict_path=os.path.join(models_folder,'arousal-NRC-VAD-Lexicon.txt')
dominance_dict_path=os.path.join(models_folder,'dominance-NRC-VAD-Lexicon.txt')
dnn_net = cv2.dnn.readNetFromCaffe(os.path.join(models_folder,"deploy.prototxt"), os.path.join(models_folder,"res10_300x300_ssd_iter_140000.caffemodel"))
predictor = dlib.shape_predictor(os.path.join(models_folder,"shape_predictor_68_face_landmarks.dat"))
video_paths=['videos/a3.webm','videos/a3.webm']

import time
st=time.time()


eye=[]
fer=[]
blinks=[]
class_wise_frame_counts=[]
ser_major_emotions=[]
speech_data=[]
word_weights_list=[]
speech_emotions=[]

for count in range(len(video_paths)):
    video_path=video_paths[count]
    output_dir='output'
    folder_name = f"vid{count+1}"
    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    meta_ind={}
    #output paths
    fer_log_path = os.path.join(folder_path,"fer_log.csv")
    Speech_log_path = os.path.join(folder_path,"speech_log.csv")
    eye_log_path = os.path.join(folder_path,"eyetrack_log.csv")
    word_path = os.path.join(folder_path,"word_log.csv")
    json_path = os.path.join(folder_path, "data.json")
    valence_plot=os.path.join(folder_path,"valence.png")
    arousal_plot=os.path.join(folder_path,"arousal.png")
    stress_plot=os.path.join(folder_path,"stress.png")

    video_clip = VideoFileClip(video_path)
    video_clip = video_clip.set_fps(30)
    print("Duration: ", video_clip.duration)
    fps = video_clip.fps
    audio = video_clip.audio
    audio_path = os.path.join(folder_path,'extracted_audio.wav')
    audio.write_audiofile(audio_path)
    video_frames = [frame for frame in video_clip.iter_frames()]
    print("extracting faces")
    faces=[extract_face(frame,dnn_net,predictor) for frame in video_frames]
    new_frames=[video_frames[i] for i in range(len(video_frames)) if faces[i] is not None]
    print(f'{len([face for face in faces if face is not None])} faces found.')
    #EYE TRACKING
    fc=Facetrack()
    column=['Timestamp','Total_Blinks']
    preds,blink_durations,total_blinks=eye_track_predict(fc,faces,fps)
    eye_df=pd.DataFrame(preds,columns=column)
    eye_df.to_csv(eye_log_path,index=False)


    #FACIAL EXPRESSION RECOGNITION
    fer_df,class_wise_frame_count,em_tensors=fer_predict(faces,fps,fer_model)
    valence_list,arousal_list,stress_list=va_predict(valence_arousal_model,val_ar_feat_path,faces,list(em_tensors))
    fer_df['Arousal']=arousal_list
    fer_df['Valence']=valence_list
    fer_df['Stress']=stress_list
    timestamps=list(fer_df['Timestamp'])
    frame_index=[i+1 for i in range(len(timestamps))]
    fer_df.to_csv(fer_log_path, index=False)

    plot_graph(filter(frame_index,valence_list),'valence',valence_plot)
    plot_graph(filter(frame_index,arousal_list),'arousal',arousal_plot)
    plot_graph(filter(frame_index,stress_list),'Stress',stress_plot)
    # save_frames(new_frames,frames_folder)


    #SPEECH EMOTION RECOGNITION
    emotions_df,major_emotion,word=speech_predict(audio_path,speech_model,valence_dict_path,arousal_dict_path,dominance_dict_path,word_path)
    emotions_df.to_csv(Speech_log_path, index=False)
    meta_data={}

    meta_data['blink_durations']=blink_durations
    try:
        meta_data['avg_blink_duration']=sum(blink_durations)/len(blink_durations)
    except:
        meta_data['avg_blink_duration']=0
    meta_data['Total_blinks']=total_blinks
    try:
        avg_blink_duration=float(sum(blink_durations)/(len(blink_durations)))
        meta_data['avg_blink_durations']=avg_blink_duration
    except Exception as e:
        print(f"An error occurred: {e}")

    meta_data['fer_class_wise_frame_count']=class_wise_frame_count

    meta_data['ser_major_emotion']=str(major_emotion)
    meta_data['pause_length']=float(word['average_pause_length'])
    meta_data['articulation_rate']=float(word['articulation_rate'])
    meta_data['speaking rate']=float(word['speaking_rate'])
    meta_data['word_weights']=word['word_weights']
    with open(json_path, 'w') as json_file:
        json.dump(meta_data, json_file)

    eye.append(eye_df)
    fer.append(fer_df)
    blinks.append(blink_durations)
    class_wise_frame_counts.append(class_wise_frame_count)
    speech_data.append([word['average_pause_length'],word['articulation_rate'],word['speaking_rate']])
    ser_major_emotions.append(major_emotion)
    word_weights_list.append(word['word_weights'])
    speech_emotions.append(emotions_df)

    if count==len(video_paths)-1 and len(video_paths)>1:
        folder_name='combined'
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        combined_json_path=os.path.join(folder_path,'combined_data.json')
        combined_valence_path=os.path.join(folder_path,'combined_valence.png')
        combined_arousal_path=os.path.join(folder_path,'combined_arousal.png')
        combined_stress_path=os.path.join(folder_path,'combined_stress.png')
        current_max = 0

        for i, df in enumerate(fer):
            df['Timestamp'] = df['Timestamp'] + current_max
            current_max = df['Timestamp'].max()
        combined_fer_df=pd.concat(fer).reset_index(drop=True)
        combined_fer_df.to_csv(os.path.join(folder_path,'fer_combined.csv'),index=False)

        
        current_max = 0

        for i, df in enumerate(speech_emotions):
            df['Timestamp'] = df['Timestamp'] + current_max
            current_max = df['Timestamp'].max()
        combined_speech_df=pd.concat(speech_emotions).reset_index(drop=True)
        combined_speech_df.to_csv(os.path.join(folder_path,'speech_combined.csv'),index=False)



        valence_list=list(combined_fer_df['Valence'])
        arousal_list=list(combined_fer_df['Arousal'])
        stress_list=list(combined_fer_df['Stress'])
        timestamps=list(combined_fer_df['Timestamp'])
        frame_index=[i+1 for i in range(len(timestamps))]
        plot_graph(filter(frame_index,valence_list),'valence',combined_valence_path)
        plot_graph(filter(frame_index,arousal_list),'arousal',combined_arousal_path)
        plot_graph(filter(frame_index,stress_list),'Stress',combined_stress_path)



        meta_data={}
        current_max = 0
        c1=0
        combined_weights = Counter()
        for word_weight in word_weights_list:
            combined_weights.update(word_weight)
        # Convert back to a dictionary if needed
        combined_weights_dict = dict(combined_weights)
        meta_data['word_weights']=combined_weights_dict

        for i, df in enumerate(eye):
            df['Timestamp'] = df['Timestamp'] + current_max
            current_max = df['Timestamp'].max()
            add_value = c1
            def add_integer(val):
                if isinstance(val, (int, float)):  # Check if the value is an integer or float
                    return val + add_value
                return val  # Return the value as is if it's a string
            df['Total_Blinks'] = df['Total_Blinks'].apply(add_integer)
            c1 = len(blinks[i])
        combined_eye_df = pd.concat(eye).reset_index(drop=True)
        combined_eye_df.to_csv(os.path.join(folder_path,'eye_combined.csv'),index=False)
        flattened_list = [item for sublist in blinks for item in sublist]
        try:
            meta_data['avg_blink_duration']=float(sum(flattened_list)/len(flattened_list))
        except:
            meta_data['avg_blink_duration']=0
        numeric_values = pd.to_numeric(combined_eye_df['Total_Blinks'], errors='coerce')
        max_value = numeric_values.max()
        meta_data['Total_blinks']=int(max_value)
        dict_list = class_wise_frame_counts

	# Initialize a dictionary to store the sum of values
        result = {}

        # Sum the values for each key across all dictionaries
        for d in dict_list:
                for key, value in d.items():
                        result[key] = result.get(key, 0) + value
        meta_data['fer_class_wise_frame_count']=result
        
        meta_data['ser_major_emotion']=str(statistics.mode(ser_major_emotions))
        meta_data['ser_major_emotions_video_wise']=ser_major_emotions
        meta_data['avg_pause_length']=statistics.mean([row[0] for row in speech_data])
        meta_data['articulation_rate'] =statistics.mean([row[1] for row in speech_data])
        meta_data['speaking_rate'] = statistics.mean([row[2] for row in speech_data])
        with open(combined_json_path, 'w') as json_file:
            json.dump(meta_data, json_file)