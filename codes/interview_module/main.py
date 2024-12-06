import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
from functions.models import models_dict
from functions.helper import extract_faces_from_frames,make_pdf,make_pdf_ph
from functions.video import eyebrow,detect_blinks,detect_yawns,detect_smiles
from functions.valence_arousal import va_predict
from functions.fer import fer_predict,plot_graph
from functions.helper import plot_facial_expression_graphs
from moviepy.editor import VideoFileClip
import json 
# from trash import detect_eyes_in_faces
import pandas as pd 
from typing import Callable
from functions.audio import extract_audio_features
asrmodel=models_dict['asrmodel']
asrproc=models_dict['asrproc']
sentipipe=models_dict['sentipipe']
valence_arousal_model=models_dict['valence_fer'][1]
val_ar_feat_model=models_dict['valence_fer'][0]
fer_model=models_dict['fer']
smile_cascade=models_dict['smile_cascade']
dnn_net=models_dict['face'][0]
predictor=models_dict['face'][1]
gem_model=models_dict['gem']
fps=30
session_data={}
calibration={}
def analyze_live_video(video_path: str, uid: str, user_id: str, count: int, final: bool, log: Callable[[str], None]):
    try:
        global session_data
        global calibration
        if uid not in session_data:
            session_data[uid]={
                "vcount":[],
				"duration":[],
                    
                "audio":[],
                "pitches":[],

                "blinks":[],
                "yawn":[],
                "smile":[],
                "eyebrow":[],

                "fer": [],
				"valence":[],
				"arousal":[],
				"stress":[],

                "sentiment":[]
            }
       
        print(f"Analyzing video for question - {count}")
        print(f"UID: {uid}, User ID: {user_id}, Count: {count}, Final: {final}, Video: {video_path}")
        print(f"analysing video for question - {count}")

        output_dir = os.path.join('output', uid)
        os.makedirs(output_dir,exist_ok=True)
        
        folder_path=os.path.join(output_dir,f'{count}')
        os.makedirs(folder_path,exist_ok=True)
        meta_data_path=os.path.join(folder_path,'metadata.json')
        valence_plot=os.path.join(folder_path,"vas.png")
        word_cloud=os.path.join(folder_path,'wordcloud.jpg')
        df_path=os.path.join(folder_path,'data.csv')
        pdf_filename = os.path.join(folder_path,"formatted_output_with_plots.pdf")

        video_clip=VideoFileClip(video_path)
        video_clip=video_clip.set_fps(fps)
        duration=video_clip.duration
        print(duration)
        session_data[uid]['vcount'].append(count)
        session_data[uid]['duration'].append(duration)
        audio=video_clip.audio
        audio_path = os.path.join(folder_path,'extracted_audio.wav')
        print(audio_path)
        audio.write_audiofile(audio_path)
        video_frames=[frame for frame in video_clip.iter_frames()]
        faces, landmarks, sizes=extract_faces_from_frames(video_frames,dnn_net,predictor)


        # faces=[extract_face(frame) for frame in tqdm(video_frames)]
        af,pitches=extract_audio_features(audio_path,asrmodel,asrproc,sentipipe,duration,word_cloud,gem_model)
        pitches=[float(pitch) for pitch in pitches]


        fer_emotions,class_wise_frame_count,em_tensors=fer_predict(faces,fps,fer_model)
        valence_list,arousal_list=va_predict(valence_arousal_model,val_ar_feat_model,faces,list(em_tensors))

        timestamps=[j/fps for j in range(len(valence_list))]

        eyebrow_dist=eyebrow(landmarks,sizes)
        print('eyebrow done')

        blinks,blink_count, ear_ratios=detect_blinks(landmarks,sizes,fps)
        ear_ratios=[float(pitch) for pitch in ear_ratios]
        print('blinks done',blink_count)

        smiles, smile_ratios, total_smiles, smile_durations,smile_threshold=detect_smiles(landmarks,sizes)
        smile_ratios=[float(smile) for smile in smile_ratios]
        print('smiles done',total_smiles)

        yawns, yawn_ratios, total_yawns, yawn_durations=detect_yawns(landmarks,sizes)
        print('ywan done')

        thresholds=[smile_threshold,0.225,0.22]
        buffer = plot_facial_expression_graphs(smile_ratios, ear_ratios, yawn_ratios, thresholds)

        # print("detect_eyes : ",detect_eyes_in_faces(faces))

        y_vals = [valence_list, arousal_list,eyebrow_dist,pitches]
        labels = ['Valence', 'Arousal',"EyeBrowDistance","Pitch"]
        buf=plot_graph(timestamps, y_vals, labels, valence_plot)
        print('graph_plotted')
        meta_data={}
        meta_data['duration']=duration
        meta_data['facial_emotion_recognition'] = {
			"class_wise_frame_count": class_wise_frame_count,
		}
        meta_data['audio']=af


        make_pdf(pdf_filename,meta_data,buf,buffer)

        with open(meta_data_path, 'w') as json_file:
            json.dump(meta_data, json_file, indent=4)

        session_data[uid]['audio'].append(af)
        session_data[uid]['pitches'].append(pitches)
        session_data[uid]['fer'].append(fer_emotions)
        session_data[uid]['valence'].append(valence_list)
        session_data[uid]['arousal'].append(arousal_list)
        session_data[uid]['eyebrow'].append(eyebrow_dist)
        session_data[uid]['smile'].append(smile_ratios)
        session_data[uid]['blinks'].append(ear_ratios)
        session_data[uid]['yawn'].append(yawn_ratios)
        print(1)
        session_data[uid]['sentiment'].append(af['sentiment'][0]['label'])
        print(2,session_data[uid]['sentiment'])
        if count==1:
            try:	
                filtered_val=[item for item in valence_list if isinstance(item, (int,float))]
                filtered_aro=[item for item in arousal_list if isinstance(item, (int,float))]
                calibration_valence=sum(filtered_val)/len(valence_list)
                calibration_arousal=sum(filtered_aro)/len(arousal_list)
                calibration_pitch=sum(pitches)/len(pitches)
                calibration_eyebrow=sum(eyebrow_dist)/len(eyebrow_dist)
            except:
                calibration_arousal=0
                calibration_valence=0
                calibration_pitch=0
                calibration_eyebrow=0
            calibration['valence']=calibration_valence
            calibration['arousal']=calibration_arousal
            calibration['pitch']=calibration_pitch
            calibration['eyebrow']=calibration_eyebrow
            print(calibration)
        if not final:
            return
        videos=len(session_data[uid]['vcount'])
        final_score=0
        #combined calculation 
        combined_pdf=os.path.join(output_dir,'combined.pdf')
        transcripts=''
        combined_valence=[]
        combined_arousal=[]
        combined_fer=[]
        combined_pitch=[]
        combined_eyebrow=[]
        combined_blinks=[]
        combined_yawn=[]
        senti_list=[]
        combined_smiles=[]
        vid_index=[]
        for i in range(videos):
            timestamps=[j/fps for j in range(len(session_data[uid]['valence'][i]))]	
            for j in range(len(timestamps)):
                vid_index.append(i+1)
            transcripts+=session_data[uid]['audio'][i]['transcript']
            print('t')
            combined_pitch+=session_data[uid]['pitches'][i]
            combined_arousal+=session_data[uid]['arousal'][i]
            combined_valence+=session_data[uid]['valence'][i]
            combined_fer+=session_data[uid]['fer'][i]
            combined_blinks+=session_data[uid]['blinks'][i]
            combined_eyebrow+=session_data[uid]['eyebrow'][i]
            combined_smiles+=session_data[uid]['smile'][i]
            combined_yawn+=session_data[uid]['yawn'][i]
            senti_list.append(session_data[uid]['sentiment'][i])
        print('seli',senti_list)
        sentiment_scores = {"Positive": 1, "Negative": -1, "Neutral": 0}
        total_score = sum(sentiment_scores[sentiment] for sentiment in senti_list)
        normalized_senti_score = total_score / len(senti_list)
        neg_val=sum([1 for val in combined_valence if val<calibration['valence']])/len(combined_valence)
        neg_ar=sum([1 for val in combined_arousal if val>calibration['arousal']])/len(combined_arousal)
        neg_ya=sum([1 for val in combined_yawn if val>0.225])/len(combined_yawn)
        neg_sm=sum([1 for val in combined_smiles if val<smile_threshold])/len(combined_smiles)
        print('hi')
        avg_sentiment=(neg_ar+neg_val+neg_ya+neg_sm+normalized_senti_score)/5
        print("avgsen",avg_sentiment)
        print("score from emotional and sentiment features : ",1-avg_sentiment)

        print('hi')
        print(transcripts)
        print('hi')
        y_vals = [combined_valence, combined_arousal,combined_eyebrow,combined_pitch]
        labels = ['Valence', 'Arousal',"EyeBrowDistance","Pitch"]
        buf=plot_graph(timestamps, y_vals, labels, valence_plot)
        print('hi')
        thresholds=[smile_threshold,0.225,0.22]
        buffer = plot_facial_expression_graphs(combined_smiles, combined_blinks, combined_yawn, thresholds)
        make_pdf_ph(combined_pdf,buf,buffer,avg_sentiment)



        timestamps=[i/fps for i in range(len(combined_arousal))]
        l=len(timestamps)
        df = pd.DataFrame({
			'timestamps':timestamps,
			'video_index': vid_index,  # Add a column for video index
			'fer': combined_fer,
			'valence': combined_valence,
			'arousal': combined_arousal,
            'eyebrow':combined_eyebrow,
            'blinks':combined_blinks,
            'yawn':combined_yawn,
            'smiles':combined_smiles,
            'pitches':combined_pitch[:l]
		})
        df.to_csv(os.path.join(output_dir,'combined_data.csv'), index=False)


    except Exception as e:
            print("Error analyzing video...: ", e)
            
analyze_live_video('ganesh_sample.mp4','1',1,1,False,print)
analyze_live_video('ganesh_sample.mp4','1',1,2,True,print)
