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
import asyncio
import traceback

from app.functions.valence_arousal import va_predict
from app.functions.speech import speech_predict
from app.functions.eye_track import Facetrack, eye_track_predict
from app.functions.fer import extract_face,fer_predict,plot_graph,filter,save_frames
# from app.functions.facs import *
# from app.utils.session import send_analytics, send_individual_analytics_files, send_combined_analytics_files, send_error
# from app.utils.socket import ConnectionManager
from typing import Callable


def analyze_live_video(video_path: str, uid: str, user_id: str, count: int, final: bool, log: Callable[[str], None]):
	try:
		print(f"UID: {uid}, User ID: {user_id}, Count: {count}, Final: {final}, Video: {video_path}")
		log(f"Analyzing video for question - {count}")

		output_dir = os.path.join('output',str(uid))
		print(output_dir)
		models_folder = 'models'
		print(models_folder)
		if not os.path.exists(output_dir):
				os.makedirs(output_dir)
		
		# Wait for previous files to be written if final
		if final and count > 1:
			for i in range(1, count):
				previous_file_name = os.path.join(output_dir, f"{i}.json")
				print(previous_file_name)
				while not os.path.exists(previous_file_name):
					time.sleep(1) 
				
		speech_model=os.path.join(models_folder,'speech.keras')
		fer_model=os.path.join(models_folder,'22.6_AffectNet_10K_part2.pt')
		frames_folder='frames'
		valence_arousal_model=os.path.join(models_folder,'emotion_model.pt')
		facs_model=os.path.join(models_folder,'incept_v3_10fps_full_dp0.4.keras')
		val_ar_feat_path=os.path.join(models_folder,'resnet_features.pt')
		valence_dict_path=os.path.join(models_folder,'valence-NRC-VAD-Lexicon.txt')
		arousal_dict_path=os.path.join(models_folder,'arousal-NRC-VAD-Lexicon.txt')
		dominance_dict_path=os.path.join(models_folder,'dominance-NRC-VAD-Lexicon.txt')
		dnn_net = cv2.dnn.readNetFromCaffe(os.path.join(models_folder,"deploy.prototxt"), os.path.join(models_folder,"res10_300x300_ssd_iter_140000.caffemodel"))
		predictor = dlib.shape_predictor(os.path.join(models_folder,"shape_predictor_68_face_landmarks.dat"))

		eye=[]
		fer=[]
		blinks=[]
		class_wise_frame_counts=[]
		ser_major_emotions=[]
		speech_data=[]
		speech_emotions=[]
		word_weights_list=[ ]		# Load previous data
		if final:
			print("Gathering data from previous runs")
			log(f"Gathering data from previous runs")
			for i in range(count):
				folder_name = f"{i+1}"
				folder_path = os.path.join(output_dir, folder_name)
				if os.path.isdir(folder_path):
					for file in os.listdir(folder_path):
						file_path = os.path.join(folder_path, file)
						if file.endswith('.csv'):
							df = pd.read_csv(file_path)
							if 'eye' in file:
								eye.append(df)
							elif 'fer' in file:
								fer.append(df)
							elif 'speech' in file:
								speech_emotions.append(df)
						elif file == 'meta_data.json':
							with open(file_path, 'r') as json_file:
								data = json.load(json_file)
								
								blinks.append(data['eye_emotion_recognition']['blink_durations'])
								class_wise_frame_counts.append(data['facial_emotion_recognition']['class_wise_frame_count'])
								ser_major_emotions.append(data['speech_emotion_recognition']['major_emotion'])
								speech_data.append([data['speech_emotion_recognition']['pause_length'], data['speech_emotion_recognition']['articulation_rate'], data['speech_emotion_recognition']['speaking_rate']])
								word_weights_list.append(data['speech_emotion_recognition']['word_weights'])
				
		folder_name = f"{count}"
		folder_path = os.path.join(output_dir, folder_name)
		os.makedirs(folder_path, exist_ok=True)
		print(folder_path)
		#output paths
		fer_log_path = os.path.join(folder_path,"fer_log.csv")
		Speech_log_path = os.path.join(folder_path,"speech_log.csv")
		eye_log_path = os.path.join(folder_path,"eyetrack_log.csv")
		word_path = os.path.join(folder_path,"word_log.csv")
		json_path = os.path.join(folder_path, "meta_data.json")
		valence_plot=os.path.join(folder_path,"valence.png")
		arousal_plot=os.path.join(folder_path,"arousal.png")
		stress_plot=os.path.join(folder_path,"stress.png")
		facs_log=os.path.join(folder_path,'facs_log.csv')

		video_clip = VideoFileClip(video_path)
		video_clip = video_clip.set_fps(30)
		print("Duration: ", video_clip.duration)
		fps = video_clip.fps
		audio = video_clip.audio
		audio_path = os.path.join(folder_path,'extracted_audio.wav')
		audio.write_audiofile(audio_path)
		video_frames = [frame for frame in video_clip.iter_frames()]

		print("extracting faces")
		faces=[extract_face(frame,dnn_net,predictor) for frame in tqdm(video_frames)]
		print(f'{len([face for face in faces if face is not None])} faces found.')



		# #FACS PREDICT
		# facs_df=facs_pred(faces,facs_model)
		# facs_df.to_csv(facs_log,index=False)
		# print('facs log saved to ',facs_log)

		##EYE TRACKING
		fc=Facetrack()
		column=['timestamp','total_blinks']
		log(f"Extracting eye features for question - {count}")
		preds,blink_durations,total_blinks=eye_track_predict(fc,faces,fps)
		print("total_blinks- ",total_blinks)
		eye_df=pd.DataFrame(preds,columns=column)
		eye_df.to_csv(eye_log_path,index=False)


		#FACIAL EXPRESSION RECOGNITION
		log(f"Extracting facial features for question - {count}")
		fer_df,class_wise_frame_count,em_tensors=fer_predict(faces,fps,fer_model)
		valence_list,arousal_list,stress_list=va_predict(valence_arousal_model,val_ar_feat_path,faces,list(em_tensors))
		fer_df['arousal']=arousal_list
		fer_df['valence']=valence_list
		fer_df['stress']=stress_list
		timestamps=list(fer_df['timestamp'])
		frame_index=[i+1 for i in range(len(timestamps))]
		fer_df.to_csv(fer_log_path, index=False)
		plot_graph(filter(frame_index,valence_list),'valence',valence_plot)
		plot_graph(filter(frame_index,arousal_list),'arousal',arousal_plot)
		plot_graph(filter(frame_index,stress_list),'Stress',stress_plot)
		print("saving frames")
		# save_frames(video_frames,frames_folder)
		print("frames saved ")
		#SPEECH EMOTION RECOGNITION
		log(f"Extracting speech features for question - {count}")
		emotions_df,major_emotion,word=speech_predict(audio_path,speech_model,valence_dict_path,arousal_dict_path,dominance_dict_path,word_path)
		emotions_df.to_csv(Speech_log_path, index=False)


		log(f"Generating the metadata for question - {count}")
		# Create Meta Data
		meta_data={}
		try:
			avg_blink_duration= float(sum(blink_durations)/(len(blink_durations)))
		except:
			avg_blink_duration=0
		# try:
		# 	avg_blink_duration=float(sum(blink_durations)/(len(blink_durations)))
		# 	meta_data['avg_blink_durations']=avg_blink_duration
		# except Exception as e:
		# 	print(f"An error occurred: {e}")

		meta_data['eye_emotion_recognition'] = {
			"blink_durations": blink_durations,
			"avg_blink_duration":avg_blink_duration,
			"total_blinks": total_blinks,
			"duration":video_clip.duration
		}

		meta_data['facial_emotion_recognition'] = {
			"class_wise_frame_count": class_wise_frame_count,
		}
		meta_data['speech_emotion_recognition'] = {
		'major_emotion':str(major_emotion),
		'pause_length':float(word['average_pause_length']),
		'articulation_rate':float(word['articulation_rate']),
		'speaking_rate':float(word['speaking_rate']),
		'word_weights':word['word_weights']
		}
		with open(json_path, 'w') as json_file:
			json.dump(meta_data, json_file)

		# Save CSV Logs
		eye.append(eye_df)
		fer.append(fer_df)
		blinks.append(blink_durations)
		class_wise_frame_counts.append(class_wise_frame_count)
		speech_data.append([float(word['average_pause_length'] if word and word['average_pause_length'] else 0),float(word['articulation_rate'] if word and word['articulation_rate'] else 0),float(word['speaking_rate'] if word and word['speaking_rate'] else 0)])
		ser_major_emotions.append(major_emotion)
		speech_emotions.append(emotions_df)
		word_weights_list.append(word['word_weights'])

		file_path=audio_path
		if os.path.exists(file_path):
				os.remove(file_path)
		file_path='segment.wav'
		if os.path.exists(file_path):
				os.remove(file_path)

		print("Individual: ", meta_data)

		if not final:
			print("Not final Executing")
			log(f"Saving analytics for question - {count}")
		# 	send_analytics(valence_plot, arousal_plot,{
		# 		"uid": uid,
		# 		"user_id": user_id, 
		# 		"individual": meta_data,
		# 		"count": count
		# 	})
			print("Sent analytics")
		# 	send_individual_analytics_files(uid, output_dir, count)
			dummy_file_path = os.path.join(output_dir, f'{count}.json')
			print("Writing dummy file: ", dummy_file_path)
			with open(dummy_file_path, 'w') as dummy_file:
				json.dump({"status": "completed"}, dummy_file)
			return
	
		# Process combined
		log(f"Processing gathered data for final output")


		combined_json_path = os.path.join(output_dir, "combined_data.json")
		combined_valence_path = os.path.join(output_dir, "comb_valence.png")
		combined_arousal_path = os.path.join(output_dir, "comb_arousal.png")
		combined_stress_path=os.path.join(output_dir,'combined_stress.png')
		# Process each DataFrame and update timestamps sequentially
		current_max = 0
		for i, df in enumerate(fer):
				df['timestamp'] = df['timestamp'] + current_max
				current_max = df['timestamp'].max()
		combined_fer_df = pd.concat(fer).reset_index(drop=True)
		combined_fer_df.to_csv(os.path.join(output_dir,'fer_combined.csv'),index=False)

		current_max = 0
		for i, df in enumerate(speech_emotions):
			df['timestamp'] = df['timestamp'] + current_max
			current_max = df['timestamp'].max()
			combined_speech_df=pd.concat(speech_emotions).reset_index(drop=True)
			combined_speech_df.to_csv(os.path.join(output_dir,'speech_combined.csv'),index=False)

		combined_valence=list(combined_fer_df['valence'])
		combined_arousal=list(combined_fer_df['arousal'])
		combined_stress=list(combined_fer_df['stress'])
		timestamps=list(combined_fer_df['timestamp'])
		frame_index=[i+1 for i in range(len(timestamps))]
		plot_graph(filter(frame_index,combined_valence),'combined_valence',combined_valence_path)
		plot_graph(filter(frame_index,combined_arousal),'combined_arousal',combined_arousal_path)	
		plot_graph(filter(frame_index,combined_stress),'Stress',combined_stress_path)
		combined_meta_data={}
		current_max = 0
		c1=0
		combined_weights = Counter()
		for word_weight in word_weights_list:
			combined_weights.update(word_weight)
		combined_weights_dict = dict(combined_weights)

		# Process each DataFrame and update timestamps sequentially
		print(f"lenght of eye :{len(eye)}, length of blinks :{len(blinks)}")
		for i,df in enumerate(eye):
			df['timestamp'] = df['timestamp'] +current_max
			current_max = df['timestamp'].max()
			add_value=c1
			def add_integer(val):
				if isinstance(val,(int,float)):
					return val+add_value
				return val
			df['total_blinks'] = df['total_blinks'].apply(add_integer)
			c1=c1+len(blinks[i])

		# Combine the DataFrames into one
		combined_eye_df = pd.concat(eye).reset_index(drop=True)
		combined_eye_df.to_csv(os.path.join(output_dir,'eye_combined.csv'),index=False)
		flattened_list = [item for sublist in blinks for item in sublist]
		try:
			avg_blink_duration=float(sum(flattened_list)/len(flattened_list))
		except:
			avg_blink_duration=0
		numeric_values = pd.to_numeric(combined_eye_df['total_blinks'], errors='coerce')
		max_value = numeric_values.max()
		dict_list = class_wise_frame_counts

		result = {}
		for d in dict_list:
			for key,value in d.items():
				result[key]=result.get(key,0)+value

		combined_meta_data={}
		combined_meta_data['facial_emotion_recognition']={
			'class_wise_frame_count': result
		}
		combined_meta_data['eye_emotion_recognition'] = {
			'avg_blink_duration': avg_blink_duration,
			'total_blinks': int(max_value)
		}
		combined_meta_data['speech_emotion_recognition'] = {
			'major_emotion': str(statistics.mode(ser_major_emotions)),
			'pause_length': statistics.mean([row[0] for row in speech_data]),
			'articulation_rate': statistics.mean([row[1] for row in speech_data]),
			'speaking_rate': statistics.mean([row[2] for row in speech_data]),
			'word_weights':combined_weights_dict
		}

		with open(combined_json_path, 'w') as json_file:
			json.dump(combined_meta_data, json_file)

		# log(f"Saving analytics for final output")
		# send_analytics(valence_plot, arousal_plot,{
		# 	"uid": uid,
		# 	"user_id": user_id, 
		# 	"individual": meta_data,
		# 	"combined": combined_meta_data,
		# 	"count": count
		# })
		# send_individual_analytics_files(uid, output_dir, count)
		# send_combined_analytics_files(uid, output_dir)

		# shutil.rmtree(output_dir)
		# print(f"Deleted output directory: {output_dir}")
	except Exception as e:
		print("Error analyzing video...: ", e)
		error_trace = traceback.format_exc()
		print("Error Trace: ", error_trace)
		log(f"Error analyzing video for question - {count}")
		# send_error(uid, {
		# 	"message": str(e),
		# 	"trace": error_trace
		# })
		# shutil.rmtree(output_dir)
		print(f"Deleted output directory: {output_dir}")
		

# st=time.time()
# # analyze_live_video(video_path: video_path=, uid: str, user_id: str, count: int, final: bool, log: Callable[[str], None])
analyze_live_video('videos/s1.webm', 1,1,1,False,print)
# analyze_live_video('videos/s2.webm', 1,1,2,True,print)
# print("time taken - ",time.time()-st)