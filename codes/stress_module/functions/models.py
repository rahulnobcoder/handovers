import os 
import cv2
import speech_recognition as sr
import dlib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from functions.fer import Model
from functions.valence_arousal import load_models
from tensorflow.keras.models import load_model # type: ignore
def load_values(file_path):
    values_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            word, value = line.strip().split('\t')
            values_dict[word.lower()] = float(value)
    return values_dict

models_folder='models'

fer_model_path=os.path.join(models_folder,'22.6_AffectNet_10K_part2.pt')
arousal_dict_path=os.path.join(models_folder,'arousal-NRC-VAD-Lexicon.txt')
dominance_dict_path=os.path.join(models_folder,'dominance-NRC-VAD-Lexicon.txt')
valence_arousal_model=os.path.join(models_folder,'emotion_model.pt')
val_ar_feat_path=os.path.join(models_folder,'resnet_features.pt')
speech_model=os.path.join(models_folder,'speech.keras')
valence_dict_path=os.path.join(models_folder,'valence-NRC-VAD-Lexicon.txt')


print("Loading models ")

#Face detection models
dnn_net = cv2.dnn.readNetFromCaffe(os.path.join(models_folder,"deploy.prototxt"), os.path.join(models_folder,"res10_300x300_ssd_iter_140000.caffemodel"))
predictor = dlib.shape_predictor(os.path.join(models_folder,"shape_predictor_68_face_landmarks.dat"))
# print("face models loaded ")

#FER model
fer_model=Model(fps=30,fer_model=fer_model_path)
# print("fer model loaded ")

#Speech model 
model_s = load_model(speech_model)
# print("speech model loaded ")
recognizer = sr.Recognizer()
#Load valence,arousal,dominance_dicts
valence_dict = load_values(valence_dict_path)
arousal_dict = load_values(arousal_dict_path)
dominance_dict = load_values(dominance_dict_path)


#Loading valence_arousal_models
resnet,emotion_model=load_models(valence_arousal_model,val_ar_feat_path)

models_dict={
    'face':(dnn_net,predictor),
    'speech':model_s,
    'fer':fer_model,
    'vad':(valence_dict,arousal_dict,dominance_dict),
    "valence_fer":(resnet,emotion_model),
    'recognizer':recognizer
    }   
print("models loaded")