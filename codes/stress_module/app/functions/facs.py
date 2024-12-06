import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import pandas as pd


threshold = [
    0.6827917, 0.7136434, 0.510756, 0.56771123, 0.49417764, 0.45892453,
    0.32996163, 0.5038406, 0.44855, 0.32959282, 0.45619836, 0.4969851
]

au_to_movements = {
    'au1': 'inner brow raiser',
    'au2': 'outer brow raiser',
    'au4': 'brow lowerer',
    'au5': 'upper lid raiser',
    'au6': 'cheek raiser',
    'au9': 'nose wrinkler',
    'au12': 'lip corner puller',
    'au15': 'lip corner depressor',
    'au17': 'chin raiser',
    'au20': 'lip stretcher',
    'au25': 'lips part',
    'au26': 'jaw drop'
}

au_labels = [
    "au1", "au12", "au15", "au17", "au2", "au20",
    "au25", "au26", "au4", "au5", "au6", "au9"
]

col = [f'{i}:{au_to_movements[i]}' for i in au_labels]

def binary_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        fl = - alpha * (y_true * (1 - y_pred)**gamma * tf.math.log(y_pred)
                       + (1 - y_true) * (y_pred**gamma) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(fl, axis=-1)
    return focal_loss

loss = binary_focal_loss(gamma=2.0, alpha=0.25)
from moviepy.editor import VideoFileClip

# Function to process frames and make predictions
def process_frames(faces, model):
    frames=[face for face in faces if face is not None]
    frame_array = np.array(frames)
    preds = model.predict(frame_array)
    predicted_labels = np.zeros_like(preds, dtype='int')
    for i in range(12):
        predicted_labels[:, i] = (preds[:, i] > threshold[i]).astype(int)
    return predicted_labels

# Function to save predictions to a CSV file with timestamps
def save_predictions_to_csv(predictions, timestamps, filename="predictions.csv"):
    df = pd.DataFrame(predictions, columns=col)
    df['timestamp'] = timestamps
    df.set_index('timestamp', inplace=True)
    return df

# Load your Keras model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'binary_focal_loss': binary_focal_loss})
    return model

def facs_pred(faces,model_path):
    model=load_model(model_path)
    predictions = process_frames(faces, model)
    timestamps = [frame_count / 30 for frame_count in range(len(predictions))]
    df = save_predictions_to_csv(predictions, timestamps)
    return df
