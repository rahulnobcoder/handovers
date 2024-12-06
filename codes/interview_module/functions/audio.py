import librosa
import numpy as np
import torch
from collections import Counter
import nltk
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def get_pitch_list(y,sr):
    hop_length = int(sr / 30)  # hop_length determines how far apart the frames are

    # Extract the pitch (F0) using librosa's piptrack method
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)

    # Get the pitch frequencies from the pitch array
    pitch_frequencies = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()  # Get the index of the maximum magnitude
        pitch = pitches[index, t]
        
        pitch_frequencies.append(pitch)

    # Convert pitch_frequencies to a NumPy arrays
    pitch_frequencies = np.array(pitch_frequencies)
    print("shape : ",pitch_frequencies.shape)
    return pitch_frequencies


def extract_audio_features(audio_path, asrmodel, asrproc, sentipipe, duration, wordcloud_path,gem_model):
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = asrproc(y, sampling_rate=sr, return_tensors="pt").input_features
    inputs = inputs.to(device, dtype=torch_dtype)
    with torch.no_grad():
        generated_ids = asrmodel.generate(inputs)
        transcript = asrproc.batch_decode(generated_ids, skip_special_tokens=True)[0]

    prompt = f"""
    Analyze the following interview transcript and assess the candidate's qualities based on these characteristics: 
    Teamwork, Communication Skills, Problem-Solving Ability, Adaptability, Hunger for Knowledge, Strong Work Ethic, Leadership Potential, Attention to Detail, Time Management, and Positive Attitude.
    Identify the top 3 characteristics the candidate demonstrated and mention only these in the answer not any thing else.
    Question:
    'Can you tell me about a time when you had to work as part of a team to achieve a goal? What was your role, and what was the outcome?'

    Transcript:{transcript}"""
    # Generate a response
    response = gem_model.generate_content(prompt)
    print("Top3 characteristics:", response.text)

    # Sound intensity (RMS)
    rms = librosa.feature.rms(y=y)
    sound_intensity = np.mean(rms)

    # Pitch list
    pitches=get_pitch_list(y,sr)

    # Fundamental frequency (F0)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    fundamental_frequency = np.nanmean(f0)

    # Spectral energy (based on STFT)
    S = np.abs(librosa.stft(y))
    spectral_energy = np.mean(np.sum(S ** 2, axis=0))

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    avg_spectral_centroid = np.mean(spectral_centroid)

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate = np.mean(zcr)

    # Pause detection
    silence_threshold = -40
    silent_intervals = librosa.effects.split(y, top_db=silence_threshold)
    pause_duration = 0
    for start, end in silent_intervals:
        pause_duration += (end - start) / sr

    total_duration = librosa.get_duration(y=y, sr=sr)
    pause_rate = (pause_duration / total_duration) * 60  # Convert to pauses per minute

    # Transcript processing
    words = nltk.word_tokenize(transcript)
    words = [word.lower() for word in words if word not in string.punctuation]
    num_words = len(words)
    unique_words = len(set(words))
    word_frequencies = Counter(words)

    # Duration in minutes
    duration_minutes = total_duration / 60
    avg_words_per_minute = num_words / duration_minutes
    avg_unique_words_per_minute = unique_words / duration_minutes

    # Filler word detection
    filler_words = [
        'uh', 'um', 'like', 'you know', 'ah', 'er', 'hmm', 'well', 'so', 
        'I mean', 'okay', 'right', 'actually', 'basically', 'you see', 
        'sort of', 'kind of', 'yeah', 'literally', 'just', 'I guess', 
        'totally', 'honestly', 'seriously', 'alright'
    ]
    filler_word_count = sum([word_frequencies.get(filler, 0) for filler in filler_words])
    filler_words_per_minute = filler_word_count / duration_minutes

    # POS tagging
    pos_tags = nltk.pos_tag(words)
    nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
    adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
    verbs = [word for word, pos in pos_tags if pos.startswith('VB')]

    # Sentiment analysis
    sentiment = sentipipe(transcript)
    sentiment_mapping = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    sentiment[0]['label'] = sentiment_mapping[sentiment[0]['label']]
    # Generate Word Cloud and Save it as an Image
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)

    # Save the Word Cloud to the provided path
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(wordcloud_path, format='png')
    plt.close()

    print("Nouns: ", nouns)
    print("Adjectives: ", adjectives)
    print("Verbs: ", verbs)

    return {
        "transcript": transcript,
        "sentiment": sentiment,
        "sound_intensity": float(sound_intensity),
        "fundamental_frequency": float(fundamental_frequency),
        "spectral_energy": float(spectral_energy),
        "spectral_centroid": float(avg_spectral_centroid),
        "zero_crossing_rate": float(zero_crossing_rate),
        "avg_words_per_minute": float(avg_words_per_minute),
        "avg_unique_words_per_minute": float(avg_unique_words_per_minute),
        "unique_word_count": int(unique_words),
        "filler_words_per_minute": float(filler_words_per_minute),
        "noun_count": len(nouns),
        "adjective_count": len(adjectives),
        "verb_count": len(verbs),
        "pause_rate": float(pause_rate),
        "top3":response.text,
    },pitches
