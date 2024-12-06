from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2

def create_emotion_model(num_ftrs, num_emotions):
    return nn.Sequential(
        nn.Linear(num_ftrs + num_emotions, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
def load_models(val_model_path,val_featmodel_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    resnet = models.resnet18(pretrained=False)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Identity()
    resnet.load_state_dict(
        torch.load(
            val_featmodel_path,
            map_location=device
        )
    )
    resnet = resnet.to(device)

    # num_ftrs = resnet.fc.in_features
    num_emotions = 1
    emotion_model = create_emotion_model(num_ftrs, num_emotions).to(device)
    emotion_model.load_state_dict(
        torch.load(
            val_model_path,
            map_location=device
        )
    )
    return resnet,emotion_model



def va_predict(emotion_model,resnet,faces,emotions):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def model_forward(images, emotions):
        resnet_features = resnet(images)
        batch_size = resnet_features.size(0)
        emotions = emotions.view(batch_size, -1)
        x = torch.cat((resnet_features, emotions), dim=1)
        output = emotion_model(x)
        return output

    arousal_list = []
    valence_list = []
    stress_list = []
    from tqdm import tqdm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for face, emotion in tqdm(zip(faces, emotions)):
        if face is not None:
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            # print(emotion)
            # print(emotion)
            emotion = emotion.to(device)
            print(emotion)
            output_va = model_forward(face_tensor, emotion)
            arousal = output_va[0][0].item()
            norm_arousal = float(output_va[0][0].item()) / 2 + 0.5
            valence = output_va[0][1].item()
            norm_valence = float(output_va[0][1].item()) / 2 + 0.5
            stress = (1 - norm_valence) * norm_arousal
            arousal_list.append(arousal)
            valence_list.append(valence)
            stress_list.append(stress)
        else:
            arousal_list.append('frame error')
            valence_list.append('frame error')
            stress_list.append('frame error')
    return valence_list, arousal_list, stress_list
