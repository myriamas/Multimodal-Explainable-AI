import librosa
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

def audio_to_spectrogram(audio_file):
    """
    Convertit un fichier .wav en image (Mel Spectrogram) pour que les CNN puissent le lire.
    """
    y, sr = librosa.load(audio_file, sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normalisation 0-255
    S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-12)
    img_array = (S_db_norm * 255).astype(np.uint8)
    
    # Image RGB 3 canaux pour compatibilité Grad-CAM/LIME
    img_pil = Image.fromarray(img_array).convert("RGB")
    img_pil = img_pil.resize((224, 224))
    return img_pil

def preprocess_audio_tensor():
    """
    Transforme le spectrogramme en tenseur pour le modèle.
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])