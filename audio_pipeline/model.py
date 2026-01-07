import torch
import torchvision.models as models

def load_audio_model(model_name="VGG16"):
    """
    Charge un modèle pré-entraîné pour l'audio.
    Requirement: Multiple pretrained models (VGG16 et ResNet18).
    """
    if model_name == "VGG16":
        model = models.vgg16(weights="DEFAULT")
        # Sortie pour 2 classes (Real vs Fake)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    elif model_name == "ResNet18":
        model = models.resnet18(weights="DEFAULT")
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    else:
        model = models.vgg16(weights="DEFAULT")
    
    model.eval()
    return model