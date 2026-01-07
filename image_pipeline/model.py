import torchvision.models as models
import torch

def load_model(model_name="DenseNet121"):
    """
    Charge un modèle pré-entraîné pour les images.
    Requirement: Multiple pretrained models (DenseNet et AlexNet).
    """
    if model_name == "DenseNet121":
        model = models.densenet121(weights="DEFAULT")
    elif model_name == "AlexNet":
        model = models.alexnet(weights="DEFAULT")
    else:
        model = models.densenet121(weights="DEFAULT")
    
    model.eval()
    return model