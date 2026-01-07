"""
Factory pour créer des versions "SHAP-safe" des modèles PyTorch.
Cette approche clône le modèle et reconfigure tous les ReLU inplace.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import copy


def _replace_inplace_relu(module):
    """
    Parcourt récursivement un module et remplace tous les ReLU inplace par des versions non-inplace.
    Fonctionne aussi avec les modules contents dans des Sequential, ModuleList, etc.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            # Remplacer le ReLU inplace par un ReLU sans inplace
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            # Récursif sur les enfants
            _replace_inplace_relu(child)


def get_shap_safe_model(model_name: str, model_type: str = "image"):
    """
    Charge un modèle pré-entraîné et le configure pour SHAP.
    
    Args:
        model_name: "DenseNet121", "AlexNet", "VGG16", "ResNet18"
        model_type: "image" ou "audio"
    
    Returns:
        model: Modèle configuré pour SHAP (sans inplace ReLU, eval mode)
    """
    if model_type == "image":
        if model_name == "DenseNet121":
            model = models.densenet121(weights="DEFAULT")
        elif model_name == "AlexNet":
            model = models.alexnet(weights="DEFAULT")
        else:
            model = models.densenet121(weights="DEFAULT")
    else:  # audio
        if model_name == "VGG16":
            model = models.vgg16(weights="DEFAULT")
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
        elif model_name == "ResNet18":
            model = models.resnet18(weights="DEFAULT")
            model.fc = nn.Linear(model.fc.in_features, 2)
        else:
            model = models.vgg16(weights="DEFAULT")
    
    # Remplacer tous les ReLU inplace
    _replace_inplace_relu(model)
    
    # Désactiver explicitement inplace sur tous les modules qui auraient l'attribut
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False
    
    model.eval()
    return model


def create_compatible_model_for_xai(base_model, xai_method: str):
    """
    Adapte le modèle selon la méthode XAI.
    Pour SHAP, on s'assure que le modèle est sans inplace operations.
    
    Args:
        base_model: Modèle PyTorch de base
        xai_method: "Grad-CAM", "LIME", "SHAP"
    
    Returns:
        model: Modèle adapté
    """
    if xai_method == "SHAP":
        # Cloner le modèle et reconfigurer pour SHAP
        model = copy.deepcopy(base_model)
        _replace_inplace_relu(model)
        for module in model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
        model.eval()
        return model
    else:
        # Pour Grad-CAM et LIME, le modèle original fonctionne bien
        return base_model
