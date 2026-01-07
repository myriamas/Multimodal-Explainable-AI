"""
Module SHAP pour les images avec gestion robuste des erreurs inplace.
"""

import numpy as np
import torch
import copy
from core.shap_safe_engine import ShapExplainerEngine


def shap_explain_image(model, x, background, nsamples=2):
    """
    Explique les prédictions du modèle pour une image en utilisant SHAP.
    
    Args:
        model: Modèle PyTorch (sera cloné pour SHAP-safety)
        x: Tenseur d'entrée (1, C, H, W)
        background: Tenseur de contexte (N, C, H, W)
        nsamples: Nombre d'échantillons pour DeepExplainer
    
    Returns:
        heatmap: Numpy array normalisé (H, W) avec valeurs dans [0, 1]
    """
    # Cloner le modèle pour créer une version SHAP-safe
    safe_model = copy.deepcopy(model)
    safe_model.eval()
    
    # Créer l'engine SHAP
    engine = ShapExplainerEngine(safe_model, device="cpu")
    
    # Cloner les inputs
    x_input = x.clone().detach().float()
    background_input = background.clone().detach().float()
    
    # Expliquer
    heatmap = engine.explain(
        x_input,
        background_input,
        class_idx=None,
        nsamples=nsamples
    )
    
    # Cleanup
    engine.reset()
    del safe_model, engine
    
    return heatmap