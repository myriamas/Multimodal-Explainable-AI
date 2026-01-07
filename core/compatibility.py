# I centralize the compatibility rules between input modalities and XAI methods

XAI_COMPATIBILITY = {
    "image": ["Grad-CAM", "LIME", "SHAP"],
    "audio": ["LIME", "SHAP"]  # SHAP fonctionne aussi avec audio grâce à shap_safe_engine
}

def get_compatible_xai(input_type: str):
    """
    Return the list of XAI methods compatible with the given input type.
    """
    return XAI_COMPATIBILITY.get(input_type.lower(), [])
