from image_pipeline.lime_image import lime_explain_image

def lime_explain_audio(spectrogram_pil, predict_fn):
    """
    Applique LIME sur le spectrogramme audio. 
    Utilise la version accélérée (n_segments=50) de lime_image.
    """
    # On définit 100 échantillons pour garder un bon compromis vitesse/précision sur CPU
    return lime_explain_image(spectrogram_pil, predict_fn, num_samples=100)