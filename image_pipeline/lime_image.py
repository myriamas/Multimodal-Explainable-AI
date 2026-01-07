import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic

def lime_explain_image(pil_img, predict_fn, top_labels=1, num_samples=100):
    img_np = np.array(pil_img)

    explainer = lime_image.LimeImageExplainer()
    
    # On définit une segmentation plus simple (50 zones au lieu de 200+)
    # Cela réduit massivement le nombre de combinaisons que le CPU doit tester
    segmenter = lambda x: slic(x, n_segments=50, compactness=10, sigma=1)

    explanation = explainer.explain_instance(
        img_np,
        classifier_fn=predict_fn,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmenter # Utilisation de notre segmenter rapide
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=5, # On ne garde que les 5 zones les plus importantes
        hide_rest=False
    )

    vis = (mark_boundaries(temp / 255.0, mask) * 255).astype(np.uint8)
    return vis, int(top_label)