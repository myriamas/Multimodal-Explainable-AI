import torchvision.transforms as T

def preprocess_image():
    """
    Pipeline de transformation pour les images (Radiographies).
    """
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])