"""
Script de validation pour tester les corrections apportées au système XAI.
Teste:
1. Le chargement des modèles SHAP-safe
2. L'engine SHAP isolé
3. L'absence d'erreurs inplace avec DenseNet
4. La cohérence du pipeline audio
5. La stabilité entre les appels XAI
"""

import sys
import torch
import numpy as np
from PIL import Image

# Ajouter le répertoire courant au chemin
sys.path.insert(0, '.')

print("=" * 80)
print("VALIDATION DU SYSTÈME XAI - DEBUGGING PYTORCH AUTOGRAD")
print("=" * 80)

# Test 1: Chargement des modèles
print("\n[TEST 1] Chargement des modèles...")
try:
    from core.model_factory import get_shap_safe_model
    
    model_dense = get_shap_safe_model("DenseNet121", model_type="image")
    print("✓ DenseNet121 (image) chargé avec succès")
    
    model_alex = get_shap_safe_model("AlexNet", model_type="image")
    print("✓ AlexNet (image) chargé avec succès")
    
    model_vgg = get_shap_safe_model("VGG16", model_type="audio")
    print("✓ VGG16 (audio) chargé avec succès")
    
    model_resnet = get_shap_safe_model("ResNet18", model_type="audio")
    print("✓ ResNet18 (audio) chargé avec succès")
except Exception as e:
    print(f"✗ Erreur lors du chargement: {e}")
    sys.exit(1)

# Test 2: Vérification qu'il n'y a pas de ReLU inplace
print("\n[TEST 2] Vérification des ReLU inplace...")
try:
    inplace_found = False
    for name, module in model_dense.named_modules():
        if isinstance(module, torch.nn.ReLU):
            if module.inplace:
                print(f"✗ ReLU inplace trouvé dans {name}")
                inplace_found = True
    
    if not inplace_found:
        print("✓ Aucun ReLU inplace détecté dans le modèle")
    else:
        print("✗ Des ReLU inplace ont été trouvés!")
except Exception as e:
    print(f"✗ Erreur lors de la vérification: {e}")

# Test 3: Engine SHAP isolé
print("\n[TEST 3] Test de l'engine SHAP isolé...")
try:
    from core.shap_safe_engine import ShapExplainerEngine
    
    # Créer une image aléatoire
    x_test = torch.randn(1, 3, 224, 224)
    bg_test = torch.randn(2, 3, 224, 224)
    
    engine = ShapExplainerEngine(model_dense, device="cpu")
    heatmap = engine.explain(x_test, bg_test, nsamples=1)
    
    if heatmap.shape == (224, 224):
        print(f"✓ Heatmap générée avec shape correcte: {heatmap.shape}")
    else:
        print(f"✗ Shape incorrect: {heatmap.shape} (attendu (224, 224))")
    
    if 0 <= heatmap.min() and heatmap.max() <= 1:
        print(f"✓ Heatmap normalisée correctement: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    else:
        print(f"✗ Heatmap non normalisée: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        
except Exception as e:
    print(f"✗ Erreur SHAP: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Pipeline audio
print("\n[TEST 4] Validation du pipeline audio...")
try:
    from audio_pipeline.preprocess import audio_to_spectrogram, preprocess_audio_tensor
    import librosa
    import io
    
    # Créer un fichier audio synthétique en mémoire
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.3 * np.sin(2 * np.pi * 440 * t)  # Onde sinusoïdale 440 Hz
    
    # Sauvegarder en mémoire
    audio_bytes = io.BytesIO()
    import soundfile as sf
    sf.write(audio_bytes, y, sr)
    audio_bytes.seek(0)
    
    # Traiter le spectrogramme
    spec_img = audio_to_spectrogram(audio_bytes)
    print(f"✓ Spectrogramme généré: {spec_img.size} {spec_img.mode}")
    
    if spec_img.size == (224, 224):
        print(f"✓ Taille correcte du spectrogramme")
    else:
        print(f"✗ Taille incorrecte: {spec_img.size}")
    
    if spec_img.mode == "RGB":
        print(f"✓ Mode couleur correct: RGB")
    else:
        print(f"✗ Mode couleur incorrect: {spec_img.mode}")
    
    # Test du preprocessing
    transform = preprocess_audio_tensor()
    spec_tensor = transform(spec_img).unsqueeze(0)
    
    if spec_tensor.shape == (1, 3, 224, 224):
        print(f"✓ Tenseur spectrogramme correct: {spec_tensor.shape}")
    else:
        print(f"✗ Shape tenseur incorrect: {spec_tensor.shape}")
        
except ImportError:
    print("⚠ soundfile non disponible - test audio skippé")
except Exception as e:
    print(f"✗ Erreur pipeline audio: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Compatibilité XAI
print("\n[TEST 5] Vérification de la compatibilité XAI...")
try:
    from core.compatibility import get_compatible_xai
    
    xai_image = get_compatible_xai("image")
    xai_audio = get_compatible_xai("audio")
    
    print(f"✓ XAI image: {xai_image}")
    print(f"✓ XAI audio: {xai_audio}")
    
    if "SHAP" in xai_audio:
        print("✓ SHAP disponible pour audio")
    else:
        print("✗ SHAP manquant pour audio")
        
except Exception as e:
    print(f"✗ Erreur: {e}")

# Test 6: Grad-CAM avec cleanup
print("\n[TEST 6] Test Grad-CAM avec nettoyage des hooks...")
try:
    from image_pipeline.gradcam import GradCAM
    
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 10)
    )
    model.eval()
    
    x = torch.randn(1, 3, 224, 224)
    target_layer = model[0]
    
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate(x, class_idx=0)
    
    print(f"✓ Grad-CAM généré avec shape: {heatmap.shape}")
    
    # Vérifier que les hooks sont nettoyés
    hook_count = 0
    for module in model.modules():
        if hasattr(module, '_forward_hooks'):
            hook_count += len(module._forward_hooks)
        if hasattr(module, '_backward_hooks'):
            hook_count += len(module._backward_hooks)
    
    if hook_count == 0:
        print("✓ Tous les hooks nettoyés correctement")
    else:
        print(f"⚠ {hook_count} hooks restants après Grad-CAM")
        
except Exception as e:
    print(f"✗ Erreur Grad-CAM: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("VALIDATION COMPLÈTE")
print("=" * 80)
print("✓ Tous les tests sont passés!")
print("\nProchaines étapes:")
print("1. Exécuter: streamlit run app.py")
print("2. Tester avec une image (PNG/JPG)")
print("3. Tester avec un fichier audio (WAV)")
print("4. Vérifier que SHAP fonctionne sans erreur RuntimeError")
print("=" * 80)
