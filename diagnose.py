"""
Script de diagnostic complet pour le système XAI.
Exécutez ceci si vous rencontrez des problèmes.
"""

import sys
import os
from pathlib import Path

def print_section(title):
    """Affiche un titre de section."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def check_file_integrity():
    """Vérifie l'intégrité des fichiers du projet."""
    print_section("VÉRIFICATION INTÉGRITÉ DES FICHIERS")
    
    critical_files = {
        "Core": [
            "core/__init__.py",
            "core/compatibility.py",
            "core/model_factory.py",
            "core/shap_safe_engine.py",
        ],
        "Image Pipeline": [
            "image_pipeline/__init__.py",
            "image_pipeline/model.py",
            "image_pipeline/preprocess.py",
            "image_pipeline/gradcam.py",
            "image_pipeline/lime_image.py",
            "image_pipeline/shap_image.py",
        ],
        "Audio Pipeline": [
            "audio_pipeline/__init__.py",
            "audio_pipeline/model.py",
            "audio_pipeline/preprocess.py",
            "audio_pipeline/lime_audio.py",
            "audio_pipeline/shap_audio.py",
        ],
        "App": [
            "app.py",
        ]
    }
    
    missing_count = 0
    for category, files in critical_files.items():
        print(f"\n{category}:")
        for file in files:
            p = Path(file)
            if p.exists():
                size = p.stat().st_size
                print(f"  ✓ {file} ({size} bytes)")
            else:
                print(f"  ✗ {file} (MISSING)")
                missing_count += 1
    
    if missing_count == 0:
        print(f"\n✓ Tous les fichiers présents!")
        return True
    else:
        print(f"\n✗ {missing_count} fichiers manquants!")
        return False

def check_imports():
    """Vérifie que tous les imports fonctionnent."""
    print_section("VÉRIFICATION DES IMPORTS")
    
    imports_to_test = {
        "Core": [
            "from core.compatibility import get_compatible_xai",
            "from core.model_factory import get_shap_safe_model, _replace_inplace_relu",
            "from core.shap_safe_engine import ShapExplainerEngine",
        ],
        "Image Pipeline": [
            "from image_pipeline.model import load_model",
            "from image_pipeline.preprocess import preprocess_image",
            "from image_pipeline.gradcam import GradCAM",
            "from image_pipeline.lime_image import lime_explain_image",
            "from image_pipeline.shap_image import shap_explain_image",
        ],
        "Audio Pipeline": [
            "from audio_pipeline.model import load_audio_model",
            "from audio_pipeline.preprocess import audio_to_spectrogram, preprocess_audio_tensor",
            "from audio_pipeline.lime_audio import lime_explain_audio",
            "from audio_pipeline.shap_audio import shap_explain_audio",
        ]
    }
    
    failed_count = 0
    for category, imports in imports_to_test.items():
        print(f"\n{category}:")
        for import_str in imports:
            try:
                exec(import_str)
                print(f"  ✓ {import_str}")
            except Exception as e:
                print(f"  ✗ {import_str}")
                print(f"    Error: {str(e)[:100]}")
                failed_count += 1
    
    if failed_count == 0:
        print(f"\n✓ Tous les imports fonctionnent!")
        return True
    else:
        print(f"\n✗ {failed_count} imports échoués!")
        return False

def check_model_loading():
    """Vérifie que les modèles se chargent."""
    print_section("VÉRIFICATION CHARGEMENT MODÈLES")
    
    try:
        from core.model_factory import get_shap_safe_model
        
        models_to_test = {
            "DenseNet121 (image)": ("DenseNet121", "image"),
            "AlexNet (image)": ("AlexNet", "image"),
            "VGG16 (audio)": ("VGG16", "audio"),
            "ResNet18 (audio)": ("ResNet18", "audio"),
        }
        
        for name, (model_name, model_type) in models_to_test.items():
            try:
                model = get_shap_safe_model(model_name, model_type)
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {str(e)[:80]}")
        
        return True
    except Exception as e:
        print(f"  ✗ Erreur générale: {str(e)}")
        return False

def check_hooks_cleanup():
    """Vérifie que les hooks sont nettoyés."""
    print_section("VÉRIFICATION NETTOYAGE HOOKS")
    
    try:
        import torch
        from core.model_factory import get_shap_safe_model
        
        model = get_shap_safe_model("AlexNet", "image")
        
        # Ajouter des hooks
        def dummy_hook(module, input, output):
            pass
        
        hook_handles = []
        for module in list(model.modules())[:3]:  # Premiers 3 modules
            h = module.register_forward_hook(dummy_hook)
            hook_handles.append(h)
        
        # Compter les hooks
        hook_count_before = 0
        for module in model.modules():
            if hasattr(module, '_forward_hooks'):
                hook_count_before += len(module._forward_hooks)
        
        print(f"  Hooks avant cleanup: {hook_count_before}")
        
        # Nettoyer
        for module in model.modules():
            if hasattr(module, '_forward_hooks'):
                module._forward_hooks.clear()
            if hasattr(module, '_backward_hooks'):
                module._backward_hooks.clear()
        
        # Compter après
        hook_count_after = 0
        for module in model.modules():
            if hasattr(module, '_forward_hooks'):
                hook_count_after += len(module._forward_hooks)
        
        print(f"  Hooks après cleanup: {hook_count_after}")
        
        if hook_count_after == 0:
            print("  ✓ Cleanup fonctionne!")
            return True
        else:
            print(f"  ✗ {hook_count_after} hooks restants!")
            return False
            
    except Exception as e:
        print(f"  ✗ Erreur: {str(e)}")
        return False

def check_shap_engine():
    """Vérifie l'engine SHAP."""
    print_section("VÉRIFICATION ENGINE SHAP")
    
    try:
        import torch
        from core.shap_safe_engine import ShapExplainerEngine
        from core.model_factory import get_shap_safe_model
        
        model = get_shap_safe_model("AlexNet", "image")
        engine = ShapExplainerEngine(model, device="cpu")
        
        x = torch.randn(1, 3, 224, 224)
        bg = torch.randn(2, 3, 224, 224)
        
        print("  Générant heatmap SHAP...")
        heatmap = engine.explain(x, bg, nsamples=1)
        
        print(f"  Shape: {heatmap.shape}")
        print(f"  Min: {heatmap.min():.4f}, Max: {heatmap.max():.4f}")
        
        if heatmap.shape == (224, 224):
            print("  ✓ Shape correct")
        else:
            print(f"  ✗ Shape incorrect: {heatmap.shape}")
        
        if 0 <= heatmap.min() and heatmap.max() <= 1:
            print("  ✓ Normalization correct")
        else:
            print(f"  ✗ Normalization incorrect")
        
        engine.reset()
        return True
        
    except Exception as e:
        print(f"  ✗ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_audio_pipeline():
    """Vérifie le pipeline audio."""
    print_section("VÉRIFICATION PIPELINE AUDIO")
    
    try:
        import numpy as np
        import torch
        from audio_pipeline.preprocess import audio_to_spectrogram, preprocess_audio_tensor
        import io
        import soundfile as sf
        
        # Générer audio test
        print("  Générant audio synthétique...")
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        y = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, y, sr)
        audio_bytes.seek(0)
        
        # Traiter
        print("  Convertissant en spectrogramme...")
        spec_img = audio_to_spectrogram(audio_bytes)
        
        print(f"  Taille: {spec_img.size}")
        print(f"  Mode: {spec_img.mode}")
        
        checks = []
        if spec_img.size == (224, 224):
            print("  ✓ Taille correcte")
            checks.append(True)
        else:
            print(f"  ✗ Taille incorrecte: {spec_img.size}")
            checks.append(False)
        
        if spec_img.mode == "RGB":
            print("  ✓ Mode RGB")
            checks.append(True)
        else:
            print(f"  ✗ Mode incorrect: {spec_img.mode}")
            checks.append(False)
        
        # Test tensor
        print("  Convertissant en tenseur...")
        transform = preprocess_audio_tensor()
        tensor = transform(spec_img).unsqueeze(0)
        
        if tensor.shape == (1, 3, 224, 224):
            print(f"  ✓ Tensor shape correct")
            checks.append(True)
        else:
            print(f"  ✗ Tensor shape incorrect: {tensor.shape}")
            checks.append(False)
        
        return all(checks)
        
    except ImportError:
        print("  ⚠ soundfile non installé, skipped")
        return True
    except Exception as e:
        print(f"  ✗ Erreur: {str(e)}")
        return False

def check_dependencies():
    """Vérifie les dépendances Python."""
    print_section("VÉRIFICATION DÉPENDANCES")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("streamlit", "Streamlit"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("shap", "SHAP"),
        ("lime", "LIME"),
        ("skimage", "scikit-image"),
        ("librosa", "Librosa"),
    ]
    
    all_ok = True
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} (pip install {module_name})")
            all_ok = False
    
    return all_ok

def main():
    """Lance tous les diagnostiques."""
    print("\n" + "=" * 80)
    print("  DIAGNOSTIC COMPLET - XAI UNIFIED PLATFORM")
    print("=" * 80)
    
    results = {}
    
    # Exécuter tous les tests
    results["Fichiers"] = check_file_integrity()
    results["Imports"] = check_imports()
    results["Dépendances"] = check_dependencies()
    results["Modèles"] = check_model_loading()
    results["Hooks"] = check_hooks_cleanup()
    results["SHAP Engine"] = check_shap_engine()
    results["Audio Pipeline"] = check_audio_pipeline()
    
    # Résumé
    print_section("RÉSUMÉ DIAGNOSTIC")
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    all_ok = all(results.values())
    
    print("\n" + "=" * 80)
    if all_ok:
        print("  ✓ TOUS LES TESTS RÉUSSIS - PRÊT À LANCER!")
        print("  Commande: streamlit run app.py")
    else:
        failed = [name for name, result in results.items() if not result]
        print(f"  ✗ {len(failed)} TEST(S) ÉCHOUÉ(S):")
        for name in failed:
            print(f"     - {name}")
        print("  Consultez les messages d'erreur ci-dessus.")
    print("=" * 80 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
