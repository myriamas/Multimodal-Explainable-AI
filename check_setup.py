"""
Vérificateur de dépendances et de configuration.
À exécuter avant de lancer l'application Streamlit.
"""

import sys
import importlib
from pathlib import Path

def check_module(name, package_name=None):
    """Vérifie qu'un module Python est importable."""
    if package_name is None:
        package_name = name
    try:
        importlib.import_module(package_name)
        print(f"✓ {name}")
        return True
    except ImportError as e:
        print(f"✗ {name}: {e}")
        return False

def check_file(path):
    """Vérifie qu'un fichier existe."""
    p = Path(path)
    if p.exists():
        print(f"✓ {path}")
        return True
    else:
        print(f"✗ {path} (MISSING)")
        return False

print("=" * 80)
print("VÉRIFICATION DES DÉPENDANCES ET CONFIGURATION")
print("=" * 80)

# Dépendances Python
print("\n[DÉPENDANCES PYTHON]")
dependencies = [
    ("PyTorch", "torch"),
    ("TorchVision", "torchvision"),
    ("Streamlit", "streamlit"),
    ("NumPy", "numpy"),
    ("PIL", "PIL"),
    ("SHAP", "shap"),
    ("LIME", "lime"),
    ("scikit-image", "skimage"),
    ("Librosa", "librosa"),
]

all_deps_ok = True
for name, module in dependencies:
    if not check_module(name, module):
        all_deps_ok = False

# Fichiers du projet
print("\n[FICHIERS DU PROJET]")
project_files = [
    "app.py",
    "core/__init__.py",
    "core/compatibility.py",
    "core/model_factory.py",
    "core/shap_safe_engine.py",
    "image_pipeline/__init__.py",
    "image_pipeline/model.py",
    "image_pipeline/preprocess.py",
    "image_pipeline/gradcam.py",
    "image_pipeline/lime_image.py",
    "image_pipeline/shap_image.py",
    "audio_pipeline/__init__.py",
    "audio_pipeline/model.py",
    "audio_pipeline/preprocess.py",
    "audio_pipeline/lime_audio.py",
    "audio_pipeline/shap_audio.py",
]

all_files_ok = True
for file in project_files:
    if not check_file(file):
        all_files_ok = False

# Résumé
print("\n" + "=" * 80)
if all_deps_ok and all_files_ok:
    print("✓ TOUT EST OK - VOUS POUVEZ LANCER L'APPLICATION")
    print("\nCommande pour démarrer:")
    print("  streamlit run app.py")
else:
    print("✗ PROBLÈMES DÉTECTÉS")
    if not all_deps_ok:
        print("\nInstallez les dépendances manquantes:")
        print("  pip install torch torchvision streamlit shap lime scikit-image librosa soundfile")
    if not all_files_ok:
        print("\nRecréez les fichiers manquants ou retirez les appels correspondants.")
print("=" * 80)
