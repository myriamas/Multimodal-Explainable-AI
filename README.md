
# Multimodal Explainable AI

End-to-end Streamlit platform for multimodal classification with robust Explainable AI (Grad-CAM, LIME, SHAP).

---

## Project Overview

This project is a production-ready multimodal Explainable AI (XAI) platform built with Streamlit.
It enables transparent and interpretable deep learning predictions on:

- Radiographic images
- Audio spectrograms

The platform integrates three complementary XAI methods:

- Grad-CAM for spatial localization
- LIME for local surrogate explanations
- SHAP for theoretically grounded feature attribution

A strong emphasis is placed on robustness and stability, especially for PyTorch + SHAP,
solving several well-known incompatibilities encountered in real-world projects.

---

## Objectives

- Provide reliable explanations for deep neural networks
- Support multimodal inputs (image and audio)
- Ensure SHAP compatibility with PyTorch models
- Avoid conflicts between XAI methods
- Offer a clean, modular, and extensible architecture
- Deliver an interactive Streamlit-based interface

---

## Key Features

- Multimodal classification (images and audio)
- Grad-CAM, LIME, and SHAP explanations
- SHAP-safe PyTorch models (no inplace ReLU)
- Robust SHAP engine with automatic fallbacks
- Hook isolation between Grad-CAM and SHAP
- Unified processing pipeline for image and audio
- Interactive and user-friendly Streamlit UI

---

## Project Structure
    
    unified_xai/
    ├── app.py # Streamlit application (UI & orchestration)
    ├── core/ # Central logic (model safety, SHAP engine)
    ├── image_pipeline/ # Image models, preprocessing and XAI
    ├── audio_pipeline/ # Audio spectrogram models and XAI
    ├── validate_fixes.py # Full regression test suite
    ├── check_setup.py # Dependency verification
    ├── diagnose.py # System diagnostics
    ├── requirements.txt
    └── README.md

---

## Supported Models

### Image Models
- DenseNet121 (default, SHAP-safe)
- AlexNet
- VGG16

### Audio Model
- ResNet18 applied to Mel spectrograms

All models are deep-copied and patched to remove inplace operations before SHAP execution.

---

## SHAP-Safe Design

This project introduces a custom SHAP execution engine that:

- Clones models to avoid side effects
- Replaces all ReLU(inplace=True) operations recursively
- Tries multiple explainers safely:
  - DeepExplainer
  - GradientExplainer
  - Manual gradient-based fallback
- Temporarily enables gradients when required
- Restores the original model state afterward
- Always returns a normalized heatmap

This guarantees stable SHAP explanations, even in constrained environments.

---

## Application Workflow

1. User uploads an image or audio file
2. Input is preprocessed into a (1, 3, 224, 224) tensor
3. Model predicts class and confidence
4. User selects an XAI method (Grad-CAM, LIME, SHAP)
5. Explanation heatmap is generated
6. Hooks and gradients are cleaned
7. Results are visualized in the UI

---

## Audio as Vision

Audio signals are converted into visual Mel spectrograms, enabling:

- Reuse of CNN-based vision models
- Unified explanation logic across modalities
- Compatibility with LIME and SHAP

Pipeline:

  Audio → Mel Spectrogram → RGB Image → CNN → XAI


---

## Validation & Diagnostics

The project includes several scripts to ensure robustness:

- check_setup.py – verify dependencies
- validate_shap_integration.py – SHAP autograd validation
- validate_fixes.py – full regression testing
- diagnose.py – system and model diagnostics

All scripts are designed to fail explicitly if an issue is detected.

---

## Quick Start

Install dependencies:
  
  pip install -r requirements.txt

Optional (for SHAP DeepExplainer support):

  pip install tensorflow
  
Verify environment:

  python check_setup.py

Launch the application:

  streamlit run app.py

Open your browser at:

  http://localhost:8501



---

## Typical Performance

| Operation | Time |
|---------|------|
| Inference | ~0.5 s |
| Grad-CAM | ~1–2 s |
| LIME | ~8–15 s |
| SHAP (DeepExplainer) | ~15–30 s |
| SHAP (fallback) | ~3–5 s |

---

## Project Status

- Multimodal support (image and audio)
- Grad-CAM, LIME and SHAP integrated
- SHAP-safe PyTorch models
- Robust fallback mechanisms
- Validation and diagnostics included

Status: Stable · Robust · Production-ready

---

## Author

Myriam  
Explainable AI · Multimodal Deep Learning · PyTorch
