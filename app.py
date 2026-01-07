import io
import os
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F

# Imports des modules du projet 
from core.compatibility import get_compatible_xai
from core.model_factory import get_shap_safe_model
from image_pipeline.model import load_model
from image_pipeline.preprocess import preprocess_image
from image_pipeline.gradcam import GradCAM
from image_pipeline.lime_image import lime_explain_image
from image_pipeline.shap_image import shap_explain_image
from audio_pipeline.preprocess import audio_to_spectrogram, preprocess_audio_tensor
from audio_pipeline.model import load_audio_model
from audio_pipeline.shap_audio import shap_explain_audio

# Configuration CPU : Utilisation de tes 6 coeurs physiques
torch.set_num_threads(6)

# --- STYLE CSS PERSONNALISÉ ---
st.set_page_config(page_title="XAI Unified Platform", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    [data-testid="stSidebar"] { background-color: #1e293b !important; }
    [data-testid="stSidebar"] * { color: white !important; }
    div[data-baseweb="select"] * { color: #1e293b !important; }
    .stMetric { background-color: white; border-radius: 10px; padding: 15px; border: 1px solid #e2e8f0; }
    h1 { color: #0f172a; font-family: 'Inter', sans-serif; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

st.title("Unified Explainable AI Interface")
st.markdown("---")

# =========================
# Fonctions Utilitaires
# ========================= 

def _cleanup_hooks(model):
    """Nettoie tous les hooks du modèle pour éviter les conflits."""
    for module in model.modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
        if hasattr(module, '_backward_hooks'):
            module._backward_hooks.clear()
        if hasattr(module, '_forward_pre_hooks'):
            module._forward_pre_hooks.clear()


@st.cache_resource
def _get_model(input_type, model_name):
    """
    Charge le modèle avec cache.
    Les versions SHAP-safe sont créées à la demande dans shap_image/shap_audio.
    """
    if input_type == "image":
        model = load_model(model_name)
    else:
        model = load_audio_model(model_name)
    
    model.eval()
    
    # Désactiver les gradients par défaut
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def _predict(model, input_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probas = F.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probas))
    return idx, float(probas[idx]), probas

def _overlay_cam_on_image(img_pil: Image.Image, cam_2d: np.ndarray, alpha: float = 0.5):
    """ Transforme un tenseur/array XAI en image visualisable """
    img = np.array(img_pil).astype(np.float32) / 255.0
    # Normalisation de la heatmap
    cam = (cam_2d - cam_2d.min()) / (cam_2d.max() - cam_2d.min() + 1e-12)
    # Redimensionnement pour correspondre à l'image d'origine
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]))).astype(np.float32) / 255.0
    heat = np.zeros_like(img)
    heat[..., 0] = cam_resized # Couche rouge pour la chaleur
    return np.clip((1 - alpha) * img + alpha * heat, 0.0, 1.0)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Configuration")
    input_type = st.radio("Modality", ["image", "audio"])
    
    if input_type == "image":
        model_name = st.selectbox("Architecture", ["DenseNet121", "AlexNet"])
    else:
        model_name = st.selectbox("Architecture", ["VGG16", "ResNet18"])
    
    model = _get_model(input_type, model_name)
    xai_options = get_compatible_xai(input_type)
    xai_method = st.selectbox("XAI Method", xai_options)

tab1, tab2 = st.tabs(["Classification & XAI", "Comparative Analysis"])

# =========================
# TAB 1 — Classification
# =========================
with tab1:
    col_input, col_res = st.columns([1, 1.5], gap="large")
    
    if "raw_img" not in st.session_state: st.session_state.raw_img = None

    with col_input:
        st.subheader("Data Upload")
        f_types = ["png", "jpg", "jpeg"] if input_type == "image" else ["wav"]
        uploaded = st.file_uploader(f"Select {input_type} file", type=f_types)
        
        if uploaded:
            if input_type == "image":
                st.session_state.raw_img = Image.open(uploaded).convert("RGB").resize((224, 224))
                st.session_state.transform = preprocess_image()
            else:
                st.audio(uploaded)
                st.session_state.raw_img = audio_to_spectrogram(uploaded)
                st.session_state.transform = preprocess_audio_tensor()
            
            st.image(st.session_state.raw_img, caption="Input Data", use_container_width=True)
            st.session_state.x = st.session_state.transform(st.session_state.raw_img).unsqueeze(0)

    with col_res:
        if uploaded:
            st.subheader("Model Prediction")
            pred_idx, pred_conf, _ = _predict(model, st.session_state.x)
            
            c1, c2 = st.columns(2)
            c1.metric("Result Index", pred_idx)
            c2.metric("Confidence Score", f"{pred_conf:.2%}")
            
            st.markdown("---")
            st.subheader(f"Explanation: {xai_method}")
            
            # Nettoyer les hooks avant chaque XAI
            _cleanup_hooks(model)
            
            if xai_method == "Grad-CAM":
                target = model.features.denseblock4 if "DenseNet" in model_name else (model.features[-1] if hasattr(model, 'features') else model.layer4[-1])
                cam = GradCAM(model, target).generate(st.session_state.x, pred_idx)
                # Conversion du tenseur en image pour streamlit
                st.image(_overlay_cam_on_image(st.session_state.raw_img, cam[0].numpy()), use_container_width=True)
            
            elif xai_method == "LIME":
                with st.spinner("Computing LIME..."):
                    def _lime_fn(np_imgs):
                        batch = torch.stack([st.session_state.transform(Image.fromarray(i.astype(np.uint8))).clone() for i in np_imgs])
                        model.eval()
                        with torch.no_grad():
                            return F.softmax(model(batch), dim=1).detach().cpu().numpy()
                    vis, _ = lime_explain_image(st.session_state.raw_img, _lime_fn)
                    st.image(vis, use_container_width=True)

            elif xai_method == "SHAP":
                with st.spinner("Computing SHAP (cela peut prendre un moment)..."):
                    try:
                        # Nettoyer les hooks avant SHAP
                        _cleanup_hooks(model)
                        
                        # Création d'un background stable pour SHAP
                        bg = st.session_state.x.repeat(2, 1, 1, 1).clone()
                        
                        # Appel à SHAP selon le type d'entrée
                        if input_type == "image":
                            shap_map = shap_explain_image(model, st.session_state.x, bg, nsamples=2)
                        else:
                            shap_map = shap_explain_audio(model, st.session_state.x, bg, nsamples=2)
                        
                        # Affichage
                        vis_shap = _overlay_cam_on_image(st.session_state.raw_img, shap_map)
                        st.image(vis_shap, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur SHAP: {str(e)}")
                        st.info("Astuce: Essayez une segmentation LIME ou Grad-CAM à la place.")
                    finally:
                        # Cleanup après SHAP
                        _cleanup_hooks(model)


# =========================
# TAB 2 — Comparative View
# =========================
with tab2:
    if uploaded and st.session_state.raw_img:
        st.subheader("Side-by-Side Evaluation")
        pred_idx, _, _ = _predict(model, st.session_state.x)
        c1, c2, c3 = st.columns(3)
        
        # Fonctions pour LIME dans l'onglet comparatif
        def _cmp_lime_fn(np_imgs):
            batch = torch.stack([st.session_state.transform(Image.fromarray(i.astype(np.uint8))).clone() for i in np_imgs])
            model.eval()
            with torch.no_grad():
                return F.softmax(model(batch), dim=1).detach().cpu().numpy()

        with c1:
            st.markdown("**Grad-CAM**")
            try:
                _cleanup_hooks(model)
                target = model.features.denseblock4 if "DenseNet" in model_name else (model.features[-1] if hasattr(model, 'features') else model.layer4[-1])
                cam = GradCAM(model, target).generate(st.session_state.x, pred_idx)
                st.image(_overlay_cam_on_image(st.session_state.raw_img, cam[0].numpy()), use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM error: {str(e)}")
            finally:
                _cleanup_hooks(model)

        with c2:
            st.markdown("**LIME**")
            try:
                _cleanup_hooks(model)
                with st.spinner("LIME..."):
                    vis, _ = lime_explain_image(st.session_state.raw_img, _cmp_lime_fn, num_samples=30)
                    st.image(vis, use_container_width=True)
            except Exception as e:
                st.error(f"LIME error: {str(e)}")
            finally:
                _cleanup_hooks(model)

        with c3:
            st.markdown("**SHAP**")
            try:
                _cleanup_hooks(model)
                with st.spinner("SHAP..."):
                    bg = st.session_state.x.repeat(2, 1, 1, 1).clone() + 0.01 * torch.randn_like(st.session_state.x)
                    if input_type == "image":
                        shap_map = shap_explain_image(model, st.session_state.x, bg, nsamples=1)
                    else:
                        shap_map = shap_explain_audio(model, st.session_state.x, bg, nsamples=1)
                    st.image(_overlay_cam_on_image(st.session_state.raw_img, shap_map), use_container_width=True)
            except Exception as e:
                st.error(f"SHAP error: {str(e)}")
            finally:
                _cleanup_hooks(model)
    else:
        st.info("Please upload a file to enable comparison.")