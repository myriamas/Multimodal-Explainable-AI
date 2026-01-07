"""
Engine SHAP isolé et robuste pour éviter les conflits avec Grad-CAM.
Gère les dimensions audio/image et les convertit correctement.
"""

import numpy as np
import torch
import importlib


class ShapExplainerEngine:
    """
    Engine SHAP sécurisé qui fonctionne de manière isolée sans interférence avec d'autres XAI.
    """
    
    def __init__(self, model, device="cpu"):
        """
        Initialise l'engine SHAP.
        
        Args:
            model: Modèle PyTorch pré-configuré pour SHAP (sans inplace)
            device: "cpu" ou "cuda"
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        # Ne pas modifier l'état `requires_grad` des paramètres ici.
        # L'état sera temporairement activé pendant l'explication puis restauré.
        self._orig_requires_grad = None
    
    def explain(self, x_input, background, class_idx=None, nsamples=10):
        """
        Explique la prédiction du modèle pour x_input.
        
        Args:
            x_input: Tenseur d'entrée (1, C, H, W)
            background: Tenseur de contexte pour SHAP (N, C, H, W)
            class_idx: Index de classe (si None, utilise la classe prédite)
            nsamples: Nombre d'échantillons pour DeepExplainer
        
        Returns:
            heatmap: Numpy array (H, W) normalisé [0, 1]
        """
        # Cloner pour éviter les modifications inplace
        x_input_clone = x_input.clone().detach().float().to(self.device)
        background_clone = background.clone().detach().float().to(self.device)

        # Inférer la classe si nécessaire (sans gradients)
        if class_idx is None:
            with torch.no_grad():
                output = self.model(x_input_clone)
                class_idx = torch.argmax(output, dim=1).item()

        # Wrapper pour SHAP (peut recevoir numpy ou torch)
        def model_fn(x_batch):
            if isinstance(x_batch, np.ndarray):
                x_tensor = torch.from_numpy(x_batch).float().to(self.device)
            else:
                x_tensor = x_batch.to(self.device)
            x_tensor = x_tensor.requires_grad_(True)
            output = self.model(x_tensor)
            try:
                return output.detach().cpu().numpy()
            except Exception:
                return output
            
        # Tenter d'utiliser SHAP si disponible; sinon fallback sur un attribut par gradient
        shap_values = None
        # Sauvegarder et activer temporairement les gradients sur les paramètres
        orig_requires = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad = True
        try:
            shap = importlib.import_module('shap')
            try:
                explainer = shap.DeepExplainer(self._model_wrapper, background_clone)
                shap_values = explainer.shap_values(x_input_clone)
            except ModuleNotFoundError as e:
                # Cas fréquent: shap essaie d'importer tensorflow qui n'est pas installé
                if 'tensorflow' in str(e).lower():
                    print(f"DeepExplainer requires TensorFlow (missing): {e}. Using gradient fallback...")
                else:
                    print(f"DeepExplainer failed: {e}. Trying GradientExplainer...")
            except Exception as e:
                # D'autres erreurs pendant la création/usage de DeepExplainer
                print(f"DeepExplainer error: {e}. Trying GradientExplainer...")

            if shap_values is None:
                try:
                    explainer = shap.GradientExplainer(self._model_wrapper, background_clone)
                    shap_values = explainer.shap_values(x_input_clone)
                except Exception as e:
                    print(f"GradientExplainer failed: {e}. Falling back to gradient-based attribution...")
                    shap_values = None
        except ModuleNotFoundError as e:
            # shap package itself is not installed
            print(f"SHAP library not available: {e}. Falling back to gradient-based attribution...")
        except Exception as e:
            print(f"Unexpected error importing shap: {e}. Falling back to gradient-based attribution...")
        finally:
            # Restaurer l'état requires_grad des paramètres
            for p, r in zip(self.model.parameters(), orig_requires):
                p.requires_grad = r

        # Si shap_values n'a pas pu être obtenu, calculer une attribution par gradient simple
        if shap_values is None:
            x_grad = x_input_clone.clone().detach().requires_grad_(True)
            # S'assurer que les gradients sont activés
            self.model.zero_grad()
            out = self.model(x_grad)
            score = out[0, class_idx]
            score.backward()
            grads = x_grad.grad.detach().cpu().numpy()
            # grads shape: (1, C, H, W) -> squeeze
            g = np.abs(grads.squeeze())
            if g.ndim == 3:
                heat = g.mean(axis=0)
            elif g.ndim == 2:
                heat = g
            else:
                # fallback reshape
                size = int(np.sqrt(g.size))
                heat = g.flatten()[:size*size].reshape(size, size)
            # normalize
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
            return heat.astype(np.float32)
        
        # Traiter les shap_values (peuvent être liste ou array)
        if isinstance(shap_values, list):
            sv = shap_values[int(class_idx)]
        else:
            sv = shap_values
        
        # Convertir en numpy et normaliser
        sv = np.array(sv).squeeze()
        
        # Gérer les dimensions pour audio et image
        if sv.ndim == 3:  # (C, H, W)
            # Moyenne sur les canaux
            heatmap = np.abs(sv).mean(axis=0)
        elif sv.ndim == 2:  # (H, W) - déjà bon
            heatmap = np.abs(sv)
        elif sv.ndim == 1:  # Vecteur 1D - reshape en carré
            size = int(np.sqrt(len(sv)))
            heatmap = np.abs(sv[:size*size]).reshape(size, size)
        else:
            raise ValueError(f"Dimensions non supportées pour shap_values: {sv.shape}")
        
        # Normalisation [0, 1]
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max - heatmap_min > 1e-12:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = np.zeros_like(heatmap)
        
        return heatmap.astype(np.float32)
    
    def _model_wrapper(self, x_tensor):
        """
        Wrapper pour SHAP qui s'assure que les opérations ne sont jamais inplace.
        """
        # x_tensor may be numpy array or torch tensor
        if isinstance(x_tensor, np.ndarray):
            x = torch.from_numpy(x_tensor).float().to(self.device)
        else:
            x = x_tensor
        x = x.requires_grad_(True)
        output = self.model(x)
        return output
    
    def reset(self):
        """Reset le modèle à l'état eval."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
