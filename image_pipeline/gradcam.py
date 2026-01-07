import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []

        # Nettoyage préventif des anciens hooks
        if hasattr(target_layer, '_backward_hooks'):
            target_layer._backward_hooks.clear()

        self.hooks.append(target_layer.register_forward_hook(self._save_activation))
        self.hooks.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx: int):
        # On utilise .clone() pour éviter les erreurs de modification inplace
        input_clone = input_tensor.clone().detach().requires_grad_(True)
        
        self.model.zero_grad()
        output = self.model(input_clone)
        
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-12)

        # Retrait des hooks immédiatement après usage
        for hook in self.hooks:
            hook.remove()

        return cam.detach().cpu()