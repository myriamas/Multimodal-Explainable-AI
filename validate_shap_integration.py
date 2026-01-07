"""Small validation script for SHAP integration and gradient fallback.
Runs a DenseNet121 model through `ShapExplainerEngine.explain` with random tensors.
Prints outcome and any exceptions.
"""
import traceback
import torch
import torch.nn as nn
from core.shap_safe_engine import ShapExplainerEngine


class DummyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


try:
    model = DummyModel(num_classes=5)
    engine = ShapExplainerEngine(model)
    x = torch.randn(1, 3, 224, 224)
    bg = torch.randn(2, 3, 224, 224)
    print('Starting explanation (may use gradient fallback if shap missing)...')
    heat = engine.explain(x, bg, class_idx=0)
    print('Explanation completed. Type:', type(heat), 'Shape:', getattr(heat, 'shape', None))
except Exception as e:
    print('Exception during validation:')
    traceback.print_exc()
