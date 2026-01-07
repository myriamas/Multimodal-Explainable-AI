import torch
from validate_shap_integration import DummyModel
from core.shap_safe_engine import ShapExplainerEngine

m=DummyModel()
e=ShapExplainerEngine(m)
print('Calling _model_wrapper...')
out = e._model_wrapper(torch.randn(1,3,224,224))
print('Returned type:', type(out))
try:
    import numpy as np
    print('Output shape (if numpy):', getattr(out, 'shape', None))
except Exception:
    pass
