import numpy as np
from modules.base_model import DeepLearningBaseModel
from modules.nonlinear_function import SoftmaxFunction

class SoftmaxLayer(DeepLearningBaseModel):
 
    def __init__(self):
        super().__init__(
            name="softmax",
            W=np.array([0]),
            b=np.array([0]),
            activation_function_key=None,
        )
        self._softmax = SoftmaxFunction()
        self._cache: dict = {"z": None}
 
    def forward(self, z: np.ndarray) -> np.ndarray:
        self._cache["z"] = z
        return self._softmax.forward(z)
 
    def backward(self, prev_delta: np.ndarray):
        return None, None, prev_delta
 
    def update_step(self, dl_dw=None, dl_db=None, lr=None) -> None:
        pass
 
    def zero_cache(self) -> None:
        self._cache = {"z": None}
 
    def __repr__(self) -> str:
        return "SoftmaxLayer()"
    