from modules.base_model import DeepLearningBaseModel
import numpy as np

class FlattenMatrix(DeepLearningBaseModel):
    def __init__(self):
        super().__init__(
            name="flatten",
            
            # bassicaly we don't use this 
            # but the parent class (DeepLearningBaseModel) not allowed that
            
            W=np.array([0]),
            b=np.array([0]),
            activation_function_key=None,
        )
        self._cache = {"x_shape": None}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache["x_shape"] = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, prev_delta: np.ndarray):
        x_shape = self._cache["x_shape"]
        if x_shape is None:
            raise RuntimeError(
                "Cache empty. Make sure forward() called before backward()."
            )
        return None, None, prev_delta.reshape(x_shape)

    def update_step(self, dl_dw=None, dl_db=None, lr=None) -> None:
        pass

    def zero_cache(self) -> None:
        self._cache = {"x_shape": None}