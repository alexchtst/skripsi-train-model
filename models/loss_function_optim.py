import numpy as np
from modules.base_model import DeepLearningBaseModel

class LossFunction:
 
    SUPPORTED = {"mse", "binary_ce", "categorical_ce"}
    _EPS = 1e-12
 
    def __init__(self, name: str):
        if name not in self.SUPPORTED:
            raise ValueError(
                f"Loss '{name}' tidak dikenal. "
                f"Pilihan: {sorted(self.SUPPORTED)}"
            )
        self.name = name
 
 
    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        return getattr(self, f"_{self.name}_forward")(y_hat, y)
 
    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return getattr(self, f"_{self.name}_backward")(y_hat, y)
 
 
    def _mse_forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((y_hat - y) ** 2))
 
    def _mse_backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        B = y_hat.shape[0]
        return 2 * (y_hat - y) / B
 
 
    def _binary_ce_forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        y_hat = np.clip(y_hat, self._EPS, 1 - self._EPS)
        return float(-np.mean(
            y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        ))
 
    def _binary_ce_backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        B = y_hat.shape[0]
        y_hat = np.clip(y_hat, self._EPS, 1 - self._EPS)
        return (y_hat - y) / (y_hat * (1 - y_hat) * B)
 
    def _categorical_ce_forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        y_hat = np.clip(y_hat, self._EPS, 1.0)
        return float(-np.mean(np.sum(y * np.log(y_hat), axis=-1)))
 
    def _categorical_ce_backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        B = y_hat.shape[0]
        return (y_hat - y) / B
 
    def __repr__(self) -> str:
        return f"LossFunction(name='{self.name}')"
    

class SGDMomentum:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        if lr <= 0:
            raise ValueError(f"lr harus > 0, dapat: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum harus dalam [0, 1), dapat: {momentum}")
 
        self.lr       = lr
        self.momentum = momentum
        self._state: dict = {}
 
    def init_state(self, layer_index: int, w_shape: tuple, b_shape: tuple) -> None:
        self._state[layer_index] = {
            "v_w": np.zeros(w_shape),
            "v_b": np.zeros(b_shape),
        }
 
    def update(
        self,
        layer: DeepLearningBaseModel,
        layer_index: int,
        dl_dw: np.ndarray,
        dl_db: np.ndarray,
    ) -> None:
        if layer_index not in self._state:
            self.init_state(layer_index, dl_dw.shape, dl_db.shape)
 
        state = self._state[layer_index]
 
        state["v_w"] = self.momentum * state["v_w"] + (1 - self.momentum) * dl_dw
        state["v_b"] = self.momentum * state["v_b"] + (1 - self.momentum) * dl_db
 
        layer.update_step(state["v_w"], state["v_b"], self.lr)
 
    def reset_state(self) -> None:
        self._state = {}
 
    def __repr__(self) -> str:
        return f"SGDMomentum(lr={self.lr}, momentum={self.momentum})"