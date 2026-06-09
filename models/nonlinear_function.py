import numpy as np

class PairingFunction:
    def __init__(self, name: str = "sigmoid"):
        self.name = name
        self.forward_func_modules = ActivationFunction(name)
        self.deriv_func_modules = DerivativeFunction(name)

    def forward(self, z: np.ndarray) -> np.ndarray:
        return self.forward_func_modules(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return self.deriv_func_modules(z)

    def __repr__(self) -> str:
        return f"PairingFunction(name='{self.name}')"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


class ActivationFunction:

    SUPPORTED = {
        "unit_step", "sign", "piece_wise_linear",
        "sigmoid", "tanh",
        "relu", "leaky_relu", "elu", "selu",
        "softplus", "swish", "gelu",
    }

    _SELU_ALPHA = 1.6732632423543772
    _SELU_SCALE = 1.0507009873554805

    def __init__(self, name: str = "sigmoid"):
        if name not in self.SUPPORTED:
            raise ValueError(
                f"Activation '{name}' not found. "
                f"{sorted(self.SUPPORTED)}"
            )
        self.name = name

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return getattr(self, self.name)(z)


    def unit_step(self, z: np.ndarray) -> np.ndarray:
        return np.where(z < 0, 0.0, np.where(z > 0, 1.0, 0.5))

    def sign(self, z: np.ndarray) -> np.ndarray:
        return np.where(z < 0, -1.0, np.where(z > 0, 1.0, 0.0))

    def piece_wise_linear(self, z: np.ndarray) -> np.ndarray:
        return np.clip(z + 0.5, 0.0, 1.0)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return _sigmoid(z)

    def tanh(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)


    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def leaky_relu(self, z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(z > 0, z, alpha * z)

    def elu(self, z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))

    def selu(self, z: np.ndarray) -> np.ndarray:
        return self._SELU_SCALE * np.where(
            z > 0, z, self._SELU_ALPHA * (np.exp(z) - 1)
        )

    def softplus(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 20, z, np.log1p(np.exp(z)))

    def swish(self, z: np.ndarray) -> np.ndarray:
        return z * _sigmoid(z)

    def gelu(self, z: np.ndarray) -> np.ndarray:
        return 0.5 * z * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)
        ))

    def __repr__(self) -> str:
        return f"ActivationFunction(name='{self.name}')"



class DerivativeFunction:
    _SELU_ALPHA = ActivationFunction._SELU_ALPHA
    _SELU_SCALE = ActivationFunction._SELU_SCALE

    def __init__(self, name: str = "sigmoid"):
        if name not in ActivationFunction.SUPPORTED:
            raise ValueError(
                f"Derivative '{name}' not found. "
                f"{sorted(ActivationFunction.SUPPORTED)}"
            )
        self.name = name

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return getattr(self, self.name)(z)


    def unit_step(self, z: np.ndarray) -> np.ndarray:
        return np.zeros_like(z, dtype=float)

    def sign(self, z: np.ndarray) -> np.ndarray:
        return np.zeros_like(z, dtype=float)

    def piece_wise_linear(self, z: np.ndarray) -> np.ndarray:
        return np.where((z > -0.5) & (z < 0.5), 1.0, 0.0)


    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        s = _sigmoid(z)
        return s * (1 - s)

    def tanh(self, z: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(z) ** 2


    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1.0, 0.0)

    def leaky_relu(self, z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(z > 0, 1.0, alpha)

    def elu(self, z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(z > 0, 1.0, alpha * np.exp(z))

    def selu(self, z: np.ndarray) -> np.ndarray:
        return self._SELU_SCALE * np.where(
            z > 0, 1.0, self._SELU_ALPHA * np.exp(z)
        )

    def softplus(self, z: np.ndarray) -> np.ndarray:
        return _sigmoid(z)

    def swish(self, z: np.ndarray) -> np.ndarray:
        s = _sigmoid(z)
        return s + z * s * (1 - s)

    def gelu(self, z: np.ndarray) -> np.ndarray:
        c = np.sqrt(2 / np.pi)
        t = np.tanh(c * (z + 0.044715 * z ** 3))
        sech2 = 1 - t ** 2
        return 0.5 * (1 + t) + 0.5 * z * sech2 * c * (1 + 3 * 0.044715 * z ** 2)

    def __repr__(self) -> str:
        return f"DerivativeFunction(name='{self.name}')"


class SoftmaxFunction:
    def forward(self, z: np.ndarray) -> np.ndarray:
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        if z.ndim != 1:
            raise ValueError(
                "jacobian() onlu take 1-D input. "
                "calculate each sampel for batch."
            )
        s = self.forward(z)
        return np.diagflat(s) - np.outer(s, s)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.forward(z)

    def __repr__(self) -> str:
        return "SoftmaxFunction()"