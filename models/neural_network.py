import numpy as np
from modules.base_model import DeepLearningBaseModel
from modules.nonlinear_function import PairingFunction

# Xavier Uniform (Glorot 2010).
def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

# Xavier Normal (Glorot 2010).
def xavier_normal(fan_in: int, fan_out: int) -> np.ndarray:
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

# He / Kaiming Uniform (He 2015).
def he_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

# He / Kaiming Normal.
def he_normal(fan_in: int, fan_out: int) -> np.ndarray:
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std


_INIT_MAP = {
    "relu": he_uniform,
    "leaky_relu": he_uniform,
    "elu": he_normal,
    "selu": he_normal,
    "sigmoid": xavier_uniform,
    "tanh": xavier_uniform,
    "piece_wise_linear": xavier_uniform,
    "softplus": xavier_uniform,
    "swish": he_uniform,
    "gelu": he_normal,
    "unit_step": xavier_uniform,
    "sign": xavier_uniform,
}


def auto_init_weights(fan_in: int, fan_out: int, activation: str) -> np.ndarray:
    init_fn = _INIT_MAP.get(activation, xavier_uniform)
    return init_fn(fan_in, fan_out)


class NeuralNetworkTrainable(DeepLearningBaseModel):
    def __init__(
        self,
        input_size: int = 8,
        output_size: int = 8,
        activation_func: str = "sigmoid",
        weights: np.ndarray = None,
        bias: np.ndarray = None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        
        if weights is not None:
            if weights.shape != (input_size, output_size):
                raise ValueError(f"invalid shape and weights inizialization")
            W = weights
        else:
            W = auto_init_weights(input_size, output_size, activation_func)
        
        if bias is not None:
            if bias.shape != (1, output_size):
                raise ValueError(f"invalid shape and bias inizialization")
            b = bias
        else:
            b = np.zeros((1, output_size))
        
        super().__init__(
            name="neural",
            W=W,
            b=b,
            activation_function_key=activation_func,
        )
        
        self.pair_function = PairingFunction(activation_func)
        
        
        self.__cache__ = {
            'x': None,
            'a': None,
            'z': None,
        }
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.W + self.b
        a = self.pair_function.forward(z)

        self.__cache__['x'] = x
        self.__cache__['z'] = z
        self.__cache__['a'] = a

        return a
    
    def backward(self, prev_delta: np.ndarray):

        x = self.__cache__['x']
        z = self.__cache__['z']
        W = self.W
        
        batch_size = x.shape[0]

        da_dz = self.pair_function.derivative(z)
        delta = prev_delta * da_dz
        dl_dw = (x.T @ delta) / batch_size
        dl_db = np.sum(delta, axis=0, keepdims=True) / batch_size
        delta_prev = delta @ W.T

        return dl_dw, dl_db, delta_prev
    
    def update_step(
        self,
        dl_dw: np.ndarray,
        dl_db: np.ndarray,
        lr: float,
    ) -> None:
        self.W -= lr * dl_dw
        self.b -= lr * dl_db
        
    def zero_cache(self) -> None:
        self._cache = {"x": None, "z": None, "a": None}