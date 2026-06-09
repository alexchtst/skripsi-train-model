import numpy as np
from pympler import asizeof

class DeepLearningBaseModel:
    def __init__(
        self,
        name: str,
        W: np.ndarray = None,
        b: np.ndarray = None,
        activation_function_key: str = None,
    ):
        self.name = name
        self.W = W
        self.b = b
        self.activation_function_key = activation_function_key

    @property
    def get_name(self) -> str:
        return self.name

    @property
    def get_params(self):
        return self.W, self.b

    @property
    def get_param_size(self) -> int:
        if self.W is None:
            raise ValueError(f"[{self.name}] Weight is not initialized yet")
        total = self.W.size
        if self.b is not None:
            total += self.b.size
        return total

    @property
    def is_valid(self) -> bool:
        if self.W is None:
            return False
        if self.name is None:
            return False
        return True
    
    @property
    def getparam_memory(self):
        w_mem_size = asizeof.asizeof(self.W)
        b_mem_size = asizeof.asizeof(self.b)
        
        return w_mem_size + b_mem_size

    def __repr__(self) -> str:
        w_shape = self.W.shape if self.W is not None else None
        b_shape = self.b.shape if self.b is not None else None
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"W={w_shape}, b={b_shape}, "
            f"activation='{self.activation_function_key}')"
        )