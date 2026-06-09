import numpy as np
from modules.base_model import DeepLearningBaseModel
from modules.nonlinear_function import PairingFunction

def im2col(
    x: np.ndarray, 
    k_size: int, 
    stride: int = 1
    ) -> np.ndarray:
    
    B, C, H, W = x.shape
    H_out = (H - k_size) // stride + 1
    W_out = (W - k_size) // stride + 1

    shape = (B, C, H_out, W_out, k_size, k_size)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2] * stride,
        x.strides[3] * stride,
        x.strides[2],
        x.strides[3],
    )
    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    col = patches.transpose(0, 1, 4, 5, 2, 3).reshape(B, C * k_size * k_size, -1)
    return col

def col2im(
    col: np.ndarray,
    x_shape: tuple,
    k_size: int,
    stride: int = 1,
) -> np.ndarray:
    
    B, C, H, W = x_shape
    H_out = (H - k_size) // stride + 1
    W_out = (W - k_size) // stride + 1

    col_reshaped = col.reshape(B, C, k_size, k_size, H_out, W_out)

    dx = np.zeros((B, C, H, W), dtype=col.dtype)

    for i in range(k_size):
        i_max = i + stride * H_out
        for j in range(k_size):
            j_max = j + stride * W_out
            dx[:, :, i:i_max:stride, j:j_max:stride] += col_reshaped[:, :, i, j, :, :]

    return dx

class Conv2DTrainable(DeepLearningBaseModel):
    def __init__(
        self,
        in_channel: int = 3,
        out_channel: int = 3,
        activation_func: str = "sigmoid",
        kernel_size: int = 3,
        stride: int = 1,
    ):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.k_size = kernel_size
        self.stride = stride

        fan_in  = in_channel  * kernel_size * kernel_size
        fan_out = out_channel * kernel_size * kernel_size

        # Xavier Uniform (Glorot 2010).
        limit = np.sqrt(6.0 / (fan_in + fan_out))

        W = np.random.uniform(
            -limit, limit,
            (out_channel, in_channel, kernel_size, kernel_size),
        )
        b = np.random.uniform(
            -limit, limit,
            (out_channel,),
        )

        super().__init__(
            name="conv2d",
            W=W,
            b=b,
            activation_function_key=activation_func,
        )

        self.pair_function = PairingFunction(activation_func)

        self._cache: dict = {
            "x": None,
            "col": None,
            "z": None,
            "a": None,
        }


    def forward(self, x: np.ndarray) -> np.ndarray:
        B, C, H, W_img = x.shape

        if C != self.in_channel:
            raise ValueError(
                f"Channel input {C} is not valid with in_channel={self.in_channel}."
            )
        if H < self.k_size or W_img < self.k_size:
            raise ValueError(
                f"invalid kernel size"
                f"spatial input ({H}x{W_img})."
            )

        H_out = (H - self.k_size) // self.stride + 1
        W_out = (W_img - self.k_size) // self.stride + 1

        col = im2col(x, self.k_size, self.stride)
        W_col = self.W.reshape(self.out_channel, -1)

        z_flat = np.tensordot(W_col, col, axes=([1], [1]))
        z_flat = z_flat.transpose(1, 0, 2)
        z_flat += self.b[np.newaxis, :, np.newaxis]

        z = z_flat.reshape(B, self.out_channel, H_out, W_out)
        a = self.pair_function.forward(z)

        self._cache["x"] = x
        self._cache["col"] = col
        self._cache["z"] = z
        self._cache["a"] = a

        return a


    def backward(self, prev_delta: np.ndarray):
        x   = self._cache["x"]
        col = self._cache["col"]
        z   = self._cache["z"]

        if x is None:
            raise RuntimeError(
                "Cache is empty. Make sure forward() has been called before backward()."
            )

        B = x.shape[0]
        H_out, W_out = z.shape[2], z.shape[3]

        da_dz = self.pair_function.derivative(z)
        delta  = prev_delta * da_dz

        delta_flat = delta.reshape(B, self.out_channel, -1)

        dl_db = delta_flat.sum(axis=(0, 2)) / B

        dl_dw_flat = np.einsum("bfp,bcp->fc", delta_flat, col) / B
        dl_dw = dl_dw_flat.reshape(self.W.shape)

        W_col = self.W.reshape(self.out_channel, -1)

        dcol = np.einsum("fc,bfp->bcp", W_col, delta_flat)

        delta_prev = col2im(dcol, x.shape, self.k_size, self.stride)

        return dl_dw, dl_db, delta_prev

    def update_step(
        self,
        dl_dw: np.ndarray,
        dl_db: np.ndarray,
        lr: float,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be > 0 {lr}")
        self.W -= lr * dl_dw
        self.b -= lr * dl_db

    def zero_cache(self) -> None:
        self._cache = {"x": None, "col": None, "z": None, "a": None}

    def output_shape(self, h_in: int, w_in: int) -> tuple:
        h_out = (h_in - self.k_size) // self.stride + 1
        w_out = (w_in - self.k_size) // self.stride + 1
        return self.out_channel, h_out, w_out
    
