from modules.base_model import DeepLearningBaseModel
import numpy as np

SUPPORTED_MODES = {
    "max",
    "min",
    
    "mean",
    
    "global_max", 
    "global_mean"
}

class Pool2DTrainable(DeepLearningBaseModel):
    def __init__(
       self,
        mode: str = "max",
        kernel_size: int = 2,
        stride: int = None,
    ):
        if mode not in SUPPORTED_MODES:
            raise ValueError(
               f"Invalid mode '{mode}'"
               f"{sorted(SUPPORTED_MODES)}"
            )
        
        self.mode = mode
        self.k_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
        super().__init__(
            name="pool",
            
            # bassicaly we don't use this 
            # but the parent class (DeepLearningBaseModel) not allowed that
            
            # so we defined W as kernel_size
            W=np.array([kernel_size]),

            # and we defined b as kernel_size
            b=np.array([stride]),
            
            activation_function_key=None,
        )
        
        self._cache: dict = {
            "x": None,
            "mask": None, 
        }
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(
                f"Input must be 4-D (B, C, H, W), {x.shape}."
            )
        self._cache["x"] = x
        
        if self.mode == "global_max":
            return self._global_forward(x, mode="max")
        if self.mode == "global_mean":
            return self._global_forward(x, mode="mean")

        return self._local_forward(x)
    
    def _local_forward(self, x: np.ndarray) -> np.ndarray:
        B, C, H, W = x.shape
        k, s = self.k_size, self.stride

        if H < k or W < k:
            raise ValueError(
                f"Kernel ({k}x{k}) lebih besar dari input spasial ({H}x{W})."
            )

        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        
        shape = (B, C, H_out, W_out, k, k)
        strides = (
            x.strides[0],
            x.strides[1],
            x.strides[2] * s,
            x.strides[3] * s,
            x.strides[2],
            x.strides[3],
        )
        windows = np.lib.stride_tricks.as_strided(
            x, shape=shape, strides=strides
        )
        
        flat = windows.reshape(B, C, H_out, W_out, k * k)
        
        if self.mode == "max":
            argmax = flat.argmax(axis=-1)
            out = flat.max(axis=-1)

            mask = np.zeros_like(flat, dtype=bool)
            idx = np.arange(k * k)
            mask = (argmax[..., np.newaxis] == idx)
            self._cache["mask"] = mask.reshape(B, C, H_out, W_out, k, k)

        elif self.mode == "min":
            argmin = flat.argmin(axis=-1)
            out = flat.min(axis=-1)

            idx = np.arange(k * k)
            mask = (argmin[..., np.newaxis] == idx)
            self._cache["mask"] = mask.reshape(B, C, H_out, W_out, k, k)

        elif self.mode == "mean":
            out = flat.mean(axis=-1)
            self._cache["mask"] = None

        return out
    
    def _global_forward(self, x: np.ndarray, mode: str) -> np.ndarray:
        if mode == "max":
            B, C, H, W = x.shape
            flat = x.reshape(B, C, -1)
            argmax  = flat.argmax(axis=-1)
            out = flat.max(axis=-1)

            mask = np.zeros_like(flat, dtype=bool)
            np.put_along_axis(mask, argmax[..., np.newaxis], True, axis=-1)
            self._cache["mask"] = mask.reshape(B, C, H, W)

        else:
            out = x.mean(axis=(-2, -1))
            self._cache["mask"] = None

        return out
    
    def backward(self, prev_delta: np.ndarray) -> np.ndarray:
       
        x = self._cache["x"]
        if x is None:
            raise RuntimeError(
                "Cache empty. Make sure forward() called before backward()."
            )

        if self.mode == "global_max":
            return self._global_backward(prev_delta, mode="max")
        if self.mode == "global_mean":
            return self._global_backward(prev_delta, mode="mean")

        return None, None, self._local_backward(prev_delta)
    
    def _local_backward(self, prev_delta: np.ndarray) -> np.ndarray:
        x    = self._cache["x"]
        mask = self._cache["mask"]
        B, C, H, W = x.shape
        k, s = self.k_size, self.stride
        H_out, W_out = prev_delta.shape[2], prev_delta.shape[3]

        delta_prev = np.zeros_like(x)

        if self.mode in ("max", "min"):
            delta_expanded = prev_delta[:, :, :, :, np.newaxis, np.newaxis]
            grad_windows = mask * delta_expanded

            for i in range(H_out):
                for j in range(W_out):
                    r0, c0 = i * s, j * s
                    delta_prev[:, :, r0:r0+k, c0:c0+k] += grad_windows[:, :, i, j, :, :]

        elif self.mode == "mean":
            scale = 1.0 / (k * k)
            for i in range(H_out):
                for j in range(W_out):
                    r0, c0 = i * s, j * s
                    delta_prev[:, :, r0:r0+k, c0:c0+k] += (
                        prev_delta[:, :, i, j, np.newaxis, np.newaxis] * scale
                    )

        return delta_prev
    
    def _global_backward(self, prev_delta: np.ndarray, mode: str) -> np.ndarray:
        x = self._cache["x"]
        B, C, H, W = x.shape

        delta_expanded = prev_delta[:, :, np.newaxis, np.newaxis]

        if mode == "max":
            mask = self._cache["mask"]
            delta_prev = mask * delta_expanded

        else:
            scale = 1.0 / (H * W)
            delta_prev = np.broadcast_to(
                delta_expanded * scale, (B, C, H, W)
            ).copy()

        return delta_prev
    
    def update_step(self, dl_dw=None, dl_db=None, lr=None) -> None:
        pass
    
    def zero_cache(self) -> None:
        self._cache = {"x": None, "mask": None}
    
    def output_shape(self, h_in: int, w_in: int) -> tuple:
        if self.mode in ("global_max", "global_mean"):
            return None, None
        h_out = (h_in - self.k_size) // self.stride + 1
        w_out = (w_in - self.k_size) // self.stride + 1
        return h_out, w_out

    def __repr__(self) -> str:
        return (
            f"Pool2DTrainable("
            f"mode='{self.mode}', "
            f"kernel_size={self.k_size}, "
            f"stride={self.stride})"
        )
