from modules.base_model import DeepLearningBaseModel
from modules.loss_function_optim import LossFunction, SGDMomentum
from modules.softmax_layer import SoftmaxLayer
import os
import json
import numpy as np
from datetime import datetime

class SeqFrameworkTrainable:
    def __init__(self):
        
        """ __models_conf is a list that contain of
        {
            name: "conv2d" | "neural" | "pooling" | "conv2dtranspose",
            params: {
                "weights": [.....],
                "bias": [.....],
            },
            index: 0,
            hasforward: True | False,
            activation: ....
            pramnumber: ...
            size: ...
        }
        """
        self.__models_conf = []
        self.__models = []
        self.__loss_hist = []
        
        self._loss_fn: LossFunction = None
        self._optimizer: SGDMomentum = None
        self._compiled: bool = False
        
    @property
    def loss_hist(self) -> list:
        return self.__loss_hist
    
    @property
    def get_models(self) -> list:
        return self.__models
    
    @property
    def get_models_config(self) -> list:
        return self.__models_conf
    
    def __create_template_cofig__(
        self, name: str, idx: int, 
        w: np.ndarray, b: np.ndarray, 
        size: float, paramnumber: int,
        activation: str,
        hasforward: bool = True,
    ):
        return {
            "name": name,
            "params": {
                "weights": w,
                "bias": b,
            },
            "index": idx,
            "hasforward": hasforward,
            "size": size,
            "activation": activation,
            "paramnumber": paramnumber
        }
    
    def add(self, model:DeepLearningBaseModel, hasforward: bool=True):
        self.__models.append(model)
        last_idx = len(self.__models)
        
        model_weights, model_bias = model.get_params
        param_number = model.get_param_size
        param_mem_size = model.getparam_memory
        activation_function = model.activation_function_key
        hasforward = True if model.activation_function_key is not None else False
        
        conf = self.__create_template_cofig__(
            model.get_name, last_idx,
            model_weights, model_bias,
            param_mem_size, param_number,
            activation_function,
            hasforward
        )
        
        self.__models_conf.append(conf)
    
    def compile(
        self,
        loss: str = "categorical_ce",
        lr: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        
        if len(self.__models) == 0:
            raise RuntimeError("There is no layer. Call add() first.")
        
        if loss == "categorical_ce":
            last_layer = self.__models[-1]
            if not isinstance(last_layer, SoftmaxLayer):
                raise ValueError(
                    "Loss 'categorical_ce' requires SoftmaxLayer as "
                    "lLast layer. Add model.add(SoftmaxLayer())"
                    "before compile()."
                )
        
        if loss == "binary_ce":
            last_layer = self.__models[-1]
            if hasattr(last_layer, "output_size") and last_layer.output_size != 1:
                raise ValueError(
                    f"Loss 'binary_ce' requires output_size=1,"
                    f"{last_layer.output_size}."
                )
                
        self._loss_fn   = LossFunction(loss)
        self._optimizer = SGDMomentum(lr=lr, momentum=momentum)
        self._compiled  = True
        
        print(f"Compiled — loss: {loss}, optimizer: SGDMomentum(lr={lr}, momentum={momentum})")
        self.summary()
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int     = 10,
        batch_size: int = 32,
    ) -> None:
        
        if not self._compiled:
            raise RuntimeError(
                "The model hasn't been compiled yet. Call compile() first."
            )
        N = X.shape[0]
        n_batches = int(np.ceil(N / batch_size))
        is_cat_ce = self._loss_fn.name == "categorical_ce"
        
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(N)
            X_shuf = X[perm]
            y_shuf = y[perm]
            
            epoch_loss = 0.0
            
            for batch_idx in range(n_batches):
                
                start = batch_idx * batch_size
                end   = min(start + batch_size, N)
                
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]
                
                a = X_batch
                for layer in self.__models:
                    a = layer.forward(a)
                y_hat = a
                
                batch_loss = self._loss_fn.forward(y_hat, y_batch)
                epoch_loss += batch_loss
                delta = self._loss_fn.backward(y_hat, y_batch)
                
                for layer_idx, layer in enumerate(reversed(self.__models)):
                    actual_idx = len(self.__models) - 1 - layer_idx
                    if is_cat_ce and isinstance(layer, SoftmaxLayer):
                        _, _, delta = layer.backward(delta)
                        continue
                    
                    # print(self.__models_conf[layer_idx]["name"])
                    # print(self.__models_conf[layer_idx]["index"])
                    # print(actual_idx)
                    dl_dw, dl_db, delta = layer.backward(delta)
                    if dl_dw is not None:
                        self._optimizer.update(layer, actual_idx, dl_dw, dl_db)
                
                print(
                    f"Epoch {epoch}/{epochs} "
                    f"| Batch {batch_idx + 1}/{n_batches} "
                    f"| Loss: {batch_loss:.6f}",
                    end="\r"
                )
                
            avg_loss = epoch_loss / n_batches
            self.__loss_hist.append(avg_loss)
            print(
                f"Epoch {epoch}/{epochs} "
                f"| Avg Loss: {avg_loss:.6f}"
                + " " * 20
            )
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._compiled:
            raise RuntimeError("The model has not been compiled yet.")
        a = X
        for layer in self.__models:
            a = layer.forward(a)
        return a
    
    def summary(self) -> None:
        """Tampilkan ringkasan arsitektur model."""
        print("\n" + "=" * 58)
        print(f"{'Layer':<14} {'Activation':<16} {'Params':>8} {'Memory':>10}")
        print("=" * 58)
 
        total_params  = 0
        total_mem     = 0
 
        for conf in self.__models_conf:
            name       = conf["name"]
            activation = conf["activation"] or "-"
            params     = conf["paramnumber"]
            mem_bytes  = conf["size"]
 
            total_params += params
            total_mem    += mem_bytes
 
            mem_str = (
                f"{mem_bytes / 1024:.2f} KB"
                if mem_bytes >= 1024
                else f"{mem_bytes} B"
            )
            print(
                f"{name:<14} {activation:<16} {params:>8} {mem_str:>10}"
            )
 
        print("-" * 58)
        total_mem_str = (
            f"{total_mem / 1024:.2f} KB"
            if total_mem >= 1024
            else f"{total_mem} B"
        )
        print(f"{'Total':<14} {'':<16} {total_params:>8} {total_mem_str:>10}")
        print("=" * 58 + "\n")
    
    def save_model_config(
        self,
        train_acc: float = 0.0,
        test_acc: float = 0.0,
        base_folder: str = "folder_results"
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{base_folder}_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)

        config_output = []

        for conf in self.__models_conf:
            name = conf["name"]
            idx = conf["index"]

            has_forward = conf["hasforward"]

            w_filename = None
            b_filename = None

            if has_forward:
                weights = conf["params"]["weights"]
                bias = conf["params"]["bias"]

                w_filename = f"{name}-{idx}-weights.npz"
                b_filename = f"{name}-{idx}-bias.npz"

                w_path = os.path.join(folder_name, w_filename)
                b_path = os.path.join(folder_name, b_filename)

                np.savez_compressed(w_path, weights=weights)
                np.savez_compressed(b_path, bias=bias)

            config_output.append({
                "name": f"{name}-{idx}",
                "wparam": w_filename,
                "bparam": b_filename,
                "activation": conf["activation"],
                "bytesize": str(conf["size"]),
                "forward": has_forward
            })

        result_json = {
            "timestamp": timestamp,
            "trainaccuracy": train_acc,
            "testaccuracy": test_acc,
            "configuration": config_output
        }

        json_path = os.path.join(folder_name, "model_results.json")
        with open(json_path, "w") as f:
            json.dump(result_json, f, indent=4)

        print(f"Model saved in folder: {folder_name}")
        