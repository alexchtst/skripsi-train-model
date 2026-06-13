from models.base_model import DeepLearningBaseModel
from models.loss_function_optim import LossFunction, SGDMomentum
from models.softmax_layer import SoftmaxLayer
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
        self.__acc_hist = []
        
        self._loss_fn: LossFunction = None
        self._optimizer: SGDMomentum = None
        self._compiled: bool = False
        
    @property
    def loss_hist(self) -> list:
        return self.__loss_hist

    @property
    def acc_hist(self) -> list:
        return self.__acc_hist
    
    @property
    def get_models(self) -> list:
        return self.__models
    
    @property
    def get_models_config(self) -> list:
        return self.__models_conf
    
    def __create_template_cofig__(
        self,
        name: str,
        idx: int,
        w: np.ndarray,
        b: np.ndarray,
        size: float,
        paramnumber: int,
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
    
    def add(self, model: DeepLearningBaseModel, hasforward: bool = True):
        self.__models.append(model)
        last_idx = len(self.__models)
        
        model_weights, model_bias = model.get_params
        param_number = model.get_param_size
        param_mem_size = model.getparam_memory
        activation_function = model.activation_function_key

        hasforward = True if model.activation_function_key is not None else False
        
        conf = self.__create_template_cofig__(
            model.get_name,
            last_idx,
            model_weights,
            model_bias,
            param_mem_size,
            param_number,
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
            raise RuntimeError("Tidak ada layer. Panggil add() dulu.")
        
        if loss == "categorical_ce":
            last_layer = self.__models[-1]

            if not isinstance(last_layer, SoftmaxLayer):
                raise ValueError(
                    "Loss 'categorical_ce' membutuhkan SoftmaxLayer sebagai "
                    "layer terakhir. Tambahkan model.add(SoftmaxLayer()) "
                    "sebelum compile()."
                )
        
        if loss == "binary_ce":
            last_layer = self.__models[-1]

            if hasattr(last_layer, "output_size") and last_layer.output_size != 1:
                raise ValueError(
                    f"Loss 'binary_ce' membutuhkan output_size=1, "
                    f"dapat {last_layer.output_size}."
                )
                
        self._loss_fn = LossFunction(loss)
        self._optimizer = SGDMomentum(lr=lr, momentum=momentum)
        self._compiled = True
        
        print(
            f"Compiled — loss: {loss}, "
            f"optimizer: SGDMomentum(lr={lr}, momentum={momentum})"
        )

        self.summary()

    def _accuracy_score(
        self,
        y_hat: np.ndarray,
        y_true: np.ndarray,
    ):
        if self._loss_fn is None:
            return None

        y_hat = np.asarray(y_hat)
        y_true = np.asarray(y_true)

        if self._loss_fn.name == "categorical_ce":
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(1, -1)

            y_pred_label = np.argmax(y_hat, axis=1)

            if y_true.ndim == 2:
                y_true_label = np.argmax(y_true, axis=1)
            else:
                y_true_label = y_true.astype(int).reshape(-1)

            return np.mean(y_pred_label == y_true_label)

        elif self._loss_fn.name == "binary_ce":
            y_pred_label = (y_hat.reshape(-1) >= 0.5).astype(int)
            y_true_label = y_true.reshape(-1).astype(int)

            return np.mean(y_pred_label == y_true_label)

        return None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        
        if not self._compiled:
            raise RuntimeError(
                "Model belum di-compile. Panggil compile() dulu."
            )

        N = X.shape[0]
        n_batches = int(np.ceil(N / batch_size))
        is_cat_ce = self._loss_fn.name == "categorical_ce"
        
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(N)

            X_shuf = X[perm]
            y_shuf = y[perm]
            
            epoch_loss = 0.0
            epoch_acc_sum = 0.0
            total_seen = 0
            
            for batch_idx in range(n_batches):
                
                start = batch_idx * batch_size
                end = min(start + batch_size, N)
                
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                current_batch_size = X_batch.shape[0]
                
                a = X_batch

                for layer in self.__models:
                    a = layer.forward(a)

                y_hat = a
                
                batch_loss = self._loss_fn.forward(y_hat, y_batch)
                epoch_loss += batch_loss

                batch_acc = self._accuracy_score(y_hat, y_batch)

                if batch_acc is not None:
                    epoch_acc_sum += batch_acc * current_batch_size
                    total_seen += current_batch_size
                
                delta = self._loss_fn.backward(y_hat, y_batch)
                
                for layer_idx, layer in enumerate(reversed(self.__models)):
                    actual_idx = len(self.__models) - 1 - layer_idx

                    if is_cat_ce and isinstance(layer, SoftmaxLayer):
                        _, _, delta = layer.backward(delta)
                        continue
                    
                    dl_dw, dl_db, delta = layer.backward(delta)

                    if dl_dw is not None:
                        self._optimizer.update(
                            layer,
                            actual_idx,
                            dl_dw,
                            dl_db
                        )
                
                if batch_acc is not None:
                    print(
                        f"Epoch {epoch}/{epochs} "
                        f"| Batch {batch_idx + 1}/{n_batches} "
                        f"| Loss: {batch_loss:.6f} "
                        f"| Acc: {batch_acc:.4f}",
                        end="\r"
                    )
                else:
                    print(
                        f"Epoch {epoch}/{epochs} "
                        f"| Batch {batch_idx + 1}/{n_batches} "
                        f"| Loss: {batch_loss:.6f}",
                        end="\r"
                    )
                
            avg_loss = epoch_loss / n_batches
            self.__loss_hist.append(avg_loss)

            if total_seen > 0:
                avg_acc = epoch_acc_sum / total_seen
                self.__acc_hist.append(avg_acc)

                print(
                    f"Epoch {epoch}/{epochs} "
                    f"| Avg Loss: {avg_loss:.6f} "
                    f"| Avg Acc: {avg_acc:.4f}"
                    + " " * 20
                )
            else:
                print(
                    f"Epoch {epoch}/{epochs} "
                    f"| Avg Loss: {avg_loss:.6f}"
                    + " " * 20
                )
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._compiled:
            raise RuntimeError("Model belum di-compile.")

        a = X

        for layer in self.__models:
            a = layer.forward(a)

        return a

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        if not self._compiled:
            raise RuntimeError("Model belum di-compile.")

        y_hat = self.predict(X)

        loss = self._loss_fn.forward(y_hat, y)
        acc = self._accuracy_score(y_hat, y)

        if acc is not None:
            print(f"Loss: {loss:.6f} | Accuracy: {acc:.4f}")
        else:
            print(f"Loss: {loss:.6f}")

        return loss, acc
    
    def summary(self) -> None:
        """Tampilkan ringkasan arsitektur model."""
        print("\n" + "=" * 58)
        print(f"{'Layer':<14} {'Activation':<16} {'Params':>8} {'Memory':>10}")
        print("=" * 58)
 
        total_params = 0
        total_mem = 0
 
        for conf in self.__models_conf:
            name = conf["name"]
            activation = conf["activation"] or "-"
            params = conf["paramnumber"]
            mem_bytes = conf["size"]
 
            total_params += params
            total_mem += mem_bytes
 
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