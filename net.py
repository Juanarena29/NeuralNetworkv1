"""
RED NEURONAL MANUAL — modular, arquitectura configurable
=========================================================
Arquitectura parametrizada vía lista de dimensiones:
  layer_dims = [d_in, h1, h2, ..., d_out]
 
Activaciones:
  - Capas ocultas : ReLU
  - Capa de salida: lineal (regresión)
Loss: MSE
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ─────────────────────────────────────────────────────────────
# ACTIVACIONES
# ─────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


# ─────────────────────────────────────────────────────────────
# CLASE RED NEURONAL
# ─────────────────────────────────────────────────────────────

class NeuralNet:
    """
    Red neuronal fully-connected con ReLU en capas ocultas
    y activación lineal en la salida (regresión, loss MSE).

    Parámetros
    ----------
    layer_dims : list[int]
        Dimensiones incluyendo entrada y salida.
        Ej: [3, 16, 8, 1]  →  entrada 3, hidden 16, hidden 8, salida 1
    lr : float
        Learning rate para SGD. Hiperparámetro de la red, no del paso.
    """

    def __init__(self, layer_dims: list, lr: float = 0.01):
        self.L = len(layer_dims) - 1   # número de capas con pesos
        self.lr = lr
        self.params = {}               # W1, b1, W2, b2, ...
        self.cache = {}                # A0, Z1, A1, Z2, A2, ...
        self.grads = {}                # dW1, db1, ...

        for l in range(1, self.L + 1):
            n_in = layer_dims[l - 1]
            n_out = layer_dims[l]
            # He initialization — apropiada para ReLU
            self.params[f"W{l}"] = np.random.randn(
                n_in, n_out) * np.sqrt(2 / n_in)
            self.params[f"b{l}"] = np.zeros((1, n_out))

    # ─────────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante.

        X shape : (N, n_features)
        Retorna : y_pred shape (N, 1)
        Guarda Z y A de cada capa en self.cache para el backward.
        """
        self.cache["A0"] = X
        A = X

        for l in range(1, self.L + 1):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]

            # (N, n_in) @ (n_in, n_out) → (N, n_out)
            Z = A @ W + b
            self.cache[f"Z{l}"] = Z

            # última capa lineal, resto ReLU
            A = Z if l == self.L else relu(Z)
            self.cache[f"A{l}"] = A

        return A   # y_pred

    # ─────────────────────────────────────────────────────────
    def backward(self, y_true: np.ndarray) -> None:
        """
        Retropropagación. Guarda gradientes en self.grads.

        MSE = (1/N) * Σ (y_pred - y_true)²
        dL/dZ_out = (2/N) * (y_pred - y_true)   [última capa lineal → dA=dZ]
        """
        N = y_true.shape[0]
        y_pred = self.cache[f"A{self.L}"]

        dZ = (2 / N) * (y_pred - y_true)   # gradiente inicial, shape (N, 1)

        for l in reversed(range(1, self.L + 1)):
            A_prev = self.cache[f"A{l - 1}"]
            W = self.params[f"W{l}"]

            # Gradientes de pesos y bias de esta capa
            # (n_in, N) @ (N, n_out) → (n_in, n_out)
            self.grads[f"dW{l}"] = A_prev.T @ dZ
            self.grads[f"db{l}"] = dZ.sum(axis=0, keepdims=True)

            # Propagar gradiente hacia la capa anterior (no hace falta en l=1)
            if l > 1:
                dA_prev = dZ @ W.T                               # (N, n_in)
                dZ = dA_prev * relu_deriv(self.cache[f"Z{l-1}"])  # mask ReLU

    # ─────────────────────────────────────────────────────────
    def update(self) -> None:
        """SGD: W = W - lr * dW"""
        for l in range(1, self.L + 1):
            self.params[f"W{l}"] -= self.lr * self.grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.lr * self.grads[f"db{l}"]

    # ─────────────────────────────────────────────────────────
    def train(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 32,
        log_every: int = 100,
        y_std: float = 1.0,
        y_mean: float = 0.0,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> dict:
        """
        Loop de entrenamiento con mini-batch gradient descent.

        Parámetros
        ----------
        X, y_true     : datos de entrenamiento normalizados
        epochs        : número de epochs (pasadas completas por el dataset)
        batch_size    : tamaño de cada mini-batch
        log_every     : cada cuántas epochs imprimir métricas
        y_std, y_mean : para desnormalizar y reportar en USD
        X_val, y_val  : datos de validación normalizados (opcionales)

        Retorna
        -------
        dict con claves "train" y (si se pasa val) "val",
        cada una con la lista de MSE por epoch
        """
        N = X.shape[0]
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Shuffle: permutamos índices para que cada epoch
            # vea los mini-batches en orden distinto
            indices = np.random.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y_true[indices]

            epoch_losses = []

            # Recorremos el dataset en bloques de batch_size
            for start in range(0, N, batch_size):
                X_batch = X_shuffled[start: start + batch_size]
                y_batch = y_shuffled[start: start + batch_size]

                # Forward → loss → backward → update sobre este mini-batch
                y_pred_batch = self.forward(X_batch)
                batch_loss = np.mean((y_pred_batch - y_batch) ** 2)
                self.backward(y_batch)
                self.update()

                epoch_losses.append(batch_loss)

            # Pérdida de entrenamiento = promedio de todos los mini-batches
            epoch_loss = float(np.mean(epoch_losses))
            train_losses.append(epoch_loss)

            # Pérdida de validación: solo forward, sin backward ni update
            if X_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = float(np.mean((y_val_pred - y_val) ** 2))
                val_losses.append(val_loss)

            if epoch % log_every == 0:
                y_pred_all = self.forward(X)
                y_pred_usd = y_pred_all * y_std + y_mean
                y_true_usd = y_true * y_std + y_mean
                mae = np.mean(np.abs(y_pred_usd - y_true_usd))
                rmse = np.sqrt(np.mean((y_pred_usd - y_true_usd) ** 2))
                val_info = f"  Val MSE: {val_loss:.4f}" if X_val is not None else ""
                print(
                    f"Epoch {epoch:4d}  Train MSE: {epoch_loss:.4f}{val_info}  "
                    f"MAE: {mae:>12,.0f} USD  RMSE: {rmse:>12,.0f} USD")

        history = {"train": train_losses}
        if X_val is not None:
            history["val"] = val_losses
        return history

    # ─────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Solo forward, sin tocar cache de entrenamiento."""
        return self.forward(X)

    # ─────────────────────────────────────────────────────────
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_mean: float,
        y_std: float,
    ) -> dict:
        """
        Evalúa sobre el test set y devuelve MAE, RMSE en USD.

        X_test debe estar normalizado igual que en train.
        y_test está en escala original (USD).
        """
        y_pred_norm = self.predict(X_test)
        y_pred = y_pred_norm * y_std + y_mean
        mae = float(np.mean(np.abs(y_pred - y_test)))
        rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
        return {"mae": mae, "rmse": rmse, "y_pred": y_pred}
