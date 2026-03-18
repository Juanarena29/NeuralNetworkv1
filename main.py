from linear_regression_baseline import LinearRegressionModel
from net import NeuralNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# ─────────────────────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────────────────────

path_csv = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "housing.csv")

if not os.path.exists(path_csv):
    raise FileNotFoundError(f"Archivo no encontrado: {path_csv}")

df = pd.read_csv(path_csv)

# Filtrar outliers 
p99 = df["price"].quantile(0.99)
df = df[df["price"] <= p99].reset_index(drop=True)

X = df.drop(columns=["price"]).values
y = df["price"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────────────────────
# NORMALIZACIÓN — solo para la red neuronal
# ─────────────────────────────────────────────────────────────

feature_pipeline = Pipeline([("scaler", StandardScaler())])
X_train_prep = feature_pipeline.fit_transform(X_train)
X_test_prep = feature_pipeline.transform(X_test)

y_mean = float(y_train.mean())
y_std = float(y_train.std())
y_train_norm = (y_train - y_mean) / y_std

# ─────────────────────────────────────────────────────────────
# ENTRENAR MODELOS
# ─────────────────────────────────────────────────────────────

np.random.seed(42)
net = NeuralNet([3, 16, 8, 1], lr=0.01)
history = net.train(X_train_prep, y_train_norm, epochs=800,
                    log_every=200, y_std=y_std, y_mean=y_mean)

lr_model = LinearRegressionModel()
lr_model.train(X_train, y_train)

# ─────────────────────────────────────────────────────────────
# EVALUAR SOBRE TEST SET
# ─────────────────────────────────────────────────────────────

nn_results = net.evaluate(X_test_prep, y_test, y_mean, y_std)
lr_results = lr_model.evaluate(X_test, y_test)

# ─────────────────────────────────────────────────────────────
# COMPARACIÓN
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 48)
print(f"{'Modelo':<20} {'MAE':>12} {'RMSE':>12}")
print("=" * 48)
print(
    f"{'Neural Net':<20} {nn_results['mae']:>12,.0f} {nn_results['rmse']:>12,.0f}")
print(
    f"{'Linear Regression':<20} {lr_results['mae']:>12,.0f} {lr_results['rmse']:>12,.0f}")
print("=" * 48)

# ─────────────────────────────────────────────────────────────
# CURVA DE PÉRDIDA
# ─────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 5))
plt.plot(history["train"], label="Train Loss (MSE)", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Curva de Pérdida — Neural Net")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("loss_curve.png", dpi=150)
plt.show()
print("\nCurva de pérdida guardada en loss_curve.png")

# ─────────────────────────────────────────────────────────────
# GRÁFICO
# ─────────────────────────────────────────────────────────────

y_test_flat = y_test.flatten()
min_val = y_test_flat.min()
max_val = y_test_flat.max()
perfect = [min_val, max_val]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Real vs Predicho — Test Set", fontsize=13)

for ax, results, title in zip(
    axes,
    [nn_results, lr_results],
    ["Neural Net (numpy)", "Linear Regression (sklearn)"],
):
    ax.scatter(y_test_flat, results["y_pred"].flatten(),
               alpha=0.75, edgecolors="k", linewidths=0.4)
    ax.plot(perfect, perfect, "r--", linewidth=1.5)
    ax.set_xlabel("Precio real (USD)")
    ax.set_ylabel("Precio predicho (USD)")
    ax.set_title(
        f"{title}\nMAE: ${results['mae']:,.0f}  -  RMSE: ${results['rmse']:,.0f}")
    ax.legend()

plt.tight_layout()
plt.savefig("comparacion.png", dpi=150)
plt.show()
print("\nGráfico guardado en comparacion.png")
