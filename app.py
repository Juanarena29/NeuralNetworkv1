"""
Streamlit Demo — Neural Network vs Linear Regression
=====================================================
Dataset: housing.csv (hardcoded, m2 · bedrooms · floors → price)
Configurable hyperparameters: hidden layers, neurons/layer, lr, epochs
"""

import contextlib
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from linear_regression_baseline import LinearRegressionModel
from net import NeuralNet

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neural Net Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Neural Network vs Linear Regression")
st.caption("Dataset: housing.csv — predice precio de vivienda (USD)")

# ─────────────────────────────────────────────────────────────
# DATOS (hardcodeados, se cachean)
# ─────────────────────────────────────────────────────────────


@st.cache_data
def load_data():
    path_csv = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "housing.csv")
    df = pd.read_csv(path_csv)
    p99 = df["price"].quantile(0.99)
    df = df[df["price"] <= p99].reset_index(drop=True)
    X = df.drop(columns=["price"]).values
    y = df["price"].values.reshape(-1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=42)


X_train, X_test, y_train, y_test = load_data()

# ─────────────────────────────────────────────────────────────
# SIDEBAR — HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Hiperparámetros")

n_hidden = st.sidebar.slider(
    "Capas ocultas", min_value=1, max_value=10, value=2,
    help="Cantidad de capas ocultas (ReLU). La capa de salida es lineal.",
)

st.sidebar.markdown("**Neuronas por capa oculta:**")
neurons_per_layer = [
    st.sidebar.slider(
        f"Capa oculta {i + 1}",
        min_value=1,
        max_value=36,
        value=16,
        key=f"neurons_{i}",
    )
    for i in range(n_hidden)
]

lr = st.sidebar.select_slider(
    "Learning rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    value=0.01,
    format_func=lambda x: f"{x:.4f}".rstrip("0"),
)

epochs = st.sidebar.slider(
    "Épocas (iteraciones)", min_value=100, max_value=3000,
    value=800, step=100,
)

# Arquitectura derivada
INPUT_DIM = 3   # m2, bedrooms, floors — fijo
layer_dims = [INPUT_DIM] + neurons_per_layer + [1]

st.sidebar.markdown("---")
st.sidebar.markdown("**Arquitectura resultante:**")

arch_lines = []
arch_lines.append(f"Input  →  {INPUT_DIM}  (fijo)")
for i, d in enumerate(layer_dims[1:-1], start=1):
    arch_lines.append(f"Hidden {i}  →  {d}  [ReLU]")
arch_lines.append(f"Output  →  1  [lineal]")

st.sidebar.code("\n".join(arch_lines), language=None)

st.sidebar.markdown("---")
train_btn = st.sidebar.button(
    "🚀 Entrenar y Comparar", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────

if train_btn:
    # Normalización de features
    feature_pipeline = Pipeline([("scaler", StandardScaler())])
    X_train_prep = feature_pipeline.fit_transform(X_train)
    X_test_prep = feature_pipeline.transform(X_test)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    y_train_norm = (y_train - y_mean) / y_std

    # ── Neural Net ────────────────────────────────────────────
    np.random.seed(42)
    with st.spinner("Entrenando red neuronal…"):
        net = NeuralNet(layer_dims, lr=lr)
        # Redirigimos stdout para silenciar los prints del loop de entrenamiento
        with contextlib.redirect_stdout(io.StringIO()):
            history = net.train(
                X_train_prep, y_train_norm,
                epochs=epochs,
                log_every=epochs + 1,   # sin prints intermedios
                y_std=y_std,
                y_mean=y_mean,
            )
    nn_results = net.evaluate(X_test_prep, y_test, y_mean, y_std)

    # ── Linear Regression ────────────────────────────────────
    with st.spinner("Entrenando regresión lineal…"):
        lr_model = LinearRegressionModel()
        lr_model.train(X_train, y_train)
    lr_results = lr_model.evaluate(X_test, y_test)

    # Guardar en session_state
    st.session_state["history"] = history
    st.session_state["nn_results"] = nn_results
    st.session_state["lr_results"] = lr_results
    st.session_state["trained"] = True
    st.session_state["config"] = {
        "layer_dims": layer_dims,
        "lr": lr,
        "epochs": epochs,
    }

# ─────────────────────────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────────────────────────

if st.session_state.get("trained"):
    history = st.session_state["history"]
    nn_results = st.session_state["nn_results"]
    lr_results = st.session_state["lr_results"]
    config = st.session_state["config"]

    nn_mae = nn_results["mae"]
    nn_rmse = nn_results["rmse"]
    lr_mae = lr_results["mae"]
    lr_rmse = lr_results["rmse"]

    # ── Métricas ─────────────────────────────────────────────
    st.subheader("Métricas — Test Set")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Neural Net — MAE",
        f"${nn_mae:,.0f}",
    )
    col2.metric(
        "Neural Net — RMSE",
        f"${nn_rmse:,.0f}",
    )
    col3.metric(
        "Lin. Reg. — MAE",
        f"${lr_mae:,.0f}",
        delta=f"${lr_mae - nn_mae:+,.0f} vs NN",
        delta_color="inverse",
        help="Negativo = Lin. Reg. mejor que NN",
    )
    col4.metric(
        "Lin. Reg. — RMSE",
        f"${lr_rmse:,.0f}",
        delta=f"${lr_rmse - nn_rmse:+,.0f} vs NN",
        delta_color="inverse",
        help="Negativo = Lin. Reg. mejor que NN",
    )

    st.markdown("---")

    # ── Gráficos en dos columnas ──────────────────────────────
    col_loss, col_scatter = st.columns([1, 2])

    # Loss curve
    with col_loss:
        st.subheader("Curva de Pérdida")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.plot(history["train"], color="#1f77b4",
                 linewidth=2, label="Train MSE")
        ax1.set_xlabel("Época")
        ax1.set_ylabel("MSE (norm.)")
        ax1.set_title(f"lr={config['lr']}  ·  {config['epochs']} épocas")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # Scatter Real vs Predicho
    with col_scatter:
        st.subheader("Real vs Predicho")
        y_test_flat = y_test.flatten()
        min_val = y_test_flat.min()
        max_val = y_test_flat.max()

        fig2, (ax_nn, ax_lr) = plt.subplots(1, 2, figsize=(10, 4))
        fig2.suptitle("Real vs Predicho — Test Set", fontsize=12)

        for ax, results, title, color in zip(
            [ax_nn, ax_lr],
            [nn_results, lr_results],
            ["Neural Net (numpy)", "Linear Regression (sklearn)"],
            ["#1f77b4", "#ff7f0e"],
        ):
            ax.scatter(
                y_test_flat,
                results["y_pred"].flatten(),
                alpha=0.65,
                edgecolors="k",
                linewidths=0.3,
                color=color,
            )
            ax.plot([min_val, max_val], [min_val, max_val], "r--",
                    linewidth=1.5, label="Predicción perfecta")
            ax.set_xlabel("Precio real (USD)")
            ax.set_ylabel("Precio predicho (USD)")
            ax.set_title(
                f"{title}\nMAE: ${results['mae']:,.0f}  ·  RMSE: ${results['rmse']:,.0f}"
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

else:
    st.info("⬅️  Configurá los hiperparámetros en el panel izquierdo y presioná **Entrenar y Comparar**.")
