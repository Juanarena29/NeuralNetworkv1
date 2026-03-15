# Red Neuronal desde cero con NumPy — vs scikit-learn

Implementación de una red neuronal fully-connected en **NumPy puro**: forward pass, backpropagation y mini-batch gradient descent derivados a mano, sin ninguna librería de deep learning. El modelo incluye características acordes a la magnitud del proyecto — inicialización He, normalización sin data leakage, mini-batches con shuffle, y monitoreo de validación — y se benchmarkea directamente contra la Regresión Lineal de scikit-learn, alcanzando una precisión equivalente a una implementación optimizada y battle-tested.

---

## Estructura del proyecto

```
├── net.py                      # Red neuronal implementada desde cero
├── linear_regression_baseline.py  # Baseline con sklearn
├── main.py                     # Script principal: datos, entrenamiento, evaluación
├── loss_curve.png              # Curva de pérdida train vs validación
└── comparacion.png             # Scatter real vs predicho (NN vs LR)
```
---

## Instalación y uso

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

# Instalar dependencias
pip install numpy pandas scikit-learn matplotlib

# Ejecutar
python main.py
```

El script entrena la red neuronal y la regresión lineal, imprime las métricas en consola, y guarda `loss_curve.png` y `comparacion.png` en el directorio actual.

---

## La red neuronal — `net.py`

### Arquitectura configurable

La red se define con una lista de dimensiones `layer_dims`, que especifica el tamaño de cada capa incluyendo la entrada y la salida. Por ejemplo, `[3, 16, 8, 1]` genera la siguiente arquitectura:

```
Input (3) → Hidden (16) → Hidden (8) → Output (1)
```

Esto permite cambiar la profundidad y el ancho de la red sin modificar nada más. Las capas ocultas usan **ReLU** como función de activación, y la capa de salida es **lineal** (sin activación), lo cual es apropiado para regresión.

![Arquitectura de la red](architecture.png)

---

### Referencia matemática — cálculos matriciales

Para una red con dos capas ocultas y un batch de **N** ejemplos, los tensores tienen las siguientes dimensiones:

| Símbolo | Descripción | Shape |
|---|---|---|
| `X` | Input | `(N, 3)` |
| `W1`, `b1` | Pesos y bias capa 1 | `(3, 16)`, `(1, 16)` |
| `W2`, `b2` | Pesos y bias capa 2 | `(16, 8)`, `(1, 8)` |
| `W3`, `b3` | Pesos y bias capa salida | `(8, 1)`, `(1, 1)` |
| `Ŷ` | Predicción | `(N, 1)` |

#### Forward Pass

$$Z^{[1]} = X W^{[1]} + b^{[1]} \qquad (N,3)\cdot(3,16) \rightarrow (N,16)$$

$$A^{[1]} = \text{ReLU}(Z^{[1]}) \qquad (N,16)$$

$$Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]} \qquad (N,16)\cdot(16,8) \rightarrow (N,8)$$

$$A^{[2]} = \text{ReLU}(Z^{[2]}) \qquad (N,8)$$

$$Z^{[3]} = A^{[2]} W^{[3]} + b^{[3]} \qquad (N,8)\cdot(8,1) \rightarrow (N,1)$$

$$\hat{Y} = Z^{[3]} \qquad \text{(salida lineal)}$$

#### Loss

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

#### Backward Pass

**Gradiente inicial** (desde la pérdida hacia la última capa):

$$dZ^{[3]} = \frac{2}{N}(\hat{Y} - Y) \qquad (N,1)$$

**Capa 3 → 2:**

$$dW^{[3]} = (A^{[2]})^T \cdot dZ^{[3]} \qquad (8,N)\cdot(N,1) \rightarrow (8,1)$$

$$db^{[3]} = \sum_{\text{batch}} dZ^{[3]} \qquad (1,1)$$

$$dA^{[2]} = dZ^{[3]} \cdot (W^{[3]})^T \qquad (N,1)\cdot(1,8) \rightarrow (N,8)$$

$$dZ^{[2]} = dA^{[2]} \odot \mathbb{1}[Z^{[2]} > 0] \qquad (N,8)$$

**Capa 2 → 1:**

$$dW^{[2]} = (A^{[1]})^T \cdot dZ^{[2]} \qquad (16,N)\cdot(N,8) \rightarrow (16,8)$$

$$db^{[2]} = \sum_{\text{batch}} dZ^{[2]} \qquad (1,8)$$

$$dA^{[1]} = dZ^{[2]} \cdot (W^{[2]})^T \qquad (N,8)\cdot(8,16) \rightarrow (N,16)$$

$$dZ^{[1]} = dA^{[1]} \odot \mathbb{1}[Z^{[1]} > 0] \qquad (N,16)$$

**Capa 1:**

$$dW^{[1]} = X^T \cdot dZ^{[1]} \qquad (3,N)\cdot(N,16) \rightarrow (3,16)$$

$$db^{[1]} = \sum_{\text{batch}} dZ^{[1]} \qquad (1,16)$$

#### Actualización de pesos (SGD)

$$W^{[l]} \leftarrow W^{[l]} - \alpha \cdot dW^{[l]}$$

$$b^{[l]} \leftarrow b^{[l]} - \alpha \cdot db^{[l]}$$

donde $\alpha$ es el learning rate. El símbolo $\odot$ denota multiplicación elemento a elemento (Hadamard), y $\mathbb{1}[Z > 0]$ es la derivada de ReLU — una máscara binaria que vale 1 donde la pre-activación fue positiva y 0 donde fue ≤ 0.


### Inicialización de pesos — He Initialization

```python
self.params[f"W{l}"] = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
self.params[f"b{l}"] = np.zeros((1, n_out))
```

Los pesos se inicializan con **He initialization** (también llamada Kaiming initialization), escalando por `√(2 / n_in)`. Esta técnica está diseñada específicamente para redes con ReLU: evita que los gradientes se desvanezcan o exploten en las primeras iteraciones, y es considerada la inicialización de facto para este tipo de activación.

Los biases se inicializan en cero, lo cual es una práctica estándar.

---

### Forward Pass

```python
Z = A @ W + b        # Pre-activación: (N, n_in) @ (n_in, n_out) → (N, n_out)
A = relu(Z)          # Activación (excepto en la última capa)
```

En cada capa `l`, se computa la **pre-activación** `Z` y luego se aplica ReLU. La última capa devuelve `Z` directamente (lineal). Todos los valores de `Z` y `A` se guardan en `self.cache` porque el backward pass los necesitará para calcular los gradientes.

---

### Backward Pass — Backpropagation

Este es el corazón del entrenamiento. Se calcula el gradiente de la función de pérdida respecto a cada parámetro de la red, recorriendo las capas en sentido inverso.

**Función de pérdida:** MSE (Mean Squared Error)

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

**Gradiente inicial** (en la salida lineal, `dA = dZ`):

```python
dZ = (2 / N) * (y_pred - y_true)   # shape: (N, 1)
```

**Por cada capa (de la última hacia la primera):**

```python
# Gradiente de los pesos
dW = A_prev.T @ dZ           # (n_in, N) @ (N, n_out) → (n_in, n_out)

# Gradiente del bias — se suma sobre el batch
db = dZ.sum(axis=0, keepdims=True)

# Propagar el gradiente hacia la capa anterior
dA_prev = dZ @ W.T                              # (N, n_in)
dZ      = dA_prev * relu_deriv(Z[l-1])          # máscara ReLU
```

El gradiente del bias se suma sobre el eje del batch (`axis=0`) porque el bias `b` es **compartido** por todos los ejemplos del batch en el forward pass: cada ejemplo contribuye independientemente al gradiente, y esas contribuciones se acumulan por suma.

La derivada de ReLU actúa como una **máscara binaria**: deja pasar el gradiente donde la pre-activación fue positiva, y lo bloquea donde fue ≤ 0.

---

### Mini-Batch Gradient Descent

```python
for start in range(0, N, batch_size):
    X_batch = X_shuffled[start : start + batch_size]
    y_batch = y_shuffled[start : start + batch_size]

    y_pred_batch = self.forward(X_batch)
    self.backward(y_batch)
    self.update()
```

En lugar de calcular el gradiente sobre todo el dataset (batch gradient descent) o sobre un solo ejemplo (SGD puro), se implementa **mini-batch gradient descent**: el dataset se divide en bloques de tamaño `batch_size`, y se hace un paso de optimización por bloque.

> **¿Por qué mini-batches si el dataset es pequeño?**
>
> En este proyecto concreto, con pocos cientos de ejemplos, los mini-batches no producen ninguna ventaja observable respecto al batch completo. Sin embargo, se implementan deliberadamente para demostrar su funcionamiento, ya que en redes neuronales de mayor escala son **indispensables**:
>
> - Permiten entrenar con datasets que no caben en memoria RAM/VRAM.
> - Introducen ruido estocástico en el gradiente que ayuda a escapar de mínimos locales.
> - Aprovechan la paralelización de las GPUs de forma mucho más eficiente que un ejemplo a la vez.

Además, en cada epoch el dataset se **permuta aleatoriamente** antes de armar los batches, lo que garantiza que la red no vea siempre los mismos mini-batches en el mismo orden.

---

### Validation Set

Durante el entrenamiento, en cada epoch se evalúa la pérdida sobre un **conjunto de validación separado** que la red nunca usa para actualizar sus pesos:

```python
if X_val is not None:
    y_val_pred = self.forward(X_val)
    val_loss = float(np.mean((y_val_pred - y_val) ** 2))
```

Esto sirve para monitorear si la red está **generalizando** o está memorizando el set de entrenamiento (overfitting). Si la pérdida de entrenamiento sigue bajando pero la de validación comienza a subir, es una señal clara de overfitting.

En este proyecto se usó un split **70% / 15% / 15%** (train / val / test), donde las estadísticas de normalización se calculan exclusivamente sobre el set de entrenamiento y se aplican a val y test, evitando cualquier filtración de información.

---

## Curva de pérdida

![Loss curve](loss_curve.png)

La curva muestra que la red converge rápidamente en los primeros ~100 epochs, pasando de un MSE normalizado de ~1.45 a valores cercanos a 0.03. A partir de allí, el descenso se vuelve mucho más gradual.

Lo más relevante es que la curva de validación **sigue de cerca a la de entrenamiento** durante toda la ejecución, sin separarse. Esto indica que la red está generalizando bien y no muestra señales de overfitting.

---

## Comparación contra scikit-learn

![Comparación](comparacion.png)

| Modelo | MAE | RMSE |
|---|---|---|
| **Neural Net (NumPy — desde cero)** | **9,022** | **11,097** |
| Linear Regression (sklearn) | 8,963 | 10,889 |

El dataset utilizado es un conjunto básico de house pricing con tres features: superficie (m²), número de habitaciones y número de pisos. La simplicidad del dataset es intencional — con una relación esencialmente lineal entre features y target, la regresión lineal representa un baseline difícil de superar, lo que hace la comparación más exigente para la red neuronal.

La diferencia es de **~$59 en MAE** y **~$208 en RMSE**. Prácticamente cero.

Esto significa que una red neuronal implementada a mano, en NumPy puro, sin ningún framework de deep learning, **iguala en precisión a la implementación optimizada de scikit-learn**. La red derivada a mano produce el mismo resultado que décadas de ingeniería de software aplicada a regresión lineal.

Los scatter plots lo confirman visualmente: en ambos casos los puntos caen sobre la diagonal con la misma dispersión, sin ninguna diferencia sistemática entre los dos modelos.

> **¿Por qué usar una NN si la regresión lineal da lo mismo?**
> Porque el punto de este proyecto no es maximizar métricas — es demostrar que la implementación es correcta. Si los resultados fueran radicalmente distintos, podría indicar un bug. Que ambos modelos converjan a la misma solución en un problema con relación lineal es exactamente lo que se espera de una implementación bien hecha.

---

## Conclusiones

La red neuronal implementada desde cero, con toda su matemática explícita, **alcanza la misma performance que scikit-learn** sobre el mismo dataset y el mismo split. Eso valida que cada componente funciona correctamente:

- El **forward pass** propaga las activaciones sin errores de dimensiones ni de escala.
- La **backpropagation** calcula los gradientes exactos — cualquier bug aquí se traduciría en no convergencia o en resultados peores que el baseline.
- El **mini-batch gradient descent** optimiza correctamente, con shuffle por epoch.
- La **normalización** está aplicada sin data leakage: las estadísticas de train nunca ven val ni test.
- La **curva de pérdida** muestra convergencia limpia y sin overfitting, con val siguiendo a train durante todo el entrenamiento.

El resultado no es una aproximación ni algo "cercano" a sklearn — es estadísticamente equivalente. Eso es lo que se busca demostrar.
