import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

VARIANT = 1

np.random.seed(42)
tf.random.set_seed(42)

X_bool = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1],
], dtype=np.float32)

def target_variant1(x):
    a, b, c = x[...,0], x[...,1], x[...,2]
    # (a and b) or (a and c)
    return ((a==1) & (b==1) | (a==1) & (c==1)).astype(np.float32)[...,None]

def target_variant2(x):
    a, b, c = x[...,0], x[...,1], x[...,2]
    # (a or b) xor not(b and c)
    left  = ((a==1) | (b==1))
    right = ~((b==1) & (c==1))
    y = (left ^ right).astype(np.float32)
    return y[...,None]

y = target_variant1(X_bool) if VARIANT == 1 else target_variant2(X_bool)

model = models.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def extract_weights(model):
    wb = []
    for layer in model.layers:
        if hasattr(layer, "get_weights"):
            weights = layer.get_weights()
            if len(weights) == 2:
                W, b = weights
                wb.append((W.copy(), b.copy()))
    return wb

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0.0)

# 1
def simulate_elementwise(X, weights):
    A = X.copy()
    for idx, (W, b) in enumerate(weights):
        N, in_dim = A.shape
        out_dim = W.shape[1]
        Z = np.zeros((N, out_dim), dtype=np.float32)

        for j in range(out_dim):
            col = np.zeros((N,), dtype=np.float32)
            for k in range(in_dim):
                col = col + (A[:, k] * W[k, j])  
            Z[:, j] = col + b[j]

        if idx < len(weights) - 1:
            A = relu(Z)
        else:
            A = sigmoid(Z)
    return A

# 2
def simulate_numpy(X, weights):
    """
    X: (N, in_dim)
    weights: [(W1,b1), (W2,b2), ...]
    Разрешено: @, np.dot, broadcasting, суммирования по осям и т.п.
    """
    A = X.copy()
    for idx, (W, b) in enumerate(weights):
        Z = A @ W + b  # (N, out_dim)
        if idx < len(weights) - 1:
            A = relu(Z)
        else:
            A = sigmoid(Z)
    return A

weights_init = extract_weights(model)
keras_pred_init = model.predict(X_bool, verbose=0)
elem_pred_init  = simulate_elementwise(X_bool, weights_init)
np_pred_init    = simulate_numpy(X_bool, weights_init)

print("=== ДО ОБУЧЕНИЯ ===")
print("max|keras - elementwise| =", np.max(np.abs(keras_pred_init - elem_pred_init)))
print("max|keras - numpy     | =", np.max(np.abs(keras_pred_init - np_pred_init)))

def show_table(title, arr):
    print(f"\n{title}")
    for i, row in enumerate(arr):
        print(f"{i}: {row.ravel()}")

show_table("Keras (init)", keras_pred_init)
show_table("Elem  (init)", elem_pred_init)
show_table("NumPy  (init)", np_pred_init)

H = model.fit(X_bool, y, epochs=600, batch_size=8, verbose=0)

weights_trained = extract_weights(model)
keras_pred_tr = model.predict(X_bool, verbose=0)
elem_pred_tr  = simulate_elementwise(X_bool, weights_trained)
np_pred_tr    = simulate_numpy(X_bool, weights_trained)

print("\n=== ПОСЛЕ ОБУЧЕНИЯ ===")
print("max|keras - elementwise| =", np.max(np.abs(keras_pred_tr - elem_pred_tr)))
print("max|keras - numpy     | =", np.max(np.abs(keras_pred_tr - np_pred_tr)))

show_table("Keras (trained)", keras_pred_tr)
show_table("Elem  (trained)", elem_pred_tr)
show_table("NumPy  (trained)", np_pred_tr)

bin_keras = (keras_pred_tr >= 0.5).astype(int)
print("\nTargets:", y.ravel().astype(int))
print("Keras  :", bin_keras.ravel())
