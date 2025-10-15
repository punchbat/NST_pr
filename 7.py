import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

def gen_sequence(seq_len=1000):
    seq = [math.sin(i/5)/2 + math.cos(i/3)/2 + random.normalvariate(0, 0.04)
           for i in range(seq_len)]
    return np.array(seq, dtype=np.float32)

def gen_data_from_sequence(seq, lookback, horizon=1):
    X, y = [], []
    for i in range(len(seq) - lookback - horizon + 1):
        X.append(seq[i:i+lookback])
        y.append(seq[i+lookback:i+lookback+horizon])
    return np.array(X)[..., None], np.array(y)

def chronological_split(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def build_model(lookback, horizon=1):
    inp = layers.Input(shape=(lookback, 1))
    x = layers.GRU(64, return_sequences=True)(inp)
    x = layers.LSTM(64)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(horizon, activation="linear")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def run_rnn_forecast():
    start_time = time.time()
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    SEQ_LEN = 2000
    LOOKBACK = 40
    HORIZON = 1
    EPOCHS = 30
    BATCH = 64

    seq = gen_sequence(SEQ_LEN)
    mean, std = seq.mean(), seq.std() + 1e-8
    seq_norm = (seq - mean) / std

    X, y = gen_data_from_sequence(seq_norm, LOOKBACK, HORIZON)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = chronological_split(X, y)

    model = build_model(LOOKBACK, HORIZON)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1,
        callbacks=[es]
    )

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)

    y_test_denorm = y_test * std + mean
    y_pred_denorm = y_pred * std + mean

    plt.figure(figsize=(10, 4))
    plt.plot(y_test_denorm.ravel(), label="Истинное значение")
    plt.plot(y_pred_denorm.ravel(), label="Прогноз модели")
    plt.title("RNN прогноз на тестовой выборке (Вариант 1)")
    plt.xlabel("Временной шаг")
    plt.ylabel("Значение сигнала")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_var1_rnn_forecast.png", dpi=150)
    plt.show()

    elapsed = time.time() - start_time
    print(f"\nИтого:")
    print(f"  Количество точек: {len(seq)}")
    print(f"  Тестовая MSE = {test_loss:.6f}")
    print(f"  Тестовая MAE = {test_mae:.6f}")
    print(f"  Время обучения: {elapsed:.2f} секунд")

run_rnn_forecast()