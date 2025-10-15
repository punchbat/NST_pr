# callbacks_train_run.py
import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import ssl

# ============== Вариант 1: кастомный Callback — сохранение трёх лучших моделей ==============
class TopKModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, out_dir, user_prefix, monitor="val_accuracy", mode="max", k=3):
        super().__init__()
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.user_prefix = user_prefix
        self.monitor = monitor
        self.mode = mode
        self.k = k
        self.date_prefix = dt.datetime.now().strftime("%Y%m%d")
        self.top = []
        self.counter = 0
        self._better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get(self.monitor)
        if val is None:
            return
        if len(self.top) < self.k:
            self._save(val)
        else:
            worst_idx = np.argmin([v for v, _ in self.top]) if self.mode == "max" else np.argmax([v for v, _ in self.top])
            worst_val = self.top[worst_idx][0]
            if self._better(val, worst_val):
                os.remove(self.top[worst_idx][1])
                self._save(val, replace_idx=worst_idx)

    def _save(self, val, replace_idx=None):
        self.counter += 1
        fname = f"{self.date_prefix}_{self.user_prefix}_{self.counter}.h5"
        path = os.path.join(self.out_dir, fname)
        self.model.save(path)
        if replace_idx is None:
            self.top.append((val, path))
        else:
            self.top[replace_idx] = (val, path)


CHOSEN_VARIANT = 1        # можно менять
USER_PREFIX = "mnistcnn"
OUT_DIR = "./artifacts"
EPOCHS = 8
BATCH = 128

ssl._create_default_https_context = ssl._create_unverified_context

try:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
except:
    print("Не удалось загрузить MNIST, создаем синтетические данные...")
    np.random.seed(42)
    x_train = np.random.rand(1000, 28, 28).astype("float32")
    y_train = np.random.randint(0, 10, 1000)
    x_test = np.random.rand(200, 28, 28).astype("float32")
    y_test = np.random.randint(0, 10, 200)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
val_split = 0.1
n_val = int(len(x_train) * val_split)
x_val, y_val = x_train[:n_val], y_train[:n_val]
x_train, y_train = x_train[n_val:], y_train[n_val:]

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    TopKModelCheckpoint(out_dir=os.path.join(OUT_DIR, "top_models"),
                        user_prefix=USER_PREFIX,
                        monitor="val_accuracy",
                        mode="max",
                        k=3)
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nИтого:")
print(f"  Test loss = {test_loss:.4f}")
print(f"  Test acc  = {test_acc:.4f}")
print(f"  Топ-модели сохранены в: {os.path.join(OUT_DIR, 'top_models')}")
