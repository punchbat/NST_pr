import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

from gen_image import gen_data

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

N_SAMPLES = 2000
IMG_SIZE = 50

data, labels = gen_data(size=N_SAMPLES, img_size=IMG_SIZE)

labels = labels.reshape(-1)  # (N,)
y = (labels == 'Circle').astype(np.int32)  # бинарные метки

X = data.astype("float32")
X = (X - X.min()) / (X.max() - X.min() + 1e-8)
X = np.expand_dims(X, axis=-1)  # (N, 50, 50, 1)

idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
)

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # бинарная классификация
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy')
]

H = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

probs = model.predict(X_test[:10])
preds = (probs.ravel() >= 0.5).astype(int)
true = y_test[:10]
print("\nFirst 10 predictions (pred/true/prob):")
for i, (p, t, pr) in enumerate(zip(preds, true, probs.ravel())):
    print(f"{i:02d}: pred={p}  true={t}  prob_circle={pr:.3f}")