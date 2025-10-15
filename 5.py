import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import load_model

VARIANT = 1            
IIN_LAST2 = 42           # последние 2 цифры ИИН (пример). TARGET = (IIN_LAST2 % 7) + 1
N_SAMPLES = 5000
LATENT_DIM = 3
EPOCHS = 80
BATCH_SIZE = 64
VAL_SPLIT = 0.15
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

TARGET = (IIN_LAST2 % 7) + 1   # от 1 до 7
tag = f"v{VARIANT}_t{TARGET}"

outdir = Path(".")
STD = np.sqrt(10.0)

if VARIANT == 1:
    X = np.random.normal(loc=3.0, scale=STD, size=(N_SAMPLES, 1))
    e = np.random.normal(loc=0.0, scale=np.sqrt(0.3), size=(N_SAMPLES, 7))
    f1 = (X**2)[:,0]
    f2 = np.sin(X[:,0]/2.0)
    f3 = np.cos(2.0*X[:,0])
    f4 = X[:,0] - 3.0
    f5 = -X[:,0]
    f6 = np.abs(X[:,0])
    f7 = (X[:,0]**3)/4.0
    F = np.stack([f1,f2,f3,f4,f5,f6,f7], axis=1) + e

elif VARIANT == 2:
    X = np.random.normal(loc=-5.0, scale=STD, size=(N_SAMPLES, 1))
    e = np.random.normal(loc=0.0, scale=np.sqrt(0.3), size=(N_SAMPLES, 7))
    f1 = - (X[:,0]**3)
    # лог безопасно: ln(|X|), где |X| > 0; при X=0 заменим на очень маленькое значение
    absX = np.abs(X[:,0])
    absX_safe = np.where(absX>0, absX, 1e-8)
    f2 = np.log(absX_safe)
    f3 = np.sin(3.0*X[:,0])
    f4 = np.exp(X[:,0])  # может быть очень маленьким (т.к. X ~ N(-5,10))
    f5 = X[:,0] + 4.0
    f6 = -X[:,0] + np.sqrt(absX)
    f7 = X[:,0]
    F = np.stack([f1,f2,f3,f4,f5,f6,f7], axis=1) + e

else:
    raise ValueError("VARIANT must be 1 or 2.")

cols = [f"f{i}" for i in range(1,8)]
df_all = pd.DataFrame(F, columns=cols)
df_all.to_csv(outdir / f"pr_{tag}_data.csv", index=False)

t_idx = TARGET - 1
y_all = F[:, t_idx:t_idx+1]                # (N,1)
X_all = np.delete(F, t_idx, axis=1)        # (N,6)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, F_train, F_test = train_test_split(
    X_all, y_all, F, test_size=0.2, random_state=RANDOM_SEED
)

pd.DataFrame(X_train, columns=[c for i,c in enumerate(cols) if i!=t_idx]).to_csv(outdir/f"pr_{tag}_train_inputs.csv", index=False)
pd.DataFrame(X_test,  columns=[c for i,c in enumerate(cols) if i!=t_idx]).to_csv(outdir/f"pr_{tag}_test_inputs.csv",  index=False)

inp = Input(shape=(6,), name="inputs")

x = layers.Dense(16, activation="relu")(inp)
x = layers.Dense(12, activation="relu")(x)
z = layers.Dense(LATENT_DIM, activation="linear", name="latent")(x)

d = layers.Dense(12, activation="relu")(z)
d = layers.Dense(16, activation="relu")(d)
recon = layers.Dense(7, activation="linear", name="reconstruction")(d)

r = layers.Dense(8, activation="relu")(z)
out = layers.Dense(1, activation="linear", name="regression")(r)

full_model = Model(inp, outputs=[recon, out], name="autoencoder_regression")

full_model.compile(
    optimizer="adam",
    loss={
        "reconstruction": "mse",
        "regression": "mse"
    },
    loss_weights={"reconstruction": 1.0, "regression": 1.0},
    metrics={"reconstruction": ["mse"], "regression": ["mse"]}
)

hist = full_model.fit(
    X_train, {"reconstruction": F_train, "regression": y_train},
    validation_split=VAL_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

encoder = Model(inp, z, name="encoder")
encoder.save(outdir / f"pr_{tag}_encoder.h5")

z_inp = Input(shape=(LATENT_DIM,), name="z_inp")

# Создание декодера - более простой и надежный способ
# Найдем все Dense слои между latent и reconstruction
latent_idx = None
recon_idx = None
for i, layer in enumerate(full_model.layers):
    if layer.name == "latent":
        latent_idx = i
    elif layer.name == "reconstruction":
        recon_idx = i

# Получаем Dense слои декодера (между latent и reconstruction)
decoder_dense_layers = []
for i in range(latent_idx + 1, recon_idx):
    layer = full_model.layers[i]
    if isinstance(layer, layers.Dense):
        decoder_dense_layers.append(layer)

# Строим декодер
d = z_inp
for layer in decoder_dense_layers:
    d = layers.Dense(layer.units, activation=layer.activation)(d)
recon_out = layers.Dense(7, activation="linear", name="reconstruction")(d)
decoder = Model(z_inp, recon_out, name="decoder")
decoder.save(outdir / f"pr_{tag}_decoder.h5")

# 3) REGRESSOR: inputs -> regression
reg_out_layer = full_model.get_layer("regression")
#  Dense-слои регрессора — это два Dense после latent перед "regression"
reg_dense_layers = []
start_collect = False
for lyr in full_model.layers:
    if lyr.name == "latent":
        start_collect = True
        continue
    if start_collect and isinstance(lyr, layers.Dense):
        reg_dense_layers.append(lyr)
    if lyr.name == "regression":
        break

#  регрессор: inp -> (encoder) -> reg_dense_layers -> regression
reg_inp = Input(shape=(6,), name="reg_inp")
z_from_inp = encoder(reg_inp)                 # тот же encoder
r = z_from_inp
for layer in reg_dense_layers:
    r = layers.Dense(layer.units, activation=layer.activation)(r)
r_out = layers.Dense(1, activation="linear", name="regression")(r)
regressor = Model(reg_inp, r_out, name="regressor")
regressor.save(outdir / f"pr_{tag}_regressor.h5")

Z_train = encoder.predict(X_train, verbose=0)
Z_test  = encoder.predict(X_test,  verbose=0)
pd.DataFrame(Z_train, columns=[f"z{i}" for i in range(1, LATENT_DIM+1)]).to_csv(outdir/f"pr_{tag}_train_encoded.csv", index=False)
pd.DataFrame(Z_test,  columns=[f"z{i}" for i in range(1, LATENT_DIM+1)]).to_csv(outdir/f"pr_{tag}_test_encoded.csv",  index=False)

F_train_dec = decoder.predict(Z_train, verbose=0)
F_test_dec  = decoder.predict(Z_test,  verbose=0)
pd.DataFrame(F_train_dec, columns=cols).to_csv(outdir/f"pr_{tag}_train_decoded.csv", index=False)
pd.DataFrame(F_test_dec,  columns=cols).to_csv(outdir/f"pr_{tag}_test_decoded.csv",  index=False)

y_pred_train = regressor.predict(X_train, verbose=0)
y_pred_test  = regressor.predict(X_test,  verbose=0)

pd.DataFrame(np.hstack([y_train, y_pred_train]), columns=["y_true","y_pred"]).to_csv(outdir/f"pr_{tag}_regression_train.csv", index=False)
pd.DataFrame(np.hstack([y_test,  y_pred_test ]), columns=["y_true","y_pred"]).to_csv(outdir/f"pr_{tag}_regression_test.csv",  index=False)

print(f"\nГотово. Вариант={VARIANT}, Цель(TARGET)={TARGET} -> префикс pr_{tag}_*. Файлы сохранены в {outdir.resolve()}.")
