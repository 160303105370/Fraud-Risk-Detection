from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_autoencoder(input_dim: int, bottleneck: int = 8) -> keras.Model:
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(32, activation="relu")(x)
    # bottleneck
    z = layers.Dense(bottleneck, activation="relu", name="bottleneck")(x)
    # decoder
    x = layers.Dense(32, activation="relu")(z)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(input_dim, activation=None)(x)  # linear for reconstruction
    model = keras.Model(inp, out, name="tabular_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

def train_autoencoder(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    input_dim: int,
    bottleneck: int = 8,
    epochs: int = 50,
    batch_size: int = 256,
) -> Tuple[keras.Model, dict]:
    model = build_autoencoder(input_dim=input_dim, bottleneck=bottleneck)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    hist = model.fit(
        X_train, X_train,
        validation_data=(X_valid, X_valid),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )
    return model, {k: [float(v) for v in hist.history[k]] for k in hist.history}

def reconstruction_error(model: keras.Model, X: np.ndarray) -> np.ndarray:
    X_hat = model.predict(X, verbose=0)
    # MSE per row
    err = np.mean((X - X_hat) ** 2, axis=1)
    return err
