import ctypes
import numpy as np
import tensorflow as tf

lib = ctypes.CDLL('./sensor_sim.so')
lib.simulate_sensor_data.restype = ctypes.POINTER(ctypes.c_float)
lib.simulate_sensor_data.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int]

# Model to building/training the simple anomaly (w/ unsupervised detection)
def build_anomaly_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train on normal data (simulate once then save model)
def train_model():
    normal_data = run_test_cycle(1000, anomaly_chance=0)  # No anomalies
    model = build_anomaly_model(1)  # Per-sample detection for simplicity
    model.fit(normal_data, normal_data, epochs=50, batch_size=32)
    model.save('anomaly_model.h5')
    return model
