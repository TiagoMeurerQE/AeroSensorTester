import ctypes
import numpy as np
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

# Load C lib
lib = ctypes.CDLL('./sensor_sim.so')
lib.simulate_sensor_data.restype = ctypes.POINTER(ctypes.c_float)
lib.simulate_sensor_data.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int]

# Build/train simple anomaly model (autoencoder for unsupervised detection)
def build_anomaly_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train on normal data (simulate once, save model)
def train_model():
    normal_data = run_test_cycle(1000, anomaly_chance=0)
    # Normalize data
    mean = normal_data.mean()
    std = normal_data.std()
    normal_data = (normal_data - mean) / std if std != 0 else normal_data
    model = build_anomaly_model(1)
    model.fit(normal_data, normal_data, epochs=50, batch_size=32)
    model.save('anomaly_model.h5')
    return model, mean, std  # Return mean and std for consistent normalization

# Run simulation cycle
def run_test_cycle(num_samples=100, base_value=0.0, noise_amp=1.0, anomaly_chance=5):
    data_ptr = lib.simulate_sensor_data(num_samples, base_value, noise_amp, anomaly_chance)
    raw_data = np.ctypeslib.as_array(data_ptr, shape=(num_samples,))
    processed = signal.detrend(raw_data)
    return processed.reshape(-1, 1)

# Detect anomalies
def detect_anomalies(model, data, mean, std, threshold=0.05):
    # Normalize with same mean/std from training
    data = (data - mean) / std if std != 0 else data
    reconstructions = model.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    return mse > threshold

# Example usage
if __name__ == "__main__":
    model, mean, std = train_model()
    data = run_test_cycle(200)
    anomalies = detect_anomalies(model, data, mean, std)
    print(f"Detected {np.sum(anomalies)} anomalies in {len(data)} samples.")