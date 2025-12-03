import ctypes
import numpy as np
import tensorflow as tf
from scipy import signal

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
    normal_data = run_test_cycle(1000, anomaly_chance=0)  
    model = build_anomaly_model(1)  
    model.fit(normal_data, normal_data, epochs=50, batch_size=32)
    model.save('anomaly_model.h5')
    return model

# Run simulation cycle
def run_test_cycle(num_samples=100, base_value=0.0, noise_amp=1.0, anomaly_chance=5):
    data_ptr = lib.simulate_sensor_data(num_samples, base_value, noise_amp, anomaly_chance)
    raw_data = np.ctypeslib.as_array(data_ptr, shape=(num_samples,))
    processed = signal.detrend(raw_data)  
    return processed.reshape(-1, 1)  

# Anomalies detection
def detect_anomalies(model, data, threshold=0.1):
    reconstructions = model.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    return mse > threshold

# Example usage
if __name__ == "__main__":
    # model = tf.keras.models.load_model('anomaly_model.h5')  # Or train_model() first time
    model = train_model()  # Run training first time
    data = run_test_cycle(200)
    anomalies = detect_anomalies(model, data)
    print(f"Detected {np.sum(anomalies)} anomalies in {len(data)} samples.")