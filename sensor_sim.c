#include <stdlib.h>
#include <math.h>
#include <time.h>

// Simulate sensor data (e.g., vibration or temp) with noise/anomalies
float* simulate_sensor_data(int num_samples, float base_value, float noise_amp, int anomaly_chance) {
    srand(time(NULL));
    float* data = (float*)malloc(num_samples * sizeof(float));
    for (int i = 0; i < num_samples; i++) {
        float noise = ((float)rand() / RAND_MAX) * noise_amp * 2 - noise_amp;
        data[i] = base_value + sin(i * 0.1) * 5 + noise;  // Sinusoidal base + noise
        if (rand() % 100 < anomaly_chance) {  // Random anomaly spike
            data[i] *= 2.5;  // Exaggerate for detection
        }
    }
    return data;
}