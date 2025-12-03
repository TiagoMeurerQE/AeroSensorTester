import pytest
from main import run_test_cycle, detect_anomalies, build_anomaly_model

def test_simulation():
    data = run_test_cycle(10)
    assert len(data) == 10

def test_detection():
    model = build_anomaly_model(1)
    data = np.array([[0.5]])
    anomalies = detect_anomalies(model, data)
    assert len(anomalies) == 1  # Basic check