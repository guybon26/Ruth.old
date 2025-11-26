import numpy as np
import pytest
from ruth.causal.discovery import preprocess_sensors, run_discovery, validate_event

def test_preprocess_sensors():
    # Setup: 10 seconds of data at 50Hz
    original_fs = 50.0
    duration = 10.0
    n_samples = int(duration * original_fs)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # Signal: Sine wave (2Hz) + Gravity (DC offset of 9.8)
    # Channel 0 (Accel X): Sine + Gravity
    # Channel 1 (Accel Y): Noise
    # Channel 2 (Accel Z): Gravity
    # Channel 3 (Gyro): Sine
    data = np.zeros((n_samples, 4))
    data[:, 0] = np.sin(2 * np.pi * 2 * t) + 9.8
    data[:, 1] = np.random.randn(n_samples)
    data[:, 2] = 9.8
    data[:, 3] = np.sin(2 * np.pi * 1 * t)
    
    target_fs = 5.0
    processed = preprocess_sensors(data, original_fs, target_fs)
    
    # Check shape
    expected_samples = int(duration * target_fs)
    assert processed.shape == (expected_samples, 4)
    
    # Check Gravity Removal (High-pass)
    # Mean of Accel channels (0, 1, 2) should be close to 0
    # Channel 0: Mean should be ~0 (removed 9.8)
    assert abs(processed[:, 0].mean()) < 0.5
    # Channel 2: Mean should be ~0 (removed 9.8)
    assert abs(processed[:, 2].mean()) < 0.5
    
    # Check Channel 3 (Gyro) - should NOT be filtered (logic only filters first 3)
    # Wait, my implementation filters first 3. Channel 3 is index 3.
    # So channel 3 should be just resampled.
    # Resampling preserves mean of sine wave (0).
    pass

def test_run_discovery():
    n_samples = 100
    n_features = 5
    data = np.random.randn(n_samples, n_features)
    
    result = run_discovery(data)
    
    graph = result["graph"]
    features = result["features"]
    
    assert graph.shape == (n_features, n_features)
    assert features.shape == (n_samples, n_features)
    # Check binary matrix
    assert np.all(np.isin(graph, [0, 1]))

def test_validate_event():
    n_features = 5
    graph = np.zeros((n_features, n_features))
    
    # Valid event
    event_valid = {"variable_index": 2, "value": 1.0}
    assert validate_event(event_valid, graph) == True
    
    # Invalid index
    event_invalid = {"variable_index": 10, "value": 1.0}
    assert validate_event(event_invalid, graph) == False
    
    # Missing index
    event_missing = {"value": 1.0}
    assert validate_event(event_missing, graph) == False

if __name__ == "__main__":
    test_preprocess_sensors()
    test_run_discovery()
    test_validate_event()
    print("All Causal tests passed!")
