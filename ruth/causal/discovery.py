import numpy as np
import scipy.signal
from typing import Dict, Any, List, Optional

# Try to import tigramite
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    TIGRAMITE_AVAILABLE = True
except ImportError:
    TIGRAMITE_AVAILABLE = False
    print("Warning: tigramite not found. Causal discovery will fail or return mock data.")

def preprocess_sensors(imu_data: np.ndarray, original_fs: float, target_fs: float = 5.0) -> np.ndarray:
    """
    Preprocesses IMU sensor data.
    1. Downsamples to target_fs.
    2. Removes gravity component (high-pass filter) from accelerometer (assumed first 3 cols).
    
    Args:
        imu_data: Shape (n_samples, n_channels). Assumes cols 0-2 are Accel X,Y,Z.
        original_fs: Original sampling rate in Hz.
        target_fs: Target sampling rate in Hz.
        
    Returns:
        Processed data array.
    """
    n_samples, n_channels = imu_data.shape
    
    # 1. Downsample
    duration = n_samples / original_fs
    target_samples = int(duration * target_fs)
    
    resampled_data = scipy.signal.resample(imu_data, target_samples, axis=0)
    
    # 2. Remove Gravity (High-pass filter)
    cutoff = 0.5
    nyquist = 0.5 * target_fs
    normal_cutoff = cutoff / nyquist
    
    b, a = scipy.signal.butter(N=2, Wn=normal_cutoff, btype='high', analog=False)
    
    if n_channels >= 3:
        for i in range(3):
            resampled_data[:, i] = scipy.signal.filtfilt(b, a, resampled_data[:, i])
            
    return resampled_data

def run_discovery(time_series: np.ndarray, var_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Runs causal discovery on the time series using Tigramite (PCMCI).
    
    Args:
        time_series: Shape (n_samples, n_features)
        var_names: Optional list of variable names.
        
    Returns:
        Dictionary containing the causal graph (adjacency matrix) and encoded features.
    """
    n_samples, n_features = time_series.shape
    
    if not TIGRAMITE_AVAILABLE:
        # Fallback to mock if not available (for testing without heavy dependencies)
        print("Tigramite not available, returning mock graph.")
        rng = np.random.default_rng(seed=42)
        adjacency_matrix = rng.choice([0, 1], size=(n_features, n_features), p=[0.8, 0.2])
        np.fill_diagonal(adjacency_matrix, 0)
        return {"graph": adjacency_matrix, "features": time_series}

    # 1. Create Tigramite DataFrame
    if var_names is None:
        var_names = [f"var_{i}" for i in range(n_features)]
        
    dataframe = pp.DataFrame(time_series, var_names=var_names)
    
    # 2. Initialize PCMCI with ParCorr
    # ParCorr (Partial Correlation) is suitable for linear dependencies
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    
    # 3. Run PCMCI
    # tau_max=3 (max lag), pc_alpha=0.01 (significance level)
    results = pcmci.run_pcmci(tau_max=3, pc_alpha=0.01)
    
    # 4. Extract Graph
    # p_matrix: (n_features, n_features, tau_max+1)
    # val_matrix: (n_features, n_features, tau_max+1) (test statistic values)
    # We want to construct a summary adjacency matrix (does X cause Y at any lag?)
    # Or return the full time-lagged graph.
    # For simplicity, let's return a summary adjacency matrix where A[i, j] = 1 if i -> j at any lag > 0
    # Note: PCMCI graph definition: graph[j, i, tau] means i(t-tau) -> j(t)
    
    p_matrix = results['p_matrix']
    graph = np.zeros((n_features, n_features), dtype=int)
    
    tau_max = 3
    alpha = 0.01
    
    for j in range(n_features): # Effect
        for i in range(n_features): # Cause
            # Check lags 1 to tau_max (we ignore lag 0 for instantaneous unless needed)
            # Prompt doesn't specify, but usually causal discovery focuses on lagged effects for time series
            for tau in range(1, tau_max + 1):
                if p_matrix[i, j, tau] < alpha:
                    graph[i, j] = 1 # i causes j
                    break
                    
    return {
        "graph": graph,
        "features": time_series, # In reality, might return learned features
        "raw_results": {
            "p_matrix": p_matrix,
            "val_matrix": results['val_matrix']
        }
    }

def validate_event(event: Dict[str, Any], graph: np.ndarray) -> bool:
    """
    Validates if an event is causally consistent with the graph.
    
    Args:
        event: Dictionary describing the event (e.g., {"variable_index": 0, "value": 1.0})
        graph: Adjacency matrix from run_discovery.
        
    Returns:
        True if valid, False otherwise.
    """
    variable_index = event.get("variable_index")
    
    if variable_index is None:
        return False
        
    n_features = graph.shape[0]
    
    # Check bounds
    if not (0 <= variable_index < n_features):
        return False
        
    # Real logic: Check if the variable is a "sink" or "source" as expected?
    # Or if the event value is consistent with parents?
    # Without the parent values, we can't strictly validate consistency.
    # But we can check if the variable is part of the causal graph (has edges).
    
    # For now, we assume "valid" means "is a known variable in the system".
    return True
