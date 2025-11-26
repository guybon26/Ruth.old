import torch
import pytest
from ruth.client.runtime import ClientRuntime
from ruth.core.export import RuthEdge
from ruth.core.prng import Xoshiro256StarStar

def test_client_runtime_step():
    # Setup
    model = RuthEdge()
    seeds = [42, 123, 999]
    prng = Xoshiro256StarStar(seeds=seeds)
    epsilon = 0.1
    
    runtime = ClientRuntime(model=model, prng=prng, epsilon_schedule=epsilon, max_norm=5.0)
    
    # Dummy Data
    batch_x = torch.randn(1, 10)
    batch_y = torch.tensor([0], dtype=torch.long)
    
    # Step 1
    result1 = runtime.step(batch_x, batch_y)
    
    assert result1["seed_id"] == 42
    assert isinstance(result1["scalar"], float)
    assert isinstance(result1["loss"], float)
    assert result1["epsilon"] == 0.1
    
    # Step 2 (Check seed cycling)
    result2 = runtime.step(batch_x, batch_y)
    assert result2["seed_id"] == 123
    
    # Check Baseline Update
    # Baseline starts at 0. After step 1, baseline = 0.1 * rho1 (since beta=0.9, 1-beta=0.1)
    # rho_adj = rho - baseline
    # We can verify that baseline is changing
    assert runtime.baseline != 0.0

def test_clipping():
    model = RuthEdge()
    prng = Xoshiro256StarStar(seeds=[1])
    # Very small epsilon to force large gradients
    runtime = ClientRuntime(model=model, prng=prng, epsilon_schedule=1e-9, max_norm=0.5)
    
    batch_x = torch.randn(1, 10)
    batch_y = torch.tensor([0], dtype=torch.long)
    
    result = runtime.step(batch_x, batch_y)
    
    assert abs(result["scalar"]) <= 0.5

if __name__ == "__main__":
    test_client_runtime_step()
    test_clipping()
    print("All ClientRuntime tests passed!")
