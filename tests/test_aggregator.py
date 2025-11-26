import torch
import pytest
from ruth.server.aggregator import RobustAggregator
from ruth.core.prng import Xoshiro256StarStar

def test_reconstruction():
    shape = (10,)
    seeds = [1, 2, 3]
    prng = Xoshiro256StarStar(seeds=seeds)
    aggregator = RobustAggregator(prng=prng, shape=shape)
    
    # Create a payload
    seed_id = 1
    scalar = 2.0
    payloads = [{'seed_id': seed_id, 'scalar': scalar}]
    
    # Reconstruct
    gradients = aggregator.reconstruct(payloads)
    
    # Verify
    expected_v = prng.generate_noise_vector(seed_id, shape)
    expected_g = scalar * expected_v
    
    assert gradients.shape == (1, 10)
    assert torch.allclose(gradients[0], expected_g)

def test_trimmed_mean_aggregation():
    shape = (5,)
    prng = Xoshiro256StarStar(seeds=[1]) # Dummy
    # Trim 10% from each side. With 10 clients, k=1.
    aggregator = RobustAggregator(prng=prng, shape=shape, trim_ratio=0.1)
    
    # Create gradients: 8 normal, 2 outliers
    # Normal: all 1.0
    normal_grads = torch.ones(8, 5)
    
    # Outliers: one very small (-100), one very large (100)
    outlier_low = torch.full((1, 5), -100.0)
    outlier_high = torch.full((1, 5), 100.0)
    
    gradients = torch.cat([normal_grads, outlier_low, outlier_high], dim=0)
    # Shuffle to ensure sorting works (though robust agg sorts internally)
    # For simplicity, we just pass them as is.
    
    aggregated = aggregator.aggregate(gradients)
    
    # Expectation: Outliers removed. Mean of 8 ones is 1.0.
    assert torch.allclose(aggregated, torch.ones(5))

def test_aggregation_no_trim():
    shape = (5,)
    prng = Xoshiro256StarStar(seeds=[1])
    # Trim 10%. With 5 clients, k=0.5 -> 0. No trimming.
    aggregator = RobustAggregator(prng=prng, shape=shape, trim_ratio=0.1)
    
    gradients = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0, 5.0, 5.0]
    ])
    
    aggregated = aggregator.aggregate(gradients)
    
    # Mean should be 3.0
    assert torch.allclose(aggregated, torch.full((5,), 3.0))

if __name__ == "__main__":
    test_reconstruction()
    test_trimmed_mean_aggregation()
    test_aggregation_no_trim()
    print("All RobustAggregator tests passed!")
