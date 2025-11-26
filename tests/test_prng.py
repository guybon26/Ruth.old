import torch
import numpy as np
import pytest
from ruth.core.prng import Xoshiro256StarStar

def test_prng_reproducibility():
    prng = Xoshiro256StarStar(seeds=[42, 123])
    shape = (10, 10)
    
    # Generate twice with same seed
    v1 = prng.generate_noise_vector(seed_id=42, shape=shape)
    v2 = prng.generate_noise_vector(seed_id=42, shape=shape)
    
    assert torch.equal(v1, v2), "Vectors with same seed must be identical"
    
    # Generate with different seed
    v3 = prng.generate_noise_vector(seed_id=123, shape=shape)
    assert not torch.equal(v1, v3), "Vectors with different seeds should be different"

def test_prng_next_seed():
    seeds = [10, 20, 30]
    prng = Xoshiro256StarStar(seeds=seeds)
    
    assert prng.next_seed() == 10
    assert prng.next_seed() == 20
    assert prng.next_seed() == 30
    assert prng.next_seed() == 10 # Cycle

def test_prng_shape():
    prng = Xoshiro256StarStar(seeds=[1])
    shape = (5, 4)
    v = prng.generate_noise_vector(seed_id=1, shape=shape)
    
    assert v.ndim == 1, "Output should be flattened"
    assert v.numel() == 20, "Output should have prod(shape) elements"

def test_prng_distribution():
    prng = Xoshiro256StarStar(seeds=[1])
    shape = (1000, 1000) # Large enough for stats
    v = prng.generate_noise_vector(seed_id=1, shape=shape)
    
    mean = v.mean().item()
    std = v.std().item()
    
    assert abs(mean) < 0.01, f"Mean {mean} should be close to 0"
    assert abs(std - 1.0) < 0.01, f"Std {std} should be close to 1"

if __name__ == "__main__":
    test_prng_reproducibility()
    test_prng_next_seed()
    test_prng_shape()
    test_prng_distribution()
    print("All PRNG tests passed!")
