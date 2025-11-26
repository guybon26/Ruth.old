import numpy as np
import torch
from typing import List, Tuple

class Xoshiro256StarStar:
    """
    Deterministic PRNG wrapper for FedKSeed.
    Uses numpy's PCG64 (default_rng) for robust, architecture-independent reproducibility.
    """
    def __init__(self, seeds: List[int]):
        self.seeds = seeds
        self.current_epoch = 0

    def next_seed(self) -> int:
        """
        Returns the next seed index based on the current epoch.
        Cycles through the provided seeds list.
        """
        if not self.seeds:
            raise ValueError("No seeds provided")
        
        seed_idx = self.current_epoch % len(self.seeds)
        self.current_epoch += 1
        return self.seeds[seed_idx]

    def generate_noise_vector(self, seed_id: int, shape: Tuple[int]) -> torch.Tensor:
        """
        Generates a flattened noise vector drawn from a Standard Normal distribution.
        Reproducible across architectures given the same seed_id.
        """
        # Initialize numpy generator with the specific seed
        rng = np.random.default_rng(seed=seed_id)
        
        # Generate standard normal noise
        # We generate as float32 to match typical torch tensor types
        noise = rng.standard_normal(size=shape, dtype=np.float32)
        
        # Convert to torch tensor and flatten
        return torch.from_numpy(noise).flatten()
