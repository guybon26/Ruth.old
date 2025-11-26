import torch
from typing import List, Dict, Tuple, Any

class RobustAggregator:
    """
    Server-side aggregator for FedKSeed.
    Reconstructs gradients from seeds and aggregates them using robust statistics.
    """
    def __init__(self, prng: Any, shape: Tuple[int], trim_ratio: float = 0.1):
        self.prng = prng
        self.shape = shape
        self.trim_ratio = trim_ratio

    def reconstruct(self, payloads: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Reconstructs gradient vectors from client payloads.
        Payload format: {'seed_id': int, 'scalar': float}
        Returns a tensor of shape (num_clients, num_params).
        """
        gradients = []
        
        for payload in payloads:
            seed_id = payload['seed_id']
            scalar = payload['scalar']
            
            # Generate noise vector (v)
            v = self.prng.generate_noise_vector(seed_id, self.shape)
            
            # Reconstruct gradient approximation (g = scalar * v)
            g = scalar * v
            gradients.append(g)
            
        if not gradients:
            return torch.empty(0, *self.shape)
            
        return torch.stack(gradients)

    def aggregate(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Aggregates gradients using Coordinate-wise Trimmed Mean.
        gradients shape: (num_clients, num_params)
        """
        if gradients.numel() == 0:
            return torch.zeros(self.shape)
            
        num_clients = gradients.shape[0]
        
        # Calculate number of elements to trim from each side
        k = int(num_clients * self.trim_ratio)
        
        if k == 0:
            # Not enough clients to trim, return mean
            return torch.mean(gradients, dim=0)
            
        # Sort along client dimension
        sorted_grads, _ = torch.sort(gradients, dim=0)
        
        # Trim top and bottom k
        trimmed_grads = sorted_grads[k : num_clients - k]
        
        # Compute mean of remaining
        return torch.mean(trimmed_grads, dim=0)
