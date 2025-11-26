import torch
import numpy as np
from typing import Dict, Any, Callable, Union

class ClientRuntime:
    """
    Manages the on-device training loop using Antithetic Sampling (Forward Gradient).
    """
    def __init__(
        self, 
        model: Any, 
        prng: Any, 
        epsilon_schedule: Union[float, Callable[[int], float]], 
        max_norm: float = 1.0,
        model_path: str = None # Kept for API compatibility, ignored in prototype
    ):
        self.model = model
        self.prng = prng
        self.epsilon_schedule = epsilon_schedule
        self.max_norm = max_norm
        
        self.step_count = 0
        self.baseline = 0.0
        self.beta = 0.9 # Momentum for baseline moving average

    def _get_epsilon(self) -> float:
        if callable(self.epsilon_schedule):
            return self.epsilon_schedule(self.step_count)
        return self.epsilon_schedule

    def step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Dict[str, Any]:
        """
        Performs a single optimization step using Antithetic Sampling.
        Returns a dictionary with the seed_id, scalar update, and loss.
        """
        # 1. Get Seed and Noise
        seed_id = self.prng.next_seed()
        
        # We need to know the shape of the noise vector.
        # In a real scenario, this would be derived from the model's trainable parameters.
        # For the RuthEdge prototype, we can calculate it dynamically or assume the model provides it.
        # Let's calculate it from the model's trainable parameters (LoRA).
        # We assume the model has lora_a and lora_b as in RuthEdge.
        total_params = sum(p.numel() for p in self.model.lora_a.parameters()) + \
                       sum(p.numel() for p in self.model.lora_b.parameters())
        
        v = self.prng.generate_noise_vector(seed_id, (total_params,))
        
        # 2. Inference (Loss 0)
        # We run this for logging/metrics, though strictly not needed for the gradient estimate itself
        # if we only care about the difference. But usually we want to know the current loss.
        with torch.no_grad():
            loss0 = self.model.forward_infer(batch_x, batch_y)
        
        # 3. Antithetic Sampling
        epsilon = self._get_epsilon()
        
        with torch.no_grad():
            # Perturb +epsilon
            lossP = self.model.forward_perturb(batch_x, batch_y, v, epsilon)
            
            # Perturb -epsilon
            lossM = self.model.forward_perturb(batch_x, batch_y, v, -epsilon)
            
        # 4. Gradient Estimate (Scalar rho)
        # rho = (L+ - L-) / (2 * epsilon)
        rho = (lossP.item() - lossM.item()) / (2 * epsilon)
        
        # 5. Control Variate (Baseline subtraction)
        # Update running baseline of rho
        self.baseline = self.beta * self.baseline + (1 - self.beta) * rho
        
        # Subtract baseline (centering the update)
        rho_adj = rho - self.baseline
        
        # 6. Clipping
        if abs(rho_adj) > self.max_norm:
            rho_adj = self.max_norm * (1.0 if rho_adj > 0 else -1.0)
            
        self.step_count += 1
        
        return {
            "seed_id": seed_id,
            "scalar": rho_adj,
            "loss": loss0.item(),
            "raw_rho": rho,
            "epsilon": epsilon
        }
