import torch
import pytest
from ruth.core.export import RuthEdge

def test_ruth_edge_forward():
    model = RuthEdge()
    x = torch.randn(1, 10)
    y = torch.tensor([0], dtype=torch.long)
    
    loss = model.forward_infer(x, y)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # Scalar loss

def test_ruth_edge_perturb():
    model = RuthEdge()
    x = torch.randn(1, 10)
    y = torch.tensor([0], dtype=torch.long)
    
    # Calculate total params for v_flat
    total_params = sum(p.numel() for p in model.lora_a.parameters()) + \
                   sum(p.numel() for p in model.lora_b.parameters())
    v_flat = torch.randn(total_params)
    epsilon = 0.1
    
    # Capture original weights
    orig_wa = model.lora_a.weight.clone()
    
    loss_perturbed = model.forward_perturb(x, y, v_flat, epsilon)
    
    # Check loss is scalar
    assert isinstance(loss_perturbed, torch.Tensor)
    assert loss_perturbed.ndim == 0
    
    # Check that original weights are NOT mutated
    assert torch.allclose(model.lora_a.weight, orig_wa)

if __name__ == "__main__":
    test_ruth_edge_forward()
    test_ruth_edge_perturb()
    print("All tests passed!")
