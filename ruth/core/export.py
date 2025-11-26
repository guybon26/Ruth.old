import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
try:
    from torch.func import functional_call
except ImportError:
    print("Error: torch.func not available. Please upgrade PyTorch.")
    functional_call = None

try:
    from executorch.exir import to_edge
except ImportError:
    to_edge = None
    print("Warning: executorch not found. Export to .pte will fail.")

class RuthEdge(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        # Base model (Frozen)
        self.base_layer = nn.Linear(input_dim, hidden_dim)
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # LoRA Adapters (Trainable)
        # W' = W + B @ A
        self.lora_a = nn.Linear(input_dim, 4, bias=False)
        self.lora_b = nn.Linear(4, hidden_dim, bias=False)
        
        # Output head (Frozen)
        self.head = nn.Linear(hidden_dim, output_dim)
        for param in self.head.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.
        """
        # Base path
        base_out = self.base_layer(x)
        
        # LoRA path
        lora_out = self.lora_b(self.lora_a(x))
        
        # Combined
        hidden = F.relu(base_out + lora_out)
        logits = self.head(hidden)
        return logits

    def forward_infer(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning CrossEntropyLoss.
        """
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def forward_perturb(self, x: torch.Tensor, y: torch.Tensor, v_flat: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Stateless forward pass with perturbed weights using torch.func.functional_call.
        W' = W + epsilon * v
        """
        # 1. Construct the perturbed state dictionary
        # We only perturb LoRA weights: lora_a.weight, lora_b.weight
        # We need to reconstruct the shapes from v_flat
        
        # Get current state dict (names and params)
        params = dict(self.named_parameters())
        
        # Identify trainable params (LoRA)
        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        
        offset = 0
        perturbed_params = {}
        
        # Copy all params first (including frozen ones)
        for name, param in params.items():
            perturbed_params[name] = param

        # Apply perturbation to trainable params
        for name in trainable_names:
            param = params[name]
            numel = param.numel()
            
            # Extract v for this param
            v_param = v_flat[offset : offset + numel].view_as(param)
            offset += numel
            
            # W' = W + epsilon * v
            perturbed_params[name] = param + epsilon * v_param
            
        # 2. Functional Call
        # Execute the model using the perturbed weights
        # functional_call(module, parameter_dict, args) calls module.forward(*args)
        # Our forward only takes x and returns logits
        
        logits = functional_call(self, perturbed_params, (x,))
        loss = F.cross_entropy(logits, y)
        return loss

def export_recipes(output_dir: str = "."):
    if to_edge is None:
        print("Skipping export: executorch not available.")
        return

    model = RuthEdge()
    model.eval()
    
    # Dummy inputs
    x = torch.randn(1, 10)
    y = torch.tensor([0], dtype=torch.long)
    
    # Calculate total params for v_flat
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    v_flat = torch.randn(total_params)
    epsilon = torch.tensor(0.1)

    print("Exporting ruth_infer.pte...")
    infer_program = export(model.forward_infer, (x, y))
    edge_infer = to_edge(infer_program)
    edge_infer.to_executorch_program().save(f"{output_dir}/ruth_infer.pte")
    
    print("Exporting ruth_perturb.pte...")
    # Note: forward_perturb uses functional_call which might be tricky for export
    # But torch.export should trace through it.
    perturb_program = export(model.forward_perturb, (x, y, v_flat, epsilon))
    edge_perturb = to_edge(perturb_program)
    edge_perturb.to_executorch_program().save(f"{output_dir}/ruth_perturb.pte")
    
    print("Export complete.")

if __name__ == "__main__":
    export_recipes()
