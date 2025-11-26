import torch
import numpy as np
import pytest
from ruth.core.export import RuthEdge
from ruth.core.prng import Xoshiro256StarStar
from ruth.client.runtime import ClientRuntime
from ruth.server.aggregator import RobustAggregator

def test_federated_learning_loop():
    # --- Setup ---
    
    # 1. Shared Configuration
    input_dim = 10
    hidden_dim = 20
    output_dim = 2
    seeds = list(range(1000)) # Shared pool of seeds
    
    # 2. Initialize Server Components
    # We need the shape of the noise vector. 
    # Instantiate a dummy model to get parameter count.
    dummy_model = RuthEdge(input_dim, hidden_dim, output_dim)
    total_params = sum(p.numel() for p in dummy_model.lora_a.parameters()) + \
                   sum(p.numel() for p in dummy_model.lora_b.parameters())
    noise_shape = (total_params,)
    
    server_prng = Xoshiro256StarStar(seeds=seeds)
    # Trim 20% (1 out of 5 from each side) to handle 1 adversary out of 5
    # With 5 clients, trim_ratio=0.2 -> k=1. Removes top 1 and bottom 1.
    aggregator = RobustAggregator(prng=server_prng, shape=noise_shape, trim_ratio=0.2)
    
    # 3. Initialize Clients
    num_clients = 5
    clients = []
    client_prngs = []
    
    for i in range(num_clients):
        # Each client has its own model instance (conceptually)
        # In this sim, they start identical.
        model = RuthEdge(input_dim, hidden_dim, output_dim)
        # Sync weights (optional for this test since we only care about gradients, 
        # but good practice)
        model.load_state_dict(dummy_model.state_dict())
        
        # Client PRNG must be synced with server's seed list
        prng = Xoshiro256StarStar(seeds=seeds)
        client_prngs.append(prng)
        
        runtime = ClientRuntime(
            model=model, 
            prng=prng, 
            epsilon_schedule=0.1, 
            max_norm=5.0
        )
        clients.append(runtime)
        
    # --- Simulation Loop ---
    num_rounds = 10
    
    for round_idx in range(num_rounds):
        print(f"--- Round {round_idx} ---")
        
        # 1. Server broadcasts seeds (implicitly handled by shared PRNG logic here)
        # In real protocol, server sends seed indices. 
        # Here, clients call next_seed() which advances their local PRNG state.
        
        payloads = []
        honest_scalars = []
        
        # 2. Clients Compute Updates
        for i, client in enumerate(clients):
            # Dummy Data
            batch_x = torch.randn(1, input_dim)
            batch_y = torch.tensor([0], dtype=torch.long)
            
            result = client.step(batch_x, batch_y)
            
            # Adversarial Client (Index 4)
            if i == 4:
                # Poison the scalar
                result['scalar'] = result['scalar'] * 100.0
                print(f"Client {i} (Adversary): scalar={result['scalar']:.4f}")
            else:
                honest_scalars.append(result['scalar'])
                # print(f"Client {i} (Honest): scalar={result['scalar']:.4f}")
                
            payloads.append(result)
            
        # 3. Server Aggregates
        # Reconstruct gradients
        gradients = aggregator.reconstruct(payloads)
        
        # Check dimensions
        assert gradients.shape == (num_clients, total_params)
        
        # Aggregate
        global_grad = aggregator.aggregate(gradients)
        
        # --- Verification ---
        
        # 1. Check Global Gradient Norm
        # The adversarial update is huge. If aggregation is not robust, 
        # the global gradient will be huge.
        
        # Reconstruct honest gradients for comparison
        honest_grads = []
        for j in range(4): # First 4 are honest
            v = server_prng.generate_noise_vector(payloads[j]['seed_id'], noise_shape)
            g = payloads[j]['scalar'] * v
            honest_grads.append(g)
        honest_mean_grad = torch.stack(honest_grads).mean(dim=0)
        
        # Calculate distances
        dist_to_honest = torch.norm(global_grad - honest_mean_grad).item()
        
        # Adversarial gradient
        adv_v = server_prng.generate_noise_vector(payloads[4]['seed_id'], noise_shape)
        adv_g = payloads[4]['scalar'] * adv_v
        dist_to_adv = torch.norm(global_grad - adv_g).item()
        
        print(f"Global Grad Norm: {torch.norm(global_grad):.4f}")
        print(f"Dist to Honest Mean: {dist_to_honest:.4f}")
        print(f"Dist to Adversary: {dist_to_adv:.4f}")
        
        # The global gradient should be MUCH closer to the honest mean than the adversary
        # Because the adversary was trimmed out.
        # Note: With trimmed mean, we remove top/bottom 1. 
        # If adversary is huge positive, it's removed. 
        # If huge negative, it's removed.
        
        assert dist_to_honest < dist_to_adv, "Aggregation failed to reject adversary!"
        
        # Also check that the global gradient norm is reasonable (not exploded)
        # Honest scalars are clipped to 5.0. Noise is std normal.
        # Expected norm approx sqrt(total_params) * scalar.
        # total_params = 10*4 + 4*20 = 120. sqrt(120) ~ 11.
        # Max norm ~ 5 * 11 = 55.
        assert torch.norm(global_grad) < 100.0, "Global gradient exploded!"

if __name__ == "__main__":
    test_federated_learning_loop()
    print("Integration test passed!")
