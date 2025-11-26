from ruth.server.schema import SeedSet, ScalarUpload, AggResponse
import pytest

def test_seed_set():
    data = {
        "round_id": 1,
        "prng_config": {"type": "test"},
        "seeds": [1, 2, 3],
        "epsilon": 0.1
    }
    model = SeedSet(**data)
    assert model.round_id == 1
    assert model.seeds == [1, 2, 3]

def test_scalar_upload():
    data = {
        "round_id": 1,
        "seed_id": 42,
        "scalar": 0.5,
        "loss_local": 1.2,
        "attestation_token": "token123"
    }
    model = ScalarUpload(**data)
    assert model.scalar == 0.5
    assert model.attestation_token == "token123"

def test_agg_response():
    data = {
        "server_updates": {"global_step": 10},
        "next_round_hint": "wait"
    }
    model = AggResponse(**data)
    assert model.server_updates["global_step"] == 10

if __name__ == "__main__":
    test_seed_set()
    test_scalar_upload()
    test_agg_response()
    print("All Schema tests passed!")
