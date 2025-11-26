import pytest
from ruth.client.attestation import generate_binding_hash, verify_attestation

def test_generate_binding_hash():
    payload = {
        'seed_id': 123,
        'scalar': 0.5,
        'model_hash': 'abc'
    }
    
    # Construct expected string manually to verify format
    binding_str = "123:0.5:abc"
    import hashlib
    expected_hash = hashlib.sha256(binding_str.encode('utf-8')).hexdigest()
    
    generated_hash = generate_binding_hash(payload)
    assert generated_hash == expected_hash

def test_verify_attestation():
    valid_token = "valid_token_123"
    invalid_token = "invalid_token_123"
    binding_hash = "any_hash"
    
    assert verify_attestation(valid_token, binding_hash) == True
    assert verify_attestation(invalid_token, binding_hash) == False

if __name__ == "__main__":
    test_generate_binding_hash()
    test_verify_attestation()
    print("All Attestation tests passed!")
