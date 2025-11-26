import pytest
from ruth.client.security import SecurityManager
from ruth.server.verifier import Gatekeeper

# Mock ClientUpdate Object
class MockClientUpdate:
    def __init__(self, seed_id, scalar, round_id, signature, attestation_token):
        self.seed_id = seed_id
        self.scalar = scalar
        self.round_id = round_id
        self.signature = signature
        self.attestation_token = attestation_token

from unittest.mock import MagicMock, patch
import json

def test_security_flow():
    # Mock Response for Attestation
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({
        "isValidSignature": True,
        "basicIntegrity": True,
        # Nonce must match SHA256(payload)
        # We calculate it inside the test or make the mock dynamic?
        # Easier to mock the return value to match what we expect
        # But the nonce depends on the payload.
        # Let's use side_effect or just ensure the nonce matches the one generated in the test.
    }).encode('utf-8')
    
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None

    # We need to compute the expected nonce to put in the mock
    # But the test runs sequentially.
    
    with patch('urllib.request.urlopen', return_value=mock_response) as mock_urlopen:
        # 1. Setup
        client_sec = SecurityManager()
        gatekeeper = Gatekeeper()
        
        seed_id = 12345
        scalar = 0.1234
        round_id = 1
        
        # 2. Client Signs and Attests
        signature = client_sec.sign_update(seed_id, scalar, round_id)
        attestation = client_sec.get_attestation_token(seed_id, scalar, round_id)
        public_key = client_sec.get_public_key_bytes()
        
        update = MockClientUpdate(seed_id, scalar, round_id, signature, attestation)
        
        # Calculate expected nonce for the valid update
        import hashlib
        payload = f"{seed_id}:{scalar}:{round_id}".encode('utf-8')
        expected_nonce = hashlib.sha256(payload).hexdigest()
        
        # Update mock to return correct nonce
        mock_response.read.return_value = json.dumps({
            "isValidSignature": True,
            "basicIntegrity": True,
            "nonce": expected_nonce
        }).encode('utf-8')
        
        # 3. Server Verifies (Success Case)
        assert gatekeeper.verify_update(update, public_key) == True
        
        # 4. Tampering Test (Signature)
        # Modify scalar but keep signature
        tampered_update = MockClientUpdate(seed_id, scalar + 0.1, round_id, signature, attestation)
        assert gatekeeper.verify_update(tampered_update, public_key) == False
        
        # 5. Tampering Test (Attestation)
        # Modify scalar, re-sign, but keep old attestation (nonce mismatch)
        new_scalar = scalar + 0.1
        new_sig = client_sec.sign_update(seed_id, new_scalar, round_id)
        # Attestation is bound to old scalar
        tampered_attestation_update = MockClientUpdate(seed_id, new_scalar, round_id, new_sig, attestation)
        
        # The mock still returns the OLD nonce (valid for old payload), but we are verifying against NEW payload
        # So verify_update should calculate NEW nonce, compare with OLD nonce from mock, and fail.
        assert gatekeeper.verify_update(tampered_attestation_update, public_key) == False
        
        # 6. Invalid Attestation Verdict
        # Update mock to return failure
        mock_response.read.return_value = json.dumps({
            "isValidSignature": True,
            "basicIntegrity": False, # Fail
            "nonce": expected_nonce
        }).encode('utf-8')
        
        bad_verdict_update = MockClientUpdate(seed_id, scalar, round_id, signature, attestation)
        assert gatekeeper.verify_update(bad_verdict_update, public_key) == False

if __name__ == "__main__":
    test_security_flow()
    print("All Security tests passed!")
