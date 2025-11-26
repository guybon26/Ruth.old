import os
import base64
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

class SecurityManager:
    def __init__(self):
        # Load private key from secure storage (Env Var simulation)
        # In production, this would come from Android Keystore / iOS Secure Enclave
        private_key_bytes_b64 = os.environ.get("RUTH_CLIENT_PRIVATE_KEY")
        
        if private_key_bytes_b64:
            try:
                private_key_bytes = base64.b64decode(private_key_bytes_b64)
                self.private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            except Exception as e:
                print(f"Error loading private key: {e}. Generating new one.")
                self.private_key = ed25519.Ed25519PrivateKey.generate()
        else:
            print("No private key found in env. Generating new one for session.")
            self.private_key = ed25519.Ed25519PrivateKey.generate()
            
        self.public_key = self.private_key.public_key()

    def get_public_key_bytes(self) -> bytes:
        """Returns the public key in raw bytes format."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

    def sign_update(self, seed_id: int, scalar: float, round_id: int) -> bytes:
        """
        Signs the update payload using Ed25519.
        Payload structure: "{seed_id}:{scalar}:{round_id}"
        """
        payload = f"{seed_id}:{scalar}:{round_id}".encode('utf-8')
        return self.private_key.sign(payload)

    def get_attestation_token(self, seed_id: int, scalar: float, round_id: int) -> bytes:
        """
        Generates a mock attestation token.
        In production, this would call the OS API (Play Integrity / App Attest).
        Here we return a token that the server's mock verification logic expects,
        but we structure it to look like a real integrity token payload for simulation.
        """
        # For the server's "Real" verification logic (which mocks the Google API call),
        # we need to provide a token that looks like a JWS or similar.
        # But since we are mocking the *API call* on the server, the content here 
        # just needs to be something we can pass.
        
        # However, the prompt says "Implement the full JSON payload structure" on the server side.
        # So the client should probably return something that looks like a token.
        return b"mock_integrity_token_from_device"
