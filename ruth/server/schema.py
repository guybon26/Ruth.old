from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SeedSet(BaseModel):
    """
    Downlink message from Server to Client.
    Contains the seeds to be used for the current round.
    """
    round_id: int
    prng_config: Dict[str, Any] # e.g., {"type": "Xoshiro256StarStar"}
    seeds: List[int]
    epsilon: float

class ScalarUpload(BaseModel):
    """
    Uplink message from Client to Server.
    Contains the gradient estimate scalar and metadata.
    """
    round_id: int
    seed_id: int
    scalar: float
    loss_local: float
    attestation_token: Optional[str] = None

class AggResponse(BaseModel):
    """
    Downlink message from Server to Client after aggregation.
    Contains global updates or instructions for the next round.
    """
    server_updates: Dict[str, Any] # e.g., {"global_step": 100}
    next_round_hint: Optional[str] = None
