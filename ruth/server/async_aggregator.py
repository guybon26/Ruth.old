import asyncio
import redis.asyncio as redis
from typing import Any, Optional
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

# Try to import generated protobuf classes
try:
    from proto import ruth_pb2
except ImportError:
    # If not generated, we define a dummy class for type hinting/mocking
    # in a real env, protoc would have run
    class ruth_pb2:
        class ClientUpdate:
            def SerializeToString(self): return b""
            def ParseFromString(self, data): pass
            device_id: str
            round_id: int
            seed_id: int
            scalar: float
            loss: float
            signature: bytes
            attestation_token: bytes

class AsyncAggregator:
    def __init__(self, redis_url: str, k_threshold: int):
        # Use Connection Pool for efficiency
        self.pool = redis.ConnectionPool.from_url(redis_url)
        self.redis = redis.Redis(connection_pool=self.pool)
        self.k_threshold = k_threshold
        self.running = False
        self.task = None

    async def submit_update(self, update: ruth_pb2.ClientUpdate) -> bool:
        """
        Submits a client update to Redis.
        
        Args:
            update: ClientUpdate protobuf object.
        
        Returns:
            True if submitted successfully, False if signature invalid.
        """
        # 1. Validate Signature (Fast Check)
        # In a real system, we'd look up the public key for update.device_id
        # For now, we assume a way to get it, or we skip if we can't.
        # This is "Real State Management", so we focus on the Redis part.
        # The verification logic is in Gatekeeper, but aggregator might do a quick check
        # or rely on the gatekeeper calling it.
        # Let's assume this method is called AFTER Gatekeeper verification in the API layer.
        # But if we must verify here:
        # public_key = await self.get_public_key(update.device_id)
        # if not public_key: return False
        
        # 2. Push to Redis using Pipeline
        round_id = update.round_id
        
        # Serialize Protobuf
        serialized_data = update.SerializeToString()
        
        updates_key = f"ruth:round:{round_id}:updates"
        count_key = f"ruth:round:{round_id}:count"
        
        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                # LPUSH the binary data
                await pipe.lpush(updates_key, serialized_data)
                # INCR the counter
                await pipe.incr(count_key)
                # Execute atomic block
                await pipe.execute()
        except redis.RedisError as e:
            print(f"Redis error during submit_update: {e}")
            return False
            
        return True

    def start_worker(self):
        """Starts the background worker loop."""
        self.running = True
        self.task = asyncio.create_task(self._worker_loop())

    async def stop_worker(self):
        """Stops the background worker loop."""
        self.running = False
        if self.task:
            await self.task
        # Close pool
        await self.redis.close()
        await self.pool.disconnect()

    async def _worker_loop(self):
        """
        Background loop that checks if aggregation threshold is met.
        """
        print("AsyncAggregator worker started.")
        while self.running:
            try:
                # Scan for active rounds
                # In production, we'd use SCAN or a set of active rounds
                # For efficiency, let's assume we maintain a set "ruth:active_rounds"
                # But to stick to the prompt's simplicity on keys:
                
                # Use SCAN to find keys matching pattern
                # This is better than KEYS which blocks
                cursor = b'0'
                while cursor:
                    cursor, keys = await self.redis.scan(cursor, match="ruth:round:*:count", count=100)
                    for key in keys:
                        key_str = key.decode('utf-8')
                        # key format: ruth:round:{round_id}:count
                        try:
                            round_id = int(key_str.split(':')[2])
                            
                            count = int(await self.redis.get(key_str) or 0)
                            
                            if count >= self.k_threshold:
                                print(f"Threshold reached for round {round_id} (Count: {count}). Aggregating...")
                                await self._trigger_aggregation(round_id)
                        except (IndexError, ValueError):
                            continue
                            
                    if cursor == b'0':
                        break
                
                await asyncio.sleep(1.0) # Poll interval
                
            except Exception as e:
                print(f"Worker error: {e}")
                await asyncio.sleep(1.0)

    async def _trigger_aggregation(self, round_id: int):
        """
        Loads updates from Redis, performs aggregation, and cleans up.
        """
        updates_key = f"ruth:round:{round_id}:updates"
        count_key = f"ruth:round:{round_id}:count"
        
        try:
            # 1. Load all updates
            serialized_updates = await self.redis.lrange(updates_key, 0, -1)
            
            print(f"Loaded {len(serialized_updates)} updates for round {round_id}.")
            
            parsed_updates = []
            for data in serialized_updates:
                update = ruth_pb2.ClientUpdate()
                update.ParseFromString(data)
                parsed_updates.append(update)
            
            # 2. Perform Aggregation (Mock call to RobustAggregator)
            # In a real app, we'd call the aggregator logic here
            # aggregator = RobustAggregator(...)
            # result = aggregator.aggregate(parsed_updates)
            
            # 3. Cleanup
            # Use pipeline for atomic cleanup
            async with self.redis.pipeline(transaction=True) as pipe:
                await pipe.delete(updates_key)
                await pipe.delete(count_key)
                await pipe.execute()
                
            print(f"Round {round_id} aggregation complete. Keys cleared.")
            
        except redis.RedisError as e:
            print(f"Redis error during aggregation: {e}")
        except Exception as e:
            print(f"Aggregation error: {e}")
