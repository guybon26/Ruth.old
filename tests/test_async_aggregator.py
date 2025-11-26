import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from ruth.server.async_aggregator import AsyncAggregator

# Mock ClientUpdate Protobuf Object
class MockClientUpdate:
    def __init__(self, round_id, device_id="device1", scalar=1.0):
        self.round_id = round_id
        self.device_id = device_id
        self.scalar = scalar
        self.signature = b"dummy_signature"
        self.attestation_token = b"dummy_token"
        self.seed_id = 123
        self.loss = 0.5
        
    def SerializeToString(self):
        return f"{self.device_id}:{self.scalar}".encode('utf-8')

@pytest.mark.asyncio
async def test_submit_update():
    # Mock Redis
    mock_redis = AsyncMock()
    mock_pipeline = AsyncMock()
    
    # Configure pipeline() to return a context manager
    mock_redis.pipeline = MagicMock()
    mock_redis.pipeline.return_value.__aenter__.return_value = mock_pipeline
    
    # Mock ConnectionPool
    mock_pool = MagicMock()
    
    with patch('ruth.server.async_aggregator.redis.ConnectionPool.from_url', return_value=mock_pool), \
         patch('ruth.server.async_aggregator.redis.Redis', return_value=mock_redis):
        
        aggregator = AsyncAggregator("redis://localhost", k_threshold=10)
        
        update = MockClientUpdate(round_id=1)
        
        # Test Submit
        success = await aggregator.submit_update(update)
        assert success is True
        
        # Verify Redis calls
        mock_pipeline.lpush.assert_called_once()
        mock_pipeline.incr.assert_called_once()
        mock_pipeline.execute.assert_called_once()
        
        # Verify arguments
        args, _ = mock_pipeline.lpush.call_args
        assert args[0] == "ruth:round:1:updates"
        assert args[1] == update.SerializeToString()
        
        args, _ = mock_pipeline.incr.call_args
        assert args[0] == "ruth:round:1:count"

@pytest.mark.asyncio
async def test_worker_aggregation_trigger():
    # Mock Redis
    mock_redis = AsyncMock()
    mock_pipeline = AsyncMock()
    mock_redis.pipeline = MagicMock()
    mock_redis.pipeline.return_value.__aenter__.return_value = mock_pipeline
    
    # Mock ConnectionPool
    mock_pool = MagicMock()
    mock_pool.disconnect = AsyncMock()
    
    # Setup mock data for worker loop
    # 1. scan() returns (cursor, keys)
    # First call returns (b'0', [key]) to simulate finding one key and finishing
    mock_redis.scan.return_value = (b'0', [b"ruth:round:1:count"])
    
    # 2. get() returns count >= threshold
    mock_redis.get.return_value = b"10" # Threshold is 10
    
    # 3. lrange() returns list of updates
    # We need to return serialized data that ParseFromString can handle
    # Since we use a dummy ruth_pb2 in the source if import fails, we should match that.
    # But wait, the source code imports ruth_pb2 or defines a dummy class.
    # The dummy class has ParseFromString.
    # We should return bytes that won't crash ParseFromString.
    # The dummy ParseFromString does nothing (pass).
    mock_redis.lrange.return_value = [b"update1", b"update2"]
    
    with patch('ruth.server.async_aggregator.redis.ConnectionPool.from_url', return_value=mock_pool), \
         patch('ruth.server.async_aggregator.redis.Redis', return_value=mock_redis):
        
        aggregator = AsyncAggregator("redis://localhost", k_threshold=10)
        
        # Start worker
        aggregator.start_worker()
        
        # Let it run for a bit
        await asyncio.sleep(0.1)
        
        # Stop worker
        await aggregator.stop_worker()
        
        # Verify Aggregation Triggered
        # Should have called lrange to get updates
        mock_redis.lrange.assert_called()
        args, _ = mock_redis.lrange.call_args
        assert args[0] == "ruth:round:1:updates"
        
        # Should have called delete to cleanup (via pipeline)
        mock_pipeline.delete.assert_any_call("ruth:round:1:updates")
        mock_pipeline.delete.assert_any_call("ruth:round:1:count")

if __name__ == "__main__":
    # Manually run async tests if pytest-asyncio not working via command line
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test_submit_update())
    loop.run_until_complete(test_worker_aggregation_trigger())
    print("All Async Aggregator tests passed!")
