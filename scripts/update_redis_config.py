import argparse
import redis
import os

def update_redis(round_id, model_url):
    redis_url = os.environ.get("REDIS_URL", "redis://localhost")
    redis_password = os.environ.get("REDIS_PASSWORD", None)
    
    r = redis.from_url(redis_url, password=redis_password)
    
    key = f"ruth:round:{round_id}:model_url"
    r.set(key, model_url)
    print(f"Updated {key} -> {model_url}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, help="Round ID")
    parser.add_argument("--url", type=str, required=True, help="S3 URL of the model")
    args = parser.parse_args()
    
    update_redis(args.round, args.url)
