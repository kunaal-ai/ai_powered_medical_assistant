import json

with open("logs/app_log.jsonl", "r") as f:
    for line in f:
        log = json.loads(line)
        print("\n🕒", log["timestamp"])
        print("🩺 Prediction:", log["prediction"])
        print("💬 Q:", log["qa"]["question"])
        print("💬 A:", log["qa"]["answer"])
        print("-" * 50)
