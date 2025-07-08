import requests
import pandas as pd
from datetime import datetime, timedelta

# Constants
DAYS_BACK = 30
# Updated endpoint with query parameter to specify chain 'solana'
API_URL = "https://api.dexscreener.com/latest/dex/pairs?chain=solana"
CUTOFF = datetime.utcnow() - timedelta(days=DAYS_BACK)

# Fetch from Dexscreener
resp = requests.get(API_URL)
if resp.status_code != 200:
    print(f"Error: Received status code {resp.status_code}")
    print("Response text:", resp.text)
    exit(1)
if not resp.text.strip():
    print("Error: Empty response from API.")
    exit(1)
try:
    data = resp.json()
except Exception as e:
    print("Error decoding JSON:", e)
    print("Response text:", resp.text)
    exit(1)

# Parse data
records = []
for pair in data.get("pairs", []):
    try:
        pair_created = datetime.strptime(pair["pairCreatedAt"], "%Y-%m-%dT%H:%M:%S.%fZ")
        if pair_created < CUTOFF:
            continue

        records.append({
            "mint_address": pair.get("pairAddress"),
            "symbol": pair.get("baseToken", {}).get("symbol", ""),
            "mint_time": pair_created,
            "price_usd": float(pair.get("priceUsd", 0)),
            "volume_24h": float(pair.get("volume", {}).get("h24", 0)),
            "liquidity_usd": float(pair.get("liquidity", {}).get("usd", 0)),
            "market_cap_usd": float(pair.get("marketCapUsd", 0)),
            "age_hours": (datetime.utcnow() - pair_created).total_seconds() / 3600.0
        })

    except Exception as e:
        print("Error parsing a record:", e)

# Convert and save
df = pd.DataFrame(records)
df.to_csv("general_sol_tokens_last30d.csv", index=False)
print("Saved dataset with", len(df), "entries")
