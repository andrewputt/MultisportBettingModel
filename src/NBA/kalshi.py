import requests

resp = requests.get(
    "https://api.elections.kalshi.com/trade-api/v2/markets",
    params={"limit": 200, "status": "open", "series_ticker": "KXNBAGAME"},
)

print(resp.url)

data = resp.json()
markets = data.get("markets", [])

print(f"Found {len(markets)} NBA markets\n")

for m in markets:
    print(m.get("ticker"), "|", m.get("title"), "| yes_ask:", m.get("yes_ask_dollars"))
