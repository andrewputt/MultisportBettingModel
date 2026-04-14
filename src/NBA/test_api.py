import requests

resp = requests.get(
    "https://api.elections.kalshi.com/trade-api/v2/markets",
    params={"limit": 200, "status": "open", "series_ticker": "KXNBAGAME"},
)
markets = resp.json().get("markets", [])
print(f"Found {len(markets)} NBA game markets\n")
for m in markets:
    print(m["ticker"], "|", m.get("title"), "| yes_ask:", m.get("yes_ask_dollars"))
