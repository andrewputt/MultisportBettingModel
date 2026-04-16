import requests
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Setup pathing to your root .env
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def get_mlb_props_refined(target_date=None):
    if target_date is None:
        target_date = datetime(2026, 4, 16) 
        
    date_str = target_date.strftime("%y%b%d").lower()
    events_url = "https://api.elections.kalshi.com/trade-api/v2/events"
    
    # ⚾ THE FIX: Loop through the distinct Kalshi MLB betting categories
    target_series = [
        "KXMLBGAME",   # Moneyline / Winner
        "KXMLBSPREAD", # Run Line / Spreads
        "KXMLBTOTAL"   # Over/Under Totals
        # Note: You can add "KXMLBYRFI" or "KXMLBF5" here later if needed
    ]
    
    all_data = []
    
    print(f"Fetching Full MLB Board for {target_date.strftime('%b %d, %Y')}...\n")

    for series in target_series:
        params = {
            "limit": 100,
            "status": "open",
            "series_ticker": series,
            "with_nested_markets": "true" # This correctly pulls the sub-options for Spreads/Totals
        }

        try:
            resp = requests.get(events_url, params=params)
            resp.raise_for_status()
            events = resp.json().get("events", [])

            # Isolate today's events for the current series
            today_events = [e for e in events if date_str in e.get("event_ticker", "").lower()]
            
            for event in today_events:
                event_title = event.get("title")
                markets = event.get("markets", [])
                
                print(f"🏟️ Event Group: {event_title} [{series}]")
                
                for m in markets:
                    ticker = m.get("ticker")
                    title = m.get("title")
                    subtitle = m.get("subtitle", "")
                    price = m.get("yes_ask_dollars")
                    
                    display_title = f"{title} {subtitle}".strip()
                    print(f"   ↳ {ticker} | {display_title} | Price: ${price}")
                    
                    all_data.append(m)
            
            # Pacing to avoid rate limits between series loops
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"❌ Network/API Error fetching {series}: {e}")
        except Exception as e:
            print(f"❌ Error fetching {series}: {e}")

    print(f"\n✅ Total Markets Captured: {len(all_data)}")
    print("-" * 50)
    return all_data

if __name__ == "__main__":
    get_mlb_props_refined()