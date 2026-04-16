"""
fetch_weather.py
─────────────────────────────────────────────────────────────────────────────
Pulls hourly weather for every stadium in STADIUM_CONFIG from Open-Meteo.
Calculates Wind Factor (OUT/IN/CROSS/CALM) and predicts if retractable 
roofs are CLOSED based on temperature and precipitation.

Includes full MLB Team Abbreviation mapping (MLB Stats API & FanGraphs aliases).

Usage: python fetch_weather.py --date 2026-04-14
"""

import argparse
import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
import requests
from dotenv import load_dotenv

# ── env & paths ──────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

STADIUM_CONFIG = {
"Angel Stadium": {"lat": 33.800, "lon": -117.883, "azimuth": 55, "has_roof": False},
"Chase Field": {"lat": 33.445, "lon": -112.067, "azimuth": 343, "has_roof": True},
"Truist Park": {"lat": 33.891, "lon": -84.468, "azimuth": 120, "has_roof": False},
"Camden Yards": {"lat": 39.284, "lon": -76.622, "azimuth": 45, "has_roof": False},
"Fenway Park": {"lat": 42.346, "lon": -71.098, "azimuth": 45, "has_roof": False},
"Wrigley Field": {"lat": 41.948, "lon": -87.656, "azimuth": 45, "has_roof": False},
"Guaranteed Rate Field": {"lat": 41.830, "lon": -87.634, "azimuth": 165, "has_roof": False},
"Great American Ball Park":{"lat": 39.097, "lon": -84.507, "azimuth": 145, "has_roof": False},
"Progressive Field": {"lat": 41.496, "lon": -81.685, "azimuth": 45, "has_roof": False},
"Coors Field": {"lat": 39.756, "lon": -104.994, "azimuth": 4, "has_roof": False},
"Comerica Park": {"lat": 42.339, "lon": -83.049, "azimuth": 140, "has_roof": False},
"Daikin Park": {"lat": 29.757, "lon": -95.356, "azimuth": 0, "has_roof": True},
"Kauffman Stadium": {"lat": 39.052, "lon": -94.481, "azimuth": 40, "has_roof": False},
"Dodger Stadium": {"lat": 34.074, "lon": -118.240, "azimuth": 200, "has_roof": False},
"loanDepot park": {"lat": 25.778, "lon": -80.220, "azimuth": 348, "has_roof": True},
"American Family Field": {"lat": 43.028, "lon": -87.971, "azimuth": 125, "has_roof": True},
"Target Field": {"lat": 44.982, "lon": -93.278, "azimuth": 45, "has_roof": False},
"Citi Field": {"lat": 40.757, "lon": -73.846, "azimuth": 50, "has_roof": False},
"Yankee Stadium": {"lat": 40.830, "lon": -73.926, "azimuth": 200, "has_roof": False},
"Sutter Health Park": {"lat": 38.581, "lon": -121.505, "azimuth": 60, "has_roof": False},
"Citizens Bank Park": {"lat": 39.906, "lon": -75.166, "azimuth": 125, "has_roof": False},
"PNC Park": {"lat": 40.447, "lon": -80.006, "azimuth": 110, "has_roof": False},
"Petco Park": {"lat": 32.708, "lon": -117.157, "azimuth": 160, "has_roof": False},
"Oracle Park": {"lat": 37.778, "lon": -122.389, "azimuth": 75, "has_roof": False},
"T-Mobile Park": {"lat": 47.591, "lon": -122.332, "azimuth": 15, "has_roof": True},
"Busch Stadium": {"lat": 38.623, "lon": -90.193, "azimuth": 145, "has_roof": False},
"Tropicana Field": {"lat": 27.768, "lon": -82.654, "azimuth": 0, "has_roof": True},
"Globe Life Field": {"lat": 32.751, "lon": -97.083, "azimuth": 35, "has_roof": True},
"Rogers Centre": {"lat": 43.641, "lon": -79.389, "azimuth": 0, "has_roof": True},
"Nationals Park": {"lat": 38.873, "lon": -77.007, "azimuth": 160, "has_roof": False},

}

# ── Team → Stadium Mapping (SSOT) ──────────────────────────────────────────────
# SILENT WIND FIX: The MLB Stats API returns short-form abbreviations
# (SD, TB, SF, KC, WSH) while older references use FanGraphs/BR style
# (SDP, TBR, SFG, KCR, WSN). Both are included here so ps.team always
# resolves to a valid stadium key regardless of which feed is the source.
# Primary keys → MLB Stats API abbreviations (what data_fetcher.py receives)
# Alias keys → FanGraphs / Baseball-Reference / DraftKings alternates

TEAM_STADIUM_MAP = {

# ── Primary: MLB Stats API abbreviations ──────────────────────────────────
"ARI": "Chase Field",
"ATL": "Truist Park",
"BAL": "Camden Yards",
"BOS": "Fenway Park",
"CHC": "Wrigley Field",
"CWS": "Guaranteed Rate Field",
"CIN": "Great American Ball Park",
"CLE": "Progressive Field",
"COL": "Coors Field",
"DET": "Comerica Park",
"HOU": "Daikin Park",
"KC": "Kauffman Stadium",
"LAA": "Angel Stadium",
"LAD": "Dodger Stadium",
"MIA": "loanDepot park",
"MIL": "American Family Field",
"MIN": "Target Field",
"NYM": "Citi Field",
"NYY": "Yankee Stadium",
"OAK": "Sutter Health Park",
"PHI": "Citizens Bank Park",
"PIT": "PNC Park",
"SD": "Petco Park",
"SF": "Oracle Park",
"SEA": "T-Mobile Park",
"STL": "Busch Stadium",
"TB": "Tropicana Field",
"TEX": "Globe Life Field",
"TOR": "Rogers Centre",
"WSH": "Nationals Park",

# ── Aliases: FanGraphs / Baseball-Reference / DraftKings alternates ───────
"AZ": "Chase Field",
"KCR": "Kauffman Stadium",
"SDP": "Petco Park",
"SFG": "Oracle Park",
"TBR": "Tropicana Field",
"WSN": "Nationals Park",
"CHW": "Guaranteed Rate Field",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _angular_diff(a, b):
    diff = (a - b) % 360
    return diff - 360 if diff > 180 else diff

def predict_roof_closed(stadium_name, temp_c, is_raining, target_date):
    temp_f = (temp_c * 9/5) + 32
    if is_raining: return True
    if temp_f < 65 or temp_f > 85: return True
    if stadium_name in ["Daikin Park", "loanDepot park", "Tropicana Field"]: return True
    if stadium_name == "Rogers Centre":
        month = int(target_date.split("-")[1])
        if month < 5: return True
    return False

def compute_wind_factor(speed_mph, direction_deg, cf_azimuth):
    if speed_mph < 5.0:
        return {"category": "CALM", "magnitude": 0.0, "angle": 0.0}
    wind_to = (direction_deg + 180) % 360
    angle = _angular_diff(wind_to, cf_azimuth)
    component = speed_mph * math.cos(math.radians(angle))
    if abs(angle) <= 45.0: category = "OUT"
    elif abs(angle) >= 135.0: category = "IN"
    else: category = "CROSS"
    return {"category": category, "magnitude": round(abs(component), 1), "angle": round(angle, 1)}

# ── Core Runner ──────────────────────────────────────────────────────────────

def run(target_date):
    print(f"[fetch_weather] Processing {target_date}...")
    results = {}

    # We iterate through STADIUM_CONFIG to build the weather database
    for name, cfg in STADIUM_CONFIG.items():
        print(f"  → {name}", end="", flush=True)
        try:
            params = {
                "latitude": cfg["lat"], "longitude": cfg["lon"],
                "hourly": "windspeed_10m,winddirection_10m,temperature_2m,precipitation",
                "windspeed_unit": "mph", "timezone": "auto",
                "start_date": target_date, "end_date": target_date,
            }
            resp = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()["hourly"]

            enriched = []
            for i in range(len(data["time"])):
                temp_c, precip = data["temperature_2m"][i], data["precipitation"][i]
                speed, direction = data["windspeed_10m"][i], data["winddirection_10m"][i]
                is_closed = predict_roof_closed(name, temp_c, precip > 0, target_date) if cfg["has_roof"] else False
                wf = compute_wind_factor(speed, direction, cfg["azimuth"]) if not is_closed else {"category": "CALM", "magnitude": 0.0, "angle": 0.0}
                enriched.append({"time": data["time"][i], "temp_f": round((temp_c * 9/5) + 32, 1), "is_closed": is_closed, "wind": wf})

            game_hours = [h for h in enriched if 13 <= int(h["time"].split("T")[1][:2]) <= 22]
            if game_hours:
                cats = [h["wind"]["category"] for h in game_hours]
                dominant = max(set(cats), key=cats.count)
                avg_speed = round(sum(h["wind"]["magnitude"] for h in game_hours) / len(game_hours), 1)
            else:
                dominant, avg_speed = "UNKNOWN", 0.0

            results[name] = {"stadium": name, "has_roof": cfg["has_roof"], "summary": {"factor": dominant, "speed_mph": avg_speed}, "hourly": enriched}
            print(f"  [{dominant} @ {avg_speed} mph]")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Map teams to their stadium results for easy model lookups
    final_output = {"date": target_date, "stadiums": results, "team_map": {}}
    for abr, s_name in TEAM_STADIUM_MAP.items():
        if s_name in results:
            final_output["team_map"][abr] = results[s_name]["summary"]

    out_path = RAW_DIR / "weather.json"
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[fetch_weather] Saved to {out_path} (including {len(TEAM_STADIUM_MAP)} team aliases)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    run(args.date)