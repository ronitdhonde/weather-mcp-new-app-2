"""
Open-Meteo weather engine backing the Open-Meteo weather MCP server.

This module is a thin wrapper around Open-Meteo's free, no-API-key-required
REST APIs (https://open-meteo.com/en/docs) - geocoding, forecast, and air
quality. All HTTP calls, param-building, and response parsing live here so
open_meteo_mcp_server.py's @mcp.tool functions can stay thin pass-throughs,
mirroring how alpaca_mcp_server.py delegates to alpaca_broker.py.

No credentials are required - Open-Meteo's non-commercial endpoints are
open. If you later move to a commercial Open-Meteo plan or a different
provider that requires an API key, add a `_secret()`-style lookup here
(see alpaca_broker.py's Databricks-secret-scope pattern) and thread the
key into the request params - the function signatures below don't need
to change.

Swap-in-a-different-provider note: to point this at a different weather
API instead (e.g. NOAA, WeatherAPI, Tomorrow.io), keep the same function
signatures below (geocode, get_current_weather, get_forecast,
get_hourly_forecast, get_air_quality, get_weather_alerts) and replace the
requests.get(...) calls inside each with that provider's SDK/API - the
MCP surface in open_meteo_mcp_server.py does not need to change.
"""

import os
from functools import lru_cache

import requests

GEOCODING_BASE_URL = os.environ.get(
    "OPEN_METEO_GEOCODING_URL", "https://geocoding-api.open-meteo.com/v1/search"
)
FORECAST_BASE_URL = os.environ.get(
    "OPEN_METEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast"
)
AIR_QUALITY_BASE_URL = os.environ.get(
    "OPEN_METEO_AIR_QUALITY_URL", "https://air-quality-api.open-meteo.com/v1/air-quality"
)

REQUEST_TIMEOUT_SECONDS = 10

# WMO weather interpretation codes -> human-readable condition text.
# https://open-meteo.com/en/docs (see "WMO Weather interpretation codes")
_WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_SEVERE_WEATHER_CODES = {95, 96, 99}
_HIGH_WIND_THRESHOLD_MPH = 40
_HIGH_PRECIP_PROBABILITY_PERCENT = 80


def _describe_weather_code(code) -> str:
    """Translate a WMO weather code into human-readable condition text."""
    try:
        return _WMO_CODES.get(int(code), f"Unknown (code {code})")
    except (TypeError, ValueError):
        return "Unknown"


def _parse_lat_lon(location: str):
    """If `location` is already a 'lat,lon' pair, parse and return it, else None."""
    parts = [p.strip() for p in location.split(",")]
    if len(parts) != 2:
        return None
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    return None


@lru_cache(maxsize=256)
def geocode(query: str) -> dict:
    """
    Resolve a free-text place name to coordinates via Open-Meteo's free
    geocoding API. Cached, since the same place name will map to the same
    coordinates for the lifetime of the process.
    """
    resp = requests.get(
        GEOCODING_BASE_URL,
        params={"name": query, "count": 1, "language": "en", "format": "json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        raise ValueError(f"No location found matching '{query}'")
    top = results[0]
    return {
        "name": top.get("name"),
        "admin1": top.get("admin1"),
        "country": top.get("country"),
        "latitude": top.get("latitude"),
        "longitude": top.get("longitude"),
        "timezone": top.get("timezone"),
    }


def resolve_location(location: str) -> dict:
    """
    Resolve a location string to a dict with at least name/latitude/longitude.
    Accepts either a 'lat,lon' string or a free-text place name (geocoded
    via `geocode`).
    """
    coords = _parse_lat_lon(location)
    if coords:
        lat, lon = coords
        return {"name": location, "latitude": lat, "longitude": lon, "timezone": None}
    return geocode(location)


def get_current_weather(location: str) -> dict:
    """
    Get current weather conditions for a location from Open-Meteo's
    forecast API's `current` block.
    """
    place = resolve_location(location)
    resp = requests.get(
        FORECAST_BASE_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": ",".join(
                [
                    "temperature_2m",
                    "apparent_temperature",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "weather_code",
                    "is_day",
                ]
            ),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    current = data.get("current", {})

    return {
        "location": place.get("name"),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "timezone": data.get("timezone", place.get("timezone")),
        "temperature_f": current.get("temperature_2m"),
        "feels_like_f": current.get("apparent_temperature"),
        "humidity_percent": current.get("relative_humidity_2m"),
        "wind_mph": current.get("wind_speed_10m"),
        "wind_direction_deg": current.get("wind_direction_10m"),
        "condition": _describe_weather_code(current.get("weather_code")),
        "is_day": bool(current.get("is_day")),
        "observed_at": current.get("time"),
    }


def get_forecast(location: str, days: int = 7) -> dict:
    """Get a daily weather forecast (1-16 days) for a location."""
    days = max(1, min(int(days), 16))
    place = resolve_location(location)
    resp = requests.get(
        FORECAST_BASE_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                ]
            ),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "forecast_days": days,
            "timezone": "auto",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])

    def _at(field, i):
        values = daily.get(field) or []
        return values[i] if i < len(values) else None

    forecast = [
        {
            "date": date,
            "condition": _describe_weather_code(_at("weather_code", i)),
            "high_f": _at("temperature_2m_max", i),
            "low_f": _at("temperature_2m_min", i),
            "precipitation_probability_percent": _at("precipitation_probability_max", i),
            "precipitation_in": _at("precipitation_sum", i),
            "wind_max_mph": _at("wind_speed_10m_max", i),
        }
        for i, date in enumerate(dates)
    ]

    return {
        "location": place.get("name"),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "timezone": data.get("timezone", place.get("timezone")),
        "forecast": forecast,
    }


def get_hourly_forecast(location: str, hours: int = 24) -> dict:
    """Get an hourly weather forecast (1-384 hours) for a location."""
    hours = max(1, min(int(hours), 384))
    place = resolve_location(location)
    resp = requests.get(
        FORECAST_BASE_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "weather_code",
                    "precipitation_probability",
                    "wind_speed_10m",
                ]
            ),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "forecast_hours": hours,
            "timezone": "auto",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    def _at(field, i):
        values = hourly.get(field) or []
        return values[i] if i < len(values) else None

    forecast = [
        {
            "time": t,
            "temperature_f": _at("temperature_2m", i),
            "condition": _describe_weather_code(_at("weather_code", i)),
            "precipitation_probability_percent": _at("precipitation_probability", i),
            "wind_mph": _at("wind_speed_10m", i),
        }
        for i, t in enumerate(times)
    ]

    return {
        "location": place.get("name"),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "timezone": data.get("timezone", place.get("timezone")),
        "forecast": forecast,
    }


def get_air_quality(location: str) -> dict:
    """Get current air quality data for a location from Open-Meteo's air quality API."""
    place = resolve_location(location)
    resp = requests.get(
        AIR_QUALITY_BASE_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "us_aqi,pm2_5,pm10,ozone",
            "timezone": "auto",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    current = data.get("current", {})

    return {
        "location": place.get("name"),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "us_aqi": current.get("us_aqi"),
        "pm2_5": current.get("pm2_5"),
        "pm10": current.get("pm10"),
        "ozone": current.get("ozone"),
        "observed_at": current.get("time"),
    }


def get_weather_alerts(location: str) -> dict:
    """
    Derive simple heuristic alert strings (storms, high wind, heavy
    precipitation) from the next 24 hours of hourly forecast data. Open-Meteo
    has no dedicated alerts endpoint, so this is computed rather than fetched.
    """
    place = resolve_location(location)
    resp = requests.get(
        FORECAST_BASE_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "hourly": "weather_code,wind_speed_10m,precipitation_probability",
            "wind_speed_unit": "mph",
            "forecast_hours": 24,
            "timezone": "auto",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})
    codes = hourly.get("weather_code") or []
    winds = hourly.get("wind_speed_10m") or []
    precip_probs = hourly.get("precipitation_probability") or []

    alerts = []
    if any(c in _SEVERE_WEATHER_CODES for c in codes):
        alerts.append("Thunderstorms expected within the next 24 hours.")
    if winds and max(winds) >= _HIGH_WIND_THRESHOLD_MPH:
        alerts.append(f"High winds expected, up to {max(winds):.0f} mph.")
    if precip_probs and max(precip_probs) >= _HIGH_PRECIP_PROBABILITY_PERCENT:
        alerts.append("High chance of heavy precipitation in the next 24 hours.")

    return {
        "location": place.get("name"),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "alerts": alerts,
    }
