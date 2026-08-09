"""
Open-Meteo weather MCP server.

Exposes weather tools over MCP (Model Context Protocol) so a Databricks
Agent Bricks agent (or any MCP client) can call them like any other tool:
    - geocode_location(query)
    - get_current_weather(location)
    - get_forecast(location, days)
    - get_hourly_forecast(location, hours)
    - get_air_quality(location)
    - get_weather_alerts(location)
    - add_to_weather_watchlist(location)
    - get_weather_watchlist(limit)
    - remove_from_weather_watchlist(location_query)

These tools are backed by Open-Meteo (https://open-meteo.com), a free,
no-API-key-required weather API, so students can safely wire an Agent
Bricks agent to real weather data without signing up for a paid provider
or managing API keys/secrets.

"location" accepts either:
    - a place name / city string, e.g. "Hoboken, NJ" or "Tokyo" - it is
      resolved to coordinates via Open-Meteo's free geocoding API, or
    - a "lat,lon" string, e.g. "40.7439,-74.0324" - used directly.

All the HTTP calls and response parsing live in open_meteo_broker.py -
these @mcp.tool functions are thin pass-throughs, mirroring how
alpaca_mcp_server.py delegates to alpaca_broker.py.

The watchlist tools are the one exception: like alpaca_mcp_server.py's
stock watchlist, they talk to Lakebase directly from this file (see the
`weather_watchlist` table) rather than through the broker, and use the
same X-Forwarded-User header pattern to scope entries per end user.

Swap-in-a-different-provider note: to point this at a different weather
API instead (e.g. NOAA, WeatherAPI, Tomorrow.io), keep the same tool
signatures below and replace the open_meteo_broker.* calls inside each
tool with calls to that provider's SDK/API - the MCP surface for the
agent does not need to change.

Deploy this as its own Databricks App (same app.yaml + FastMCP entrypoint
pattern documented at
https://docs.databricks.com/aws/en/agents/mcp-tools/custom-mcp), separate
from the dashboard app, so an Agent Bricks agent (or any MCP client) can
register its URL as an external MCP server.

Run locally:
    python open_meteo_mcp_server.py
"""

import os
import logging
from contextvars import ContextVar

from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

import open_meteo_broker
import lakebase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("open-meteo-weather-mcp-server")

# Context variable to store request headers for accessing end-user identity
_request_context: ContextVar[dict] = ContextVar('request_context', default={})


def _get_end_user_email() -> str:
    """Get the actual end user's email from request headers, or fallback to service principal."""
    # Try to get from X-Forwarded-User header (Databricks App context)
    headers = _request_context.get()
    forwarded_user = headers.get('x-forwarded-user')
    if forwarded_user:
        return forwarded_user

    # Fallback: use service principal (local development or non-App contexts)
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    return w.current_user.me().user_name or 'zach@dataexpert.io'


mcp = FastMCP("open-meteo-weather")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to capture HTTP headers containing end-user identity."""
    async def dispatch(self, request: Request, call_next):
        # Capture headers that Databricks injects with user identity
        headers = {
            'x-forwarded-user': request.headers.get('x-forwarded-user'),
            'x-forwarded-email': request.headers.get('x-forwarded-email'),
        }
        _request_context.set(headers)
        response = await call_next(request)
        return response


@mcp.tool
def geocode_location(query: str) -> dict:
    """
    Resolve a free-text place name to coordinates using Open-Meteo's free
    geocoding API.

    Args:
        query: Place name, e.g. "Hoboken, NJ" or "Kyoto, Japan".

    Returns:
        A dict with name, admin1 (state/region), country, latitude,
        longitude, and timezone for the best-matching location.
    """
    return open_meteo_broker.geocode(query)


@mcp.tool
def get_current_weather(location: str) -> dict:
    """
    Get the current weather conditions for a location.

    Args:
        location: A place name (e.g. "Hoboken, NJ") or a "lat,lon"
            string (e.g. "40.7439,-74.0324").

    Returns:
        A dict with location info plus temperature_f, feels_like_f,
        humidity_percent, wind_mph, wind_direction_deg, condition,
        is_day, and observed_at (ISO timestamp).
    """
    return open_meteo_broker.get_current_weather(location)


@mcp.tool
def get_forecast(location: str, days: int = 7) -> dict:
    """
    Get a daily weather forecast for a location.

    Args:
        location: A place name (e.g. "Hoboken, NJ") or a "lat,lon" string.
        days: Number of forecast days to return, 1-16 (default 7).

    Returns:
        A dict with location info and a list of daily forecasts, each
        with date, condition, high_f, low_f, precipitation_probability_percent,
        precipitation_in, and wind_max_mph.
    """
    return open_meteo_broker.get_forecast(location, days)


@mcp.tool
def get_hourly_forecast(location: str, hours: int = 24) -> dict:
    """
    Get an hourly weather forecast for a location.

    Args:
        location: A place name (e.g. "Hoboken, NJ") or a "lat,lon" string.
        hours: Number of hours ahead to return, 1-384 (default 24).

    Returns:
        A dict with location info and a list of hourly forecasts, each
        with time, temperature_f, condition, precipitation_probability_percent,
        and wind_mph.
    """
    return open_meteo_broker.get_hourly_forecast(location, hours)


@mcp.tool
def get_air_quality(location: str) -> dict:
    """
    Get current air quality data for a location.

    Args:
        location: A place name (e.g. "Hoboken, NJ") or a "lat,lon" string.

    Returns:
        A dict with location info plus us_aqi (US Air Quality Index),
        pm2_5, pm10, ozone, and observed_at (ISO timestamp).
    """
    return open_meteo_broker.get_air_quality(location)


@mcp.tool
def get_weather_alerts(location: str) -> dict:
    """
    Check whether current or near-term conditions look severe enough to
    warrant a heads-up. Open-Meteo has no dedicated alerts endpoint, so
    this derives simple heuristic flags from weather_code, wind speed,
    and precipitation over the next 24 hours.

    Args:
        location: A place name (e.g. "Hoboken, NJ") or a "lat,lon" string.

    Returns:
        A dict with location info and a list of alert strings (empty list
        if nothing notable in the next 24 hours).
    """
    return open_meteo_broker.get_weather_alerts(location)


@mcp.tool
def add_to_weather_watchlist(location: str) -> dict:
    """
    Add a location to the watchlist by geocoding it via Open-Meteo and
    storing it in the Lakebase weather_watchlist table.

    Uses the authenticated user's email as the user_id.

    Args:
        location: A place name (e.g. "Austin, TX") or a "lat,lon" string.
            This is stored verbatim as the watchlist key, so re-adding the
            same string updates the existing entry instead of duplicating it.

    Returns:
        A dict with the resolved location data and confirmation that it
        was added to the watchlist.
    """
    try:
        # Get the actual end user's email (not the service principal)
        user_email = _get_end_user_email()

        # Resolve the location (geocode if it's a place name, pass through if lat,lon)
        place = open_meteo_broker.resolve_location(location)

        # Store in Lakebase weather_watchlist table
        sql = """
        INSERT INTO weather_watchlist (email, location_query, display_name, latitude, longitude, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (email, location_query)
        DO UPDATE SET
            display_name = EXCLUDED.display_name,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            updated_at = NOW()
        """

        lakebase.run_write(
            sql,
            (
                user_email,
                location,
                place.get("name"),
                place["latitude"],
                place["longitude"],
            ),
        )

        return {
            "status": "success",
            "message": f"Added {location} to weather watchlist for {user_email}",
            "user_email": user_email,
            "location": place,
        }
    except Exception as e:
        logger.exception(f"Failed to add {location} to weather watchlist")
        return {
            "status": "error",
            "message": f"Failed to add {location} to weather watchlist: {str(e)}",
        }


@mcp.tool
def get_weather_watchlist(limit: int = 100, email: str = 'zach@dataexpert.io') -> dict:
    """
    Retrieve all locations in the authenticated user's weather watchlist
    from Lakebase.

    Uses the authenticated user's email as the user_id.

    Args:
        limit: Maximum number of entries to return (default: 100).
        email: authenticated user's email

    Returns:
        A dict with watchlist entries sorted by most recently added/updated.
    """
    try:
        sql = """
        SELECT
            location_query,
            display_name,
            latitude,
            longitude,
            updated_at
        FROM weather_watchlist
        WHERE email = %s
        ORDER BY updated_at DESC
        LIMIT %s
        """

        rows = lakebase.run_query(sql, (email, limit))

        return {
            "status": "success",
            "user_email": email,
            "count": len(rows),
            "watchlist": rows,
        }
    except Exception as e:
        logger.exception("Failed to retrieve weather watchlist")
        return {
            "status": "error",
            "message": f"Failed to retrieve weather watchlist: {str(e)}",
        }


@mcp.tool
def remove_from_weather_watchlist(location_query: str) -> dict:
    """
    Remove a location from the authenticated user's weather watchlist.

    Uses the authenticated user's email as the user_id.

    Args:
        location_query: The exact location string originally passed to
            add_to_weather_watchlist, e.g. "Austin, TX".

    Returns:
        A dict with status and confirmation message.
    """
    try:
        # Get the actual end user's email (not the service principal)
        user_email = _get_end_user_email()

        sql = """
        DELETE FROM weather_watchlist
        WHERE email = %s AND location_query = %s
        """

        rows_affected = lakebase.run_write(sql, (user_email, location_query))

        if rows_affected > 0:
            return {
                "status": "success",
                "message": f"Removed {location_query} from weather watchlist",
                "location_query": location_query,
                "user_email": user_email,
            }
        else:
            return {
                "status": "not_found",
                "message": f"{location_query} was not in the weather watchlist",
                "location_query": location_query,
                "user_email": user_email,
            }
    except Exception as e:
        logger.exception(f"Failed to remove {location_query} from weather watchlist")
        return {
            "status": "error",
            "message": f"Failed to remove {location_query} from weather watchlist: {str(e)}",
        }


if __name__ == "__main__":
    # Add middleware to capture request headers for end-user identity
    # This must be done before mcp.run() is called
    if hasattr(mcp, 'app') and mcp.app is not None:
        mcp.app.add_middleware(RequestContextMiddleware)

    # Databricks Apps route external HTTP traffic to this port via app.yaml;
    # streamable-http is the transport Databricks' MCP client/gateway expects
    # (see the "Host your own MCP" doc linked in the module docstring above).
    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", 8000)))
    mcp.run(transport="http", host="0.0.0.0", port=port)
