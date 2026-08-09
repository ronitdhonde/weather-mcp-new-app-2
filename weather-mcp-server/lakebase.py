"""
PLACEHOLDER FILE - replace with your project's existing lakebase.py.

open_meteo_mcp_server.py imports `lakebase` for the weather_watchlist
tools (add_to_weather_watchlist / get_weather_watchlist /
remove_from_weather_watchlist), exactly the way alpaca_mcp_server.py
imports it for the stock watchlist tools. It needs to expose:

    run_query(sql: str, params: tuple) -> list[dict]
    run_write(sql: str, params: tuple) -> int   # rows affected

Reuse the same lakebase.py that already backs alpaca_mcp_server.py's
watchlist tools - copy that file into this folder in place of this one.
This stub deliberately raises so a forgotten copy fails loudly at import
time instead of silently.
"""

raise NotImplementedError(
    "lakebase.py is a placeholder - replace this file with your project's "
    "real lakebase.py (the one alpaca_mcp_server.py already imports) before deploying."
)
