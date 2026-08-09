"""
Weather Agent Bricks app.

A Databricks App that exposes a chat UI (Streamlit) for a weather agent.
The agent answers natural-language weather questions (e.g. "Will it rain
in Chicago tomorrow?", "Should I bring a jacket to Austin this weekend?")
by combining:
    - a Databricks Foundation Model API serving endpoint (a tool-calling
      LLM, e.g. "databricks-claude-sonnet-4")
    - the Open-Meteo weather MCP server (open_meteo_mcp_server.py),
      registered here as an *external* MCP tool server over streamable
      HTTP - this app does not talk to Open-Meteo directly, it only ever
      talks to the deployed MCP server's tools.

Agent loop, per user turn:
    1. Open one MCP session against WEATHER_MCP_SERVER_URL and list its
       tools (geocode_location, get_current_weather, get_forecast,
       get_hourly_forecast, get_air_quality, get_weather_alerts).
    2. Send the conversation + tool schemas to the LLM.
    3. If the LLM requests tool call(s), execute them against the open
       MCP session, append the results as tool messages, and go back to
       step 2 (bounded by MAX_TOOL_ROUNDS).
    4. Once the LLM responds without requesting a tool call, show that
       as the final answer.

Deploy this as its own Databricks App (separate from the weather MCP
server app), with WEATHER_MCP_SERVER_URL pointed at the MCP server app's
deployed URL. See app.yaml for the run command and configuration.

Run locally:
    streamlit run app.py
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import streamlit as st
from databricks.sdk import WorkspaceClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-agent-app")

# URL of the deployed Open-Meteo weather MCP server, e.g.
# "https://weather-mcp-server-<workspace>.databricksapps.com/mcp"
MCP_SERVER_URL = os.environ.get("WEATHER_MCP_SERVER_URL", "")

# Databricks Foundation Model API serving endpoint used as the agent's LLM.
# Must support OpenAI-style tool calling.
MODEL_SERVING_ENDPOINT = os.environ.get("MODEL_SERVING_ENDPOINT", "databricks-claude-sonnet-4")

# Safety bound on how many LLM <-> tool round trips a single user turn may take.
MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", 5))

APP_TITLE = "Weather Agent"


def _system_prompt() -> str:
    """Build the system prompt fresh per turn so 'today'/'tomorrow'/'this weekend' resolve correctly."""
    now = datetime.now(timezone.utc)
    return (
        "You are a helpful, concise weather assistant. You have tools that call a "
        "live weather MCP server (backed by Open-Meteo) - use them rather than "
        "guessing, for any question about current conditions, forecasts, or "
        "whether to expect rain/snow/wind/heat/cold at a specific place or time.\n\n"
        f"The current UTC date and time is {now.isoformat()}. Use this to resolve "
        "relative dates like 'today', 'tomorrow', or 'this weekend' into actual "
        "dates before deciding how many forecast days to request.\n\n"
        "If a question names a place with no state/country and it's ambiguous, "
        "make a reasonable assumption (most populous match) rather than asking "
        "a clarifying question, but mention the assumption in your answer.\n\n"
        "When you answer, be direct and practical (e.g. answer 'yes, bring a "
        "jacket' or 'no, you won't need one', not just raw numbers), and briefly "
        "justify it with the specific conditions you found."
    )


@st.cache_resource(show_spinner=False)
def _workspace_client() -> WorkspaceClient:
    return WorkspaceClient()


def _get_openai_client() -> OpenAI:
    """
    OpenAI-compatible client wired to this workspace's Model Serving endpoints.

    Built directly (base_url + bearer token) rather than via
    WorkspaceClient.serving_endpoints.get_open_ai_client() - that helper has
    been deprecated/removed across databricks-sdk versions (Databricks now
    points people at dedicated packages instead), so it isn't reliably
    present. This approach works the same whether the app authenticates via
    OAuth (the usual mode for Databricks Apps) or a PAT, and matches the
    pattern in Databricks' own "Query the Foundation Model API with the
    OpenAI SDK" docs.
    """
    w = _workspace_client()
    host = w.config.host.rstrip("/")
    # w.config.authenticate() returns the auth headers Databricks would use for
    # a raw REST call, e.g. {"Authorization": "Bearer <token>"} - reuse that
    # token here so we always get one valid for however this app is authenticated.
    auth_headers = w.config.authenticate()
    token = auth_headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        raise RuntimeError(
            "Could not obtain a Databricks auth token from WorkspaceClient - "
            "check the app's configured authentication (OAuth/service principal or PAT)."
        )
    return OpenAI(base_url=f"{host}/serving-endpoints", api_key=token)


def _mcp_tool_to_openai_schema(tool) -> dict:
    """Convert an MCP Tool definition into an OpenAI tool-calling schema."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def _tool_result_to_text(result) -> str:
    """Flatten an MCP CallToolResult's content blocks into a string for the LLM."""
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(str(block))
    text = "\n".join(parts)
    if getattr(result, "isError", False):
        return f"ERROR calling tool: {text}"
    return text


async def _run_agent_turn(history: list[dict]) -> str:
    """
    Run one full agent turn: open an MCP session, discover tools, and loop
    with the LLM (executing any requested tool calls) until it returns a
    final, non-tool-call answer. Returns that answer as plain text.
    """
    if not MCP_SERVER_URL:
        return (
            "The weather MCP server URL isn't configured (WEATHER_MCP_SERVER_URL "
            "is empty). Set it to the deployed weather MCP server's URL and try again."
        )

    openai_client = _get_openai_client()

    async with streamable_http_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            openai_tools = [_mcp_tool_to_openai_schema(t) for t in tools_result.tools]

            messages = [{"role": "system", "content": _system_prompt()}, *history]

            for _ in range(MAX_TOOL_ROUNDS):
                response = openai_client.chat.completions.create(
                    model=MODEL_SERVING_ENDPOINT,
                    messages=messages,
                    tools=openai_tools,
                )
                choice = response.choices[0]
                message = choice.message

                tool_calls = getattr(message, "tool_calls", None)
                if not tool_calls:
                    return message.content or ""

                # Record the assistant's tool-call request, then execute each
                # call against the open MCP session and feed the results back.
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    logger.info("Calling MCP tool %s with args %s", tc.function.name, args)
                    try:
                        result = await session.call_tool(tc.function.name, args)
                        result_text = _tool_result_to_text(result)
                    except Exception as exc:  # noqa: BLE001 - surface tool failures to the LLM
                        logger.exception("MCP tool call failed: %s", tc.function.name)
                        result_text = f"ERROR calling tool: {exc}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_text,
                        }
                    )

            return (
                "I wasn't able to finish looking that up within the allowed number "
                "of tool calls - try narrowing the question (e.g. a single city/day)."
            )


def _ask(history: list[dict]) -> str:
    """Sync wrapper around the async agent turn, for use inside Streamlit."""
    try:
        return asyncio.run(_run_agent_turn(history))
    except Exception as exc:  # noqa: BLE001 - show a clean error in the UI
        logger.exception("Agent turn failed")
        return f"Sorry, something went wrong answering that: {exc}"


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="\U0001F324\uFE0F")
    st.title(APP_TITLE)
    st.caption(
        "Ask about current conditions, forecasts, or what to wear/pack - "
        "backed by a live weather MCP server."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] in ("user", "assistant"):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("e.g. Will it rain in Chicago tomorrow?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Checking the weather..."):
                answer = _ask(st.session_state.messages)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
