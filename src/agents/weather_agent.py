"""Fetches current weather and forecast for the extracted destination."""

import json
from langchain_core.messages import AIMessage

from src.agents.prompts import WEATHER_RESULTS_TEMPLATE
from src.config.settings import WEATHER_DATA_CHAR_LIMIT
from src.graph.state import TravelState
from src.agents.destination import extract_destination
from src.mcp_servers.local import forecast_mcp_search, weather_mcp_search
from src.utils.async_utils import run_async, truncate


def format_forecast_data(forecast_raw) -> str:
    """Compact raw forecast into daily highlights rather than 40 3-hour entries."""
    try:
        f_text = None
        if isinstance(forecast_raw, list) and forecast_raw and isinstance(forecast_raw[0], dict) and "text" in forecast_raw[0]:
            f_text = forecast_raw[0]["text"]
        elif isinstance(forecast_raw, str):
            f_text = forecast_raw

        if f_text and f_text.strip().startswith("{"):
            data = json.loads(f_text)
            forecast_list = data.get("forecast", [])
            if forecast_list:
                daily = []
                # Sample every 8 entries (roughly once per 24 hours)
                for entry in forecast_list[::8][:5]:
                    dt = entry.get("datetime", "")
                    temp = entry.get("temperature", "")
                    cond = entry.get("weather", "")
                    daily.append(f"- {dt}: {temp}°C, {cond}")
                if daily:
                    return "\n".join(daily)
        return truncate(f_text or forecast_raw, 1000)
    except Exception:
        return truncate(forecast_raw, 1000)


def weather_agent(state: TravelState):
    try:
        city = extract_destination(state["user_query"])
        weather_data = run_async(weather_mcp_search(city))
        forecast_data = run_async(forecast_mcp_search(city))

        w_text = (
            weather_data[0]["text"]
            if isinstance(weather_data, list) and weather_data and isinstance(weather_data[0], dict) and "text" in weather_data[0]
            else str(weather_data)
        )
        f_text = format_forecast_data(forecast_data)

        weather_results = WEATHER_RESULTS_TEMPLATE.format(
            weather_data=truncate(w_text, 800),
            forecast_data=truncate(f_text, 1000),
        )[:WEATHER_DATA_CHAR_LIMIT]

    except Exception as e:
        weather_results = f"Weather information unavailable: {str(e)}"

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(content="Weather information fetched")
        ],
    }
