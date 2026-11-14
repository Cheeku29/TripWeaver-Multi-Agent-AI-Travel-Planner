"""Finds hotel options for the destination through the Tavily MCP server."""

import json
from langchain_core.messages import AIMessage

from src.agents.prompts import HOTEL_SEARCH_QUERY
from src.config.settings import HOTEL_DATA_CHAR_LIMIT
from src.graph.state import TravelState
from src.mcp_servers.remote import tavily_mcp_search
from src.utils.async_utils import bump_llm_calls, run_async, truncate


def format_hotel_results(raw_results, limit: int = HOTEL_DATA_CHAR_LIMIT) -> str:
    """Format raw Tavily search payload into concise readable hotel options."""
    try:
        data = None
        if isinstance(raw_results, list) and raw_results:
            first = raw_results[0]
            if isinstance(first, dict) and "text" in first:
                data = json.loads(first["text"])
            elif isinstance(first, dict):
                data = first
        elif isinstance(raw_results, str):
            data = json.loads(raw_results)
        elif isinstance(raw_results, dict):
            data = raw_results

        if isinstance(data, dict) and "results" in data:
            formatted = []
            for item in data["results"][:5]:
                title = item.get("title", "").strip()
                content = item.get("content", "").strip()
                url = item.get("url", "").strip()
                if title:
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    link_text = f"\n  Booking/Info: {url}" if url else ""
                    formatted.append(f"- **{title}**: {snippet}{link_text}")
            if formatted:
                return "\n\n".join(formatted)[:limit]
    except Exception:
        pass
    return truncate(raw_results, limit)


def hotel_agent(state: TravelState):
    query = HOTEL_SEARCH_QUERY.format(user_query=state["user_query"])

    try:
        raw_results = run_async(tavily_mcp_search(query))
        hotel_results = format_hotel_results(raw_results)
    except Exception as e:
        hotel_results = f"Hotel information unavailable: {str(e)}"

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": bump_llm_calls(state),
    }
