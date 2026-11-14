"""Formats everything the other agents produced into the user-facing answer."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.prompts import FINAL_AGENT_PROMPT, FINAL_SYSTEM_PROMPT
from src.clients.llm import get_llm
from src.config.settings import (
    HOTEL_DATA_CHAR_LIMIT,
    ITINERARY_DATA_CHAR_LIMIT,
    WEATHER_DATA_CHAR_LIMIT,
)
from src.graph.state import TravelState
from src.utils.async_utils import bump_llm_calls, truncate


def final_agent(state: TravelState):
    flight_results = truncate(state.get("flight_results", ""), 2500)
    hotel_results = truncate(state.get("hotel_results", ""), HOTEL_DATA_CHAR_LIMIT)
    weather_results = truncate(state.get("weather_results", ""), WEATHER_DATA_CHAR_LIMIT)
    itinerary = truncate(state.get("itinerary", ""), ITINERARY_DATA_CHAR_LIMIT)

    prompt = FINAL_AGENT_PROMPT.format(
        user_query=state["user_query"],
        flight_results=flight_results,
        hotel_results=hotel_results,
        weather_results=weather_results,
        itinerary=itinerary,
    )

    try:
        response = get_llm().invoke([
            SystemMessage(content=FINAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        messages = [response]
    except Exception as e:
        fallback_msg = (
            f"## Trip Plan\n\n"
            f"**Flight Details:**\n{flight_results}\n\n"
            f"**Hotel Suggestions:**\n{hotel_results}\n\n"
            f"**Weather:**\n{weather_results}\n\n"
            f"**Itinerary:**\n{itinerary}\n\n"
            f"*(Note: Summary generation encountered: {str(e)})*"
        )
        messages = [AIMessage(content=fallback_msg)]

    return {
        "messages": messages,
        "llm_calls": bump_llm_calls(state),
    }
