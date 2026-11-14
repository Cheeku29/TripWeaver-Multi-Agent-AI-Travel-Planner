"""Combines flight, hotel and weather results into a day-by-day itinerary."""

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.prompts import ITINERARY_AGENT_PROMPT, ITINERARY_SYSTEM_PROMPT
from src.clients.llm import get_llm
from src.config.settings import (
    HOTEL_DATA_CHAR_LIMIT,
    WEATHER_DATA_CHAR_LIMIT,
)
from src.graph.state import TravelState
from src.utils.async_utils import bump_llm_calls, truncate


def itinerary_agent(state: TravelState):
    flight_results = truncate(state.get("flight_results", ""), 2500)
    hotel_results = truncate(state.get("hotel_results", ""), HOTEL_DATA_CHAR_LIMIT)
    weather_results = truncate(state.get("weather_results", ""), WEATHER_DATA_CHAR_LIMIT)

    prompt = ITINERARY_AGENT_PROMPT.format(
        user_query=state["user_query"],
        flight_results=flight_results,
        hotel_results=hotel_results,
        weather_results=weather_results,
    )

    try:
        response = get_llm().invoke([
            SystemMessage(content=ITINERARY_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        itinerary_text = response.content
        messages = [response]
    except Exception as e:
        itinerary_text = f"Itinerary could not be generated: {str(e)}"
        messages = [HumanMessage(content=itinerary_text)]

    return {
        "itinerary": itinerary_text,
        "messages": messages,
        "llm_calls": bump_llm_calls(state),
    }
