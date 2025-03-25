from pydantic import BaseModel, Field
from typing import Optional, List
from trustcall import create_extractor
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
import configuration
from datetime import datetime

# Initialize the LLM
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# Enhanced Travel Preferences Schema
class TravelPreferences(BaseModel):
    """Stores user's travel-related preferences"""
    preferred_accommodation_types: List[str] = Field(description="Preferred hotel/housing types (e.g., boutique, hostel, luxury)")
    dietary_restrictions: List[str] = Field(description="Dietary needs (e.g., vegetarian, gluten-free)")
    budget_range: str = Field(description="Typical daily budget range (e.g., $100-200)")
    visited_destinations: List[str] = Field(description="Previously visited locations")
    travel_style: str = Field(description="Travel style (e.g., backpacking, luxury, family)")

# Create the extractor
trustcall_extractor = create_extractor(
    model,
    tools=[TravelPreferences],
    tool_choice="TravelPreferences",
)

# System Prompts
TRAVEL_SYSTEM_MESSAGE = """You are a Travel Assistant AI that helps plan trips, book accommodations, 
and remember travel preferences. Use the following traveler profile when making recommendations:

{memory}

Current Date: {current_date}
Always:
- Ask clarifying questions about destinations/dates/budget
- Suggest activities based on user preferences
- Mention any dietary restrictions when recommending restaurants
- Consider budget range when making suggestions"""

TRUSTCALL_INSTRUCTION = """Extract or update travel preferences from this conversation:"""
SUMMARY_INSTRUCTION = """Summarize the key travel planning details including destinations, dates, 
bookings made, and preferences discussed. Preserve all critical trip information:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Generate travel recommendations using memory"""
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Retrieve travel preferences
    namespace = ("travel_memory", user_id)
    existing_memory = store.get(namespace, "travel_prefs")

    # Format memory for prompt
    memory_str = ""
    if existing_memory and existing_memory.value:
        prefs = existing_memory.value
        memory_str = (
            f"Travel Style: {prefs.get('travel_style', 'Not specified')}\n"
            f"Budget: {prefs.get('budget_range', 'Not specified')}\n"
            f"Dietary Needs: {', '.join(prefs.get('dietary_restrictions', []))}\n"
            f"Visited Locations: {', '.join(prefs.get('visited_destinations', []))}"
        )

    system_msg = TRAVEL_SYSTEM_MESSAGE.format(
        memory=memory_str,
        current_date=datetime.now().strftime("%Y-%m-%d")
    )

    response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])
    return {"messages": [response]}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Update travel preferences"""
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    namespace = ("travel_memory", user_id)
    
    existing_memory = store.get(namespace, "travel_prefs")
    existing_prefs = {"TravelPreferences": existing_memory.value} if existing_memory else None
    
    result = trustcall_extractor.invoke({
        "messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"],
        "existing": existing_prefs
    })
    
    if not result.get("responses"):
        return state
        
    try:
        updated_prefs = result["responses"][0].model_dump()
        # Add new destinations to visited list
        if existing_memory and "visited_destinations" in existing_memory.value:
            updated_prefs["visited_destinations"] = list(set(
                existing_memory.value["visited_destinations"] +
                updated_prefs.get("visited_destinations", [])
            ))
        store.put(namespace, "travel_prefs", updated_prefs)
    except (IndexError, KeyError) as e:
        print(f"Memory update error: {e}")
    
    return state

def summarize_messages(state: MessagesState) -> Optional[dict]:
    """Manage conversation context for trip planning"""
    if len(state["messages"]) > 25:
        summary = model.invoke([
            SystemMessage(content=SUMMARY_INSTRUCTION),
            *state["messages"][-25:]
        ])
        return {
            "messages": [
                SystemMessage(content=f"TRIP SUMMARY:\n{summary.content}"),
                *state["messages"][-2:]
            ]
        }
    return None

# Add booking confirmation pseudo-node
def confirm_booking(state: MessagesState):
    """Simulate booking confirmation flow"""
    last_message = state["messages"][-1].content.lower()
    if "book" in last_message or "reserve" in last_message:
        return {"messages": [
            SystemMessage(content="Please confirm:\n1. Travel dates\n2. Guest details\n3. Payment method")
        ]}
    return state

# Build enhanced graph
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Add nodes
builder.add_node("travel_assistant", call_model)
builder.add_node("Memory_Extraction", write_memory)
builder.add_node("summarize", summarize_messages)
builder.add_node("booking_check", confirm_booking)

# Set up workflow
builder.add_edge(START, "travel_assistant")

# After generating response, check for booking keywords
builder.add_edge("travel_assistant", "booking_check")

# Then check if summarization needed
builder.add_conditional_edges(
    "booking_check",
    lambda state: "summarize" if len(state["messages"]) > 25 else "Memory_Extraction",
    {
        "summarize": "summarize",
        "Memory_Extraction": "Memory_Extraction"
    }
)

# Connect final nodes
builder.add_edge("summarize", "Memory_Extraction")
builder.add_edge("Memory_Extraction", END)

graph = builder.compile()