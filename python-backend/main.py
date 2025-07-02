# python-backend/main.py

from __future__ import annotations as _annotations

import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import date, datetime

from agents import (
    Agent,
    RunContextWrapper,
    function_tool,
    handoff,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from shared_types import AirlineAgentContext, RelevanceOutput, JailbreakOutput
from database import db_client

# Import conference agents
from conference_agents.conference_agents_definitions import (
    schedule_agent,
    networking_agent,
    get_conference_schedule_tool,
    search_attendees_tool,
    search_businesses_tool,
    get_user_businesses_tool,
    display_business_form_tool,
    add_business_tool,
    get_organization_info_tool,
    on_schedule_handoff,
    on_networking_handoff,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# GUARDRAILS
# =========================

@function_tool
async def relevance_guardrail(
    context: RunContextWrapper[AirlineAgentContext], user_input: str
) -> RelevanceOutput:
    """Check if user input is relevant to conference assistance."""
    # Conference-related keywords
    conference_keywords = [
        "conference", "session", "speaker", "schedule", "event", "attendee", "networking",
        "business", "company", "organization", "track", "room", "time", "date",
        "july", "presentation", "talk", "workshop", "meeting", "registration",
        "participant", "delegate", "agenda", "program", "venue", "location"
    ]
    
    user_input_lower = user_input.lower()
    
    # Check for conference-related content
    is_relevant = any(keyword in user_input_lower for keyword in conference_keywords)
    
    # Also allow greetings and basic questions
    greeting_patterns = ["hello", "hi", "help", "what", "how", "when", "where", "who", "can you"]
    if any(pattern in user_input_lower for pattern in greeting_patterns):
        is_relevant = True
    
    reasoning = (
        "User input is relevant to conference assistance" if is_relevant
        else "User input is not related to conference topics"
    )
    
    return RelevanceOutput(reasoning=reasoning, is_relevant=is_relevant)

@function_tool
async def jailbreak_guardrail(
    context: RunContextWrapper[AirlineAgentContext], user_input: str
) -> JailbreakOutput:
    """Check if user input contains jailbreak attempts."""
    jailbreak_patterns = [
        "ignore", "forget", "system", "prompt", "instruction", "override",
        "pretend", "roleplay", "act as", "you are now", "new role",
        "disregard", "bypass", "admin", "developer", "debug"
    ]
    
    user_input_lower = user_input.lower()
    contains_jailbreak = any(pattern in user_input_lower for pattern in jailbreak_patterns)
    
    reasoning = (
        "Input contains potential jailbreak attempt" if contains_jailbreak
        else "Input appears safe"
    )
    
    return JailbreakOutput(reasoning=reasoning, is_safe=not contains_jailbreak)

# =========================
# TOOLS
# =========================

@function_tool(
    name_override="get_user_info",
    description_override="Get user information by registration ID or QR code."
)
async def get_user_info_tool(
    context: RunContextWrapper[AirlineAgentContext], identifier: str
) -> str:
    """Get user information by registration ID or QR code."""
    try:
        # Try registration ID first
        user = await db_client.get_user_by_registration_id(identifier)
        
        # If not found, try QR code (UUID format)
        if not user:
            user = await db_client.get_user_by_qr_code(identifier)
        
        if not user:
            return f"No user found with identifier '{identifier}'. Please check your registration ID or QR code."
        
        # Update context with user information
        context.context.passenger_name = user.get("name")
        context.context.customer_id = user.get("id")
        context.context.account_number = user.get("account_number")
        context.context.customer_email = user.get("email")
        context.context.is_conference_attendee = user.get("is_conference_attendee", True)
        context.context.conference_name = user.get("conference_name")
        
        return f"Welcome {user.get('name')}! I have loaded your conference information."
        
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        return f"Error retrieving user information: {str(e)}"

# =========================
# AGENTS
# =========================

def triage_instructions(
    run_context: RunContextWrapper[AirlineAgentContext], agent: Agent[AirlineAgentContext]
) -> str:
    ctx = run_context.context
    user_name = ctx.passenger_name or "Attendee"
    conference_name = ctx.conference_name or "Business Conference 2025"
    
    return (
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        f"You are a Conference Triage Agent for {conference_name}. Welcome {user_name}!\n\n"
        "Your role is to understand what attendees need and route them to the right specialist:\n\n"
        "🗓️ **Schedule Agent** - For questions about:\n"
        "   • Conference sessions, events, and schedule\n"
        "   • Speaker information and presentations\n"
        "   • Room locations and timings\n"
        "   • Track information\n"
        "   • Specific dates (July 15-16, 2025)\n\n"
        "🤝 **Networking Agent** - For questions about:\n"
        "   • Finding other attendees\n"
        "   • Business connections and companies\n"
        "   • Attendee profiles and contact information\n"
        "   • Business registration and networking\n"
        "   • Organization information\n\n"
        "IMPORTANT:\n"
        "- Route users to the appropriate specialist agent based on their needs\n"
        "- If users ask about schedule/sessions/speakers → Schedule Agent\n"
        "- If users ask about people/networking/businesses → Networking Agent\n"
        "- Provide a warm, helpful welcome and explain what each agent can do\n"
        "- Don't try to answer specific questions yourself - let the specialists handle them\n\n"
        "If users ask unrelated questions, politely redirect them to conference topics."
    )

triage_agent = Agent[AirlineAgentContext](
    name="Triage Agent",
    model="groq/llama3-8b-8192",
    handoff_description="Main entry point for conference assistance",
    instructions=triage_instructions,
    tools=[get_user_info_tool],
    handoffs=[
        handoff(schedule_agent, on_schedule_handoff),
        handoff(networking_agent, on_networking_handoff),
    ],
    input_guardrails=["relevance_guardrail", "jailbreak_guardrail"],
)

# Update handoffs for other agents to include triage
schedule_agent.handoffs = [handoff(triage_agent)]
networking_agent.handoffs = [handoff(triage_agent)]

# =========================
# EXPORTED COMPONENTS
# =========================

# Export all agents for use in api.py
all_agents = [triage_agent, schedule_agent, networking_agent]

# Export all guardrails for use in api.py
all_guardrails = {
    "relevance_guardrail": relevance_guardrail,
    "jailbreak_guardrail": jailbreak_guardrail,
}

# Export the starting agent
starting_agent = triage_agent

# =========================
# MAIN RUNNER (for standalone testing)
# =========================

async def main():
    """Main function to run the conference assistant standalone."""
    print("🎉 Conference Assistant Started!")
    print("This is a standalone test mode.")
    print("For full functionality, use the web interface.")
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())