# python-backend/api.py

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import from main.py instead of trying to import run_agents
from main import (
    all_agents,
    all_guardrails,
    starting_agent,
    AirlineAgentContext,
)

# Import agents framework components
from agents import (
    Agent,
    RunContextWrapper,
    function_tool,
    handoff,
)

from database import db_client
from shared_types import AirlineAgentContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Conference Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REQUEST/RESPONSE MODELS
# =========================

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    account_number: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    current_agent: str
    messages: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[Dict[str, Any]]
    customer_info: Optional[Dict[str, Any]] = None

class CustomerInfoResponse(BaseModel):
    customer: Optional[Dict[str, Any]] = None
    bookings: List[Dict[str, Any]] = []
    current_booking: Optional[Dict[str, Any]] = None

# =========================
# CONVERSATION MANAGEMENT
# =========================

# In-memory storage for conversations (in production, use a database)
conversations: Dict[str, Dict[str, Any]] = {}

def get_or_create_conversation(conversation_id: Optional[str] = None) -> str:
    """Get existing conversation or create a new one."""
    if conversation_id and conversation_id in conversations:
        return conversation_id
    
    new_id = str(uuid.uuid4())
    conversations[new_id] = {
        "context": AirlineAgentContext(),
        "current_agent": starting_agent.name,
        "messages": [],
        "events": [],
        "guardrails": [],
    }
    return new_id

def serialize_agent(agent: Agent) -> Dict[str, Any]:
    """Serialize an agent for API response."""
    handoff_names = []
    if hasattr(agent, 'handoffs') and agent.handoffs:
        for handoff_obj in agent.handoffs:
            if hasattr(handoff_obj, 'target') and hasattr(handoff_obj.target, 'name'):
                handoff_names.append(handoff_obj.target.name)
            elif hasattr(handoff_obj, 'name'):
                handoff_names.append(handoff_obj.name)
    
    return {
        "name": agent.name,
        "description": agent.handoff_description or "No description available",
        "handoffs": handoff_names,
        "tools": [tool.__name__ for tool in agent.tools] if agent.tools else [],
        "input_guardrails": agent.input_guardrails or [],
    }

def serialize_context(context: AirlineAgentContext) -> Dict[str, Any]:
    """Serialize context for API response."""
    return context.model_dump(exclude_none=True)

async def run_single_agent_turn(
    agent: Agent,
    context: RunContextWrapper[AirlineAgentContext],
    message: str,
    conversation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Run a single turn with an agent and return the results."""
    
    # Check input guardrails first
    guardrail_results = []
    if agent.input_guardrails:
        for guardrail_name in agent.input_guardrails:
            if guardrail_name in all_guardrails:
                guardrail_func = all_guardrails[guardrail_name]
                try:
                    # Call the guardrail function properly
                    result = await guardrail_func(context, message)
                    
                    # Handle different result types
                    if hasattr(result, 'is_relevant'):
                        passed = result.is_relevant
                        reasoning = result.reasoning if hasattr(result, 'reasoning') else ""
                    elif hasattr(result, 'is_safe'):
                        passed = result.is_safe
                        reasoning = result.reasoning if hasattr(result, 'reasoning') else ""
                    else:
                        passed = bool(result)
                        reasoning = str(result)
                    
                    guardrail_check = {
                        "id": str(uuid.uuid4()),
                        "name": guardrail_name,
                        "input": message,
                        "reasoning": reasoning,
                        "passed": passed,
                        "timestamp": datetime.now().isoformat(),
                    }
                    guardrail_results.append(guardrail_check)
                    
                    # If guardrail fails, return early
                    if not passed:
                        return {
                            "messages": [{
                                "content": "Sorry, I can only answer questions related to conference topics.",
                                "agent": agent.name
                            }],
                            "events": [],
                            "guardrails": guardrail_results,
                            "handoff": None,
                        }
                except Exception as e:
                    logger.error(f"Error running guardrail {guardrail_name}: {e}")
    
    # Run the agent
    try:
        # Get agent instructions
        instructions = agent.instructions(context, agent) if callable(agent.instructions) else agent.instructions
        
        # Check if this is a handoff request
        handoff_keywords = {
            "schedule": ["schedule", "session", "speaker", "event", "time", "date", "july", "room", "track", "when", "what time"],
            "networking": ["attendee", "people", "business", "company", "network", "connect", "find", "who", "attendees", "businesses"]
        }
        
        message_lower = message.lower()
        target_agent = None
        
        # Determine if we need to handoff
        if agent.name == "Triage Agent":
            schedule_score = sum(1 for keyword in handoff_keywords["schedule"] if keyword in message_lower)
            networking_score = sum(1 for keyword in handoff_keywords["networking"] if keyword in message_lower)
            
            if schedule_score > networking_score and schedule_score > 0:
                target_agent = "Schedule Agent"
            elif networking_score > 0:
                target_agent = "Networking Agent"
        
        # Execute tools if this is a specialist agent
        tool_results = []
        if agent.name == "Schedule Agent" and agent.tools:
            # Use the schedule tool
            for tool in agent.tools:
                if tool.__name__ == "get_conference_schedule_tool":
                    try:
                        result = await tool(context, query=message)
                        tool_results.append({
                            "tool_name": tool.__name__,
                            "result": result
                        })
                        break
                    except Exception as e:
                        logger.error(f"Error executing tool {tool.__name__}: {e}")
                        tool_results.append({
                            "tool_name": tool.__name__,
                            "result": f"Error: {str(e)}"
                        })
        
        elif agent.name == "Networking Agent" and agent.tools:
            # Check if user wants to add business
            if any(phrase in message_lower for phrase in ["add business", "register business", "add my business", "register my company"]):
                # Use display business form tool
                for tool in agent.tools:
                    if tool.__name__ == "display_business_form_tool":
                        try:
                            result = await tool(context)
                            tool_results.append({
                                "tool_name": tool.__name__,
                                "result": result
                            })
                            break
                        except Exception as e:
                            logger.error(f"Error executing tool {tool.__name__}: {e}")
            
            # Check if this is business data submission
            elif "company name:" in message_lower and "industry sector:" in message_lower:
                # Parse business data and use add_business tool
                try:
                    lines = message.split('\n')
                    business_data = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip().lower().replace(' ', '_')
                            value = value.strip()
                            business_data[key] = value
                    
                    # Use add_business tool
                    for tool in agent.tools:
                        if tool.__name__ == "add_business_tool":
                            try:
                                result = await tool(
                                    context,
                                    company_name=business_data.get('company_name', ''),
                                    industry_sector=business_data.get('industry_sector', ''),
                                    sub_sector=business_data.get('sub-sector', ''),
                                    location=business_data.get('location', ''),
                                    position_title=business_data.get('position_title', ''),
                                    legal_structure=business_data.get('legal_structure', ''),
                                    establishment_year=business_data.get('establishment_year', ''),
                                    products_or_services=business_data.get('products/services', ''),
                                    brief_description=business_data.get('brief_description', ''),
                                    website=business_data.get('website')
                                )
                                tool_results.append({
                                    "tool_name": tool.__name__,
                                    "result": result
                                })
                                break
                            except Exception as e:
                                logger.error(f"Error executing tool {tool.__name__}: {e}")
                except Exception as e:
                    logger.error(f"Error parsing business data: {e}")
            
            else:
                # Use the networking search tools
                for tool in agent.tools:
                    if tool.__name__ == "search_attendees_tool":
                        try:
                            result = await tool(context, query=message)
                            tool_results.append({
                                "tool_name": tool.__name__,
                                "result": result
                            })
                            break
                        except Exception as e:
                            logger.error(f"Error executing tool {tool.__name__}: {e}")
        
        # Generate response
        if target_agent:
            # Handoff response
            response_content = f"I'll connect you with our {target_agent} who can help you with that. One moment please..."
            events = [{
                "id": str(uuid.uuid4()),
                "type": "handoff",
                "agent": agent.name,
                "content": f"Handing off to {target_agent}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "source_agent": agent.name,
                    "target_agent": target_agent
                }
            }]
        elif tool_results:
            # Tool execution response
            response_content = tool_results[0]["result"]
            events = [{
                "id": str(uuid.uuid4()),
                "type": "tool_call",
                "agent": agent.name,
                "content": f"Executed {tool_results[0]['tool_name']}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "tool_name": tool_results[0]["tool_name"],
                    "tool_result": tool_results[0]["result"]
                }
            }]
        else:
            # Default response
            if agent.name == "Triage Agent":
                response_content = (
                    "Hello! I'm here to help you with conference information. "
                    "I can help you find:\n\n"
                    "🗓️ **Schedule & Sessions** - Ask about speakers, events, times, or rooms\n"
                    "🤝 **Networking** - Find attendees, businesses, or register your company\n\n"
                    "What would you like to know about the conference?"
                )
            else:
                response_content = "I'm here to help! Please let me know what you'd like to know."
            events = []
        
        return {
            "messages": [{
                "content": response_content,
                "agent": agent.name
            }],
            "events": events,
            "guardrails": guardrail_results,
            "handoff": target_agent,
        }
        
    except Exception as e:
        logger.error(f"Error running agent {agent.name}: {e}")
        return {
            "messages": [{
                "content": "I'm sorry, I encountered an error. Please try again.",
                "agent": agent.name
            }],
            "events": [],
            "guardrails": guardrail_results,
            "handoff": None,
        }

# =========================
# API ENDPOINTS
# =========================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages and return agent responses."""
    try:
        # Get or create conversation
        conversation_id = get_or_create_conversation(request.conversation_id)
        conversation_data = conversations[conversation_id]
        
        # Load user context if account_number provided
        if request.account_number:
            try:
                # Try to get user by registration ID first
                user = await db_client.get_user_by_registration_id(request.account_number)
                if not user:
                    # Try QR code
                    user = await db_client.get_user_by_qr_code(request.account_number)
                
                if user:
                    # Update context with user information
                    conversation_data["context"].passenger_name = user.get("name")
                    conversation_data["context"].customer_id = user.get("id")
                    conversation_data["context"].account_number = user.get("account_number")
                    conversation_data["context"].customer_email = user.get("email")
                    conversation_data["context"].is_conference_attendee = user.get("is_conference_attendee", True)
                    conversation_data["context"].conference_name = user.get("conference_name")
                    
                    # Store customer info for response
                    customer_info = {
                        "customer": user,
                        "bookings": [],  # Conference doesn't use bookings
                        "current_booking": None
                    }
                else:
                    customer_info = None
            except Exception as e:
                logger.error(f"Error loading user context: {e}")
                customer_info = None
        else:
            customer_info = conversation_data.get("customer_info")
        
        # Create context wrapper
        context = RunContextWrapper(conversation_data["context"])
        
        # Get current agent
        current_agent_name = conversation_data["current_agent"]
        current_agent = next((a for a in all_agents if a.name == current_agent_name), all_agents[0])
        
        # Add user message to conversation
        user_message = {
            "content": request.message,
            "role": "user",
            "timestamp": datetime.now().isoformat()
        }
        conversation_data["messages"].append(user_message)
        
        # Run agent turn
        result = await run_single_agent_turn(current_agent, context, request.message, conversation_data)
        
        # Handle handoff
        if result.get("handoff"):
            target_agent_name = result["handoff"]
            target_agent = next((a for a in all_agents if a.name == target_agent_name), None)
            if target_agent:
                conversation_data["current_agent"] = target_agent_name
                # Run the target agent with the same message
                target_result = await run_single_agent_turn(target_agent, context, request.message, conversation_data)
                # Combine results
                result["messages"].extend(target_result["messages"])
                result["events"].extend(target_result["events"])
                result["guardrails"].extend(target_result["guardrails"])
        
        # Update conversation data
        conversation_data["messages"].extend(result["messages"])
        conversation_data["events"].extend(result["events"])
        conversation_data["guardrails"].extend(result["guardrails"])
        conversation_data["context"] = context.context
        if customer_info:
            conversation_data["customer_info"] = customer_info
        
        # Prepare response
        response = ChatResponse(
            conversation_id=conversation_id,
            current_agent=conversation_data["current_agent"],
            messages=result["messages"],
            events=result["events"],
            context=serialize_context(conversation_data["context"]),
            agents=[serialize_agent(agent) for agent in all_agents],
            guardrails=result["guardrails"],
            customer_info=customer_info
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{identifier}")
async def get_user_info(identifier: str):
    """Get user information by registration ID or QR code."""
    try:
        # Try registration ID first
        user = await db_client.get_user_by_registration_id(identifier)
        
        # If not found, try QR code
        if not user:
            user = await db_client.get_user_by_qr_code(identifier)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customer/{account_number}")
async def get_customer_info(account_number: str):
    """Get customer information by account number (backward compatibility)."""
    try:
        customer = await db_client.get_customer_by_account_number(account_number)
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Get customer bookings
        bookings = await db_client.get_bookings_by_customer_id(customer.get("id", ""))
        
        return CustomerInfoResponse(
            customer=customer,
            bookings=bookings,
            current_booking=bookings[0] if bookings else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/booking/{confirmation_number}")
async def get_booking_info(confirmation_number: str):
    """Get booking information by confirmation number."""
    try:
        booking = await db_client.get_booking_by_confirmation(confirmation_number)
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")
        
        return booking
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting booking info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Conference Assistant API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)