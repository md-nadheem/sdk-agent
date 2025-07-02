# import sys
# import os
# import logging
# import time
# from typing import Optional, List, Dict, Any
# from uuid import uuid4

# from dotenv import load_dotenv
# load_dotenv()

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from agents import (
#     Runner,
#     ItemHelpers,
#     MessageOutputItem,
#     HandoffOutputItem,
#     ToolCallItem,
#     ToolCallOutputItem,
#     InputGuardrailTripwireTriggered,
#     Handoff, # This import is not directly used in the provided snippet's logic, but fine to keep if it's used elsewhere
#     RunContextWrapper,
#     set_tracing_disabled,
#     enable_verbose_stdout_logging
# )

# enable_verbose_stdout_logging()
# set_tracing_disabled(True)

# # Import ONLY the agents and context available in the current main.py
# from main import (
#     triage_agent,
#     schedule_agent, # Only conference agent
#     create_initial_context,
#     load_user_context,  # Updated function name
#     load_customer_context,  # Keep for backward compatibility
#     AirlineAgentContext, # This context class is now imported from shared_types.py into main.py, and can be used here.
# )

# from database import db_client # Make sure your db_client is correctly set up.

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:3001"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     conversation_id: Optional[str] = None
#     message: str
#     account_number: Optional[str] = None

# class MessageResponse(BaseModel):
#     content: str
#     agent: str

# class AgentEvent(BaseModel):
#     id: str
#     type: str
#     agent: str
#     content: str
#     metadata: Optional[Dict[str, Any]] = None
#     timestamp: Optional[float] = None

# class GuardrailCheck(BaseModel):
#     id: str
#     name: str
#     input: str
#     reasoning: str
#     passed: bool
#     timestamp: float

# class CustomerDetails(BaseModel):
#     name: Optional[str] = None
#     account_number: Optional[str] = None
#     email: Optional[str] = None
#     is_conference_attendee: Optional[bool] = None
#     conference_name: Optional[str] = None
#     registration_id: Optional[str] = None
#     company: Optional[str] = None
#     location: Optional[str] = None
#     conference_package: Optional[str] = None
#     membership_type: Optional[List[str]] = None
#     primary_stream: Optional[str] = None
#     secondary_stream: Optional[str] = None

# class BookingDetails(BaseModel):
#     id: str
#     confirmation_number: str

# class CustomerInfoResponse(BaseModel):
#     customer: Optional[CustomerDetails] = None
#     bookings: List[BookingDetails] = []
#     current_booking: Optional[Dict[str, Any]] = None


# class ChatResponse(BaseModel):
#     conversation_id: str
#     current_agent: str
#     messages: List[MessageResponse]
#     events: List[AgentEvent]
#     context: Dict[str, Any]
#     agents: List[Dict[str, Any]]
#     guardrails: List[GuardrailCheck] = []
#     customer_info: Optional[CustomerInfoResponse] = None

# class ConversationStore:
#     async def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
#         raise NotImplementedError

#     async def save(self, conversation_id: str, state: Dict[str, Any]):
#         raise NotImplementedError

# class SupabaseConversationStore(ConversationStore):
#     def __init__(self):
#         self._memory_cache: Dict[str, Dict[str, Any]] = {}

#     async def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
#         if conversation_id in self._memory_cache:
#             logger.debug(f"Loaded conversation {conversation_id} from memory cache.")
#             return self._memory_cache[conversation_id]
        
#         try:
#             conversation_data = await db_client.load_conversation(conversation_id)
#             if conversation_data:
#                 context_data = conversation_data.get("context", {})
#                 if isinstance(context_data, dict):
#                     context_instance = AirlineAgentContext(**context_data)
#                 else:
#                     context_instance = create_initial_context() # Fallback

#                 state = {
#                     "input_items": conversation_data.get("history", []),
#                     "context": context_instance,
#                     "current_agent": conversation_data.get("current_agent", triage_agent.name),
#                 }
#                 self._memory_cache[conversation_id] = state
#                 logger.debug(f"Loaded conversation {conversation_id} from database.")
#                 return state
#         except Exception as e:
#             logger.error(f"Error loading conversation {conversation_id} from database: {e}", exc_info=True)
        
#         return None

#     async def save(self, conversation_id: str, state: Dict[str, Any]):
#         self._memory_cache[conversation_id] = state
#         try:
#             context_to_save = state["context"].model_dump() if isinstance(state["context"], BaseModel) else state["context"]
#             success = await db_client.save_conversation(
#                 session_id=conversation_id,
#                 history=state.get("input_items", []),
#                 context=context_to_save,
#                 current_agent=state.get("current_agent", triage_agent.name)
#             )
#             if not success:
#                 logger.warning(f"Failed to save conversation {conversation_id} to database.")
#             else:
#                 logger.debug(f"Saved conversation {conversation_id} to database.")
#         except Exception as e:
#             logger.error(f"Error saving conversation {conversation_id} to database: {e}", exc_info=True)

# conversation_store = SupabaseConversationStore()

# def get_agent_by_name(name: str):
#     # Only include agents that are currently active in main.py
#     agents = {
#         triage_agent.name: triage_agent,
#         schedule_agent.name: schedule_agent,
#     }
#     return agents.get(name, triage_agent) # Default to triage_agent if name not found

# def get_guardrail_name(g) -> str:
#     name_attr = getattr(g, "name", None)
#     if isinstance(name_attr, str) and name_attr:
#         return name_attr
#     guard_fn = getattr(g, "guardrail_function", None)
#     if guard_fn is not None and hasattr(guard_fn, "__name__"):
#         return guard_fn.__name__.replace("_", " ").title()
#     fn_name = getattr(g, "__name__", None)
#     if isinstance(fn_name, str) and fn_name:
#         return fn_name.replace("_", " ").title()
#     return str(g)

# def build_agents_list() -> List[Dict[str, Any]]:
#     # Only include agents that are currently active in main.py
#     all_agents = [
#         triage_agent,
#         schedule_agent,
#     ]
#     def make_agent_dict(agent):
#         handoff_names = []
#         for h in getattr(agent, "handoffs", []):
#             if hasattr(h, 'agent') and hasattr(h.agent, 'name'):
#                 handoff_names.append(h.agent.name)
#             elif hasattr(h, 'agent_name'):
#                 handoff_names.append(h.agent_name)
#             else:
#                 handoff_names.append(str(h))
#         tool_names = [getattr(t, "name_override", getattr(t, "__name__", str(t))) for t in getattr(agent, "tools", [])]
#         input_guardrail_names = [get_guardrail_name(g) for g in getattr(agent, "input_guardrails", [])]
#         return {
#             "name": agent.name,
#             "description": getattr(agent, "handoff_description", getattr(agent, "description", "")),
#             "handoffs": handoff_names,
#             "tools": tool_names,
#             "input_guardrails": input_guardrail_names,
#         }
#     return [make_agent_dict(agent) for agent in all_agents]

# @app.get("/user/{identifier}", response_model=Dict[str, Any])
# async def get_user(identifier: str):
#     """Get user by registration_id or QR code."""
#     try:
#         # Try to get user by registration_id first (if numeric), then by QR code
#         user_data = None
#         if identifier.isdigit():
#             user_data = await db_client.get_user_by_registration_id(identifier)
#         else:
#             user_data = await db_client.get_user_by_qr_code(identifier)
        
#         if not user_data:
#             logger.debug(f"No user found for identifier: {identifier}")
#             raise HTTPException(status_code=404, detail=f"User with identifier {identifier} not found")
        
#         logger.debug(f"Returning user data for {identifier}: {user_data}")
#         return user_data
#     except Exception as e:
#         logger.error(f"Error fetching user {identifier}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# # Keep the old customer endpoint for backward compatibility
# @app.get("/customer/{account_number}", response_model=Dict[str, Any])
# async def get_customer(account_number: str):
#     try:
#         customer = await db_client.get_customer_by_account_number(account_number)
#         if not customer:
#             logger.debug(f"No customer found for account_number: {account_number}")
#             raise HTTPException(status_code=404, detail=f"Customer with account_number {account_number} not found")
#         logger.debug(f"Returning customer data for {account_number}: {customer}")
#         return customer
#     except Exception as e:
#         logger.error(f"Error fetching customer {account_number}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(req: ChatRequest):
#     conversation_id: str = req.conversation_id or uuid4().hex
#     current_agent_name: str = triage_agent.name # Default to triage_agent
#     state: Dict[str, Any] = {
#         "input_items": [],
#         "context": create_initial_context(),
#         "current_agent": triage_agent.name,
#     }
    
#     try:
#         logger.debug(f"Received request for conversation_id: {req.conversation_id}, message: '{req.message}'")
        
#         existing_state = await conversation_store.get(req.conversation_id) if req.conversation_id else None
        
#         if existing_state:
#             state = existing_state
#             conversation_id = req.conversation_id
#             logger.debug(f"Resuming existing conversation {conversation_id} with agent {state['current_agent']}")
#         else:
#             conversation_id = uuid4().hex
#             if req.account_number:
#                 # Use the new load_user_context function
#                 state["context"] = await load_user_context(req.account_number)
#                 logger.info(f"New conversation {conversation_id}. Loaded user context for identifier: {req.account_number}")
#             else:
#                 state["context"] = create_initial_context()
#                 logger.info(f"New conversation {conversation_id}. Created initial context.")
            
#             customer_info_response = None
#             if state["context"].account_number:
#                 customer_info_response = CustomerInfoResponse(
#                     customer=CustomerDetails(
#                         name=state["context"].passenger_name,
#                         account_number=state["context"].account_number,
#                         email=state["context"].customer_email,
#                         is_conference_attendee=state["context"].is_conference_attendee,
#                         conference_name=state["context"].conference_name,
#                         registration_id=state["context"].account_number,
#                         company=state["context"].user_company_name,
#                         location=state["context"].user_contact_info.get("address") if state["context"].user_contact_info else None,
#                         conference_package=state["context"].user_personal_schedule_events[0].get("conference_package") if state["context"].user_personal_schedule_events else None,
#                         membership_type=state["context"].user_registered_tracks,
#                         primary_stream=state["context"].user_conference_interests[0] if state["context"].user_conference_interests else None,
#                         secondary_stream=state["context"].user_conference_interests[1] if len(state["context"].user_conference_interests) > 1 else None,
#                     ),
#                     bookings=[] # No bookings for conference users
#                 )

#             if not req.message.strip():
#                 await conversation_store.save(conversation_id, state)
#                 return ChatResponse(
#                     conversation_id=conversation_id,
#                     current_agent=state["current_agent"],
#                     messages=[],
#                     events=[AgentEvent(id=uuid4().hex, type="info", agent="System", content="Conversation started.")],
#                     context=state["context"].model_dump(),
#                     agents=build_agents_list(),
#                     guardrails=[],
#                     customer_info=customer_info_response,
#                 )

#         current_agent = get_agent_by_name(state["current_agent"])
#         current_agent_name = current_agent.name
        
#         state["input_items"].append({"content": req.message, "role": "user"})
        
#         old_context_dict = state["context"].model_dump().copy()
#         messages: List[MessageResponse] = []
#         events: List[AgentEvent] = []

#         logger.debug(f"Running agent: {current_agent.name}, with input: '{req.message}'")
        
#         result = await Runner.run(
#             current_agent,
#             state["input_items"],
#             context=state["context"] 
#         )

#         for item in result.new_items:
#             current_time_ms = time.time() * 1000
#             if isinstance(item, MessageOutputItem):
#                 text = ItemHelpers.text_message_output(item)
#                 messages.append(MessageResponse(content=text, agent=item.agent.name))
#                 events.append(AgentEvent(id=uuid4().hex, type="message", agent=item.agent.name, content=text, timestamp=current_time_ms))
#             elif isinstance(item, HandoffOutputItem):
#                 events.append(
#                     AgentEvent(
#                         id=uuid4().hex,
#                         type="handoff",
#                         agent=item.source_agent.name,
#                         content=f"Handoff from {item.source_agent.name} to {item.target_agent.name}",
#                         metadata={"source_agent": item.source_agent.name, "target_agent": item.target_agent.name},
#                         timestamp=current_time_ms
#                     )
#                 )
#                 # Correctly find the handoff object to call its on_handoff hook
#                 ho = next((h for h in getattr(item.source_agent, "handoffs", []) if getattr(h, "agent", None) == item.target_agent), None)
#                 if ho and ho.on_handoff:
#                     cb_name = getattr(ho.on_handoff, "__name__", repr(ho.on_handoff))
#                     events.append(AgentEvent(id=uuid4().hex, type="hook_call", agent=item.target_agent.name, content=f"Calling handoff hook: {cb_name}", timestamp=current_time_ms))
#                     await ho.on_handoff(RunContextWrapper(context=state["context"]))
#                     events.append(AgentEvent(id=uuid4().hex, type="hook_output", agent=item.target_agent.name, content=f"Handoff hook {cb_name} completed.", timestamp=current_time_ms))
#                 current_agent = item.target_agent
#                 state["current_agent"] = current_agent.name
#             elif isinstance(item, ToolCallItem):
#                 tool_name = getattr(item.raw_item, "name", "")
#                 tool_args = getattr(item.raw_item, "arguments", {})
#                 events.append(
#                     AgentEvent(
#                         id=uuid4().hex,
#                         type="tool_call",
#                         agent=item.agent.name,
#                         content=f"Calling tool: {tool_name}",
#                         metadata={"tool_name": tool_name, "tool_args": tool_args},
#                         timestamp=current_time_ms
#                     )
#                 )
#             elif isinstance(item, ToolCallOutputItem):
#                 tool_name_for_log = "UNKNOWN_TOOL"
#                 if hasattr(item, 'tool_call') and item.tool_call is not None and \
#                    hasattr(item.tool_call, 'function') and item.tool_call.function is not None and \
#                    hasattr(item.tool_call.function, 'name'):
#                     tool_name_for_log = item.tool_call.function.name

#                 events.append(
#                     AgentEvent(
#                         id=uuid4().hex,
#                         type="tool_output",
#                         agent=item.agent.name,
#                         content=f"Tool '{tool_name_for_log}' output: {str(item.output)}",
#                         metadata={"tool_result": str(item.output), "tool_name": tool_name_for_log},
#                         timestamp=current_time_ms
#                     )
#                 )
        
#         new_context_dict = state["context"].model_dump()
#         changes = {k: new_context_dict[k] for k in new_context_dict if old_context_dict.get(k) != new_context_dict[k]}
#         if changes:
#             events.append(
#                 AgentEvent(
#                     id=uuid4().hex,
#                     type="context_update",
#                     agent=current_agent.name,
#                     content=f"Context updated: {', '.join(changes.keys())}",
#                     metadata={"changes": changes},
#                     timestamp=time.time() * 1000
#                 )
#             )

#         state["input_items"] = result.to_input_list()
#         state["current_agent"] = current_agent.name

#         logger.debug(f"Attempting to save conversation {conversation_id} state.")
#         await conversation_store.save(conversation_id, state)
#         logger.debug(f"Conversation {conversation_id} state successfully saved.")

#         guardrail_checks: List[GuardrailCheck] = []
#         active_agent_for_guardrails = get_agent_by_name(state["current_agent"])
#         for g in getattr(active_agent_for_guardrails, "input_guardrails", []):
#             guardrail_checks.append(
#                 GuardrailCheck(
#                     id=uuid4().hex,
#                     name=get_guardrail_name(g),
#                     input=req.message,
#                     reasoning="Passed (no tripwire triggered)", # Assuming success until explicitly failed
#                     passed=True,
#                     timestamp=time.time() * 1000,
#                 )
#             )
        
#         customer_info_response = None
#         if state["context"].account_number:
#             customer_info_response = CustomerInfoResponse(
#                 customer=CustomerDetails(
#                     name=state["context"].passenger_name,
#                     account_number=state["context"].account_number,
#                     email=state["context"].customer_email,
#                     is_conference_attendee=state["context"].is_conference_attendee,
#                     conference_name=state["context"].conference_name,
#                     registration_id=state["context"].account_number,
#                     company=state["context"].user_company_name,
#                     location=state["context"].user_contact_info.get("address") if state["context"].user_contact_info else None,
#                     conference_package=state["context"].user_personal_schedule_events[0].get("conference_package") if state["context"].user_personal_schedule_events else None,
#                     membership_type=state["context"].user_registered_tracks,
#                     primary_stream=state["context"].user_conference_interests[0] if state["context"].user_conference_interests else None,
#                     secondary_stream=state["context"].user_conference_interests[1] if len(state["context"].user_conference_interests) > 1 else None,
#                 ),
#                 bookings=[] # No bookings for conference users
#             )

#         logger.debug(f"Returning ChatResponse for conversation {conversation_id}.")
#         return ChatResponse(
#             conversation_id=conversation_id,
#             current_agent=state["current_agent"],
#             messages=messages,
#             events=events,
#             context=new_context_dict,
#             agents=build_agents_list(),
#             guardrails=guardrail_checks,
#             customer_info=customer_info_response,
#         )

#     except InputGuardrailTripwireTriggered as e:
#         logger.warning(f"Guardrail tripped for conversation {conversation_id}: {e.guardrail_result.guardrail.name}")
#         failed_guardrail_name = get_guardrail_name(e.guardrail_result.guardrail)
#         gr_input = req.message
#         gr_timestamp = time.time() * 1000
        
#         guardrail_checks = []
#         # Populate guardrail checks based on the active agent's guardrails
#         active_agent_for_guardrails = get_agent_by_name(current_agent_name)
#         for g in getattr(active_agent_for_guardrails, "input_guardrails", []):
#             reasoning = ""
#             passed = True
#             if get_guardrail_name(g) == failed_guardrail_name:
#                 reasoning = getattr(e.guardrail_result, "reasoning", "Guardrail tripped.")
#                 if not reasoning: # Ensure reasoning is not empty if it's the tripped guardrail
#                     reasoning = "Guardrail tripped."
#                 passed = False
#             guardrail_checks.append(
#                 GuardrailCheck(
#                     id=uuid4().hex,
#                     name=get_guardrail_name(g),
#                     input=gr_input,
#                     reasoning=reasoning,
#                     passed=passed,
#                     timestamp=gr_timestamp,
#                 )
#             )

#         refusal = "Sorry, I can only answer questions related to the conference. Your message was flagged as irrelevant/unsafe."
#         state["input_items"].append({"role": "assistant", "content": refusal})
        
#         await conversation_store.save(conversation_id, state)

#         customer_info_response = None
#         if state["context"].account_number:
#             customer_info_response = CustomerInfoResponse(
#                 customer=CustomerDetails(
#                     name=state["context"].passenger_name,
#                     account_number=state["context"].account_number,
#                     email=state["context"].customer_email,
#                     is_conference_attendee=state["context"].is_conference_attendee,
#                     conference_name=state["context"].conference_name,
#                     registration_id=state["context"].account_number,
#                     company=state["context"].user_company_name,
#                     location=state["context"].user_contact_info.get("address") if state["context"].user_contact_info else None,
#                     conference_package=state["context"].user_personal_schedule_events[0].get("conference_package") if state["context"].user_personal_schedule_events else None,
#                     membership_type=state["context"].user_registered_tracks,
#                     primary_stream=state["context"].user_conference_interests[0] if state["context"].user_conference_interests else None,
#                     secondary_stream=state["context"].user_conference_interests[1] if len(state["context"].user_conference_interests) > 1 else None,
#                 ),
#                 bookings=[] # No bookings for conference users
#             )

#         return ChatResponse(
#             conversation_id=conversation_id,
#             current_agent=state["current_agent"],
#             messages=[MessageResponse(content=refusal, agent=state["current_agent"])],
#             events=[AgentEvent(id=uuid4().hex, type="guardrail_refusal", agent="System", content=refusal, metadata={"guardrail_name": failed_guardrail_name}, timestamp=gr_timestamp)],
#             context=state["context"].model_dump(),
#             agents=build_agents_list(),
#             guardrails=guardrail_checks,
#             customer_info=customer_info_response,
#         )
#     except Exception as e:
#         logger.error(f"Unexpected error in chat_endpoint for conversation {conversation_id}: {str(e)}", exc_info=True)
#         try:
#             error_message_for_user = "An unexpected internal error occurred. Please try again or contact support."
#             if not isinstance(state.get("input_items"), list):
#                 state["input_items"] = []
#             state["input_items"].append({"role": "assistant", "content": error_message_for_user})
#             await conversation_store.save(conversation_id, state)
#         except Exception as save_e:
#             logger.error(f"Failed to save state after unexpected error: {save_e}", exc_info=True)
            
#         raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")






















# import sys
# import os
# import logging
# import time
# from typing import Optional, List, Dict, Any
# from uuid import uuid4

# from dotenv import load_dotenv
# load_dotenv()

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from agents import (
#     Runner,
#     ItemHelpers,
#     MessageOutputItem,
#     HandoffOutputItem,
#     ToolCallItem,
#     ToolCallOutputItem,
#     InputGuardrailTripwireTriggered,
#     Handoff, # This import is not directly used in the provided snippet's logic, but fine to keep if it's used elsewhere
#     RunContextWrapper,
#     set_tracing_disabled,
#     enable_verbose_stdout_logging
# )

# enable_verbose_stdout_logging()
# set_tracing_disabled(True)

# # Import ONLY the agents and context available in the current main.py
# from main import (
#     triage_agent,
#     schedule_agent, # Conference schedule agent
#     networking_agent, # New networking agent
#     create_initial_context,
#     load_user_context,  # Updated function name
#     load_customer_context,  # Keep for backward compatibility
#     AirlineAgentContext, # This context class is now imported from shared_types.py into main.py, and can be used here.
# )

# from database import db_client # Make sure your db_client is correctly set up.

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:3001"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     conversation_id: Optional[str] = None
#     message: str
#     account_number: Optional[str] = None

# class MessageResponse(BaseModel):
#     content: str
#     agent: str

# class AgentEvent(BaseModel):
#     id: str
#     type: str
#     agent: str
#     content: str
#     metadata: Optional[Dict[str, Any]] = None
#     timestamp: Optional[float] = None

# class GuardrailCheck(BaseModel):
#     id: str
#     name: str
#     input: str
#     reasoning: str
#     passed: bool
#     timestamp: float

# class CustomerDetails(BaseModel):
#     name: Optional[str] = None
#     account_number: Optional[str] = None
#     email: Optional[str] = None
#     is_conference_attendee: Optional[bool] = None
#     conference_name: Optional[str] = None
#     registration_id: Optional[str] = None
#     company: Optional[str] = None
#     location: Optional[str] = None
#     conference_package: Optional[str] = None
#     membership_type: Optional[List[str]] = None
#     primary_stream: Optional[str] = None
#     secondary_stream: Optional[str] = None

# class BookingDetails(BaseModel):
#     id: str
#     confirmation_number: str

# class CustomerInfoResponse(BaseModel):
#     customer: Optional[CustomerDetails] = None
#     bookings: List[BookingDetails] = []
#     current_booking: Optional[Dict[str, Any]] = None


# class ChatResponse(BaseModel):
#     conversation_id: str
#     current_agent: str
#     messages: List[MessageResponse]
#     events: List[AgentEvent]
#     context: Dict[str, Any]
#     agents: List[Dict[str, Any]]
#     guardrails: List[GuardrailCheck] = []
#     customer_info: Optional[CustomerInfoResponse] = None

# class ConversationStore:
#     async def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
#         raise NotImplementedError

#     async def save(self, conversation_id: str, state: Dict[str, Any]):
#         raise NotImplementedError

# class SupabaseConversationStore(ConversationStore):
#     def __init__(self):
#         self._memory_cache: Dict[str, Dict[str, Any]] = {}

#     async def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
#         if conversation_id in self._memory_cache:
#             logger.debug(f"Loaded conversation {conversation_id} from memory cache.")
#             return self._memory_cache[conversation_id]
        
#         try:
#             conversation_data = await db_client.load_conversation(conversation_id)
#             if conversation_data:
#                 context_data = conversation_data.get("context", {})
#                 if isinstance(context_data, dict):
#                     context_instance = AirlineAgentContext(**context_data)
#                 else:
#                     context_instance = create_initial_context() # Fallback

#                 state = {
#                     "input_items": conversation_data.get("history", []),
#                     "context": context_instance,
#                     "current_agent": conversation_data.get("current_agent", triage_agent.name),
#                 }
#                 self._memory_cache[conversation_id] = state
#                 logger.debug(f"Loaded conversation {conversation_id} from database.")
#                 return state
#         except Exception as e:
#             logger.error(f"Error loading conversation {conversation_id} from database: {e}", exc_info=True)
        
#         return None

#     async def save(self, conversation_id: str, state: Dict[str, Any]):
#         self._memory_cache[conversation_id] = state
#         try:
#             context_to_save = state["context"].model_dump() if isinstance(state["context"], BaseModel) else state["context"]
#             success = await db_client.save_conversation(
#                 session_id=conversation_id,
#                 history=state.get("input_items", []),
#                 context=context_to_save,
#                 current_agent=state.get("current_agent", triage_agent.name)
#             )
#             if not success:
#                 logger.warning(f"Failed to save conversation {conversation_id} to database.")
#             else:
#                 logger.debug(f"Saved conversation {conversation_id} to database.")
#         except Exception as e:
#             logger.error(f"Error saving conversation {conversation_id} to database: {e}", exc_info=True)

# conversation_store = SupabaseConversationStore()

# def get_agent_by_name(name: str):
#     # Include all active agents
#     agents = {
#         triage_agent.name: triage_agent,
#         schedule_agent.name: schedule_agent,
#         networking_agent.name: networking_agent,
#     }
#     return agents.get(name, triage_agent) # Default to triage_agent if name not found

# def get_guardrail_name(g) -> str:
#     name_attr = getattr(g, "name", None)
#     if isinstance(name_attr, str) and name_attr:
#         return name_attr
#     guard_fn = getattr(g, "guardrail_function", None)
#     if guard_fn is not None and hasattr(guard_fn, "__name__"):
#         return guard_fn.__name__.replace("_", " ").title()
#     fn_name = getattr(g, "__name__", None)
#     if isinstance(fn_name, str) and fn_name:
#         return fn_name.replace("_", " ").title()
#     return str(g)

# def build_agents_list() -> List[Dict[str, Any]]:
#     # Include all active agents
#     all_agents = [
#         triage_agent,
#         schedule_agent,
#         networking_agent,
#     ]
#     def make_agent_dict(agent):
#         handoff_names = []
#         for h in getattr(agent, "handoffs", []):
#             if hasattr(h, 'agent') and hasattr(h.agent, 'name'):
#                 handoff_names.append(h.agent.name)
#             elif hasattr(h, 'agent_name'):
#                 handoff_names.append(h.agent_name)
#             else:
#                 handoff_names.append(str(h))
#         tool_names = [getattr(t, "name_override", getattr(t, "__name__", str(t))) for t in getattr(agent, "tools", [])]
#         input_guardrail_names = [get_guardrail_name(g) for g in getattr(agent, "input_guardrails", [])]
#         return {
#             "name": agent.name,
#             "description": getattr(agent, "handoff_description", getattr(agent, "description", "")),
#             "handoffs": handoff_names,
#             "tools": tool_names,
#             "input_guardrails": input_guardrail_names,
#         }
#     return [make_agent_dict(agent) for agent in all_agents]

# @app.get("/user/{identifier}", response_model=Dict[str, Any])
# async def get_user(identifier: str):
#     """Get user by registration_id or QR code."""
#     try:
#         # Try to get user by registration_id first (if numeric), then by QR code
#         user_data = None
#         if identifier.isdigit():
#             user_data = await db_client.get_user_by_registration_id(identifier)
#         else:
#             user_data = await db_client.get_user_by_qr_code(identifier)
        
#         if not user_data:
#             logger.debug(f"No user found for identifier: {identifier}")
#             raise HTTPException(status_code=404, detail=f"User with identifier {identifier} not found")
        
#         logger.debug(f"Returning user data for {identifier}: {user_data}")
#         return user_data
#     except Exception as e:
#         logger.error(f"Error fetching user {identifier}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# # Keep the old customer endpoint for backward compatibility
# @app.get("/customer/{account_number}", response_model=Dict[str, Any])
# async def get_customer(account_number: str):
#     try:
#         customer = await db_client.get_customer_by_account_number(account_number)
#         if not customer:
#             logger.debug(f"No customer found for account_number: {account_number}")
#             raise HTTPException(status_code=404, detail=f"Customer with account_number {account_number} not found")
#         logger.debug(f"Returning customer data for {account_number}: {customer}")
#         return customer
#     except Exception as e:
#         logger.error(f"Error fetching customer {account_number}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(req: ChatRequest):
#     conversation_id: str = req.conversation_id or uuid4().hex
#     current_agent_name: str = triage_agent.name # Default to triage_agent
#     state: Dict[str, Any] = {
#         "input_items": [],
#         "context": create_initial_context(),
#         "current_agent": triage_agent.name,
#     }
    
#     try:
#         logger.debug(f"Received request for conversation_id: {req.conversation_id}, message: '{req.message}'")
        
#         existing_state = await conversation_store.get(req.conversation_id) if req.conversation_id else None
        
#         if existing_state:
#             state = existing_state
#             conversation_id = req.conversation_id
#             logger.debug(f"Resuming existing conversation {conversation_id} with agent {state['current_agent']}")
#         else:
#             conversation_id = uuid4().hex
#             if req.account_number:
#                 # Use the new load_user_context function
#                 state["context"] = await load_user_context(req.account_number)
#                 logger.info(f"New conversation {conversation_id}. Loaded user context for identifier: {req.account_number}")
#             else:
#                 state["context"] = create_initial_context()
#                 logger.info(f"New conversation {conversation_id}. Created initial context.")
            
#             customer_info_response = None
#             if state["context"].account_number:
#                 customer_info_response = CustomerInfoResponse(
#                     customer=CustomerDetails(
#                         name=state["context"].passenger_name,
#                         account_number=state["context"].account_number,
#                         email=state["context"].customer_email,
#                         is_conference_attendee=state["context"].is_conference_attendee,
#                         conference_name=state["context"].conference_name,
#                         registration_id=state["context"].account_number,
#                         company=state["context"].user_company_name,
#                         location=state["context"].user_contact_info.get("address") if state["context"].user_contact_info else None,
#                         conference_package=state["context"].user_personal_schedule_events[0].get("conference_package") if state["context"].user_personal_schedule_events else None,
#                         membership_type=state["context"].user_registered_tracks,
#                         primary_stream=state["context"].user_conference_interests[0] if state["context"].user_conference_interests else None,
#                         secondary_stream=state["context"].user_conference_interests[1] if len(state["context"].user_conference_interests) > 1 else None,
#                     ),
#                     bookings=[] # No bookings for conference users
#                 )

#             if not req.message.strip():
#                 await conversation_store.save(conversation_id, state)
#                 return ChatResponse(
#                     conversation_id=conversation_id,
#                     current_agent=state["current_agent"],
#                     messages=[],
#                     events=[AgentEvent(id=uuid4().hex, type="info", agent="System", content="Conversation started.")],
#                     context=state["context"].model_dump(),
#                     agents=build_agents_list(),
#                     guardrails=[],
#                     customer_info=customer_info_response,
#                 )

#         current_agent = get_agent_by_name(state["current_agent"])
#         current_agent_name = current_agent.name
        
#         state["input_items"].append({"content": req.message, "role": "user"})
        
#         old_context_dict = state["context"].model_dump().copy()
#         messages: List[MessageResponse] = []
#         events: List[AgentEvent] = []

#         logger.debug(f"Running agent: {current_agent.name}, with input: '{req.message}'")
        
#         result = await Runner.run(
#             current_agent,
#             state["input_items"],
#             context=state["context"] 
#         )

#         for item in result.new_items:
#             current_time_ms = time.time() * 1000
#             if isinstance(item, MessageOutputItem):
#                 text = ItemHelpers.text_message_output(item)
#                 messages.append(MessageResponse(content=text, agent=item.agent.name))
#                 events.append(AgentEvent(id=uuid4().hex, type="message", agent=item.agent.name, content=text, timestamp=current_time_ms))
#             elif isinstance(item, HandoffOutputItem):
#                 events.append(
#                     AgentEvent(
#                         id=uuid4().hex,
#                         type="handoff",
#                         agent=item.source_agent.name,
#                         content=f"Handoff from {item.source_agent.name} to {item.target_agent.name}",
#                         metadata={"source_agent": item.source_agent.name, "target_agent": item.target_agent.name},
#                         timestamp=current_time_ms
#                     )
#                 )
#                 # Correctly find the handoff object to call its on_handoff hook
#                 ho = next((h for h in getattr(item.source_agent, "handoffs", []) if getattr(h, "agent", None) == item.target_agent), None)
#                 if ho and ho.on_handoff:
#                     cb_name = getattr(ho.on_handoff, "__name__", repr(ho.on_handoff))
#                     events.append(AgentEvent(id=uuid4().hex, type="hook_call", agent=item.target_agent.name, content=f"Calling handoff hook: {cb_name}", timestamp=current_time_ms))
#                     await ho.on_handoff(RunContextWrapper(context=state["context"]))
#                     events.append(AgentEvent(id=uuid4().hex, type="hook_output", agent=item.target_agent.name, content=f"Handoff hook {cb_name} completed.", timestamp=current_time_ms))
#                 current_agent = item.target_agent
#                 state["current_agent"] = current_agent.name
#             elif isinstance(item, ToolCallItem):
#                 tool_name = getattr(item.raw_item, "name", "")
#                 tool_args = getattr(item.raw_item, "arguments", {})
#                 events.append(
#                     AgentEvent(
#                         id=uuid4().hex,
#                         type="tool_call",
#                         agent=item.agent.name,
#                         content=f"Calling tool: {tool_name}",
#                         metadata={"tool_name": tool_name, "tool_args": tool_args},
#                         timestamp=current_time_ms
#                     )
#                 )
#             elif isinstance(item, ToolCallOutputItem):
#                 tool_name_for_log = "UNKNOWN_TOOL"
#                 if hasattr(item, 'tool_call') and item.tool_call is not None and \
#                    hasattr(item.tool_call, 'function') and item.tool_call.function is not None and \
#                    hasattr(item.tool_call.function, 'name'):
#                     tool_name_for_log = item.tool_call.function.name

#                 events.append(
#                     AgentEvent(
#                         id=uuid4().hex,
#                         type="tool_output",
#                         agent=item.agent.name,
#                         content=f"Tool '{tool_name_for_log}' output: {str(item.output)}",
#                         metadata={"tool_result": str(item.output), "tool_name": tool_name_for_log},
#                         timestamp=current_time_ms
#                     )
#                 )
        
#         new_context_dict = state["context"].model_dump()
#         changes = {k: new_context_dict[k] for k in new_context_dict if old_context_dict.get(k) != new_context_dict[k]}
#         if changes:
#             events.append(
#                 AgentEvent(
#                     id=uuid4().hex,
#                     type="context_update",
#                     agent=current_agent.name,
#                     content=f"Context updated: {', '.join(changes.keys())}",
#                     metadata={"changes": changes},
#                     timestamp=time.time() * 1000
#                 )
#             )

#         state["input_items"] = result.to_input_list()
#         state["current_agent"] = current_agent.name

#         logger.debug(f"Attempting to save conversation {conversation_id} state.")
#         await conversation_store.save(conversation_id, state)
#         logger.debug(f"Conversation {conversation_id} state successfully saved.")

#         guardrail_checks: List[GuardrailCheck] = []
#         active_agent_for_guardrails = get_agent_by_name(state["current_agent"])
#         for g in getattr(active_agent_for_guardrails, "input_guardrails", []):
#             guardrail_checks.append(
#                 GuardrailCheck(
#                     id=uuid4().hex,
#                     name=get_guardrail_name(g),
#                     input=req.message,
#                     reasoning="Passed (no tripwire triggered)", # Assuming success until explicitly failed
#                     passed=True,
#                     timestamp=time.time() * 1000,
#                 )
#             )
        
#         customer_info_response = None
#         if state["context"].account_number:
#             customer_info_response = CustomerInfoResponse(
#                 customer=CustomerDetails(
#                     name=state["context"].passenger_name,
#                     account_number=state["context"].account_number,
#                     email=state["context"].customer_email,
#                     is_conference_attendee=state["context"].is_conference_attendee,
#                     conference_name=state["context"].conference_name,
#                     registration_id=state["context"].account_number,
#                     company=state["context"].user_company_name,
#                     location=state["context"].user_contact_info.get("address") if state["context"].user_contact_info else None,
#                     conference_package=state["context"].user_personal_schedule_events[0].get("conference_package") if state["context"].user_personal_schedule_events else None,
#                     membership_type=state["context"].user_registered_tracks,
#                     primary_stream=state["context"].user_conference_interests[0] if state["context"].user_conference_interests else None,
#                     secondary_stream=state["context"].user_conference_interests[1] if len(state["context"].user_conference_interests) > 1 else None,
#                 ),
#                 bookings=[] # No bookings for conference users
#             )

#         logger.debug(f"Returning ChatResponse for conversation {conversation_id}.")
#         return ChatResponse(
#             conversation_id=conversation_id,
#             current_agent=state["current_agent"],
#             messages=messages,
#             events=events,
#             context=new_context_dict,
#             agents=build_agents_list(),
#             guardrails=guardrail_checks,
#             customer_info=customer_info_response,
#         )

#     except InputGuardrailTripwireTriggered as e:
#         logger.warning(f"Guardrail tripped for conversation {conversation_id}: {e.guardrail_result.guardrail.name}")
#         failed_guardrail_name = get_guardrail_name(e.guardrail_result.guardrail)
#         gr_input = req.message
#         gr_timestamp = time.time() * 1000
        
#         guardrail_checks = []
#         # Populate guardrail checks based on the active agent's guardrails
#         active_agent_for_guardrails = get_agent_by_name(current_agent_name)
#         for g in getattr(active_agent_for_guardrails, "input_guardrails", []):
#             reasoning = ""
#             passed = True
#             if get_guardrail_name(g) == failed_guardrail_name:
#                 reasoning = getattr(e.guardrail_result, "reasoning", "Guardrail tripped.")
#                 if not reasoning: # Ensure reasoning is not empty if it's the tripped guardrail
#                     reasoning = "Guardrail tripped."
#                 passed = False
#             guardrail_checks.append(
#                 GuardrailCheck(
#                     id=uuid4().hex,
#                     name=get_guardrail_name(g),
#                     input=gr_input,
#                     reasoning=reasoning,
#                     passed=passed,
#                     timestamp=gr_timestamp,
#                 )
#             )

#         refusal = "Sorry, I can only answer questions related to the conference. Your message was flagged as irrelevant/unsafe."
#         state["input_items"].append({"role": "assistant", "content": refusal})
        
#         await conversation_store.save(conversation_id, state)

#         customer_info_response = None
#         if state["context"].account_number:
#             customer_info_response = CustomerInfoResponse(
#                 customer=CustomerDetails(
#                     name=state["context"].passenger_name,
#                     account_number=state["context"].account_number,
#                     email=state["context"].customer_email,
#                     is_conference_attendee=state["context"].is_conference_attendee,
#                     conference_name=state["context"].conference_name,
#                     registration_id=state["context"].account_number,
#                     company=state["context"].user_company_name,
#                     location=state["context"].user_contact_info.get("address") if state["context"].user_contact_info else None,
#                     conference_package=state["context"].user_personal_schedule_events[0].get("conference_package") if state["context"].user_personal_schedule_events else None,
#                     membership_type=state["context"].user_registered_tracks,
#                     primary_stream=state["context"].user_conference_interests[0] if state["context"].user_conference_interests else None,
#                     secondary_stream=state["context"].user_conference_interests[1] if len(state["context"].user_conference_interests) > 1 else None,
#                 ),
#                 bookings=[] # No bookings for conference users
#             )

#         return ChatResponse(
#             conversation_id=conversation_id,
#             current_agent=state["current_agent"],
#             messages=[MessageResponse(content=refusal, agent=state["current_agent"])],
#             events=[AgentEvent(id=uuid4().hex, type="guardrail_refusal", agent="System", content=refusal, metadata={"guardrail_name": failed_guardrail_name}, timestamp=gr_timestamp)],
#             context=state["context"].model_dump(),
#             agents=build_agents_list(),
#             guardrails=guardrail_checks,
#             customer_info=customer_info_response,
#         )
#     except Exception as e:
#         logger.error(f"Unexpected error in chat_endpoint for conversation {conversation_id}: {str(e)}", exc_info=True)
#         try:
#             error_message_for_user = "An unexpected internal error occurred. Please try again or contact support."
#             if not isinstance(state.get("input_items"), list):
#                 state["input_items"] = []
#             state["input_items"].append({"role": "assistant", "content": error_message_for_user})
#             await conversation_store.save(conversation_id, state)
#         except Exception as save_e:
#             logger.error(f"Failed to save state after unexpected error: {save_e}", exc_info=True)
            
#         raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")











# import os
# import logging
# from typing import Optional, Dict, Any, List
# from datetime import datetime
# import uuid

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# # Import database client
# from database import db_client

# # Import agents and context
# from shared_types import AirlineAgentContext
# from agents import Agent, RunContextWrapper, function_tool
# from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# # Import conference agents
# from conference_agents.conference_agents_definitions import (
#     schedule_agent,
#     networking_agent,
#     on_schedule_handoff,
#     on_networking_handoff,
#     add_business_tool
# )

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="Conference Assistant API")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:3001"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # =========================
# # PYDANTIC MODELS
# # =========================

# class ChatRequest(BaseModel):
#     message: str
#     conversation_id: Optional[str] = None
#     account_number: Optional[str] = None

# class ChatResponse(BaseModel):
#     conversation_id: str
#     current_agent: str
#     messages: List[Dict[str, Any]]
#     events: List[Dict[str, Any]]
#     context: Dict[str, Any]
#     agents: List[Dict[str, Any]]
#     guardrails: List[Dict[str, Any]]
#     customer_info: Optional[Dict[str, Any]] = None

# # =========================
# # GUARDRAILS
# # =========================

# @function_tool
# async def relevance_guardrail(input_text: str) -> Dict[str, Any]:
#     """Check if the input is relevant to conference assistance."""
#     conference_keywords = [
#         'conference', 'schedule', 'session', 'speaker', 'networking', 'business',
#         'attendee', 'meeting', 'presentation', 'workshop', 'track', 'room',
#         'registration', 'event', 'agenda', 'program', 'participant'
#     ]
    
#     input_lower = input_text.lower()
#     is_relevant = any(keyword in input_lower for keyword in conference_keywords)
    
#     # Also allow greetings and basic questions
#     greetings = ['hello', 'hi', 'help', 'what', 'how', 'when', 'where', 'who', 'can you']
#     if any(greeting in input_lower for greeting in greetings):
#         is_relevant = True
    
#     return {
#         "reasoning": f"Input contains conference-related keywords: {is_relevant}",
#         "is_relevant": is_relevant
#     }

# @function_tool
# async def jailbreak_guardrail(input_text: str) -> Dict[str, Any]:
#     """Check for jailbreak attempts."""
#     jailbreak_patterns = [
#         'ignore', 'forget', 'system', 'prompt', 'instruction', 'override',
#         'bypass', 'pretend', 'roleplay', 'act as', 'you are now'
#     ]
    
#     input_lower = input_text.lower()
#     is_safe = not any(pattern in input_lower for pattern in jailbreak_patterns)
    
#     return {
#         "reasoning": f"No jailbreak patterns detected: {is_safe}",
#         "is_safe": is_safe
#     }

# # =========================
# # MAIN TRIAGE AGENT
# # =========================

# def triage_instructions(
#     run_context: RunContextWrapper[AirlineAgentContext], agent: Agent[AirlineAgentContext]
# ) -> str:
#     ctx = run_context.context
#     user_name = ctx.passenger_name or "Attendee"
#     conference_name = ctx.conference_name or "Business Conference 2025"
    
#     return (
#         f"{RECOMMENDED_PROMPT_PREFIX}\n"
#         f"You are a Conference Assistant for {conference_name}. Welcome {user_name}!\n\n"
#         "You help conference attendees with:\n"
#         "1. **Conference Schedule**: Finding sessions, speakers, timings, rooms, and tracks\n"
#         "2. **Networking**: Connecting with other attendees and exploring business opportunities\n"
#         "3. **General Information**: Basic conference details and assistance\n\n"
#         "**Handoff Guidelines:**\n"
#         "- For schedule-related questions (sessions, speakers, timings, rooms, tracks) → Transfer to Schedule Agent\n"
#         "- For networking questions (finding attendees, businesses, adding business) → Transfer to Networking Agent\n"
#         "- For general greetings and basic info → Handle directly\n\n"
#         "**Important**: Always be helpful and conference-focused. If users ask about non-conference topics, politely redirect them back to conference-related assistance.\n\n"
#         "Keep responses concise and friendly. **Do not describe tool usage or agent transfers in your responses.**"
#     )

# # Create handoff functions
# from agents import handoff

# @handoff
# async def transfer_to_schedule_agent(
#     context: RunContextWrapper[AirlineAgentContext]
# ) -> Agent[AirlineAgentContext]:
#     """Transfer to the Schedule Agent for conference schedule inquiries."""
#     await on_schedule_handoff(context)
#     return schedule_agent

# @handoff
# async def transfer_to_networking_agent(
#     context: RunContextWrapper[AirlineAgentContext]
# ) -> Agent[AirlineAgentContext]:
#     """Transfer to the Networking Agent for networking and business inquiries."""
#     await on_networking_handoff(context)
#     return networking_agent

# # Create the triage agent
# triage_agent = Agent[AirlineAgentContext](
#     name="Triage Agent",
#     model="groq/llama3-8b-8192",
#     handoff_description="Main conference assistant that routes requests to specialized agents.",
#     instructions=triage_instructions,
#     tools=[],
#     handoffs=[transfer_to_schedule_agent, transfer_to_networking_agent],
#     input_guardrails=[relevance_guardrail, jailbreak_guardrail]
# )

# # Update other agents to include handoff back to triage
# schedule_agent.handoffs = [
#     handoff(lambda context: triage_agent, description="Transfer back to main conference assistant")
# ]
# networking_agent.handoffs = [
#     handoff(lambda context: triage_agent, description="Transfer back to main conference assistant")
# ]

# # =========================
# # CONVERSATION MANAGEMENT
# # =========================

# # In-memory storage for conversations (in production, use a proper database)
# conversations: Dict[str, Dict[str, Any]] = {}

# def get_or_create_conversation(conversation_id: Optional[str], account_number: Optional[str]) -> Dict[str, Any]:
#     """Get existing conversation or create a new one."""
#     if not conversation_id:
#         conversation_id = str(uuid.uuid4())
    
#     if conversation_id not in conversations:
#         conversations[conversation_id] = {
#             "id": conversation_id,
#             "context": AirlineAgentContext(),
#             "current_agent": "Triage Agent",
#             "messages": [],
#             "events": [],
#             "account_number": account_number
#         }
    
#     return conversations[conversation_id]

# # =========================
# # API ENDPOINTS
# # =========================

# @app.get("/user/{identifier}")
# async def get_user_info(identifier: str):
#     """Get user information by registration ID or QR code (user ID)."""
#     try:
#         # First try to get user by registration_id
#         user = await db_client.get_user_by_registration_id(identifier)
        
#         # If not found, try by QR code (user ID)
#         if not user:
#             user = await db_client.get_user_by_qr_code(identifier)
        
#         if not user:
#             raise HTTPException(status_code=404, detail="User not found")
        
#         return user
#     except Exception as e:
#         logger.error(f"Error fetching user info for {identifier}: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(request: ChatRequest):
#     """Main chat endpoint for conference assistant."""
#     try:
#         # Get or create conversation
#         conversation = get_or_create_conversation(request.conversation_id, request.account_number)
        
#         # Load user context if account_number is provided
#         if request.account_number and not conversation["context"].customer_id:
#             user = await db_client.get_user_by_registration_id(request.account_number)
#             if not user:
#                 user = await db_client.get_user_by_qr_code(request.account_number)
            
#             if user:
#                 # Update context with user information
#                 conversation["context"].passenger_name = user.get("name")
#                 conversation["context"].customer_id = user.get("id")
#                 conversation["context"].account_number = user.get("account_number")
#                 conversation["context"].customer_email = user.get("email")
#                 conversation["context"].is_conference_attendee = user.get("is_conference_attendee", True)
#                 conversation["context"].conference_name = user.get("conference_name", "Business Conference 2025")
        
#         # Handle business form submission
#         if "I want to add my business with the following details:" in request.message:
#             # Parse business details from message
#             lines = request.message.split('\n')
#             business_data = {}
            
#             for line in lines[1:]:  # Skip first line
#                 if ':' in line:
#                     key, value = line.split(':', 1)
#                     key = key.strip()
#                     value = value.strip()
                    
#                     # Map display names to database field names
#                     field_mapping = {
#                         "Company Name": "companyName",
#                         "Industry Sector": "industrySector", 
#                         "Sub-Sector": "subSector",
#                         "Location": "location",
#                         "Position Title": "positionTitle",
#                         "Legal Structure": "legalStructure",
#                         "Establishment Year": "establishmentYear",
#                         "Products/Services": "productsOrServices",
#                         "Brief Description": "briefDescription",
#                         "Website": "web"
#                     }
                    
#                     if key in field_mapping:
#                         business_data[field_mapping[key]] = value
            
#             # Add business using the tool
#             if business_data and conversation["context"].customer_id:
#                 try:
#                     success = await db_client.add_business(
#                         conversation["context"].customer_id,
#                         business_data
#                     )
                    
#                     if success:
#                         response_message = f"Successfully added your business '{business_data.get('companyName', 'Unknown')}' to the business directory! Other attendees can now find and connect with your business for networking opportunities."
#                     else:
#                         response_message = "I encountered an issue adding your business. Please try again or contact support."
                        
#                 except Exception as e:
#                     response_message = f"Error adding business: {str(e)}"
#             else:
#                 response_message = "I couldn't process your business information. Please make sure all required fields are filled out."
            
#             # Add messages to conversation
#             conversation["messages"].append({
#                 "content": request.message,
#                 "agent": "User"
#             })
#             conversation["messages"].append({
#                 "content": response_message,
#                 "agent": "Networking Agent"
#             })
            
#             # Update current agent
#             conversation["current_agent"] = "Networking Agent"
        
#         else:
#             # Regular message processing would go here
#             # For now, we'll simulate a simple response
#             conversation["messages"].append({
#                 "content": request.message,
#                 "agent": "User"
#             })
            
#             # Simple routing logic
#             message_lower = request.message.lower()
            
#             if any(word in message_lower for word in ['schedule', 'session', 'speaker', 'time', 'room', 'track', 'when']):
#                 agent_name = "Schedule Agent"
#                 response = "I can help you find conference schedule information. What specific session, speaker, or time are you looking for?"
#             elif any(word in message_lower for word in ['network', 'business', 'attendee', 'connect', 'company', 'find people']):
#                 agent_name = "Networking Agent"
#                 response = "I can help you connect with other attendees and explore business opportunities. Are you looking for specific people, companies, or would you like to add your own business?"
#             else:
#                 agent_name = "Triage Agent"
#                 response = f"Hello! I'm your conference assistant. I can help you with:\n\n• **Conference Schedule** - Find sessions, speakers, and timings\n• **Networking** - Connect with attendees and businesses\n\nWhat would you like to know about the conference?"
            
#             conversation["messages"].append({
#                 "content": response,
#                 "agent": agent_name
#             })
            
#             conversation["current_agent"] = agent_name
        
#         # Prepare customer info
#         customer_info = None
#         if conversation["context"].customer_id:
#             customer_info = {
#                 "customer": {
#                     "name": conversation["context"].passenger_name,
#                     "account_number": conversation["context"].account_number,
#                     "email": conversation["context"].customer_email,
#                     "is_conference_attendee": conversation["context"].is_conference_attendee,
#                     "conference_name": conversation["context"].conference_name
#                 },
#                 "bookings": []
#             }
        
#         # Prepare response
#         response = ChatResponse(
#             conversation_id=conversation["id"],
#             current_agent=conversation["current_agent"],
#             messages=conversation["messages"][-2:] if len(conversation["messages"]) >= 2 else conversation["messages"],
#             events=conversation.get("events", []),
#             context=conversation["context"].model_dump(),
#             agents=[
#                 {
#                     "name": "Triage Agent",
#                     "description": "Main conference assistant",
#                     "handoffs": ["Schedule Agent", "Networking Agent"],
#                     "tools": [],
#                     "input_guardrails": ["relevance_guardrail", "jailbreak_guardrail"]
#                 },
#                 {
#                     "name": "Schedule Agent", 
#                     "description": "Conference schedule information",
#                     "handoffs": ["Triage Agent"],
#                     "tools": ["get_conference_schedule"],
#                     "input_guardrails": []
#                 },
#                 {
#                     "name": "Networking Agent",
#                     "description": "Networking and business connections", 
#                     "handoffs": ["Triage Agent"],
#                     "tools": ["search_attendees", "search_businesses", "add_business"],
#                     "input_guardrails": []
#                 }
#             ],
#             guardrails=[],
#             customer_info=customer_info
#         )
        
#         return response
        
#     except Exception as e:
#         logger.error(f"Error in chat endpoint: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @app.get("/")
# async def root():
#     """Health check endpoint."""
#     return {"message": "Conference Assistant API is running"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

































import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import database client
from database import db_client

# Import main components
from main import (
    conversations,
    get_or_create_conversation,
    load_user_context,
    execute_schedule_agent,
    execute_networking_agent
)

# Import shared types
from shared_types import AirlineAgentContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Conference Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PYDANTIC MODELS
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

# =========================
# API ENDPOINTS
# =========================

@app.get("/user/{identifier}")
async def get_user_info(identifier: str):
    """Get user information by registration ID or QR code (user ID)."""
    try:
        # First try to get user by registration_id
        user = await db_client.get_user_by_registration_id(identifier)
        
        # If not found, try by QR code (user ID)
        if not user:
            user = await db_client.get_user_by_qr_code(identifier)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    except Exception as e:
        logger.error(f"Error fetching user info for {identifier}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for conference assistant."""
    try:
        # Get or create conversation
        conversation = get_or_create_conversation(request.conversation_id, request.account_number)
        
        # Load user context if account_number is provided and not already loaded
        if request.account_number and not conversation["context"].customer_id:
            await load_user_context(conversation, request.account_number)
        
        # Handle business form submission
        if "I want to add my business with the following details:" in request.message:
            # Parse business details from message
            lines = request.message.split('\n')
            business_data = {}
            
            for line in lines[1:]:  # Skip first line
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Map display names to database field names
                    field_mapping = {
                        "Company Name": "companyName",
                        "Industry Sector": "industrySector", 
                        "Sub-Sector": "subSector",
                        "Location": "location",
                        "Position Title": "positionTitle",
                        "Legal Structure": "legalStructure",
                        "Establishment Year": "establishmentYear",
                        "Products/Services": "productsOrServices",
                        "Brief Description": "briefDescription",
                        "Website": "web"
                    }
                    
                    if key in field_mapping:
                        business_data[field_mapping[key]] = value
            
            # Add business using the database client
            if business_data and conversation["context"].customer_id:
                try:
                    success = await db_client.add_business(
                        conversation["context"].customer_id,
                        business_data
                    )
                    
                    if success:
                        response_message = f"✅ Successfully added your business '{business_data.get('companyName', 'Unknown')}' to the business directory!\n\nYour business is now visible to other conference attendees for networking opportunities. Other participants can find your business when searching by industry, location, or company name.\n\nIs there anything else I can help you with regarding networking or the conference?"
                    else:
                        response_message = "❌ I encountered an issue adding your business. Please try again or contact support for assistance."
                        
                except Exception as e:
                    logger.error(f"Error adding business: {e}")
                    response_message = f"❌ Error adding business: {str(e)}"
            else:
                response_message = "❌ I couldn't process your business information. Please make sure all required fields are filled out correctly."
            
            # Add messages to conversation
            conversation["messages"].append({
                "content": request.message,
                "agent": "User"
            })
            conversation["messages"].append({
                "content": response_message,
                "agent": "Networking Agent"
            })
            
            # Update current agent
            conversation["current_agent"] = "Networking Agent"
        
        else:
            # Regular message processing
            conversation["messages"].append({
                "content": request.message,
                "agent": "User"
            })
            
            # Improved routing logic based on message content
            message_lower = request.message.lower()
            
            # Determine which agent to use with better keyword detection
            if any(word in message_lower for word in ['schedule', 'session', 'speaker', 'time', 'room', 'track', 'when', 'agenda', 'program', 'july', 'date', 'events', 'september', 'tell me about speaker', 'about speaker']):
                agent_name = "Schedule Agent"
                try:
                    # Convert context to dict for agent execution
                    context_dict = {
                        'customer_id': conversation["context"].customer_id,
                        'passenger_name': conversation["context"].passenger_name,
                        'account_number': conversation["context"].account_number,
                        'customer_email': conversation["context"].customer_email,
                        'is_conference_attendee': conversation["context"].is_conference_attendee,
                        'conference_name': conversation["context"].conference_name,
                        'user_company_name': conversation["context"].user_company_name,
                        'user_location': conversation["context"].user_location,
                        'user_registration_id': conversation["context"].user_registration_id,
                        'user_conference_package': conversation["context"].user_conference_package,
                        'user_primary_stream': conversation["context"].user_primary_stream,
                        'user_secondary_stream': conversation["context"].user_secondary_stream
                    }
                    response = await execute_schedule_agent(request.message, context_dict)
                except Exception as e:
                    logger.error(f"Error executing agent {agent_name}: {e}")
                    response = "I'm having trouble processing your request. Please try again or rephrase your question."
                    
            elif any(word in message_lower for word in ['network', 'business', 'attendee', 'connect', 'company', 'find people', 'add business', 'register business', 'show attendees', 'my business', 'tell about', 'tell me about', 'about']):
                agent_name = "Networking Agent"
                try:
                    # Convert context to dict for agent execution
                    context_dict = {
                        'customer_id': conversation["context"].customer_id,
                        'passenger_name': conversation["context"].passenger_name,
                        'account_number': conversation["context"].account_number,
                        'customer_email': conversation["context"].customer_email,
                        'is_conference_attendee': conversation["context"].is_conference_attendee,
                        'conference_name': conversation["context"].conference_name,
                        'user_company_name': conversation["context"].user_company_name,
                        'user_location': conversation["context"].user_location,
                        'user_registration_id': conversation["context"].user_registration_id,
                        'user_conference_package': conversation["context"].user_conference_package,
                        'user_primary_stream': conversation["context"].user_primary_stream,
                        'user_secondary_stream': conversation["context"].user_secondary_stream
                    }
                    response = await execute_networking_agent(request.message, context_dict)
                except Exception as e:
                    logger.error(f"Error executing agent {agent_name}: {e}")
                    response = "I'm having trouble processing your request. Please try again or rephrase your question."
                    
            else:
                agent_name = "Triage Agent"
                if 'hello' in message_lower or 'hi' in message_lower:
                    user_name = conversation["context"].passenger_name or "there"
                    response = f"Hello {user_name}! 👋 Welcome to Business Conference 2025!\n\nI'm your conference assistant and I'm here to help you with:\n\n🗓️ **Conference Schedule** - Find sessions, speakers, timings, and rooms\n🤝 **Networking** - Connect with attendees and explore business opportunities\n\nWhat would you like to know about the conference today?"
                else:
                    response = f"I'm your conference assistant for Business Conference 2025. I can help you with:\n\n🗓️ **Conference Schedule** - Find sessions, speakers, timings, and rooms\n🤝 **Networking** - Connect with attendees and explore business opportunities\n\nWhat would you like to know about the conference?"
            
            conversation["messages"].append({
                "content": response,
                "agent": agent_name
            })
            
            conversation["current_agent"] = agent_name
        
        # Prepare customer info
        customer_info = None
        if conversation["context"].customer_id:
            customer_info = {
                "customer": {
                    "name": conversation["context"].passenger_name,
                    "account_number": conversation["context"].account_number,
                    "email": conversation["context"].customer_email,
                    "is_conference_attendee": conversation["context"].is_conference_attendee,
                    "conference_name": conversation["context"].conference_name,
                    "registration_id": conversation["context"].user_registration_id,
                    "company": conversation["context"].user_company_name,
                    "location": conversation["context"].user_location,
                    "conference_package": conversation["context"].user_conference_package,
                    "primary_stream": conversation["context"].user_primary_stream,
                    "secondary_stream": conversation["context"].user_secondary_stream
                },
                "bookings": []
            }
        
        # Prepare response
        response = ChatResponse(
            conversation_id=conversation["id"],
            current_agent=conversation["current_agent"],
            messages=conversation["messages"][-2:] if len(conversation["messages"]) >= 2 else conversation["messages"],
            events=conversation.get("events", []),
            context=conversation["context"].model_dump() if hasattr(conversation["context"], 'model_dump') else conversation["context"].__dict__,
            agents=[
                {
                    "name": "Triage Agent",
                    "description": "Main conference assistant that routes requests",
                    "handoffs": ["Schedule Agent", "Networking Agent"],
                    "tools": [],
                    "input_guardrails": []
                },
                {
                    "name": "Schedule Agent", 
                    "description": "Conference schedule and session information",
                    "handoffs": ["Triage Agent"],
                    "tools": ["get_conference_schedule"],
                    "input_guardrails": []
                },
                {
                    "name": "Networking Agent",
                    "description": "Networking, attendees, and business connections", 
                    "handoffs": ["Triage Agent"],
                    "tools": ["search_attendees", "search_businesses", "add_business", "display_business_form"],
                    "input_guardrails": []
                }
            ],
            guardrails=[],
            customer_info=customer_info
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Conference Assistant API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)