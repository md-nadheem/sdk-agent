# python-backend/conference_agents/conference_agents_definitions.py

from __future__ import annotations as _annotations

from typing import Optional, List, Dict, Any
from datetime import date, datetime
import re

# Import shared context type from shared_types.py
from shared_types import AirlineAgentContext # Import the shared context model
from database import db_client # Assumes db_client is accessible or imported from a shared module
from agents import (
    Agent,
    RunContextWrapper,
    function_tool,
    handoff,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# =========================
# HELPER FUNCTIONS
# =========================

def parse_date_from_text(text: str) -> Optional[date]:
    """Parse date from natural language text."""
    text = text.lower().strip()
    
    # Handle various date formats
    date_patterns = [
        # July 15th, July 15, 15th July, 15 July
        (r'july\s+(\d{1,2})(?:st|nd|rd|th)?', lambda m: date(2025, 7, int(m.group(1)))),
        (r'(\d{1,2})(?:st|nd|rd|th)?\s+july', lambda m: date(2025, 7, int(m.group(1)))),
        # 2025-07-15, 07-15-2025, 15-07-2025
        (r'2025[-/]07[-/](\d{1,2})', lambda m: date(2025, 7, int(m.group(1)))),
        (r'07[-/](\d{1,2})[-/]2025', lambda m: date(2025, 7, int(m.group(1)))),
        (r'(\d{1,2})[-/]07[-/]2025', lambda m: date(2025, 7, int(m.group(1)))),
        # Handle "15th", "16th" etc when context is July 2025
        (r'(\d{1,2})(?:st|nd|rd|th)', lambda m: date(2025, 7, int(m.group(1)))),
    ]
    
    for pattern, date_func in date_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return date_func(match)
            except ValueError:
                continue
    
    # Handle "today", "tomorrow" relative to conference dates
    if 'today' in text:
        return date(2025, 7, 15)  # Assume conference start date
    elif 'tomorrow' in text:
        return date(2025, 7, 16)  # Next day
    
    return None

def extract_search_terms(text: str) -> Dict[str, Any]:
    """Extract search terms from natural language query."""
    text = text.lower().strip()
    
    search_params = {}
    
    # Extract speaker names (look for "by", "speaker", "presented by", etc.)
    speaker_patterns = [
        r'(?:by|speaker|presented by|talk by)\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|;)',
        r'([a-zA-Z\s]+?)(?:\s+is\s+speaking|\s+speaking|\s+presentation)',
        r'speaker\s*:\s*([a-zA-Z\s]+?)(?:\s|$|,|\.|;)',
    ]
    
    for pattern in speaker_patterns:
        match = re.search(pattern, text)
        if match:
            speaker_name = match.group(1).strip()
            if len(speaker_name) > 2:  # Avoid single letters
                search_params['speaker_name'] = speaker_name
                break
    
    # Extract room information
    room_patterns = [
        r'(?:in|at|room)\s+(room\s*[a-zA-Z0-9]+|[a-zA-Z0-9]+\s*room|hall\s*[a-zA-Z0-9]+|[a-zA-Z0-9]+\s*hall)',
        r'room\s*:\s*([a-zA-Z0-9\s]+?)(?:\s|$|,|\.|;)',
    ]
    
    for pattern in room_patterns:
        match = re.search(pattern, text)
        if match:
            room_name = match.group(1).strip()
            search_params['conference_room_name'] = room_name
            break
    
    # Extract track information
    track_patterns = [
        r'(?:track|stream)\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|;)',
        r'([a-zA-Z\s]+?)\s+track',
    ]
    
    for pattern in track_patterns:
        match = re.search(pattern, text)
        if match:
            track_name = match.group(1).strip()
            if len(track_name) > 2:
                search_params['track_name'] = track_name
                break
    
    # Extract topic/subject
    topic_patterns = [
        r'(?:about|on|topic|subject)\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|;)',
        r'session\s+on\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|;)',
    ]
    
    for pattern in topic_patterns:
        match = re.search(pattern, text)
        if match:
            topic = match.group(1).strip()
            if len(topic) > 2:
                search_params['topic'] = topic
                break
    
    # Extract date
    parsed_date = parse_date_from_text(text)
    if parsed_date:
        search_params['conference_date'] = parsed_date
    
    return search_params

# =========================
# TOOLS (Conference-specific)
# =========================

@function_tool(
    name_override="get_conference_schedule",
    description_override="Get conference schedule information by speaker, topic, room, track, or date. Can handle natural language queries."
)
async def get_conference_schedule_tool(
    context: RunContextWrapper[AirlineAgentContext],
    query: Optional[str] = None,
    speaker_name: Optional[str] = None,
    topic: Optional[str] = None,
    conference_room_name: Optional[str] = None,
    track_name: Optional[str] = None,
    conference_date: Optional[str] = None
) -> str:
    """Get conference schedule information based on various filters or natural language query."""
    try:
        # If query is provided, extract search parameters from it
        if query:
            search_params = extract_search_terms(query)
            # Override individual parameters with extracted ones if not already provided
            if not speaker_name and 'speaker_name' in search_params:
                speaker_name = search_params['speaker_name']
            if not topic and 'topic' in search_params:
                topic = search_params['topic']
            if not conference_room_name and 'conference_room_name' in search_params:
                conference_room_name = search_params['conference_room_name']
            if not track_name and 'track_name' in search_params:
                track_name = search_params['track_name']
            if not conference_date and 'conference_date' in search_params:
                conference_date = search_params['conference_date'].isoformat()

        # Convert date string to date object if provided
        parsed_date = None
        if conference_date:
            if isinstance(conference_date, str):
                try:
                    # Try parsing ISO format first
                    parsed_date = datetime.strptime(conference_date, "%Y-%m-%d").date()
                except ValueError:
                    # Try parsing from natural language
                    parsed_date = parse_date_from_text(conference_date)
            else:
                parsed_date = conference_date

        # Get schedule from database
        schedule = await db_client.get_conference_schedule(
            speaker_name=speaker_name,
            topic=topic,
            conference_room_name=conference_room_name,
            track_name=track_name,
            conference_date=parsed_date
        )

        if not schedule:
            # Provide helpful suggestions
            filters = []
            if speaker_name: filters.append(f"speaker '{speaker_name}'")
            if topic: filters.append(f"topic '{topic}'")
            if conference_room_name: filters.append(f"room '{conference_room_name}'")
            if track_name: filters.append(f"track '{track_name}'")
            if parsed_date: filters.append(f"date '{parsed_date}'")
            
            filter_text = " and ".join(filters) if filters else "your criteria"
            
            # Get all available dates to suggest alternatives
            all_sessions = await db_client.get_conference_schedule()
            available_dates = set()
            for session in all_sessions:
                if session.get('conference_date'):
                    available_dates.add(session['conference_date'])
            
            result = f"No conference sessions found for {filter_text}."
            if available_dates:
                dates_str = ", ".join(sorted(available_dates))
                result += f"\n\nThe conference has sessions on: {dates_str}"
                result += "\n\nTry asking about sessions on these specific dates, or ask about specific speakers, topics, or rooms."
            
            return result

        # Format the schedule information
        result = f"Found {len(schedule)} conference session(s):\n\n"
        
        # Group by date for better organization
        sessions_by_date = {}
        for session in schedule:
            session_date = session.get('conference_date', 'Unknown Date')
            if session_date not in sessions_by_date:
                sessions_by_date[session_date] = []
            sessions_by_date[session_date].append(session)
        
        for session_date in sorted(sessions_by_date.keys()):
            if len(sessions_by_date) > 1:  # Only show date header if multiple dates
                result += f"**{session_date}**\n\n"
            
            # Sort sessions by start time
            date_sessions = sorted(sessions_by_date[session_date], 
                                 key=lambda x: x.get('start_time', ''))
            
            for session in date_sessions:
                start_time = session.get('start_time', 'TBD')
                end_time = session.get('end_time', 'TBD')
                
                # Format datetime strings if they exist
                if isinstance(start_time, str) and 'T' in start_time:
                    try:
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00')).strftime('%I:%M %p')
                    except:
                        pass
                if isinstance(end_time, str) and 'T' in end_time:
                    try:
                        end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00')).strftime('%I:%M %p')
                    except:
                        pass
                
                result += f"🎯 **{session.get('topic', 'Unknown Topic')}**\n"
                result += f"👤 Speaker: {session.get('speaker_name', 'TBD')}\n"
                result += f"⏰ Time: {start_time} - {end_time}\n"
                result += f"📍 Room: {session.get('conference_room_name', 'TBD')}\n"
                result += f"🏷️ Track: {session.get('track_name', 'TBD')}\n"
                
                if session.get('description'):
                    result += f"📝 Description: {session.get('description')}\n"
                
                result += "\n"

        return result

    except Exception as e:
        return f"Error retrieving conference schedule: {str(e)}"

@function_tool(
    name_override="search_attendees",
    description_override="Search for conference attendees by name, company, location, or get all attendees."
)
async def search_attendees_tool(
    context: RunContextWrapper[AirlineAgentContext],
    query: Optional[str] = None,
    name: Optional[str] = None,
    company: Optional[str] = None,
    location: Optional[str] = None,
    limit: int = 20
) -> str:
    """Search for conference attendees by various criteria."""
    try:
        attendees = []
        
        if name:
            # Search by name
            attendees = await db_client.get_user_details_by_name(name)
        elif query:
            # Try to search by name first, then by company if no results
            attendees = await db_client.get_user_details_by_name(query)
            if not attendees:
                # Search in company field within details
                all_attendees = await db_client.get_all_attendees(limit=100)
                attendees = [
                    attendee for attendee in all_attendees
                    if query.lower() in (attendee.get('details', {}).get('company', '')).lower()
                ]
        else:
            # Get all attendees
            attendees = await db_client.get_all_attendees(limit=limit)

        # Filter by additional criteria if provided
        if company and attendees:
            attendees = [
                attendee for attendee in attendees
                if company.lower() in (attendee.get('details', {}).get('company', '')).lower()
            ]
        
        if location and attendees:
            attendees = [
                attendee for attendee in attendees
                if location.lower() in (attendee.get('details', {}).get('location', '')).lower()
            ]

        if not attendees:
            search_criteria = []
            if name: search_criteria.append(f"name '{name}'")
            if company: search_criteria.append(f"company '{company}'")
            if location: search_criteria.append(f"location '{location}'")
            if query: search_criteria.append(f"'{query}'")
            
            search_text = " and ".join(search_criteria) if search_criteria else "matching your criteria"
            return f"No attendees found {search_text}. Try searching with different terms or ask to see all attendees."

        # Format attendee information
        result = f"Found {len(attendees)} attendee(s):\n\n"
        
        for i, attendee in enumerate(attendees[:limit], 1):
            details = attendee.get('details', {})
            user_name = details.get('user_name') or f"{details.get('firstName', '')} {details.get('lastName', '')}".strip()
            
            result += f"**{i}. {user_name}**\n"
            
            if details.get('company'):
                result += f"🏢 Company: {details.get('company')}\n"
            if details.get('location'):
                result += f"📍 Location: {details.get('location')}\n"
            if details.get('title'):
                result += f"💼 Title: {details.get('title')}\n"
            if details.get('primary_stream'):
                result += f"🎯 Primary Stream: {details.get('primary_stream')}\n"
            if details.get('secondary_stream'):
                result += f"🎯 Secondary Stream: {details.get('secondary_stream')}\n"
            if details.get('conference_package'):
                result += f"🎫 Package: {details.get('conference_package')}\n"
            if details.get('registered_email'):
                result += f"📧 Email: {details.get('registered_email')}\n"
            
            result += "\n"

        if len(attendees) == limit:
            result += f"\n*Showing first {limit} results. Use more specific search terms to narrow down results.*"

        return result

    except Exception as e:
        return f"Error searching attendees: {str(e)}"

@function_tool(
    name_override="search_businesses",
    description_override="Search for businesses by company name, sector, location, or other criteria."
)
async def search_businesses_tool(
    context: RunContextWrapper[AirlineAgentContext],
    query: Optional[str] = None,
    sector: Optional[str] = None,
    location: Optional[str] = None,
    limit: int = 20
) -> str:
    """Search for businesses by various criteria."""
    try:
        businesses = await db_client.search_businesses(
            query=query,
            sector=sector,
            location=location
        )

        if not businesses:
            search_criteria = []
            if query: search_criteria.append(f"'{query}'")
            if sector: search_criteria.append(f"sector '{sector}'")
            if location: search_criteria.append(f"location '{location}'")
            
            search_text = " and ".join(search_criteria) if search_criteria else "your criteria"
            return f"No businesses found for {search_text}. Try different search terms or ask to see all businesses."

        # Format business information
        result = f"Found {len(businesses)} business(es):\n\n"
        
        for i, business in enumerate(businesses[:limit], 1):
            details = business.get('details', {})
            
            result += f"**{i}. {details.get('companyName', 'Unknown Company')}**\n"
            
            if details.get('industrySector'):
                result += f"🏭 Industry: {details.get('industrySector')}\n"
            if details.get('subSector'):
                result += f"🔧 Sub-sector: {details.get('subSector')}\n"
            if details.get('location'):
                result += f"📍 Location: {details.get('location')}\n"
            if details.get('establishmentYear'):
                result += f"📅 Established: {details.get('establishmentYear')}\n"
            if details.get('legalStructure'):
                result += f"⚖️ Legal Structure: {details.get('legalStructure')}\n"
            if details.get('briefDescription'):
                result += f"📝 Description: {details.get('briefDescription')}\n"
            if details.get('productsOrServices'):
                result += f"🛍️ Products/Services: {details.get('productsOrServices')}\n"
            if details.get('web'):
                result += f"🌐 Website: {details.get('web')}\n"
            
            result += "\n"

        if len(businesses) == limit:
            result += f"\n*Showing first {limit} results. Use more specific search terms to narrow down results.*"

        return result

    except Exception as e:
        return f"Error searching businesses: {str(e)}"

@function_tool(
    name_override="get_user_businesses",
    description_override="Get all businesses for a specific user by name or ID."
)
async def get_user_businesses_tool(
    context: RunContextWrapper[AirlineAgentContext],
    user_name: Optional[str] = None
) -> str:
    """Get all businesses for a specific user."""
    try:
        # If no user_name provided, use current user
        if not user_name:
            user_id = context.context.customer_id
            if not user_id:
                return "No user specified and no current user context available. Please provide a user name."
        else:
            # Search for user by name first
            users = await db_client.get_user_details_by_name(user_name)
            if not users:
                return f"No user found with name '{user_name}'. Please check the spelling or try a different name."
            user_id = users[0].get('id')

        businesses = await db_client.get_user_businesses(user_id)

        if not businesses:
            user_text = user_name or "the current user"
            return f"No businesses found for {user_text}. They may not have registered any businesses yet."

        # Format business information
        user_text = user_name or "the current user"
        result = f"Found {len(businesses)} business(es) for {user_text}:\n\n"
        
        for i, business in enumerate(businesses, 1):
            details = business.get('details', {})
            
            result += f"**{i}. {details.get('companyName', 'Unknown Company')}**\n"
            
            if details.get('industrySector'):
                result += f"🏭 Industry: {details.get('industrySector')}\n"
            if details.get('subSector'):
                result += f"🔧 Sub-sector: {details.get('subSector')}\n"
            if details.get('location'):
                result += f"📍 Location: {details.get('location')}\n"
            if details.get('positionTitle'):
                result += f"💼 Position: {details.get('positionTitle')}\n"
            if details.get('establishmentYear'):
                result += f"📅 Established: {details.get('establishmentYear')}\n"
            if details.get('briefDescription'):
                result += f"📝 Description: {details.get('briefDescription')}\n"
            if details.get('web'):
                result += f"🌐 Website: {details.get('web')}\n"
            
            result += "\n"

        return result

    except Exception as e:
        return f"Error retrieving user businesses: {str(e)}"

@function_tool(
    name_override="display_business_form",
    description_override="Display a business registration form for the user to fill out."
)
async def display_business_form_tool(
    context: RunContextWrapper[AirlineAgentContext]
) -> str:
    """Trigger the UI to show a business registration form."""
    return "DISPLAY_BUSINESS_FORM"

@function_tool(
    name_override="add_business",
    description_override="Add a new business for the current user."
)
async def add_business_tool(
    context: RunContextWrapper[AirlineAgentContext],
    company_name: str,
    industry_sector: str,
    sub_sector: str,
    location: str,
    position_title: str,
    legal_structure: str,
    establishment_year: str,
    products_or_services: str,
    brief_description: str,
    website: Optional[str] = None
) -> str:
    """Add a new business for the current user."""
    try:
        user_id = context.context.customer_id
        if not user_id:
            return "Unable to add business: No user context available. Please log in first."

        # Prepare business details
        business_details = {
            "companyName": company_name,
            "industrySector": industry_sector,
            "subSector": sub_sector,
            "location": location,
            "positionTitle": position_title,
            "legalStructure": legal_structure,
            "establishmentYear": establishment_year,
            "productsOrServices": products_or_services,
            "briefDescription": brief_description
        }
        
        if website:
            business_details["web"] = website

        # Add business to database
        success = await db_client.add_business(user_id, business_details)

        if success:
            return f"✅ Successfully added business '{company_name}' to your profile! The business is now listed in the business directory and available for networking."
        else:
            return f"❌ Failed to add business '{company_name}'. Please try again or contact support."

    except Exception as e:
        return f"Error adding business: {str(e)}"

@function_tool(
    name_override="get_organization_info",
    description_override="Get information about an organization."
)
async def get_organization_info_tool(
    context: RunContextWrapper[AirlineAgentContext],
    organization_id: Optional[str] = None
) -> str:
    """Get organization information."""
    try:
        # If no organization_id provided, use current user's organization
        if not organization_id:
            organization_id = context.context.get('organization_id')
            if not organization_id:
                return "No organization specified and no current organization context available."

        organization = await db_client.get_organization_details(organization_id)

        if not organization:
            return f"No organization found with ID '{organization_id}'."

        # Format organization information
        result = f"**Organization Information**\n\n"
        result += f"📋 Name: {organization.get('name', 'Unknown')}\n"
        
        details = organization.get('details', {})
        if details:
            for key, value in details.items():
                if value:
                    formatted_key = key.replace('_', ' ').title()
                    result += f"📌 {formatted_key}: {value}\n"

        return result

    except Exception as e:
        return f"Error retrieving organization information: {str(e)}"

# =========================
# AGENTS (Conference-specific)
# =========================

def schedule_agent_instructions(
    run_context: RunContextWrapper[AirlineAgentContext], agent: Agent[AirlineAgentContext]
) -> str:
    ctx = run_context.context
    user_name = ctx.passenger_name or "[unknown]"
    return (
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a Conference Schedule Agent. Help attendees find information about conference sessions, speakers, schedules, and events.\n"
        f"Current user: {user_name}\n\n"
        "You can help with:\n"
        "• Finding sessions by speaker name, topic, room, track, or date\n"
        "• Getting schedule information for specific dates (July 15-16, 2025)\n"
        "• Providing details about conference rooms and tracks\n"
        "• Answering questions about session timings and descriptions\n"
        "• Searching for specific speakers or topics\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "- Always use the get_conference_schedule tool to search for sessions\n"
        "- When users ask about dates, parse them carefully (e.g., 'July 15th' should be 2025-07-15)\n"
        "- For natural language queries, pass the entire query to the tool's 'query' parameter\n"
        "- Provide detailed, well-formatted responses with emojis for better readability\n"
        "- If no sessions are found, suggest alternative dates or search terms\n"
        "- The conference is on July 15-16, 2025\n\n"
        "Examples of queries you should handle:\n"
        "- 'Events on July 15th' → Search for sessions on 2025-07-15\n"
        "- 'Sessions by John Smith' → Search for speaker 'John Smith'\n"
        "- 'What's in Room A today?' → Search for Room A sessions\n"
        "- 'Technology track sessions' → Search for Technology track\n\n"
        "**Do not describe tool usage in your responses.**\n"
        "If the user asks unrelated questions, transfer back to the triage agent."
    )

schedule_agent = Agent[AirlineAgentContext](
    name="Schedule Agent",
    model="groq/llama3-8b-8192",
    handoff_description="An agent to provide conference schedule information and help find sessions.",
    instructions=schedule_agent_instructions,
    tools=[get_conference_schedule_tool],
    handoffs=[],
)

def networking_agent_instructions(
    run_context: RunContextWrapper[AirlineAgentContext], agent: Agent[AirlineAgentContext]
) -> str:
    ctx = run_context.context
    user_name = ctx.passenger_name or "[unknown]"
    return (
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a Networking Agent. Help attendees connect with other participants and explore business opportunities.\n"
        f"Current user: {user_name}\n\n"
        "You can help with:\n"
        "• Finding other conference attendees by name, company, or location\n"
        "• Searching for businesses by company name, industry sector, or location\n"
        "• Getting information about specific users' businesses\n"
        "• Helping users register their own businesses (show registration form)\n"
        "• Providing organization information\n"
        "• Connecting people with similar interests or business sectors\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "- Use search_attendees to find people by name, company, or location\n"
        "- Use search_businesses to find companies and business opportunities\n"
        "- Use get_user_businesses to see what businesses a specific person has\n"
        "- Use display_business_form when users want to add their business\n"
        "- Provide detailed, well-formatted responses with emojis for better readability\n"
        "- If no results found, suggest alternative search terms or broader criteria\n"
        "- Be helpful in connecting people and facilitating networking\n\n"
        "Examples of queries you should handle:\n"
        "- 'Find attendees from Mumbai' → Search attendees by location\n"
        "- 'Show me tech companies' → Search businesses in technology sector\n"
        "- 'Who is John Smith?' → Search for attendee named John Smith\n"
        "- 'I want to add my business' → Show business registration form\n\n"
        "**Do not describe tool usage in your responses.**\n"
        "If the user wants to add a business, use display_business_form to show them the registration form.\n"
        "If the user asks unrelated questions, transfer back to the triage agent."
    )

networking_agent = Agent[AirlineAgentContext](
    name="Networking Agent",
    model="groq/llama3-8b-8192",
    handoff_description="An agent to help with networking, finding attendees, and business connections.",
    instructions=networking_agent_instructions,
    tools=[
        search_attendees_tool,
        search_businesses_tool,
        get_user_businesses_tool,
        display_business_form_tool,
        add_business_tool,
        get_organization_info_tool
    ],
    handoffs=[],
)

# =========================
# HOOKS (Conference-specific)
# =========================

async def on_schedule_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    """Load user details when handed off to schedule agent."""
    pass

async def on_networking_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    """Load user details when handed off to networking agent."""
    pass