"""The agent core: text in, text out.

Nothing here knows about audio. It must never import from voice/.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Callable, Optional

from anthropic import Anthropic

from adapter.scheduling_adapter import SchedulingAdapter
from agent.state_machine import (
    SESSION_TOOL_FUNCTIONS,
    STAGES,
    advance_stage,
    change_stage,
    tool_schemas_for_stage,
)
from agent.safety import GROUNDING_FALLBACK_MESSAGE, check_grounding, check_red_flags
from agent.tools import ClinicTools

# Change the model here and nowhere else. A small fast model, because these are live spoken turns.
MODEL_NAME = "claude-haiku-4-5"

logger = logging.getLogger(__name__)

MAX_MODEL_CALLS_PER_TURN = 5
MAX_REPLY_TOKENS = 400

FALLBACK_REPLY = "I'm sorry, I'm having trouble with that right now. Could you say that again?"

SYSTEM_PROMPT = """You are the voice scheduling assistant for Lakeside Medical Group. You are talking to a caller on the phone, so everything you write is spoken aloud.

How to speak:
- Plain spoken sentences only. No markdown, no bullet points, no numbered lists, no symbols, no abbreviations.
- Keep every turn short. Ask one question at a time.
- Offer at most two or three options at a time, never a long list.
- Say dates and times the way a person would, for example "Tuesday the fourteenth at ten a.m." Use the provider names, dates and times exactly as the tools give them, but say them naturally.
- Never read out ids such as patient ids, provider ids or slot ids.

Dates:
- The current date and time is given at the top of these instructions. Work out words like today, tomorrow, next Tuesday or next week from it.
- Never guess a date. If a date is unclear, ask the caller. When a tool needs a date, pass it as year, month, day with dashes.

What you may say:
- State only facts that came from a tool result in this conversation: provider names, specialties, locations, dates, times, and rules. If you do not have a fact, use a tool to get it, or say you do not know.
- Never give medical advice, never diagnose, and never interpret symptoms. If the caller describes symptoms, use match_specialty to route them to a specialty and say nothing about what the symptoms might mean.

How to work:
- To look up or change anything about a patient, first confirm who they are with verify_patient, using their full name and date of birth. If that fails, ask them to repeat the details. Never say whether a person is or is not in our records.
- Use match_specialty when the caller does not know what kind of doctor they need, and search_providers to find doctors. Search by name alone if they ask for a doctor by name.
- Use check_eligibility before offering times with a provider. If the result is blocked, explain the reason in plain, kind words and offer the alternatives it gives.
- Use get_availability to find times, then hold_slot for the time the caller picks.
- You never book, change or cancel an appointment yourself. The system books only after the caller has confirmed. Never tell the caller an appointment is booked unless the session state shows an appointment.
- If you cannot help, or the caller asks for a person, use transfer_to_human.
"""

_clinic_tools = None
_anthropic_client = None


def get_clinic_tools() -> ClinicTools:
    """Return the shared tools object, creating it on first use.

    It is shared by every session because it owns the in-memory clinic data;
    creating it per session would give each caller their own private clinic.
    """
    global _clinic_tools
    if _clinic_tools is None:
        _clinic_tools = ClinicTools(SchedulingAdapter())
    return _clinic_tools


def get_anthropic_client() -> Anthropic:
    """Return the shared Anthropic client. Reads ANTHROPIC_API_KEY from the environment."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic()
    return _anthropic_client


def new_session() -> dict:
    """Create an empty session. The same dict is passed through every turn."""
    return {
        "session_id": uuid.uuid4().hex[:8],
        "stage": "greet",
        "intent": None,
        "patient_id": None,
        "collected": {
            "specialty": None,
            "provider_id": None,
            "location": None,
            "timing": None,
            "problem": None,
        },
        "hold_id": None,
        "slot_id": None,
        "confirmation": None,
        "appointment": None,
        "escalation_id": None,
        "preference_changed": False,
        "stage_entry_index": 0,
        "messages": [],
        "tool_results": [],
    }


def block_to_dict(block) -> dict:
    """Turn an Anthropic response block into a plain dict, so the session stays plain data."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    raise ValueError("Unexpected content block type: " + block.type)


def describe_current_time(current_time: datetime) -> str:
    """One sentence giving the clinic's current date and time, e.g. 'Tuesday 20 October 2026, 5:50 PM Central'."""
    hour_on_twelve_hour_clock = current_time.hour % 12
    if hour_on_twelve_hour_clock == 0:
        hour_on_twelve_hour_clock = 12
    if current_time.hour < 12:
        am_or_pm = "AM"
    else:
        am_or_pm = "PM"
    return (
        f"The current date and time is {current_time.strftime('%A')} {current_time.day} "
        f"{current_time.strftime('%B %Y')}, {hour_on_twelve_hour_clock}:{current_time.minute:02d} "
        f"{am_or_pm} Central."
    )


def build_system_prompt(session: dict) -> str:
    """Build the system prompt: live date and time, the standing rules, this stage's goal, and the session state.

    Built fresh for every model call, so the time and the stage are never stale.
    """
    current_time = get_clinic_tools().now_function()
    stage_name = session["stage"]
    # These are facts the code recorded from tool results; the model needs the ids to call tools.
    state_for_model = {
        "patient_id": session["patient_id"],
        "collected": session["collected"],
        "hold_id": session["hold_id"],
        "slot_id": session["slot_id"],
        "appointment": session["appointment"],
    }
    return (
        describe_current_time(current_time)
        + "\n\n"
        + SYSTEM_PROMPT
        + f"\nCurrent stage: {stage_name}. {STAGES[stage_name]['goal']}\n"
        + "\nSession state, recorded by the system: "
        + json.dumps(state_for_model)
    )


def run_tool(session: dict, tool_name: str, arguments: dict) -> dict:
    """Run one tool the model asked for. Never raises: any failure becomes a message the model can act on.

    This is the one place we catch broadly, on purpose: a malformed tool call from the model
    must cost one turn, not end the phone call. The real error is logged with its traceback.
    """
    if tool_name not in STAGES[session["stage"]]["allowed_tools"]:
        return {"message": "There is no tool called " + tool_name + " available right now. Use one of the tools you were given."}

    try:
        if tool_name in SESSION_TOOL_FUNCTIONS:
            return SESSION_TOOL_FUNCTIONS[tool_name](session, **arguments)
        tool_method = getattr(get_clinic_tools(), tool_name)
        if tool_name == "hold_slot":
            # The session id comes from code, never from the model.
            return tool_method(session_id=session["session_id"], **arguments)
        return tool_method(**arguments)
    except TypeError as error:
        # Missing, extra or wrongly typed arguments. Python's message names the argument, which helps the model.
        logger.exception("Tool %s was called with bad arguments %s", tool_name, arguments)
        return {
            "message": (
                f"{tool_name} was called with missing or invalid arguments ({error}). "
                f"Check the required fields and their types, then call it again."
            )
        }
    except Exception:
        logger.exception("Tool %s failed with arguments %s", tool_name, arguments)
        return {
            "message": (
                f"The {tool_name} tool failed unexpectedly. Check the values you passed, such as ids and "
                f"dates, and try once more. If it keeps failing, tell the caller you are having trouble "
                f"and offer to transfer them to staff."
            )
        }


def record_session_effects(session: dict, tool_name: str, result: dict) -> None:
    """Copy facts that code can read straight off a tool result into the session.

    The stage is not touched here: only the state machine moves it.
    """
    if tool_name == "verify_patient" and result.get("status") == "verified":
        session["patient_id"] = result["patient_id"]
    if tool_name == "create_patient" and result.get("status") == "created":
        session["patient_id"] = result["patient_id"]
    if tool_name == "hold_slot" and result.get("ok"):
        session["hold_id"] = result["hold_id"]
        session["slot_id"] = result["slot_id"]


def reply_with_fallback(session: dict) -> str:
    """Record the fallback as the assistant's turn, so the message history keeps alternating roles."""
    session["messages"].append({"role": "assistant", "content": [{"type": "text", "text": FALLBACK_REPLY}]})
    return FALLBACK_REPLY


def reply_with_scripted_text(session: dict, scripted_text: str) -> str:
    """Record a fixed message from safety.py as the assistant's turn, so the message history keeps alternating roles."""
    session["messages"].append({"role": "assistant", "content": [{"type": "text", "text": scripted_text}]})
    return scripted_text


def move_to_stage(session: dict, new_stage: str, on_stage_change: Optional[Callable[[str, str], None]]) -> None:
    """Move to a stage on behalf of a safety check, and tell the caller of handle_turn about it."""
    old_stage = session["stage"]
    change_stage(session, new_stage, get_clinic_tools())
    if on_stage_change is not None:
        on_stage_change(old_stage, new_stage)


def create_escalation_record(session: dict, red_flag: dict, user_text: str) -> None:
    """Call transfer_to_human from code, so a staff record exists before the scripted message promises a human."""
    reason = f"Red flag, tier {red_flag['tier']}: {red_flag['reason']}"
    summary = (
        f"Caller said: {user_text!r}. Stage when it happened: {session['stage']}. "
        f"Patient id: {session['patient_id']}. Intent: {session['intent']}. "
        f"Collected so far: {json.dumps(session['collected'])}."
    )
    transfer_result = get_clinic_tools().transfer_to_human(reason, summary)
    session["tool_results"].append(
        {
            "tool": "transfer_to_human",
            "arguments": {"reason": reason, "summary": summary},
            "result": transfer_result,
        }
    )
    session["escalation_id"] = transfer_result["task_id"]


def handle_turn(
    session: dict,
    user_text: str,
    on_tool_call: Optional[Callable[[str, dict, dict], None]] = None,
    on_stage_change: Optional[Callable[[str, str], None]] = None,
) -> tuple[str, dict]:
    """Run one caller turn through the model and its tools; return (reply_text, session).

    on_tool_call, if given, is called as each tool finishes with (name, arguments, result).
    on_stage_change, if given, is called for each stage transition with (old_stage, new_stage).
    The CLI uses both to show what the agent is doing.
    """
    session["messages"].append({"role": "user", "content": [{"type": "text", "text": user_text}]})

    # Safety first, in every stage including escalate and transfer, and before any model call.
    red_flag = check_red_flags(user_text)
    if red_flag is not None:
        logger.warning("Red flag tier %s (%s) in caller text: %r", red_flag["tier"], red_flag["reason"], user_text)
        # Already in escalate: say the script again but do not announce a stage change that did not happen.
        # The record is made before the stage moves, so its summary names the stage the caller was in.
        create_escalation_record(session, red_flag, user_text)
        if session["stage"] != "escalate":
            move_to_stage(session, "escalate", on_stage_change)
        return reply_with_scripted_text(session, red_flag["message"]), session

    client = get_anthropic_client()
    clinic_tools = get_clinic_tools()

    # Each turn starts by checking whether the last turn's facts move us to a new stage.
    for old_stage, new_stage in advance_stage(session, clinic_tools):
        if on_stage_change is not None:
            on_stage_change(old_stage, new_stage)

    grounding_retry_used = False
    correction_note = ""

    # A hard cap: a loop that never ends is dead air on a phone call.
    for call_number in range(MAX_MODEL_CALLS_PER_TURN):
        # Prompt and tools are rebuilt from the stage on every call, since a tool result can change the stage.
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_REPLY_TOKENS,
            system=build_system_prompt(session) + correction_note,
            messages=session["messages"],
            tools=tool_schemas_for_stage(session["stage"]),
        )

        assistant_blocks = []
        for block in response.content:
            assistant_blocks.append(block_to_dict(block))
        session["messages"].append({"role": "assistant", "content": assistant_blocks})

        tool_use_blocks = []
        reply_pieces = []
        for block_dict in assistant_blocks:
            if block_dict["type"] == "tool_use":
                tool_use_blocks.append(block_dict)
            else:
                reply_pieces.append(block_dict["text"])

        if len(tool_use_blocks) == 0:
            reply_text = " ".join(reply_pieces).strip()
            if reply_text == "":
                # The model said nothing. Drop its empty turn and use the fallback instead.
                session["messages"].pop()
                return reply_with_fallback(session), session

            grounding = check_grounding(
                reply_text,
                session["tool_results"],
                clinic_tools.adapter.canonical_locations,
                describe_current_time(clinic_tools.now_function()),
            )
            if grounding["passed"]:
                return reply_text, session

            logger.warning(
                "Grounding check failed (rule: %s, span: %r) on reply: %r", grounding["rule"], grounding["span"], reply_text
            )
            # The rejected reply must not stay in history, or the model would treat it as something it already said.
            session["messages"].pop()
            if not grounding_retry_used:
                grounding_retry_used = True
                correction_note = (
                    f"\nYour previous reply was rejected by a safety check (rule: {grounding['rule']}, "
                    f"words: {grounding['span']}). Write the reply again using only facts from tool results, "
                    f"with no medical advice, no guesses about symptoms, and no reassurance."
                )
                continue

            # Two failures in a row: stop improvising and hand over. Never downgrade an escalation.
            if session["stage"] != "escalate" and session["stage"] != "transfer":
                move_to_stage(session, "transfer", on_stage_change)
            return reply_with_scripted_text(session, GROUNDING_FALLBACK_MESSAGE), session

        tool_result_blocks = []
        for tool_use_block in tool_use_blocks:
            tool_name = tool_use_block["name"]
            arguments = tool_use_block["input"]
            result = run_tool(session, tool_name, arguments)
            record_session_effects(session, tool_name, result)
            session["tool_results"].append({"tool": tool_name, "arguments": arguments, "result": result})
            if on_tool_call is not None:
                on_tool_call(tool_name, arguments, result)
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_block["id"],
                    "content": json.dumps(result),
                }
            )
        session["messages"].append({"role": "user", "content": tool_result_blocks})

        # Check again now: a tool result may have moved us on, and the next model call must use the new stage's tools.
        for old_stage, new_stage in advance_stage(session, clinic_tools):
            if on_stage_change is not None:
                on_stage_change(old_stage, new_stage)

    return reply_with_fallback(session), session
