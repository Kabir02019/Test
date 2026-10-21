"""The conversation's stages, the tools allowed in each, and the rules for moving between them.

Why this exists: the model only ever sees the tools for the CURRENT stage, and every write
(booking) is done by code in this module after the patient's confirmation has been recorded.
book_appointment, reschedule_appointment and cancel_appointment appear in no stage's tool list
and in no schema.

Stages are data (the STAGES dict). Each entry has:
  goal           one paragraph added to the system prompt while in this stage
  allowed_tools  names of the tools the model may use in this stage
  exit_check     function(session) -> next stage name, or None to stay
  on_enter       function(session, clinic_tools) run when the stage is entered, or None

Print every stage and its tools with:  python -m agent.state_machine   (from the project root)
"""

from typing import Optional

from agent.tools import TOOL_SCHEMAS, ClinicTools

# The model's answer to "do you confirm?" must be at least this sure before code books.
CONFIRMATION_CONFIDENCE_THRESHOLD = 0.8

MAX_FAILED_VERIFICATIONS = 2

# Stop a bug from becoming an endless loop of transitions inside one check.
MAX_TRANSITIONS_PER_CHECK = 10

WRITE_TOOL_NAMES = ("book_appointment", "reschedule_appointment", "cancel_appointment")

# In these stages a changed specialty or provider throws away work already done.
STAGES_AFTER_CHOOSING = ("check_rules", "offer_slots", "confirm")


# ----------------------------------------------------------------------
# Tools that only write into the session. They touch no clinic data.
# ----------------------------------------------------------------------


def record_intent(session: dict, intent: str) -> dict:
    """Store why the caller is calling. Does nothing else."""
    session["intent"] = intent
    return {"recorded": True}


def same_text(first_text: Optional[str], second_text: Optional[str]) -> bool:
    """Case-insensitive equality that treats None as 'nothing'."""
    if first_text is None or second_text is None:
        return False
    return first_text.strip().lower() == second_text.strip().lower()


def record_preferences(
    session: dict,
    specialty: Optional[str] = None,
    provider_id: Optional[str] = None,
    location: Optional[str] = None,
    timing: Optional[str] = None,
) -> dict:
    """Store what the caller has chosen. Does nothing else, except flag a change of mind.

    A new specialty makes the old provider invalid, so the provider is cleared (unless a new
    one is given in the same call). If the specialty or provider changed after a choice had
    already been acted on, the flag tells the state machine to go back and redo that work.
    """
    collected = session["collected"]
    specialty_changed = collected["specialty"] is not None and specialty is not None and not same_text(
        collected["specialty"], specialty
    )
    provider_changed = (
        collected["provider_id"] is not None and provider_id is not None and collected["provider_id"] != provider_id
    )

    if specialty is not None:
        collected["specialty"] = specialty
        if specialty_changed:
            collected["provider_id"] = None
    if provider_id is not None:
        collected["provider_id"] = provider_id
    if location is not None:
        collected["location"] = location
    if timing is not None:
        collected["timing"] = timing

    if session["stage"] in STAGES_AFTER_CHOOSING and (specialty_changed or provider_changed):
        session["preference_changed"] = True
    return {"recorded": True}


def record_confirmation(session: dict, confirmed: bool, confidence: float) -> dict:
    """Store the model's reading of the caller's yes/no. Does nothing else.

    Code, not the model, decides what happens next (see exit_from_confirm).
    """
    session["confirmation"] = {"confirmed": confirmed, "confidence": confidence}
    return {"recorded": True}


SESSION_TOOL_FUNCTIONS = {
    "record_intent": record_intent,
    "record_preferences": record_preferences,
    "record_confirmation": record_confirmation,
}

SESSION_TOOL_SCHEMAS = {
    "record_intent": {
        "name": "record_intent",
        "description": "Record why the caller is calling. Use 'book' if they want to make an appointment, 'other' for anything else.",
        "input_schema": {
            "type": "object",
            "properties": {"intent": {"type": "string", "enum": ["book", "other"]}},
            "required": ["intent"],
        },
    },
    "record_preferences": {
        "name": "record_preferences",
        "description": (
            "Record what the caller has chosen so far. Pass only what they have actually chosen. "
            "Use the provider_id from search results, never a name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "specialty": {"type": "string", "description": "The specialty, such as Cardiology."},
                "provider_id": {"type": "string", "description": "The chosen provider's provider_id."},
                "location": {"type": "string", "description": "A location the caller prefers."},
                "timing": {"type": "string", "description": "When the caller wants to come, in their words."},
            },
            "required": [],
        },
    },
    "record_confirmation": {
        "name": "record_confirmation",
        "description": (
            "Record the caller's answer to 'shall I book this?'. confirmed is true for a yes and false for a no. "
            "confidence is from 0 to 1: how sure you are that you understood their answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "confirmed": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["confirmed", "confidence"],
        },
    },
}

# Every schema the state machine can hand out: the clinic tools plus the session-only tools.
ALL_TOOL_SCHEMAS = dict(TOOL_SCHEMAS)
ALL_TOOL_SCHEMAS.update(SESSION_TOOL_SCHEMAS)


# ----------------------------------------------------------------------
# Helpers that read the session
# ----------------------------------------------------------------------


def count_failed_verifications(session: dict) -> int:
    """How many verify_patient calls came back not_found this session."""
    failures = 0
    for entry in session["tool_results"]:
        if entry["tool"] == "verify_patient" and entry["result"].get("status") == "not_found":
            failures = failures + 1
    return failures


def transfer_was_requested(session: dict) -> bool:
    """True if the model has called transfer_to_human at any point."""
    for entry in session["tool_results"]:
        if entry["tool"] == "transfer_to_human":
            return True
    return False


def latest_eligibility_result(session: dict) -> Optional[dict]:
    """The newest check_eligibility result for this patient and provider made since entering this stage."""
    results_since_entry = session["tool_results"][session["stage_entry_index"] :]
    for entry in reversed(results_since_entry):
        if entry["tool"] != "check_eligibility":
            continue
        asked_about_patient = entry["arguments"].get("patient_id") == session["patient_id"]
        asked_about_provider = entry["arguments"].get("provider_id") == session["collected"]["provider_id"]
        if asked_about_patient and asked_about_provider:
            return entry["result"]
    return None


def release_hold(session: dict, clinic_tools: ClinicTools) -> None:
    """Drop the slot hold and everything tied to it.

    tools.py has no 'release hold' method, so we remove it from the holds dict directly.
    """
    if session["hold_id"] is not None:
        clinic_tools.holds.pop(session["hold_id"], None)
    session["hold_id"] = None
    session["slot_id"] = None
    session["confirmation"] = None


# ----------------------------------------------------------------------
# exit_checks: each returns the next stage name, or None to stay
# ----------------------------------------------------------------------


def exit_from_greet(session: dict) -> Optional[str]:
    if session["intent"] is None:
        return None
    if session["intent"] == "book":
        return "identify"
    # Rescheduling, cancelling and questions have no flow yet, so a person has to take them.
    return "transfer"


def exit_from_identify(session: dict) -> Optional[str]:
    if session["patient_id"] is not None:
        return "understand_need"
    if count_failed_verifications(session) >= MAX_FAILED_VERIFICATIONS:
        return "transfer"
    return None


def exit_from_understand_need(session: dict) -> Optional[str]:
    # A provider is required, not just a specialty: the rules (insurance, new patients,
    # referral) are checked per provider.
    if session["collected"]["provider_id"] is not None:
        return "check_rules"
    return None


def exit_from_check_rules(session: dict) -> Optional[str]:
    eligibility_result = latest_eligibility_result(session)
    if eligibility_result is None:
        return None
    if eligibility_result.get("allowed") is True:
        return "offer_slots"

    # Blocked: remember why, forget this provider, and let the agent offer an alternative.
    session["collected"]["problem"] = {
        "kind": "not_eligible",
        "blocked_by": eligibility_result.get("blocked_by"),
        "message": eligibility_result.get("message"),
        "alternatives": eligibility_result.get("alternatives"),
    }
    session["collected"]["provider_id"] = None
    return "understand_need"


def exit_from_offer_slots(session: dict) -> Optional[str]:
    if session["hold_id"] is not None:
        return "confirm"
    return None


def exit_from_confirm(session: dict) -> Optional[str]:
    confirmation = session["confirmation"]
    if confirmation is None:
        return None
    if confirmation["confirmed"] is False:
        return "offer_slots"
    if confirmation["confirmed"] is True and confirmation["confidence"] >= CONFIRMATION_CONFIDENCE_THRESHOLD:
        return "book"
    # A yes we are not sure about: stay here, and the agent asks again.
    return None


def exit_from_book(session: dict) -> Optional[str]:
    if session["appointment"] is not None:
        return "wrap_up"
    return "offer_slots"


def never_exits(session: dict) -> Optional[str]:
    return None


# ----------------------------------------------------------------------
# on_enter actions
# ----------------------------------------------------------------------


def enter_check_rules(session: dict, clinic_tools: ClinicTools) -> None:
    session["collected"]["problem"] = None


def enter_offer_slots(session: dict, clinic_tools: ClinicTools) -> None:
    # Arriving here from confirm means the patient said no; arriving after a failed booking
    # means the hold is gone anyway. Either way, start this stage with no hold.
    release_hold(session, clinic_tools)


def enter_confirm(session: dict, clinic_tools: ClinicTools) -> None:
    session["confirmation"] = None


def enter_book(session: dict, clinic_tools: ClinicTools) -> None:
    """The only place a booking is made. Called by code after a clear, recorded yes."""
    # The key is built from the session and the slot, so a repeat of this call cannot book twice.
    idempotency_key = "book:" + session["session_id"] + ":" + session["slot_id"]
    visit_reason = "visit"
    if session["collected"]["specialty"] is not None:
        visit_reason = session["collected"]["specialty"] + " visit"

    booking_result = clinic_tools.book_appointment(
        session["hold_id"], session["patient_id"], idempotency_key, visit_reason
    )
    session["tool_results"].append(
        {
            "tool": "book_appointment",
            "arguments": {
                "hold_id": session["hold_id"],
                "patient_id": session["patient_id"],
                "idempotency_key": idempotency_key,
            },
            "result": booking_result,
        }
    )

    if booking_result["ok"]:
        session["appointment"] = booking_result["appointment"]
        session["hold_id"] = None
        session["slot_id"] = None
        return
    session["collected"]["problem"] = {"kind": "booking_failed", "message": booking_result["message"]}


# ----------------------------------------------------------------------
# The stages
# ----------------------------------------------------------------------

STAGES = {
    "greet": {
        "goal": (
            "You are greeting the caller. Welcome them to Lakeside Medical Group in one short sentence and "
            "find out what they are calling about. As soon as you know, call record_intent: use book if they "
            "want to make an appointment, and other for anything else."
        ),
        "allowed_tools": ["record_intent", "transfer_to_human"],
        "exit_check": exit_from_greet,
        "on_enter": None,
    },
    "identify": {
        "goal": (
            "You are finding out who the caller is. Ask for their full name and date of birth, one at a time, "
            "then call verify_patient. If it does not work, ask them to say the details again. If they say they "
            "are a new patient, ask for their full name, date of birth, phone number and insurance plan, one at "
            "a time, then call create_patient. If create_patient reports a possible duplicate, ask whether that "
            "is them, and if so use verify_patient instead."
        ),
        "allowed_tools": ["verify_patient", "create_patient", "transfer_to_human"],
        "exit_check": exit_from_identify,
        "on_enter": None,
    },
    "understand_need": {
        "goal": (
            "You are finding out what the caller needs. If they are unsure what kind of doctor to see, use "
            "match_specialty. Use search_providers to find doctors and offer two or three. When the caller has "
            "picked a doctor, call record_preferences with the specialty and the provider_id, and with any "
            "location or timing they mentioned. If the session state shows a problem, first explain it to the "
            "caller in plain words, then help them choose another option."
        ),
        "allowed_tools": ["match_specialty", "search_providers", "record_preferences", "transfer_to_human"],
        "exit_check": exit_from_understand_need,
        "on_enter": None,
    },
    "check_rules": {
        "goal": (
            "The caller has chosen a doctor. Call check_eligibility right away with the patient_id and the "
            "provider_id from the session state. If the caller changes their mind about the doctor or the "
            "specialty, call record_preferences with the new choice."
        ),
        "allowed_tools": ["check_eligibility", "record_preferences", "transfer_to_human"],
        "exit_check": exit_from_check_rules,
        "on_enter": enter_check_rules,
    },
    "offer_slots": {
        "goal": (
            "The caller is allowed to book this doctor. Use get_availability for the chosen provider, applying "
            "any date, location or time of day the caller prefers, and offer two or three times. When the caller "
            "picks one, call hold_slot with its slot_id. If the session state shows a problem, such as a time "
            "that was just taken, explain it first. If the caller wants a different doctor or specialty, call "
            "record_preferences with the new choice."
        ),
        "allowed_tools": ["get_availability", "hold_slot", "record_preferences", "transfer_to_human"],
        "exit_check": exit_from_offer_slots,
        "on_enter": enter_offer_slots,
    },
    "confirm": {
        "goal": (
            "A time is being held for the caller. Read back the doctor, the day and time, and the location, then "
            "ask whether they want to book it. When they answer, call record_confirmation with confirmed true or "
            "false and your confidence from 0 to 1 that you understood their answer. Never say the appointment is "
            "booked: the system books it after a clear yes. If you are still in this stage after recording, the "
            "answer was not clear enough, so ask them to confirm again."
        ),
        "allowed_tools": ["record_confirmation", "transfer_to_human"],
        "exit_check": exit_from_confirm,
        "on_enter": enter_confirm,
    },
    "book": {
        "goal": "The system is booking the appointment now. Say nothing.",
        "allowed_tools": [],
        "exit_check": exit_from_book,
        "on_enter": enter_book,
    },
    "wrap_up": {
        "goal": (
            "The appointment is booked. Its details are in the session state under appointment. Tell the caller "
            "the doctor, the day and time, and the location from there, ask if there is anything else, and say "
            "goodbye. Do not offer to change or cancel it."
        ),
        "allowed_tools": ["find_appointments", "transfer_to_human"],
        "exit_check": never_exits,
        "on_enter": None,
    },
    "escalate": {
        "goal": (
            "The caller may need urgent help. Stay calm and brief, tell them you are connecting them to a person "
            "now, and call transfer_to_human."
        ),
        "allowed_tools": ["transfer_to_human"],
        "exit_check": never_exits,
        "on_enter": None,
    },
    "transfer": {
        "goal": (
            "The caller is being handed to staff. If you have not yet called transfer_to_human, call it now with "
            "a reason and a short summary. Then tell the caller exactly what its result says. Do not try to "
            "schedule anything else."
        ),
        "allowed_tools": ["transfer_to_human"],
        "exit_check": never_exits,
        "on_enter": None,
    },
}


# ----------------------------------------------------------------------
# Moving between stages
# ----------------------------------------------------------------------


def tool_schemas_for_stage(stage_name: str) -> list[dict]:
    """The tool schemas the model may see in this stage, and no others."""
    schemas = []
    for tool_name in STAGES[stage_name]["allowed_tools"]:
        schemas.append(ALL_TOOL_SCHEMAS[tool_name])
    return schemas


def change_stage(session: dict, new_stage: str, clinic_tools: ClinicTools) -> None:
    """Move the session to a stage and run that stage's on_enter. Other code, such as safety, can use this too."""
    if new_stage not in STAGES:
        raise ValueError("There is no stage called " + new_stage)
    session["stage"] = new_stage
    session["stage_entry_index"] = len(session["tool_results"])
    on_enter = STAGES[new_stage]["on_enter"]
    if on_enter is not None:
        on_enter(session, clinic_tools)


def find_next_stage(session: dict, clinic_tools: ClinicTools) -> Optional[str]:
    """Decide the next stage, or None. Checks that apply in every stage come before the stage's own exit_check."""
    if session["stage"] != "transfer" and transfer_was_requested(session):
        return "transfer"

    if session["preference_changed"]:
        # The caller changed their mind. Throw away the hold and slot, keep patient and the other preferences.
        session["preference_changed"] = False
        release_hold(session, clinic_tools)
        return "understand_need"

    exit_check = STAGES[session["stage"]]["exit_check"]
    return exit_check(session)


def advance_stage(session: dict, clinic_tools: ClinicTools) -> list[tuple]:
    """Apply transitions until the stage settles. Returns the (old_stage, new_stage) pairs that happened."""
    transitions = []
    for step in range(MAX_TRANSITIONS_PER_CHECK):
        next_stage = find_next_stage(session, clinic_tools)
        if next_stage is None:
            return transitions
        old_stage = session["stage"]
        change_stage(session, next_stage, clinic_tools)
        transitions.append((old_stage, next_stage))
    raise RuntimeError("Stage transitions did not settle; the last stage was " + session["stage"])


if __name__ == "__main__":
    for stage_name, stage in STAGES.items():
        tool_names = ", ".join(stage["allowed_tools"])
        if tool_names == "":
            tool_names = "(none)"
        print(f"{stage_name:16} {tool_names}")

    exposed_tool_names = []
    for stage in STAGES.values():
        for tool_name in stage["allowed_tools"]:
            assert tool_name in ALL_TOOL_SCHEMAS, tool_name + " is allowed in a stage but has no schema"
            exposed_tool_names.append(tool_name)
    for write_tool_name in WRITE_TOOL_NAMES:
        assert write_tool_name not in exposed_tool_names, write_tool_name + " is exposed to the model"
        assert write_tool_name not in ALL_TOOL_SCHEMAS, write_tool_name + " has a schema"
    print("\nNo write tool is allowed in any stage or has a schema: " + ", ".join(WRITE_TOOL_NAMES))
