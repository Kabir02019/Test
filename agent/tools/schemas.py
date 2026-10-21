"""TOOL_SCHEMAS: the descriptions of the model-facing tools, in Anthropic tool-calling format.
Does not hold any implementation, and never a schema for a write tool (book, reschedule, cancel)."""

# ----------------------------------------------------------------------
# Schemas for Anthropic tool calling. Keyed by tool name so the state machine can pick a
# subset per stage. The three write tools are intentionally absent.
# ----------------------------------------------------------------------

TOOL_SCHEMAS = {
    "verify_patient": {
        "name": "verify_patient",
        "description": (
            "Verify an existing patient from their full name and date of birth. Returns a patient_id "
            "on success. On failure it does not say whether the person exists; do not guess or hint."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The caller's full name, first and last."},
                "dob": {"type": "string", "description": "Date of birth as YYYY-MM-DD."},
            },
            "required": ["name", "dob"],
        },
    },
    "create_patient": {
        "name": "create_patient",
        "description": (
            "Register a new patient. Only use after verify_patient failed and the caller confirms they are new. "
            "If a record with the same date of birth and phone exists, you get a possible_duplicate result instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name, first and last."},
                "dob": {"type": "string", "description": "Date of birth as YYYY-MM-DD."},
                "phone": {"type": "string", "description": "10-digit phone number."},
                "insurance": {
                    "type": "string",
                    "description": "Insurance plan name, such as Aetna, BCBS, Cigna, Medicare, Medicaid or UnitedHealth. Empty if none.",
                },
                "plan_type": {
                    "type": "string",
                    "enum": ["HMO", "PPO"],
                    "description": "Plan type if the caller knows it. Omit if unknown.",
                },
            },
            "required": ["name", "dob", "phone", "insurance"],
        },
    },
    "match_specialty": {
        "name": "match_specialty",
        "description": (
            "Turn what the caller describes (symptoms, body part, kind of visit) into ranked candidate "
            "specialties with a confidence and a reason. Use when the caller does not know which doctor they need."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "The caller's own words about what they need."},
            },
            "required": ["description"],
        },
    },
    "search_providers": {
        "name": "search_providers",
        "description": (
            "Search providers. Every filter is optional and filters combine. Use name alone when the caller "
            "asks for a doctor by name. Several matches means ask the caller which one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Provider name or part of it, such as 'Kim'."},
                "specialty": {"type": "string", "description": "A specialty name such as Cardiology."},
                "location": {"type": "string", "description": "Lakeview, Downtown or Oak Park."},
                "language": {"type": "string", "description": "A language the provider speaks."},
                "gender": {"type": "string", "description": "Provider gender: F or M."},
                "insurance": {"type": "string", "description": "Insurance plan the provider must accept."},
                "accepting_new": {"type": "boolean", "description": "True to only return providers accepting new patients."},
            },
            "required": [],
        },
    },
    "find_appointments": {
        "name": "find_appointments",
        "description": "List a verified patient's upcoming appointments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "The patient_id from verify_patient."},
            },
            "required": ["patient_id"],
        },
    },
    "check_eligibility": {
        "name": "check_eligibility",
        "description": (
            "Check whether a patient may book a provider (insurance, referral, age, new-patient rules). "
            "Call before offering times. If it is blocked, explain the reason in plain words and offer the alternatives."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "The patient_id."},
                "provider_id": {"type": "string", "description": "The provider_id."},
                "appointment_date": {
                    "type": "string",
                    "description": "The date of the visit as YYYY-MM-DD, if a slot has been chosen. Omit it before that.",
                },
            },
            "required": ["patient_id", "provider_id"],
        },
    },
    "get_availability": {
        "name": "get_availability",
        "description": "Get the earliest open appointment slots for one or more providers. Returns at most 5.",
        "input_schema": {
            "type": "object",
            "properties": {
                "provider_ids": {"type": "array", "items": {"type": "string"}, "description": "One or more provider_ids."},
                "date_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "description": "First acceptable date, YYYY-MM-DD."},
                        "end": {"type": "string", "description": "Last acceptable date, YYYY-MM-DD."},
                    },
                },
                "time_of_day": {"type": "string", "enum": ["morning", "afternoon", "any"]},
            },
            "required": ["provider_ids"],
        },
    },
    "hold_slot": {
        "name": "hold_slot",
        "description": "Hold a slot for a few minutes while the patient decides. Returns a hold_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "slot_id": {"type": "string", "description": "The slot_id from get_availability."},
            },
            "required": ["slot_id"],
        },
    },
    "transfer_to_human": {
        "name": "transfer_to_human",
        "description": "Hand the caller to clinic staff. Use when you cannot help, or the caller asks for a person.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why the caller is being transferred."},
                "summary": {"type": "string", "description": "A short summary of the conversation so far for the staff member."},
            },
            "required": ["reason", "summary"],
        },
    },
}
