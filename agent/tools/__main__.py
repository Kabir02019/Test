"""The manual checks for the tools: python -m agent.tools (from the project root).
Does not hold any tool logic; it only exercises ClinicTools and prints the results."""

import json
from datetime import datetime, timedelta

from adapter.scheduling_adapter import SchedulingAdapter
from agent.tools import TOOL_SCHEMAS, ClinicTools
from agent.tools.helpers import CLINIC_TIMEZONE
from agent.tools.writes import HOLD_SECONDS


# ----------------------------------------------------------------------
# Manual checks
# ----------------------------------------------------------------------


class FakeClock:
    """A clock that only moves when the test moves it."""

    def __init__(self, start: datetime):
        self.current_time = start

    def now(self) -> datetime:
        return self.current_time


def show(label: str, result: dict) -> None:
    """Print a labelled tool result."""
    print(f"\n>>> {label}")
    print(json.dumps(result, indent=2))


def failing_cancel(appointment_id: str, idempotency_key: str) -> dict:
    """Stands in for adapter.cancel to exercise the reschedule failure path."""
    return {"ok": False, "reason": "The scheduling system did not respond."}


if __name__ == "__main__":
    clock = FakeClock(datetime(2026, 10, 20, 10, 0, tzinfo=CLINIC_TIMEZONE))
    adapter = SchedulingAdapter()
    tools = ClinicTools(adapter, now_function=clock.now)

    print("\n=========== verify_patient ===========")
    show("verify Maria Delgado", tools.verify_patient("Maria Delgado", "1979-11-02"))
    wrong_dob = tools.verify_patient("Maria Delgado", "1980-01-01")
    no_such_person = tools.verify_patient("Zelda Nobody", "1980-01-01")
    show("wrong DOB", wrong_dob)
    print("wrong DOB and unknown person give identical results:", wrong_dob == no_such_person)

    print("\n=========== create_patient ===========")
    show("duplicate (Jon Smith, same DOB+phone as PAT-2001)", tools.create_patient("Jon Smith", "1964-03-12", "312-555-0141", "BCBS"))
    show("new patient", tools.create_patient("Test Person", "1990-01-01", "312-555-0000", "Cigna", "PPO"))

    print("\n=========== match_specialty ===========")
    show("knee", tools.match_specialty("my knee hurts after running and my shoulder is sore"))
    show("rash", tools.match_specialty("I have an itchy rash"))
    show("named", tools.match_specialty("heart"))
    show("vague", tools.match_specialty("I just feel off"))

    print("\n=========== search_providers ===========")
    show("Cardiology", tools.search_providers(specialty="Cardiology"))
    show("name=Kim, accepting_new=True", tools.search_providers(name="Dr Kim", accepting_new=True))
    show("not a specialty", tools.search_providers(specialty="Podiatry"))
    show("Cardiology + French", tools.search_providers(specialty="Cardiology", language="French"))
    show("unknown name", tools.search_providers(name="Nguyen"))

    print("\n=========== find_appointments ===========")
    show("PAT-2003", tools.find_appointments("PAT-2003"))
    show("PAT-2007 (none)", tools.find_appointments("PAT-2007"))

    print("\n=========== check_eligibility ===========")
    eligibility_cases = [
        ("HMO + valid referral", "PAT-2003", "PRV-1001", "2026-10-23"),
        ("HMO referral, visit after window", "PAT-2003", "PRV-1001", "2026-12-05"),
        ("HMO referral expired by visit date", "PAT-2014", "PRV-1007", "2026-10-30"),
        ("HMO, no cardiology referral", "PAT-2005", "PRV-1001", "2026-10-28"),
        ("HMO, dermatology referral valid", "PAT-2005", "PRV-1005", "2026-10-28"),
        ("child, pediatrics", "PAT-2006", "PRV-1013", "2026-10-28"),
        ("adult, pediatrics", "PAT-2012", "PRV-1014", "2026-10-28"),
        ("turns 18 before the visit", "PAT-2009", "PRV-1014", "2027-03-01"),
        ("still 17 on the visit date", "PAT-2009", "PRV-1014", "2026-11-01"),
        ("plan not accepted", "PAT-2007", "PRV-1003", "2026-10-28"),
        ("not accepting new patients", "PAT-2001", "PRV-1003", "2026-10-28"),
        ("no insurance on file", "PAT-2011", "PRV-1004", "2026-10-28"),
        ("provider has no insurance data", "PAT-2001", "PRV-1016", "2026-10-28"),
        ("no date given", "PAT-2007", "PRV-1004", None),
    ]
    for case_label, case_patient, case_provider, case_date in eligibility_cases:
        show(case_label, tools.check_eligibility(case_patient, case_provider, case_date))

    # R5 allowed path: no real patient has seen Dr Kim, so make Dr Kim PAT-2001's PCP for this one check.
    adapter.patients["PAT-2001"]["pcp_provider_id"] = "PRV-1003"
    show("existing patient of a closed provider", tools.check_eligibility("PAT-2001", "PRV-1003", "2026-10-28"))
    adapter.patients["PAT-2001"]["pcp_provider_id"] = "PRV-1009"

    print("\n=========== get_availability ===========")
    afternoon_result = tools.get_availability(["PRV-1001", "PRV-1004"], {"start": "2026-10-23", "end": "2026-10-30"}, "afternoon")
    show("two cardiologists, afternoons", afternoon_result)
    show("nothing in range", tools.get_availability(["PRV-1004"], {"start": "2027-06-01", "end": "2027-06-30"}))

    print("\n=========== hold_slot ===========")
    open_slot_ids = []
    for slot in tools.get_availability(["PRV-1004"])["slots"]:
        open_slot_ids.append(slot["slot_id"])
    first_hold = tools.hold_slot(open_slot_ids[0], "session-A")
    show("session A holds slot 0", first_hold)
    show("session B tries the same slot", tools.hold_slot(open_slot_ids[0], "session-B"))
    show("session A holds it again (same hold)", tools.hold_slot(open_slot_ids[0], "session-A"))
    clock.current_time = clock.current_time + timedelta(seconds=HOLD_SECONDS + 1)
    show("session B after expiry", tools.hold_slot(open_slot_ids[0], "session-B"))
    show("session A books an expired hold", tools.book_appointment(first_hold["hold_id"], "PAT-2007", "demo-book-expired"))

    print("\n=========== write tools (state machine only) ===========")
    hold_for_booking = tools.hold_slot(open_slot_ids[1], "session-A")
    booked = tools.book_appointment(hold_for_booking["hold_id"], "PAT-2007", "demo-book-1", "new patient visit")
    show("book_appointment", booked)
    show("book_appointment, same key again", tools.book_appointment(hold_for_booking["hold_id"], "PAT-2007", "demo-book-1", "new patient visit"))
    booked_appointment_id = booked["appointment"]["appointment_id"]

    hold_for_move = tools.hold_slot(open_slot_ids[2], "session-A")
    show("reschedule_appointment", tools.reschedule_appointment(booked_appointment_id, hold_for_move["hold_id"], "demo-move-1"))
    show("PAT-2007 appointments after reschedule (expect exactly 1)", tools.find_appointments("PAT-2007"))
    moved_appointment_id = tools.find_appointments("PAT-2007")["appointments"][0]["appointment_id"]

    hold_for_failed_move = tools.hold_slot(open_slot_ids[3], "session-A")
    real_cancel = adapter.cancel
    adapter.cancel = failing_cancel
    show("reschedule where the cancel fails", tools.reschedule_appointment(moved_appointment_id, hold_for_failed_move["hold_id"], "demo-move-2"))
    adapter.cancel = real_cancel
    print("staff tasks so far:", json.dumps(tools.staff_tasks, indent=2))
    show("PAT-2007 now has two appointments until staff act", tools.find_appointments("PAT-2007"))
    show("cancel_appointment (what staff would do)", tools.cancel_appointment(moved_appointment_id, "cleanup after failed reschedule", "demo-cancel-1"))
    show("PAT-2007 appointments (expect exactly 1)", tools.find_appointments("PAT-2007"))
    show("cancel an appointment that is already cancelled", tools.cancel_appointment(moved_appointment_id, "again", "demo-cancel-2"))

    print("\n=========== transfer_to_human ===========")
    show("during staffed hours", tools.transfer_to_human("caller asked for a person", "Wants to discuss billing."))
    clock.current_time = datetime(2026, 10, 20, 19, 30, tzinfo=CLINIC_TIMEZONE)
    show("after hours", tools.transfer_to_human("caller asked for a person", "Wants to discuss billing."))
    clock.current_time = datetime(2026, 10, 24, 11, 0, tzinfo=CLINIC_TIMEZONE)
    show("Saturday", tools.transfer_to_human("caller asked for a person", "Wants to discuss billing."))

    print("\n=========== schemas ===========")
    write_tool_names = ["book_appointment", "reschedule_appointment", "cancel_appointment"]
    for write_tool_name in write_tool_names:
        assert write_tool_name not in TOOL_SCHEMAS, write_tool_name + " must never have a schema"
    for schema_name in TOOL_SCHEMAS:
        assert hasattr(tools, schema_name), schema_name + " has a schema but no method"
    print("schemas:", ", ".join(TOOL_SCHEMAS.keys()))
    print("no write tool appears in TOOL_SCHEMAS; every schema has a matching method")
