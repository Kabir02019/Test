"""WriteTools: slot holds, the three booking writes, and the idempotency store (write_results).
Does not hold read-only lookups or eligibility rules. The booking writes have no schema; only the state machine calls them."""

from datetime import timedelta
from typing import Optional

from agent.tools.speech import speak_time

HOLD_SECONDS = 300


class WriteTools:
    """Holds and writes. Relies on self.adapter, self.now_function, self.holds and self.write_results."""

    def _remove_expired_holds(self) -> None:
        """Forget holds past their expiry so their slots can be held again."""
        current_time = self.now_function()
        expired_hold_ids = []
        for hold_id, hold in self.holds.items():
            if current_time >= hold["expires_at"]:
                expired_hold_ids.append(hold_id)
        for hold_id in expired_hold_ids:
            del self.holds[hold_id]

    def _active_hold(self, hold_id: str) -> Optional[dict]:
        """Return the hold if it exists and has not expired."""
        self._remove_expired_holds()
        return self.holds.get(hold_id)

    def hold_slot(self, slot_id: str, session_id: str) -> dict:
        """Reserve an open slot for this session for a few minutes while the patient decides.

        session_id is filled in by the state machine; it is not in the model's schema.
        """
        self._remove_expired_holds()
        slot = self.adapter.get_slot(slot_id)
        if slot is None:
            return {"ok": False, "message": f"There is no slot with id {slot_id}. Check availability again."}
        if slot["status"] != "open":
            return {"ok": False, "message": "That slot has just been taken. Check availability and offer another time."}

        for existing_hold in self.holds.values():
            if existing_hold["slot_id"] != slot_id:
                continue
            if existing_hold["session_id"] == session_id:
                return self._hold_result(existing_hold, slot)
            return {"ok": False, "message": "Someone else is holding that slot right now. Offer another time."}

        hold_id = f"HLD-{self.next_hold_number:04d}"
        self.next_hold_number = self.next_hold_number + 1
        new_hold = {
            "hold_id": hold_id,
            "slot_id": slot_id,
            "session_id": session_id,
            "expires_at": self.now_function() + timedelta(seconds=HOLD_SECONDS),
        }
        self.holds[hold_id] = new_hold
        return self._hold_result(new_hold, slot)

    def _hold_result(self, hold: dict, slot: dict) -> dict:
        seconds_left = int((hold["expires_at"] - self.now_function()).total_seconds())
        return {
            "ok": True,
            "hold_id": hold["hold_id"],
            "slot_id": hold["slot_id"],
            "provider_name": self._provider_name(slot["provider_id"]),
            "location": slot["location"],
            "spoken_time": speak_time(slot["start"]),
            "expires_in_seconds": seconds_left,
            "message": "Slot held. Read the details back to the patient and ask them to confirm.",
        }

    def book_appointment(self, hold_id: str, patient_id: str, idempotency_key: str, visit_reason: str = "") -> dict:
        """Turn a live hold into a booked appointment."""
        earlier_result = self.write_results.get(idempotency_key)
        if earlier_result is not None:
            return earlier_result

        hold = self._active_hold(hold_id)
        if hold is None:
            return {"ok": False, "message": "That hold has expired or does not exist. Check availability and hold a slot again."}

        booking = self.adapter.book(patient_id, hold["slot_id"], visit_reason, idempotency_key)
        del self.holds[hold_id]
        if not booking["ok"]:
            return {"ok": False, "message": booking["reason"]}

        summary = self._appointment_summary(booking["appointment"])
        result = {
            "ok": True,
            "appointment": summary,
            "message": f"Booked with {summary['provider_name']} on {summary['spoken_time']} at {summary['location']}.",
        }
        self.write_results[idempotency_key] = result
        return result

    def reschedule_appointment(self, appointment_id: str, hold_id: str, idempotency_key: str) -> dict:
        """Move an appointment: book the new slot first, then cancel the old one.

        Booking first means a failure can never leave the patient with no appointment.
        If only the cancel fails, the booking still stands and a staff task is raised to remove the old one.
        """
        earlier_result = self.write_results.get(idempotency_key)
        if earlier_result is not None:
            return earlier_result

        old_appointment = self.adapter.get_appointment(appointment_id)
        if old_appointment is None:
            return {"ok": False, "message": f"There is no appointment with id {appointment_id}."}
        if old_appointment["status"] != "booked":
            return {"ok": False, "message": f"That appointment is {old_appointment['status']}, so it cannot be moved."}

        hold = self._active_hold(hold_id)
        if hold is None:
            return {
                "ok": False,
                "message": "That hold has expired or does not exist. Hold the new slot again. The original appointment is unchanged.",
            }

        booking = self.adapter.book(
            old_appointment["patient_id"],
            hold["slot_id"],
            old_appointment["visit_reason"],
            idempotency_key + ":book",
        )
        del self.holds[hold_id]
        if not booking["ok"]:
            return {"ok": False, "message": booking["reason"] + " The original appointment is unchanged."}

        new_summary = self._appointment_summary(booking["appointment"])
        cancellation = self.adapter.cancel(appointment_id, idempotency_key + ":cancel")

        if cancellation["ok"]:
            result = {
                "ok": True,
                "appointment": new_summary,
                "staff_task_needed": False,
                "message": f"Moved to {new_summary['provider_name']} on {new_summary['spoken_time']} at {new_summary['location']}. The old appointment is cancelled.",
            }
        else:
            self.staff_tasks.append(
                {
                    "kind": "cancel_old_appointment",
                    "reason": f"Reschedule booked {new_summary['appointment_id']} but cancelling {appointment_id} failed: {cancellation['reason']}",
                    "summary": f"Patient {old_appointment['patient_id']} may have two appointments until staff cancel {appointment_id}.",
                }
            )
            result = {
                "ok": True,
                "appointment": new_summary,
                "staff_task_needed": True,
                "message": (
                    f"The new appointment is booked for {new_summary['spoken_time']}. The old one could not be "
                    f"cancelled automatically, so a staff task was created to cancel it. Tell the patient the new "
                    f"time is confirmed and that staff will cancel the old one."
                ),
            }
        self.write_results[idempotency_key] = result
        return result

    def cancel_appointment(self, appointment_id: str, reason: str, idempotency_key: str) -> dict:
        """Cancel a booked appointment and reopen its slot."""
        earlier_result = self.write_results.get(idempotency_key)
        if earlier_result is not None:
            return earlier_result

        cancellation = self.adapter.cancel(appointment_id, idempotency_key)
        if not cancellation["ok"]:
            return {"ok": False, "message": cancellation["reason"]}

        print(f"[tools] cancelled {appointment_id}; reason given: {reason}")
        summary = self._appointment_summary(cancellation["appointment"])
        result = {
            "ok": True,
            "appointment": summary,
            "message": f"Cancelled the appointment with {summary['provider_name']} on {summary['spoken_time']}.",
        }
        self.write_results[idempotency_key] = result
        return result
