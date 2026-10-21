"""Loads the clinic's files and exposes scheduling operations over in-memory state.

This is the only module that knows the data lives in CSV/JSON files rather than an EHR.
Run the smoke checks with:  python -m adapter.scheduling_adapter   (from the project root)
"""

import copy
import csv
import json
from pathlib import Path
from typing import Optional

from adapter.normalize import normalize_location, normalize_phone, normalize_specialty

MAX_SLOTS_RETURNED = 5


def split_list(text: str, separator: str) -> list[str]:
    """Split a delimited CSV cell into trimmed items; an empty cell gives an empty list."""
    items = []
    for piece in text.split(separator):
        trimmed_piece = piece.strip()
        if trimmed_piece != "":
            items.append(trimmed_piece)
    return items


def read_csv_rows(path: Path) -> list[dict]:
    """Read a CSV file into a list of dicts keyed by column name."""
    with open(path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def provider_sort_key(provider: dict) -> tuple:
    """Sort providers accepting new patients first, then alphabetically by name."""
    if provider["accepting_new_patients"]:
        accepting_rank = 0
    else:
        accepting_rank = 1
    return (accepting_rank, provider["full_name"])


def slot_start_time(slot: dict) -> str:
    """Sort key for slots. ISO timestamps sort correctly as plain strings."""
    return slot["start"]


def appointment_start_time(appointment: dict) -> str:
    """Sort key for appointments."""
    return appointment["start"]


def list_contains_ignoring_case(items: list[str], wanted: str) -> bool:
    """True if wanted equals any item, ignoring case and surrounding spaces."""
    for item in items:
        if item.lower() == wanted.strip().lower():
            return True
    return False


def next_id(prefix: str, existing_ids: list[str]) -> str:
    """Build the next sequential id such as APT-3011 from ids like APT-3010."""
    highest_number = 0
    for existing_id in existing_ids:
        number = int(existing_id.split("-")[1])
        if number > highest_number:
            highest_number = number
    return prefix + "-" + str(highest_number + 1)


class SchedulingAdapter:
    """In-memory view of the clinic's data. Writes change memory only, never the files."""

    def __init__(self, data_directory: Optional[Path] = None):
        if data_directory is None:
            data_directory = Path(__file__).resolve().parent.parent / "data"

        with open(data_directory / "clinic_rules.json", encoding="utf-8") as rules_file:
            self.clinic_rules = json.load(rules_file)
        self.canonical_specialties = list(self.clinic_rules["specialties"])
        self.canonical_locations = list(self.clinic_rules["locations"].keys())

        self.providers = self._load_providers(data_directory / "providers.csv")
        self.patients = self._load_patients(data_directory / "patients.csv")
        self.slots = self._load_slots(data_directory / "slots.csv")
        self.appointments = self._load_appointments(data_directory / "appointments.csv")
        self._link_appointments_to_slots()

        # Result of every successful write, keyed by the caller's idempotency key.
        self.completed_writes = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _canonical_location_or_fail(self, raw_location: str, where: str) -> str:
        """Normalise a location from source data. Unknown labels stop the load."""
        canonical_location = normalize_location(raw_location, self.canonical_locations)
        if canonical_location is None:
            raise ValueError(where + ": unknown location '" + raw_location + "'")
        return canonical_location

    def _load_providers(self, path: Path) -> dict:
        providers = {}
        for row in read_csv_rows(path):
            where = "providers.csv " + row["provider_id"]

            specialty = normalize_specialty(row["specialty"], self.canonical_specialties)
            if specialty is None:
                raise ValueError(where + ": unknown specialty '" + row["specialty"] + "'")

            locations = []
            for raw_location in split_list(row["locations"], ";"):
                locations.append(self._canonical_location_or_fail(raw_location, where))

            providers[row["provider_id"]] = {
                "provider_id": row["provider_id"],
                "full_name": row["full_name"],
                "specialty": specialty,
                "locations": locations,
                "languages": split_list(row["languages"], ";"),
                "gender": row["gender"].strip().upper(),
                "accepting_new_patients": row["accepting_new_patients"].strip().upper() == "TRUE",
                "insurance_accepted": split_list(row["insurance_accepted"], ","),
                "notes": row["notes"],
            }
        return providers

    def _load_patients(self, path: Path) -> dict:
        patients = {}
        for row in read_csv_rows(path):
            patients[row["patient_id"]] = dict(row)
        return patients

    def _load_slots(self, path: Path) -> dict:
        slots = {}
        for row in read_csv_rows(path):
            where = "slots.csv " + row["slot_id"]
            slot = dict(row)
            slot["location"] = self._canonical_location_or_fail(row["location"], where)
            slots[row["slot_id"]] = slot
        return slots

    def _load_appointments(self, path: Path) -> dict:
        appointments = {}
        for row in read_csv_rows(path):
            where = "appointments.csv " + row["appointment_id"]
            appointment = dict(row)
            appointment["location"] = self._canonical_location_or_fail(row["location"], where)
            appointments[row["appointment_id"]] = appointment
        return appointments

    def _link_appointments_to_slots(self) -> None:
        """Make slot state agree with the booked appointments.

        In the source files the slot_id on most appointments points at a slot for a
        different provider or time, and the matching slot is often marked open. The
        appointment's own provider and start time are the facts we trust, so we find
        the slot with that provider and start, re-link to it and mark it booked.
        """
        slot_ids_by_provider_and_start = {}
        for slot in self.slots.values():
            slot_ids_by_provider_and_start[(slot["provider_id"], slot["start"])] = slot["slot_id"]

        for appointment in self.appointments.values():
            if appointment["status"] != "booked":
                continue
            lookup_key = (appointment["provider_id"], appointment["start"])
            real_slot_id = slot_ids_by_provider_and_start.get(lookup_key)
            if real_slot_id is None:
                print(
                    "[adapter] WARNING "
                    + appointment["appointment_id"]
                    + ": no slot exists for "
                    + appointment["provider_id"]
                    + " at "
                    + appointment["start"]
                    + "; leaving it unlinked"
                )
                appointment["slot_id"] = ""
                continue
            if real_slot_id != appointment["slot_id"]:
                print(
                    "[adapter] "
                    + appointment["appointment_id"]
                    + ": slot_id "
                    + appointment["slot_id"]
                    + " in the file does not match; re-linked to "
                    + real_slot_id
                )
                appointment["slot_id"] = real_slot_id
            self.slots[real_slot_id]["status"] = "booked"

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search_providers(
        self,
        name: Optional[str] = None,
        specialty: Optional[str] = None,
        location: Optional[str] = None,
        language: Optional[str] = None,
        gender: Optional[str] = None,
        insurance: Optional[str] = None,
        accepting_new: Optional[bool] = None,
    ) -> list[dict]:
        """Find providers matching every filter given; every filter is optional.

        Returns all matches, accepting-new-patients first. Providers who are not
        accepting new patients are included (and marked) unless accepting_new=True.
        """
        wanted_specialty = None
        if specialty is not None:
            wanted_specialty = normalize_specialty(specialty, self.canonical_specialties)
            if wanted_specialty is None:
                return []

        wanted_location = None
        if location is not None:
            wanted_location = normalize_location(location, self.canonical_locations)
            if wanted_location is None:
                return []

        matching_providers = []
        for provider in self.providers.values():
            if name is not None and not self._name_matches(provider, name):
                continue
            if wanted_specialty is not None and provider["specialty"] != wanted_specialty:
                continue
            if wanted_location is not None and wanted_location not in provider["locations"]:
                continue
            if language is not None and not list_contains_ignoring_case(provider["languages"], language):
                continue
            if gender is not None and provider["gender"] != gender.strip()[:1].upper():
                # Compare first letter only so "female" and "F" both work.
                continue
            if insurance is not None and not list_contains_ignoring_case(
                provider["insurance_accepted"], insurance
            ):
                continue
            if accepting_new is not None and provider["accepting_new_patients"] != accepting_new:
                continue
            matching_providers.append(copy.deepcopy(provider))

        matching_providers.sort(key=provider_sort_key)
        return matching_providers

    def _name_matches(self, provider: dict, name_query: str) -> bool:
        """Case-insensitive substring match on the full name, ignoring "Dr" titles."""
        query_words = []
        for word in name_query.lower().replace(".", " ").split():
            if word != "dr" and word != "doctor":
                query_words.append(word)
        cleaned_query = " ".join(query_words)
        if cleaned_query == "":
            return False
        return cleaned_query in provider["full_name"].lower()

    def get_provider(self, provider_id: str) -> Optional[dict]:
        """Return one provider record, or None if the id is unknown."""
        provider = self.providers.get(provider_id)
        if provider is None:
            return None
        return copy.deepcopy(provider)

    def get_slot(self, slot_id: str) -> Optional[dict]:
        """Return one slot record, or None if the id is unknown."""
        slot = self.slots.get(slot_id)
        if slot is None:
            return None
        return copy.deepcopy(slot)

    def get_appointment(self, appointment_id: str) -> Optional[dict]:
        """Return one appointment record, or None if the id is unknown."""
        appointment = self.appointments.get(appointment_id)
        if appointment is None:
            return None
        return copy.deepcopy(appointment)

    def get_availability(
        self,
        provider_id: str,
        location: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        time_of_day: Optional[str] = None,
    ) -> list[dict]:
        """Return up to 5 open slots for a provider, earliest first.

        date_from and date_to are inclusive YYYY-MM-DD strings. time_of_day is
        "morning" (before noon) or "afternoon"; anything else means any time.
        The time filter lives here, not in the caller, because the 5-slot cap is
        applied after filtering. Filtering afterwards could leave nothing.
        """
        wanted_location = None
        if location is not None:
            wanted_location = normalize_location(location, self.canonical_locations)
            if wanted_location is None:
                return []

        open_slots = []
        for slot in self.slots.values():
            if slot["provider_id"] != provider_id:
                continue
            if slot["status"] != "open":
                continue
            if wanted_location is not None and slot["location"] != wanted_location:
                continue
            slot_date = slot["start"][:10]
            if date_from is not None and slot_date < date_from:
                continue
            if date_to is not None and slot_date > date_to:
                continue
            start_hour = int(slot["start"][11:13])
            if time_of_day == "morning" and start_hour >= 12:
                continue
            if time_of_day == "afternoon" and start_hour < 12:
                continue
            open_slots.append(slot)

        open_slots.sort(key=slot_start_time)
        return copy.deepcopy(open_slots[:MAX_SLOTS_RETURNED])

    def find_patient(
        self,
        patient_id: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        dob: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> list[dict]:
        """Return patients matching every field given (dob is YYYY-MM-DD).

        Returns a list because two records can match, e.g. a duplicate patient.
        With no fields given it returns nothing rather than the whole patient list.
        """
        if patient_id is None and first_name is None and last_name is None and dob is None and phone is None:
            return []

        matching_patients = []
        for patient in self.patients.values():
            if patient_id is not None and patient["patient_id"] != patient_id:
                continue
            if first_name is not None and patient["first_name"].lower() != first_name.strip().lower():
                continue
            if last_name is not None and patient["last_name"].lower() != last_name.strip().lower():
                continue
            if dob is not None and patient["dob"] != dob.strip():
                continue
            if phone is not None and normalize_phone(patient["phone"]) != normalize_phone(phone):
                continue
            matching_patients.append(copy.deepcopy(patient))
        return matching_patients

    def find_appointments(self, patient_id: str, status: Optional[str] = "booked") -> list[dict]:
        """Return a patient's appointments, soonest first. status=None returns every status."""
        matching_appointments = []
        for appointment in self.appointments.values():
            if appointment["patient_id"] != patient_id:
                continue
            if status is not None and appointment["status"] != status:
                continue
            matching_appointments.append(copy.deepcopy(appointment))
        matching_appointments.sort(key=appointment_start_time)
        return matching_appointments

    # ------------------------------------------------------------------
    # Writes. Each takes an idempotency key made by code, so a retried call
    # returns the original result instead of acting twice.
    # ------------------------------------------------------------------

    def _earlier_result(self, idempotency_key: str) -> Optional[dict]:
        """Return the stored result if this key already completed a write, else None."""
        stored_result = self.completed_writes.get(idempotency_key)
        if stored_result is None:
            return None
        return copy.deepcopy(stored_result)

    def _remember_result(self, idempotency_key: str, result: dict) -> dict:
        """Store a successful result under its key and return a copy of it.

        Failures are not stored, so the same call can succeed after the cause is fixed.
        """
        self.completed_writes[idempotency_key] = copy.deepcopy(result)
        return copy.deepcopy(result)

    def create_patient(
        self,
        first_name: str,
        last_name: str,
        dob: str,
        phone: str,
        idempotency_key: str,
        insurance_plan: str = "",
        plan_type: str = "",
    ) -> dict:
        """Add a new patient unless someone with the same DOB and phone already exists.

        Returns {"ok": True, "patient": ...}, or {"ok": False, "reason": ..., "existing_patient": ...}
        for a duplicate.
        """
        earlier_result = self._earlier_result(idempotency_key)
        if earlier_result is not None:
            return earlier_result

        for existing_patient in self.patients.values():
            same_dob = existing_patient["dob"] == dob.strip()
            same_phone = normalize_phone(existing_patient["phone"]) == normalize_phone(phone)
            if same_dob and same_phone:
                return {
                    "ok": False,
                    "reason": "A patient with this date of birth and phone number already exists.",
                    "existing_patient": copy.deepcopy(existing_patient),
                }

        new_patient_id = next_id("PAT", list(self.patients.keys()))
        new_patient = {
            "patient_id": new_patient_id,
            "first_name": first_name.strip(),
            "last_name": last_name.strip(),
            "dob": dob.strip(),
            "phone": phone.strip(),
            "insurance_plan": insurance_plan,
            "plan_type": plan_type,
            "pcp_provider_id": "",
            "preferred_location": "",
            "authorized_contacts": "",
            "notes": "",
        }
        self.patients[new_patient_id] = new_patient
        return self._remember_result(idempotency_key, {"ok": True, "patient": new_patient})

    def book(
        self,
        patient_id: str,
        slot_id: str,
        visit_reason: str,
        idempotency_key: str,
        booked_by: str = "agent",
    ) -> dict:
        """Book an open slot for a patient.

        Returns {"ok": True, "appointment": ...} or {"ok": False, "reason": ...}.
        """
        earlier_result = self._earlier_result(idempotency_key)
        if earlier_result is not None:
            return earlier_result

        if patient_id not in self.patients:
            return {"ok": False, "reason": "There is no patient with id " + patient_id + "."}
        slot = self.slots.get(slot_id)
        if slot is None:
            return {"ok": False, "reason": "There is no slot with id " + slot_id + "."}
        if slot["status"] != "open":
            return {"ok": False, "reason": "That slot has already been taken. Offer another slot."}

        slot["status"] = "booked"
        new_appointment_id = next_id("APT", list(self.appointments.keys()))
        new_appointment = {
            "appointment_id": new_appointment_id,
            "patient_id": patient_id,
            "provider_id": slot["provider_id"],
            "slot_id": slot_id,
            "location": slot["location"],
            "start": slot["start"],
            "status": "booked",
            "visit_reason": visit_reason,
            "booked_by": booked_by,
        }
        self.appointments[new_appointment_id] = new_appointment
        return self._remember_result(idempotency_key, {"ok": True, "appointment": new_appointment})

    def cancel(self, appointment_id: str, idempotency_key: str) -> dict:
        """Cancel a booked appointment and reopen its slot.

        Returns {"ok": True, "appointment": ...} or {"ok": False, "reason": ...}.
        """
        earlier_result = self._earlier_result(idempotency_key)
        if earlier_result is not None:
            return earlier_result

        appointment = self.appointments.get(appointment_id)
        if appointment is None:
            return {"ok": False, "reason": "There is no appointment with id " + appointment_id + "."}
        if appointment["status"] != "booked":
            return {"ok": False, "reason": "That appointment is already " + appointment["status"] + "."}

        self._reopen_slot_of(appointment)
        appointment["status"] = "cancelled"
        return self._remember_result(idempotency_key, {"ok": True, "appointment": appointment})

    def reschedule(self, appointment_id: str, new_slot_id: str, idempotency_key: str) -> dict:
        """Move a booked appointment to a different open slot.

        Returns {"ok": True, "appointment": ...} or {"ok": False, "reason": ...}.
        """
        earlier_result = self._earlier_result(idempotency_key)
        if earlier_result is not None:
            return earlier_result

        appointment = self.appointments.get(appointment_id)
        if appointment is None:
            return {"ok": False, "reason": "There is no appointment with id " + appointment_id + "."}
        if appointment["status"] != "booked":
            return {"ok": False, "reason": "That appointment is " + appointment["status"] + ", not booked."}
        new_slot = self.slots.get(new_slot_id)
        if new_slot is None:
            return {"ok": False, "reason": "There is no slot with id " + new_slot_id + "."}
        if new_slot["status"] != "open":
            return {"ok": False, "reason": "That slot has already been taken. Offer another slot."}

        # Every check has passed above, so from here the move cannot half-happen.
        self._reopen_slot_of(appointment)
        new_slot["status"] = "booked"
        appointment["slot_id"] = new_slot_id
        appointment["provider_id"] = new_slot["provider_id"]
        appointment["location"] = new_slot["location"]
        appointment["start"] = new_slot["start"]
        return self._remember_result(idempotency_key, {"ok": True, "appointment": appointment})

    def _reopen_slot_of(self, appointment: dict) -> None:
        """Mark the appointment's slot open again, but only if it is really that appointment's slot."""
        slot = self.slots.get(appointment["slot_id"])
        if slot is None:
            return
        same_provider = slot["provider_id"] == appointment["provider_id"]
        same_start = slot["start"] == appointment["start"]
        if same_provider and same_start:
            slot["status"] = "open"


def print_provider_names(label: str, providers: list[dict]) -> None:
    """Print one line summarising a provider search."""
    names = []
    for provider in providers:
        names.append(provider["full_name"] + " (" + provider["specialty"] + ")")
    print(label + " -> " + str(len(providers)) + ": " + "; ".join(names))


if __name__ == "__main__":
    adapter = SchedulingAdapter()

    print("\n--- search_providers ---")
    cardiologists = adapter.search_providers(specialty="Cardiology")
    print_provider_names("specialty='Cardiology'", cardiologists)
    print_provider_names("specialty='heart'", adapter.search_providers(specialty="heart"))
    print_provider_names("specialty='Orthopaedics'", adapter.search_providers(specialty="Orthopaedics"))
    print_provider_names("specialty='obgyn'", adapter.search_providers(specialty="obgyn"))
    print_provider_names("name='Dr Kim' (no specialty)", adapter.search_providers(name="Dr Kim"))
    print_provider_names(
        "Cardiology + accepting_new=True",
        adapter.search_providers(specialty="Cardiology", accepting_new=True),
    )
    print_provider_names(
        "Dermatology + language=Spanish + insurance=Aetna",
        adapter.search_providers(specialty="Dermatology", language="Spanish", insurance="Aetna"),
    )
    print_provider_names("location='oak park'", adapter.search_providers(location="oak park"))

    print("\n--- get_availability (PRV-1001, LAKEVIEW, from 2026-10-23) ---")
    for slot in adapter.get_availability("PRV-1001", location="Lakeview", date_from="2026-10-23"):
        print(slot["slot_id"], slot["location"], slot["start"])

    print("\n--- find_patient ---")
    print([p["patient_id"] for p in adapter.find_patient(last_name="Smith")])
    print([p["patient_id"] for p in adapter.find_patient(phone="(312) 555 0188")])
    print([p["patient_id"] for p in adapter.find_patient(dob="1964-03-12", phone="312-555-0141")])

    print("\n--- find_appointments PAT-2003 ---")
    for appointment in adapter.find_appointments("PAT-2003"):
        print(appointment["appointment_id"], appointment["provider_id"], appointment["start"], appointment["slot_id"])

    print("\n--- create_patient ---")
    duplicate_attempt = adapter.create_patient("Jon", "Smith", "1964-03-12", "312-555-0141", "demo-create-1")
    print("duplicate:", duplicate_attempt["ok"], "-", duplicate_attempt["reason"])
    new_patient_result = adapter.create_patient("Test", "Person", "1990-01-01", "312-555-0000", "demo-create-2")
    print("new:", new_patient_result["ok"], new_patient_result["patient"]["patient_id"])
    replayed_result = adapter.create_patient("Test", "Person", "1990-01-01", "312-555-0000", "demo-create-2")
    print("same key again:", replayed_result["ok"], replayed_result["patient"]["patient_id"], "(no second patient)")

    print("\n--- book / reschedule / cancel ---")
    open_slots = adapter.get_availability("PRV-1001", date_from="2026-10-23")
    first_slot = open_slots[0]
    second_slot = open_slots[1]
    booking = adapter.book("PAT-2007", first_slot["slot_id"], "demo visit", "demo-book-1")
    print("book:", booking["ok"], booking["appointment"]["appointment_id"], booking["appointment"]["slot_id"])
    repeat_booking = adapter.book("PAT-2007", first_slot["slot_id"], "demo visit", "demo-book-1")
    print("book same key:", repeat_booking["appointment"]["appointment_id"], "(same appointment)")
    taken_booking = adapter.book("PAT-2005", first_slot["slot_id"], "demo visit", "demo-book-2")
    print("book taken slot:", taken_booking["ok"], "-", taken_booking["reason"])
    moved = adapter.reschedule(booking["appointment"]["appointment_id"], second_slot["slot_id"], "demo-move-1")
    print("reschedule:", moved["ok"], moved["appointment"]["slot_id"])
    print("old slot status:", adapter.slots[first_slot["slot_id"]]["status"])
    cancelled = adapter.cancel(booking["appointment"]["appointment_id"], "demo-cancel-1")
    print("cancel:", cancelled["ok"], cancelled["appointment"]["status"])
    print("new slot status:", adapter.slots[second_slot["slot_id"]]["status"])
