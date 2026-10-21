"""ReadTools: everything the model may call that looks things up, plus the small lookups they share.
Does not hold eligibility rules (eligibility.py) or anything that books, holds or cancels (writes.py)."""

from typing import Optional

from adapter.normalize import normalize_location, normalize_phone, normalize_specialty
from agent.tools.helpers import (
    CLINIC_TIMEZONE,
    candidate_sort_key,
    parse_iso_date,
    provider_summary,
    split_full_name,
    words_only,
)
from agent.tools.speech import doctor_name, join_doctor_names, speak_time

MAX_RESULTS = 5
# Mirrors "Mon-Fri 8:00-17:00 Central" in clinic_rules.json (escalation.staffed_hours).
STAFFED_WEEKDAYS = (0, 1, 2, 3, 4)
STAFFED_OPEN_HOUR = 8
STAFFED_CLOSE_HOUR = 17

# Deterministic symptom/word -> specialty lookup. Keys must be canonical specialty names;
# match_specialty only ever returns names that are in clinic_rules.json.
# Emergency wording (chest pain etc.) is deliberately absent: agent/safety.py handles that first.
SPECIALTY_KEYWORDS = {
    "Primary Care": [
        "primary care", "family doctor", "physical", "checkup", "check up", "annual",
        "flu", "cold", "cough", "fever", "sore throat", "tired", "fatigue",
        "blood pressure", "vaccine", "vaccination", "diabetes", "headache",
    ],
    "Cardiology": [
        "cardiologist", "cardiology", "heart", "cardiac", "palpitations", "heartbeat",
        "cholesterol",
    ],
    "Dermatology": [
        "dermatologist", "dermatology", "skin", "rash", "acne", "mole", "eczema",
        "psoriasis", "itchy", "hives",
    ],
    "Orthopedics": [
        "orthopedist", "orthopedic", "orthopedics", "orthopaedic", "knee", "shoulder",
        "back pain", "joint", "bone", "fracture", "broken", "sprain", "hip", "ankle",
        "sports injury", "arthritis",
    ],
    "OB/GYN": [
        "gynecologist", "obgyn", "ob gyn", "pregnant", "pregnancy", "prenatal", "period",
        "pap smear", "menstrual", "birth control", "contraception", "menopause",
    ],
    "Pediatrics": [
        "pediatrician", "pediatrics", "child", "my son", "my daughter", "baby",
        "toddler", "infant", "kid", "well child",
    ],
}

NOT_VERIFIED_MESSAGE = (
    "I could not verify those details. Ask the caller to repeat their full name and date of "
    "birth. If they confirm both are right they may be a new patient, or you can offer to "
    "transfer them to staff. Do not say whether anyone with that name is on file."
)


class ReadTools:
    """Lookup tools. Relies on self.adapter and self.now_function, which ClinicTools sets up."""

    def _provider_name(self, provider_id: str) -> str:
        provider = self.adapter.get_provider(provider_id)
        if provider is None:
            return "an unknown provider"
        return doctor_name(provider)

    def _appointment_summary(self, appointment: dict) -> dict:
        """Appointment fields a model needs, with the time already formatted for speech."""
        return {
            "appointment_id": appointment["appointment_id"],
            "provider_id": appointment["provider_id"],
            "provider_name": self._provider_name(appointment["provider_id"]),
            "location": appointment["location"],
            "start": appointment["start"],
            "spoken_time": speak_time(appointment["start"]),
            "visit_reason": appointment["visit_reason"],
        }

    def _other_providers_in_specialty(
        self, provider: dict, insurance_plan: Optional[str], accepting_new_only: bool
    ) -> list[dict]:
        """Other providers in the same specialty, optionally limited to a plan and to new-patient availability."""
        accepting_filter = None
        if accepting_new_only:
            accepting_filter = True
        candidates = self.adapter.search_providers(
            specialty=provider["specialty"],
            insurance=insurance_plan,
            accepting_new=accepting_filter,
        )
        others = []
        for candidate in candidates:
            if candidate["provider_id"] != provider["provider_id"]:
                others.append(candidate)
        return others

    def verify_patient(self, name: str, dob: str) -> dict:
        """Match a caller's full name and date of birth to a patient record.

        Failure always returns the same message, so it never reveals whether the person exists.
        """
        name_parts = split_full_name(name)
        if name_parts is None:
            return {"status": "invalid_input", "message": "I need both a first and a last name. Ask the caller for their full name."}
        if parse_iso_date(dob) is None:
            return {"status": "invalid_input", "message": "The date of birth must be a real date in YYYY-MM-DD format. Ask the caller to repeat it."}

        matching_patients = self.adapter.find_patient(
            first_name=name_parts[0], last_name=name_parts[1], dob=dob
        )
        if len(matching_patients) != 1:
            # Zero matches and two-or-more matches look identical to the caller on purpose.
            return {"status": "not_found", "message": NOT_VERIFIED_MESSAGE}

        patient = matching_patients[0]
        return {
            "status": "verified",
            "patient_id": patient["patient_id"],
            "first_name": patient["first_name"],
            "message": "Identity verified. Use this patient_id for everything that follows.",
        }

    def create_patient(self, name: str, dob: str, phone: str, insurance: str = "", plan_type: str = "") -> dict:
        """Register a new patient, unless the same date of birth and phone already exist.

        Returns a patient_id, or a possible_duplicate result that names the existing record.
        """
        name_parts = split_full_name(name)
        if name_parts is None:
            return {"status": "invalid_input", "message": "I need both a first and a last name. Ask the caller for their full name."}
        if parse_iso_date(dob) is None:
            return {"status": "invalid_input", "message": "The date of birth must be a real date in YYYY-MM-DD format. Ask the caller to repeat it."}
        phone_digits = normalize_phone(phone)
        if len(phone_digits) != 10:
            return {"status": "invalid_input", "message": "The phone number needs 10 digits including area code. Ask the caller to repeat it."}

        # Same person + same details -> same key, so a retry cannot create a second record.
        idempotency_key = f"create_patient:{dob}:{phone_digits}"
        adapter_result = self.adapter.create_patient(
            first_name=name_parts[0],
            last_name=name_parts[1],
            dob=dob,
            phone=phone,
            idempotency_key=idempotency_key,
            insurance_plan=insurance,
            plan_type=plan_type,
        )

        if adapter_result["ok"]:
            patient = adapter_result["patient"]
            return {
                "status": "created",
                "patient_id": patient["patient_id"],
                "message": "New patient record created. Use this patient_id for everything that follows.",
            }

        existing_patient = adapter_result["existing_patient"]
        existing_name = existing_patient["first_name"] + " " + existing_patient["last_name"]
        return {
            "status": "possible_duplicate",
            "existing_patient_id": existing_patient["patient_id"],
            "existing_patient_name": existing_name,
            "message": (
                f"A record with the same date of birth and phone number already exists under the name "
                f"{existing_name}. Ask the caller whether that is them. If yes, call verify_patient with "
                f"that name and their date of birth instead of creating a new record. If no, offer to "
                f"transfer them to staff."
            ),
        }

    def find_appointments(self, patient_id: str) -> dict:
        """List a patient's upcoming booked appointments, soonest first (at most 5)."""
        if len(self.adapter.find_patient(patient_id=patient_id)) == 0:
            return {"appointments": [], "message": "There is no patient with that id. Verify the patient first."}

        appointments = self.adapter.find_appointments(patient_id, status="booked")
        if len(appointments) == 0:
            return {
                "appointments": [],
                "message": "This patient has no upcoming appointments. They can book a new one.",
            }

        summaries = []
        for appointment in appointments[:MAX_RESULTS]:
            summaries.append(self._appointment_summary(appointment))
        message = f"The patient has {len(appointments)} upcoming appointment(s)."
        if len(appointments) > MAX_RESULTS:
            message = message + f" Showing the first {MAX_RESULTS}."
        return {"appointments": summaries, "message": message}

    def match_specialty(self, description: str) -> dict:
        """Rank which specialties fit a caller's description, using a fixed word list (no model call)."""
        canonical_specialties = self.adapter.canonical_specialties

        named_specialty = normalize_specialty(description, canonical_specialties)
        if named_specialty is not None:
            return {
                "candidates": [
                    {
                        "specialty": named_specialty,
                        "confidence": "high",
                        "reason": "The caller asked for this specialty by name.",
                    }
                ],
                "message": "The caller named the specialty directly.",
            }

        padded_description = " " + words_only(description) + " "
        scored_candidates = []
        for specialty in canonical_specialties:
            matched_phrases = []
            for phrase in SPECIALTY_KEYWORDS.get(specialty, []):
                if " " + phrase + " " in padded_description:
                    matched_phrases.append(phrase)
            if len(matched_phrases) > 0:
                scored_candidates.append(
                    {"specialty": specialty, "match_count": len(matched_phrases), "phrases": matched_phrases}
                )
        scored_candidates.sort(key=candidate_sort_key)

        if len(scored_candidates) == 0:
            return {
                "candidates": [
                    {
                        "specialty": "Primary Care",
                        "confidence": "low",
                        "reason": "Nothing in the description points to one specialty; Primary Care can assess it and refer on.",
                    }
                ],
                "message": "No clear match. Ask the caller to say a little more about what is going on, or offer Primary Care.",
            }

        candidates = []
        for scored_candidate in scored_candidates[:MAX_RESULTS]:
            if scored_candidate["match_count"] >= 2:
                confidence = "high"
            else:
                confidence = "medium"
            reason = "The caller mentioned: " + ", ".join(scored_candidate["phrases"]) + "."
            if scored_candidate["specialty"] == "Pediatrics":
                reason = reason + " Pediatrics is only for patients under 18."
            candidates.append(
                {"specialty": scored_candidate["specialty"], "confidence": confidence, "reason": reason}
            )

        message = "Best match first."
        if len(candidates) > 1:
            message = "More than one specialty fits. Ask one short question to decide which."
        return {"candidates": candidates, "message": message}

    def search_providers(
        self,
        name: Optional[str] = None,
        specialty: Optional[str] = None,
        location: Optional[str] = None,
        language: Optional[str] = None,
        gender: Optional[str] = None,
        insurance: Optional[str] = None,
        accepting_new: Optional[bool] = None,
    ) -> dict:
        """Find providers by any combination of filters; returns up to 5 plus guidance on what to say next."""
        canonical_specialties = self.adapter.canonical_specialties
        if specialty is not None and normalize_specialty(specialty, canonical_specialties) is None:
            return {
                "providers": [],
                "message": (
                    f"'{specialty}' is not a specialty at this clinic. The specialties are: "
                    f"{', '.join(canonical_specialties)}. If the caller described symptoms, use match_specialty."
                ),
            }
        if location is not None and normalize_location(location, self.adapter.canonical_locations) is None:
            return {
                "providers": [],
                "message": (
                    f"'{location}' is not one of our locations. The locations are: "
                    f"{', '.join(self.adapter.canonical_locations)}."
                ),
            }

        filters = {}
        if name is not None:
            filters["name"] = name
        if specialty is not None:
            filters["specialty"] = specialty
        if location is not None:
            filters["location"] = location
        if language is not None:
            filters["language"] = language
        if gender is not None:
            filters["gender"] = gender
        if insurance is not None:
            filters["insurance"] = insurance
        if accepting_new is not None:
            filters["accepting_new"] = accepting_new

        matches = self.adapter.search_providers(**filters)

        # If the caller only wants new-patient availability but the provider they named is closed
        # to new patients, show that provider anyway so we can explain and suggest alternatives.
        if len(matches) == 0 and accepting_new is True:
            relaxed_filters = dict(filters)
            del relaxed_filters["accepting_new"]
            matches = self.adapter.search_providers(**relaxed_filters)

        if len(matches) == 0:
            return {"providers": [], "message": self._explain_no_provider_match(filters)}

        shown_providers = matches[:MAX_RESULTS]
        message_parts = []
        for provider in shown_providers:
            if provider["accepting_new_patients"]:
                continue
            alternatives = self._other_providers_in_specialty(provider, insurance, accepting_new_only=True)
            if len(alternatives) > 0:
                message_parts.append(
                    f"{doctor_name(provider)} is not accepting new patients, so only patients who have seen "
                    f"them before can book. In the same specialty, accepting new patients: "
                    f"{join_doctor_names(alternatives)}."
                )
            else:
                message_parts.append(
                    f"{doctor_name(provider)} is not accepting new patients and no one else in that "
                    f"specialty is either. Offer to transfer to staff."
                )
        if len(shown_providers) > 1:
            message_parts.append("Several providers match. Read the options and ask which one the caller prefers.")
        if len(matches) > MAX_RESULTS:
            message_parts.append(
                f"There are {len(matches)} matches and only the first {MAX_RESULTS} are shown. "
                f"Ask a narrowing question such as location, language or gender."
            )
        if len(message_parts) == 0:
            message_parts.append("One provider matches.")

        summaries = []
        for provider in shown_providers:
            summaries.append(provider_summary(provider))
        return {"providers": summaries, "message": " ".join(message_parts)}

    def _explain_no_provider_match(self, filters: dict) -> str:
        """Say which single filter, if dropped, would give results, so the model can offer a way forward."""
        if len(filters) == 1:
            only_name = list(filters.keys())[0]
            if only_name == "name":
                return (
                    f"No provider is named '{filters['name']}'. Ask the caller to spell the name, "
                    f"or search by specialty instead."
                )
            return f"No provider matches {only_name} = {filters[only_name]}. Offer to loosen it or transfer to staff."

        hints = []
        for filter_name in filters:
            relaxed_filters = dict(filters)
            del relaxed_filters[filter_name]
            found_providers = self.adapter.search_providers(**relaxed_filters)
            if len(found_providers) > 0:
                hints.append(
                    f"Without the {filter_name} filter ({filters[filter_name]}) there are "
                    f"{len(found_providers)} matches, for example {join_doctor_names(found_providers)}."
                )
        if len(hints) == 0:
            return "No provider matches those filters. Ask the caller what matters most, or offer to transfer to staff."
        return "No provider matches all of those filters. " + " ".join(hints[:3])

    def get_availability(
        self,
        provider_ids: list[str],
        date_range: Optional[dict] = None,
        time_of_day: Optional[str] = None,
    ) -> dict:
        """Earliest open slots (at most 5) across the given providers.

        date_range is {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}, both optional.
        time_of_day is "morning", "afternoon" or "any".
        """
        if len(provider_ids) == 0:
            return {"slots": [], "message": "I need at least one provider_id. Search for a provider first."}
        for provider_id in provider_ids:
            if self.adapter.get_provider(provider_id) is None:
                return {"slots": [], "message": f"There is no provider with id {provider_id}. Search for the provider again."}

        date_from = None
        date_to = None
        if date_range is not None:
            date_from = date_range.get("start")
            date_to = date_range.get("end")
        if time_of_day is not None:
            time_of_day = time_of_day.strip().lower()
        if time_of_day is not None and time_of_day not in ("morning", "afternoon", "any"):
            return {"slots": [], "message": "time_of_day must be morning, afternoon or any."}
        if time_of_day == "any":
            time_of_day = None

        found_slots = []
        for provider_id in provider_ids:
            found_slots.extend(
                self.adapter.get_availability(
                    provider_id, date_from=date_from, date_to=date_to, time_of_day=time_of_day
                )
            )
        found_slots.sort(key=self._slot_start)

        if len(found_slots) == 0:
            return {"slots": [], "message": self._explain_no_slots(provider_ids)}

        slot_results = []
        for slot in found_slots[:MAX_RESULTS]:
            slot_results.append(
                {
                    "slot_id": slot["slot_id"],
                    "provider_id": slot["provider_id"],
                    "provider_name": self._provider_name(slot["provider_id"]),
                    "location": slot["location"],
                    "start": slot["start"],
                    "spoken_time": speak_time(slot["start"]),
                }
            )
        return {
            "slots": slot_results,
            "message": "Earliest first. Offer two or three of these, not all of them. Use slot_id when holding.",
        }

    def _slot_start(self, slot: dict) -> str:
        return slot["start"]

    def _explain_no_slots(self, provider_ids: list[str]) -> str:
        """No slot matched the filters: report the earliest opening ignoring them, if there is one."""
        any_slots = []
        for provider_id in provider_ids:
            any_slots.extend(self.adapter.get_availability(provider_id))
        any_slots.sort(key=self._slot_start)
        if len(any_slots) == 0:
            return "These providers have no open appointments in the schedule. Offer another provider or transfer to staff."
        earliest = any_slots[0]
        return (
            f"Nothing is open in that range or time of day. The earliest opening is "
            f"{self._provider_name(earliest['provider_id'])} on {speak_time(earliest['start'])} at "
            f"{earliest['location']}. Offer it, or ask the caller to widen the range."
        )

    def transfer_to_human(self, reason: str, summary: str) -> dict:
        """Hand the caller to staff, or record a callback task if the clinic is closed."""
        current_time = self.now_function().astimezone(CLINIC_TIMEZONE)
        is_staffed_day = current_time.weekday() in STAFFED_WEEKDAYS
        is_staffed_hour = STAFFED_OPEN_HOUR <= current_time.hour < STAFFED_CLOSE_HOUR

        if is_staffed_day and is_staffed_hour:
            task_kind = "live_transfer"
            message = "Tell the caller you are connecting them to a member of staff now, and that the staff member will have a summary."
        else:
            task_kind = "callback"
            hours_text = self.adapter.clinic_rules["escalation"]["staffed_hours"]
            message = (
                f"The clinic is closed right now (staffed {hours_text}), so no live transfer is possible. "
                f"Tell the caller a callback request has been created and staff will call them back."
            )
        task_id = "STAFF-" + f"{len(self.staff_tasks) + 1:04d}"
        self.staff_tasks.append({"task_id": task_id, "kind": task_kind, "reason": reason, "summary": summary})
        return {"status": task_kind, "task_id": task_id, "message": message}
