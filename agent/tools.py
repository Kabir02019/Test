"""Tools for the scheduling agent, plus the schemas that describe them to the model.

Two groups:
  - Model tools: listed in TOOL_SCHEMAS. The state machine hands a per-stage subset to the model.
  - Write tools (book_appointment, reschedule_appointment, cancel_appointment): methods only.
    They have NO schema, so the model can never call them. The state machine calls them in
    Python after the patient has confirmed.

Every tool returns a dict. A failure is a dict whose "message" tells the model what to do next.
Run the checks with:  python -m agent.tools   (from the project root)
"""

import json
from datetime import date, datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from adapter.normalize import normalize_location, normalize_phone, normalize_specialty
from adapter.scheduling_adapter import SchedulingAdapter, list_contains_ignoring_case

MAX_RESULTS = 5
HOLD_SECONDS = 300

CLINIC_TIMEZONE = ZoneInfo("America/Chicago")
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


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def clinic_now() -> datetime:
    """Current time in the clinic's timezone."""
    return datetime.now(CLINIC_TIMEZONE)


def speak_date(day: date) -> str:
    """Format a date the way it should be said aloud, e.g. 'Friday, October 23'."""
    return f"{day.strftime('%A, %B')} {day.day}"


def speak_time(start: str) -> str:
    """Format an ISO start time for speech, e.g. 'Friday, October 23 at 10:30 AM'.

    Done in code so the model never has to work out a weekday itself.
    """
    moment = datetime.fromisoformat(start)
    hour_on_twelve_hour_clock = moment.hour % 12
    if hour_on_twelve_hour_clock == 0:
        hour_on_twelve_hour_clock = 12
    if moment.hour < 12:
        am_or_pm = "AM"
    else:
        am_or_pm = "PM"
    return (
        f"{speak_date(moment.date())} at "
        f"{hour_on_twelve_hour_clock}:{moment.minute:02d} {am_or_pm}"
    )


def doctor_name(provider: dict) -> str:
    """Name as it should be spoken: 'Dr. Susan Kim'."""
    return "Dr. " + provider["full_name"]


def join_doctor_names(providers: list[dict]) -> str:
    """'Dr. A, Dr. B, Dr. C' for up to three providers."""
    names = []
    for provider in providers[:3]:
        names.append(doctor_name(provider))
    return ", ".join(names)


def split_full_name(full_name: str) -> Optional[tuple]:
    """Return (first_name, last_name) from the first and last word, or None if there is only one word."""
    words = full_name.split()
    if len(words) < 2:
        return None
    return (words[0], words[-1])


def parse_iso_date(text: str) -> Optional[date]:
    """Parse YYYY-MM-DD, returning None if the text is not a valid date."""
    try:
        return date.fromisoformat(text.strip())
    except ValueError:
        return None


def age_on(birth_date: date, on_date: date) -> int:
    """Age in whole years on a given date."""
    age = on_date.year - birth_date.year
    had_birthday_yet = (on_date.month, on_date.day) >= (birth_date.month, birth_date.day)
    if not had_birthday_yet:
        age = age - 1
    return age


def words_only(text: str) -> str:
    """Lowercase text with punctuation turned into spaces, so 'Check-up!' becomes 'check up'."""
    characters = []
    for character in text.lower():
        if character.isalnum():
            characters.append(character)
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def candidate_sort_key(candidate: dict) -> int:
    """Sort candidates with the most matched phrases first. Python's sort is stable, so ties keep list order."""
    return -candidate["match_count"]


def provider_summary(provider: dict) -> dict:
    """The few provider fields a model needs to talk about a provider."""
    return {
        "provider_id": provider["provider_id"],
        "name": provider["full_name"],
        "specialty": provider["specialty"],
        "locations": provider["locations"],
        "languages": provider["languages"],
        "gender": provider["gender"],
        "accepting_new_patients": provider["accepting_new_patients"],
    }


def blocked_result(code: str, message: str, alternatives: Optional[list] = None) -> dict:
    """Shape for a failed eligibility rule. `code` is for logs/tests; `message` is for the model."""
    result = {"allowed": False, "blocked_by": code, "message": message}
    if alternatives is not None:
        result["alternatives"] = alternatives
    return result


class ClinicTools:
    """Tool implementations over one SchedulingAdapter. Holds the in-memory holds and staff tasks."""

    def __init__(self, adapter: SchedulingAdapter, now_function: Optional[Callable[[], datetime]] = None):
        self.adapter = adapter
        if now_function is None:
            now_function = clinic_now
        # Injectable so scenarios can move time to test hold expiry and after-hours handoff.
        self.now_function = now_function

        self.holds = {}
        self.next_hold_number = 1
        self.staff_tasks = []
        self.write_results = {}

    # ------------------------------------------------------------------
    # Shared lookups
    # ------------------------------------------------------------------

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

    def _clinic_rule(self, rule_id: str) -> dict:
        """Look up a rule in clinic_rules.json by its id, so thresholds are not hard-coded here."""
        for rule in self.adapter.clinic_rules["eligibility_rules"]:
            if rule["rule_id"] == rule_id:
                return rule
        raise KeyError("clinic_rules.json has no rule " + rule_id)

    # ------------------------------------------------------------------
    # Patient tools
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Specialty and provider search
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------

    def check_eligibility(self, patient_id: str, provider_id: str, appointment_date: Optional[str] = None) -> dict:
        """Check whether a patient may book a provider, applying the rules in clinic_rules.json.

        appointment_date (YYYY-MM-DD) matters: age and referral validity are judged on the visit
        date, not today. If it is omitted we use the provider's earliest opening and say so.
        """
        patient_matches = self.adapter.find_patient(patient_id=patient_id)
        if len(patient_matches) == 0:
            return blocked_result("unknown_patient", "There is no patient with that id. Verify the patient first.")
        provider = self.adapter.get_provider(provider_id)
        if provider is None:
            return blocked_result("unknown_provider", "There is no provider with that id. Search for the provider again.")
        patient = patient_matches[0]

        date_was_assumed = False
        if appointment_date is None:
            earliest_slots = self.adapter.get_availability(provider_id)
            if len(earliest_slots) == 0:
                return blocked_result(
                    "no_availability",
                    f"{doctor_name(provider)} has no open appointments right now. Offer another provider or transfer to staff.",
                )
            visit_date = parse_iso_date(earliest_slots[0]["start"][:10])
            date_was_assumed = True
        else:
            visit_date = parse_iso_date(appointment_date)
            if visit_date is None:
                return blocked_result("invalid_date", "The appointment date must be a real date in YYYY-MM-DD format.")

        blocked = self._check_pediatrics_age(patient, provider, visit_date)
        if blocked is not None:
            return blocked
        blocked = self._check_insurance(patient, provider)
        if blocked is not None:
            return blocked
        blocked = self._check_new_patient_rule(patient, provider)
        if blocked is not None:
            return blocked
        blocked = self._check_hmo_referral(patient, provider, visit_date)
        if blocked is not None:
            return blocked

        message = f"{doctor_name(provider)} can be booked for this patient."
        if date_was_assumed:
            message = (
                message + f" No date was given, so this was checked against the earliest opening on "
                f"{speak_date(visit_date)}. Check again once the patient picks a slot."
            )
        return {"allowed": True, "checked_for_date": visit_date.isoformat(), "message": message}

    def _check_pediatrics_age(self, patient: dict, provider: dict, visit_date: date) -> Optional[dict]:
        """Rules R2/R3: Pediatrics is for patients under 18 on the day of the visit."""
        age_rule = self._clinic_rule("R2")
        if provider["specialty"] not in age_rule["applies_to_specialties"]:
            return None
        age_on_visit_date = age_on(parse_iso_date(patient["dob"]), visit_date)
        if age_on_visit_date <= age_rule["max_age"]:
            return None

        primary_care_options, pcp_text = self._primary_care_options(patient)
        return blocked_result(
            "pediatrics_age",
            f"Pediatrics is only for patients under 18, and this patient will be {age_on_visit_date} on "
            f"{speak_date(visit_date)}. They should see Primary Care instead. {pcp_text}",
            primary_care_options,
        )

    def _primary_care_options(self, patient: dict) -> tuple:
        """Return (provider summaries, sentence) naming where to send a patient who needs Primary Care.

        The PCP on file is only used if they really are Primary Care: in the data a patient's
        "PCP" can be a pediatrician, which is wrong advice once the patient is 18.
        """
        pcp = self.adapter.get_provider(patient["pcp_provider_id"])
        if pcp is not None and pcp["specialty"] == "Primary Care":
            return [provider_summary(pcp)], f"Their primary care doctor on file is {doctor_name(pcp)}."

        insurance_filter = None
        if patient["insurance_plan"] != "":
            insurance_filter = patient["insurance_plan"]
        candidates = self.adapter.search_providers(
            specialty="Primary Care", insurance=insurance_filter, accepting_new=True
        )
        if len(candidates) == 0:
            return [], "Search for a Primary Care provider."

        summaries = []
        for candidate in candidates[:3]:
            summaries.append(provider_summary(candidate))
        return summaries, f"Primary Care providers who could see them: {join_doctor_names(candidates)}."

    def _check_insurance(self, patient: dict, provider: dict) -> Optional[dict]:
        """Rule R4: the provider must accept the patient's plan."""
        plan = patient["insurance_plan"]
        if plan == "":
            return blocked_result(
                "no_insurance_on_file",
                "There is no insurance on file for this patient. Ask which plan they have. If they have "
                "none, or the plan is not one we recognise, offer to transfer to staff.",
            )
        if len(provider["insurance_accepted"]) == 0:
            return blocked_result(
                "provider_insurance_unknown",
                f"We have no insurance information on file for {doctor_name(provider)}, so I cannot "
                f"confirm they take {plan}. Offer another {provider['specialty']} provider, or transfer to staff.",
            )
        if list_contains_ignoring_case(provider["insurance_accepted"], plan):
            return None

        alternatives = self._other_providers_in_specialty(provider, plan, accepting_new_only=False)
        alternative_summaries = []
        for alternative in alternatives[:3]:
            alternative_summaries.append(provider_summary(alternative))
        message = f"{doctor_name(provider)} does not accept {plan}."
        if len(alternatives) > 0:
            message = message + f" In the same specialty, these providers do: {join_doctor_names(alternatives)}."
        else:
            message = message + " No other provider in that specialty takes this plan. Offer to transfer to staff."
        return blocked_result("insurance_not_accepted", message, alternative_summaries)

    def _check_new_patient_rule(self, patient: dict, provider: dict) -> Optional[dict]:
        """Rule R5: a provider closed to new patients can only be booked by patients who have seen them."""
        if provider["accepting_new_patients"]:
            return None
        if self._has_seen_provider(patient, provider["provider_id"]):
            return None

        alternatives = self._other_providers_in_specialty(
            provider, patient["insurance_plan"], accepting_new_only=True
        )
        alternative_summaries = []
        for alternative in alternatives[:3]:
            alternative_summaries.append(provider_summary(alternative))
        message = (
            f"{doctor_name(provider)} is not accepting new patients, and this patient has not seen them before."
        )
        if len(alternatives) > 0:
            message = message + (
                f" In the same specialty, accepting new patients and taking their plan: "
                f"{join_doctor_names(alternatives)}."
            )
        else:
            message = message + " No one else in that specialty is accepting new patients. Offer to transfer to staff."
        return blocked_result("not_accepting_new_patients", message, alternative_summaries)

    def _has_seen_provider(self, patient: dict, provider_id: str) -> bool:
        """True if the provider is the patient's PCP or the patient has any non-cancelled appointment with them."""
        if patient["pcp_provider_id"] == provider_id:
            return True
        for appointment in self.adapter.find_appointments(patient["patient_id"], status=None):
            if appointment["provider_id"] == provider_id and appointment["status"] != "cancelled":
                return True
        return False

    def _check_hmo_referral(self, patient: dict, provider: dict, visit_date: date) -> Optional[dict]:
        """Rule R1: HMO patients need a referral that is valid on the visit date to see a specialist."""
        referral_rule = self._clinic_rule("R1")
        specialty = provider["specialty"]
        if specialty not in referral_rule["applies_to_specialties"]:
            return None
        if patient["plan_type"] == "":
            return blocked_result(
                "plan_type_unknown",
                "The record does not say whether this patient's plan is an HMO, and HMO plans need a referral "
                "for specialists. Offer to transfer to staff to confirm.",
            )
        if patient["plan_type"] not in referral_rule["applies_to_plan_types"]:
            return None

        referrals_for_specialty = []
        for referral in self.adapter.clinic_rules["referrals_on_file"]:
            if referral["patient_id"] == patient["patient_id"] and referral["specialty"] == specialty:
                referrals_for_specialty.append(referral)

        if len(referrals_for_specialty) == 0:
            primary_care_options, pcp_text = self._primary_care_options(patient)
            return blocked_result(
                "referral_required",
                f"This patient's plan is an HMO, which needs a referral from a primary care doctor before "
                f"seeing a {specialty} specialist, and there is no referral on file. {pcp_text} Offer to book "
                f"a Primary Care visit to get one, or transfer to staff.",
                primary_care_options,
            )

        for referral in referrals_for_specialty:
            valid_from = parse_iso_date(referral["valid_from"])
            valid_until = parse_iso_date(referral["valid_until"])
            if valid_from <= visit_date <= valid_until:
                return None

        referral = referrals_for_specialty[0]
        return blocked_result(
            "referral_outside_window",
            f"This patient's {specialty} referral is only valid from {speak_date(parse_iso_date(referral['valid_from']))} "
            f"to {speak_date(parse_iso_date(referral['valid_until']))}, and the visit would be on "
            f"{speak_date(visit_date)}, which is outside that. Offer a date inside the window, or suggest "
            f"they ask their primary care doctor for a new referral.",
        )

    # ------------------------------------------------------------------
    # Availability and holds
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Handoff
    # ------------------------------------------------------------------

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
        self.staff_tasks.append({"kind": task_kind, "reason": reason, "summary": summary})
        return {"status": task_kind, "message": message}

    # ------------------------------------------------------------------
    # Write tools: NO schema. Only the state machine calls these, after the patient confirms.
    # ------------------------------------------------------------------

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
