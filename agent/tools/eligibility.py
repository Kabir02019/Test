"""EligibilityTools: check_eligibility and the one helper per rule it applies (age, insurance, new-patient, referral).
Does not hold searching, availability or any write."""

from datetime import date
from typing import Optional

from adapter.scheduling_adapter import list_contains_ignoring_case
from agent.tools.helpers import age_on, blocked_result, parse_iso_date, provider_summary
from agent.tools.speech import doctor_name, join_doctor_names, speak_date


class EligibilityTools:
    """Rule checks. Relies on self.adapter and on ReadTools helpers such as _other_providers_in_specialty."""

    def _clinic_rule(self, rule_id: str) -> dict:
        """Look up a rule in clinic_rules.json by its id, so thresholds are not hard-coded here."""
        for rule in self.adapter.clinic_rules["eligibility_rules"]:
            if rule["rule_id"] == rule_id:
                return rule
        raise KeyError("clinic_rules.json has no rule " + rule_id)

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
