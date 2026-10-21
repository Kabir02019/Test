"""Small pure functions shared by the tool classes: parsing, ages, result shapes, the clinic clock.
Does not hold anything that touches the adapter or the session, or text meant to be spoken."""

from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

CLINIC_TIMEZONE = ZoneInfo("America/Chicago")


def clinic_now() -> datetime:
    """Current time in the clinic's timezone."""
    return datetime.now(CLINIC_TIMEZONE)


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
