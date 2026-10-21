"""Maps the clinic's inconsistent source labels onto the canonical names in clinic_rules.json.

Nothing outside adapter/ should ever see a raw label.
"""

from typing import Optional

# Keys are comparison keys (see make_comparison_key), not display strings.
# Only labels that cannot be matched to a canonical name by spelling alone belong here.
SPECIALTY_ALIASES = {
    "cardio": "Cardiology",
    "heart": "Cardiology",
    "orthopaedics": "Orthopedics",
}


def make_comparison_key(label: str) -> str:
    """Lowercase the label and keep only letters and digits.

    Doing this means "OB/GYN", "OBGYN" and "ob-gyn" all compare equal.
    """
    kept_characters = []
    for character in label.lower():
        if character.isalnum():
            kept_characters.append(character)
    return "".join(kept_characters)


def normalize_specialty(raw_label: Optional[str], canonical_specialties: list[str]) -> Optional[str]:
    """Return the canonical specialty for a raw label, or None if it is not recognised."""
    if raw_label is None:
        return None
    label_key = make_comparison_key(raw_label)
    if label_key == "":
        return None

    for canonical_specialty in canonical_specialties:
        if make_comparison_key(canonical_specialty) == label_key:
            return canonical_specialty

    aliased_specialty = SPECIALTY_ALIASES.get(label_key)
    if aliased_specialty is not None and aliased_specialty in canonical_specialties:
        return aliased_specialty
    return None


def normalize_location(raw_label: Optional[str], canonical_locations: list[str]) -> Optional[str]:
    """Return the canonical location name for a raw label, or None if it is not recognised."""
    if raw_label is None:
        return None
    label_key = make_comparison_key(raw_label)
    if label_key == "":
        return None

    for canonical_location in canonical_locations:
        if make_comparison_key(canonical_location) == label_key:
            return canonical_location
    return None


def normalize_phone(raw_phone: Optional[str]) -> str:
    """Return only the digits of a phone number, dropping a leading US country code.

    Used so "312-555-0141" and "(312) 555 0141" count as the same number.
    """
    if raw_phone is None:
        return ""
    digits = []
    for character in raw_phone:
        if character.isdigit():
            digits.append(character)
    digit_string = "".join(digits)
    if len(digit_string) == 11 and digit_string.startswith("1"):
        return digit_string[1:]
    return digit_string
