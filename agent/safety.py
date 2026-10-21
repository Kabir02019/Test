"""Safety checks that run in plain Python, with no model call anywhere.

check_red_flags runs on what the caller said, before the model sees it.
check_grounding runs on what the model wants to say, before the caller hears it.
Because neither asks a model, a confused turn or a prompt injection cannot switch them off.
This module imports nothing from the rest of the project.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Scripted messages. Fixed strings, never generated.
# ----------------------------------------------------------------------

RED_FLAG_MESSAGES = {
    "emergency": (
        "What you are describing may be an emergency. Please hang up and call 911 right now, "
        "or have someone take you to the nearest emergency room. Please do not wait for an appointment."
    ),
    "mental_health": (
        "I am concerned about your safety. Please call or text 988, the Suicide and Crisis Lifeline, "
        "right now. If you are in immediate danger, call 911. You are not alone, and help is "
        "available at any hour."
    ),
    "urgent": (
        "Thank you for telling me. A nurse from our team will contact you today. If things get "
        "worse or become severe before then, please call 911 or go to the nearest emergency room."
    ),
}

GROUNDING_FALLBACK_MESSAGE = "Let me get someone from our team to confirm that for you."

# ----------------------------------------------------------------------
# Red-flag rules. Tuned for recall: a false escalation is cheap, a missed one is not.
# ----------------------------------------------------------------------

# Whole phrases, matched against the caller's words with punctuation removed and apostrophes
# dropped, so "can't breathe" is written here as "cant breathe".
MENTAL_HEALTH_PHRASES = [
    "kill myself", "killing myself", "end my life", "ending my life", "end it all",
    "want to die", "wanna die", "dont want to live", "dont want to be alive",
    "better off dead", "take my own life", "hurt myself", "harm myself", "self harm",
    "no reason to live",
]
MENTAL_HEALTH_WORD_STEMS = ["suicid"]

EMERGENCY_PHRASES = [
    "heart attack", "cant breathe", "cannot breathe", "can not breathe", "unable to breathe",
    "short of breath", "shortness of breath", "catch my breath", "stopped breathing",
    "not breathing", "choking", "choked", "worst headache",
    "lot of blood", "blood everywhere", "coughing up blood", "vomiting blood",
    "throwing up blood", "losing blood",
]
EMERGENCY_WORD_STEMS = ["slurr"]

# Each rule is (reason, anchor words, nearby word stems, window).
# It fires when an anchor word has a nearby stem within `window` words on either side.
# This catches vague wording such as "my chest feels weird" without a phrase for each variant.
EMERGENCY_WINDOW_RULES = [
    (
        "chest symptom",
        ["chest"],
        [
            "pain", "hurt", "ache", "aching", "tight", "pressure", "heav", "squeez", "crush",
            "discomfort", "weird", "funny", "strange", "odd", "burn", "sharp", "stab",
            "flutter", "pound", "rac",
        ],
        5,
    ),
    (
        "trouble breathing",
        ["breath", "breathe", "breathing"],
        ["cant", "cannot", "trouble", "difficult", "hard", "short", "struggl", "gasp", "unable", "wheez"],
        4,
    ),
    (
        "stroke sign: one-sided or facial weakness",
        ["face", "facial", "arm", "arms", "leg", "legs", "hand", "hands", "side"],
        ["droop", "numb", "weak", "limp", "paraly", "sag"],
        4,
    ),
    (
        "stroke sign: speech",
        ["speech", "speak", "speaking", "talking", "words"],
        ["slur", "trouble", "difficult", "garbl", "mumbl", "confus"],
        3,
    ),
    (
        "stroke sign: sudden confusion",
        ["sudden", "suddenly"],
        ["confus", "disorient", "vision"],
        4,
    ),
    (
        "heavy bleeding",
        ["bleeding", "bleed", "bleeds", "bled", "hemorrhaging"],
        ["heav", "lot", "wont", "cant", "soak", "gush", "pour", "profus", "uncontroll", "severe", "everywhere", "stop"],
        4,
    ),
]

URGENT_WORD_STEMS = ["worse", "worsen", "fever", "febrile", "infected"]
URGENT_EXACT_WORDS = ["pus"]
URGENT_PHRASES = ["red streak", "red streaks"]
URGENT_WINDOW_RULES = [
    (
        "wound change",
        ["wound", "wounds", "incision", "stitch", "stitches", "suture", "sutures", "scab", "cut", "bite"],
        ["red", "swol", "swell", "drain", "ooz", "open", "hot", "warm", "smell", "leak", "bleed", "spread"],
        6,
    ),
]


def normalise_text(text: str) -> str:
    """Lowercase, drop apostrophes, turn every other non-alphanumeric character into a space."""
    lowered = text.lower()
    lowered = lowered.replace("'", "").replace("\u2019", "")
    characters = []
    for character in lowered:
        if character.isalnum():
            characters.append(character)
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def word_starts_with_any(word: str, stems: list[str]) -> bool:
    """True if the word begins with any of the stems."""
    for stem in stems:
        if word.startswith(stem):
            return True
    return False


def find_phrase(normalised_text: str, phrases: list[str]) -> Optional[str]:
    """Return the first phrase found as whole words in the text, or None."""
    padded_text = " " + normalised_text + " "
    for phrase in phrases:
        if " " + phrase + " " in padded_text:
            return phrase
    return None


def find_word_stem(words: list[str], stems: list[str]) -> Optional[str]:
    """Return the first word that begins with any stem, or None."""
    for word in words:
        if word_starts_with_any(word, stems):
            return word
    return None


def find_window_match(words: list[str], rule: tuple) -> Optional[str]:
    """Apply one (reason, anchors, nearby stems, window) rule. Returns the words that matched, or None."""
    reason, anchor_words, nearby_stems, window = rule
    for index in range(len(words)):
        if words[index] not in anchor_words:
            continue
        first_index = max(0, index - window)
        last_index = min(len(words) - 1, index + window)
        for other_index in range(first_index, last_index + 1):
            if other_index == index:
                continue
            if word_starts_with_any(words[other_index], nearby_stems):
                return words[index] + " ... " + words[other_index]
    return None


def red_flag_result(tier: int, message_key: str, reason: str) -> dict:
    """Build the dict check_red_flags returns."""
    return {"tier": tier, "message": RED_FLAG_MESSAGES[message_key], "reason": reason}


def check_red_flags(user_text: str) -> Optional[dict]:
    """Look for emergency (tier 0) or urgent (tier 1) wording in what the caller said.

    Returns None if nothing matched, else {"tier", "message", "reason"}. Mental-health wording
    is checked first so those callers get 988 rather than the 911 script.
    """
    normalised_text = normalise_text(user_text)
    words = normalised_text.split()

    phrase = find_phrase(normalised_text, MENTAL_HEALTH_PHRASES)
    if phrase is not None:
        return red_flag_result(0, "mental_health", "suicidal ideation or self-harm: " + phrase)
    stem_word = find_word_stem(words, MENTAL_HEALTH_WORD_STEMS)
    if stem_word is not None:
        return red_flag_result(0, "mental_health", "suicidal ideation or self-harm: " + stem_word)

    phrase = find_phrase(normalised_text, EMERGENCY_PHRASES)
    if phrase is not None:
        return red_flag_result(0, "emergency", "emergency phrase: " + phrase)
    stem_word = find_word_stem(words, EMERGENCY_WORD_STEMS)
    if stem_word is not None:
        return red_flag_result(0, "emergency", "stroke sign: " + stem_word)
    for rule in EMERGENCY_WINDOW_RULES:
        matched_words = find_window_match(words, rule)
        if matched_words is not None:
            return red_flag_result(0, "emergency", rule[0] + ": " + matched_words)

    stem_word = find_word_stem(words, URGENT_WORD_STEMS)
    if stem_word is not None:
        return red_flag_result(1, "urgent", "urgent wording: " + stem_word)
    for exact_word in URGENT_EXACT_WORDS:
        if exact_word in words:
            return red_flag_result(1, "urgent", "urgent wording: " + exact_word)
    phrase = find_phrase(normalised_text, URGENT_PHRASES)
    if phrase is not None:
        return red_flag_result(1, "urgent", "urgent phrase: " + phrase)
    for rule in URGENT_WINDOW_RULES:
        matched_words = find_window_match(words, rule)
        if matched_words is not None:
            return red_flag_result(1, "urgent", rule[0] + ": " + matched_words)

    return None


# ----------------------------------------------------------------------
# Grounding check
# ----------------------------------------------------------------------

FORBIDDEN_PHRASES = {
    "medication advice": [
        "mg", "milligram", "milligrams", "dose", "doses", "dosage", "ibuprofen", "tylenol",
        "acetaminophen", "aspirin", "antibiotic", "antibiotics", "painkiller", "painkillers",
        "over the counter", "you should take", "try taking",
    ],
    "diagnostic language": [
        "sounds like", "thats probably", "that is probably", "its probably", "it is probably",
        "this is probably", "it could be", "it might be", "thats likely", "that is likely",
        "thats most likely", "you probably have", "you likely have", "you may have",
        "could be a sign",
    ],
    "reassurance about symptoms": [
        "nothing to worry about", "thats normal", "that is normal", "perfectly normal",
        "i wouldnt worry", "i would not worry", "nothing serious", "not serious", "not a big deal",
    ],
}

MONTH_NUMBERS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
UNIT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
TEEN_WORDS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
ORDINAL_UNIT_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
}
ORDINAL_OTHER_WORDS = {
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30,
}

# "Dr. Kim", "Dr Susan Kim", "Doctor Kim". Capital letters only, so "Dr. Kim is" captures just "Kim".
DOCTOR_PATTERN = re.compile(r"\b(?:Dr|Doctor)\.?\s+([A-Z][A-Za-z'\u2019-]*)(?:\s+([A-Z][A-Za-z'\u2019-]*))?")
ISO_DATE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
ISO_TIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T(\d{2}):(\d{2})")
COLON_TIME_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})\b")

# Fields in tool results that hold a provider's name.
NAME_KEYS = ["name", "provider_name", "full_name"]


def number_from_word(token: str) -> Optional[int]:
    """Whole number for a digit string or number word ('9', 'nine', 'twenty', 'third'), else None."""
    if token.isdigit():
        return int(token)
    if len(token) > 2 and token[:-2].isdigit() and token[-2:] in ("st", "nd", "rd", "th"):
        return int(token[:-2])
    for word_table in (UNIT_WORDS, TEEN_WORDS, TENS_WORDS, ORDINAL_UNIT_WORDS, ORDINAL_OTHER_WORDS):
        if token in word_table:
            return word_table[token]
    return None


def claim_tokens(text: str) -> list[str]:
    """Split text into tokens with spoken numbers turned into digits, e.g. 'ten thirty a.m.' -> ['10', '30', 'am'].

    Spoken replies say 'October twenty-third', so the check has to read number words as well as digits.
    """
    lowered = text.lower()
    for marker in ("a.m.", "p.m.", "a.m", "p.m"):
        lowered = lowered.replace(marker, marker.replace(".", ""))
    raw_tokens = normalise_text(lowered).split()

    tokens = []
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]
        next_token = ""
        if index + 1 < len(raw_tokens):
            next_token = raw_tokens[index + 1]

        # "oh five" is the minutes 05.
        if token == "oh" and next_token in UNIT_WORDS:
            tokens.append("0" + str(UNIT_WORDS[next_token]))
            index = index + 2
            continue

        value = number_from_word(token)
        if value is None:
            tokens.append(token)
            index = index + 1
            continue

        # "twenty" followed by "one" / "first" is 21.
        if token in TENS_WORDS:
            if next_token in UNIT_WORDS:
                tokens.append(str(value + UNIT_WORDS[next_token]))
                index = index + 2
                continue
            if next_token in ORDINAL_UNIT_WORDS:
                tokens.append(str(value + ORDINAL_UNIT_WORDS[next_token]))
                index = index + 2
                continue
        tokens.append(str(value))
        index = index + 1
    return tokens


def extract_dates(text: str) -> set:
    """Specific dates in the text as (month, day) pairs. 'Tuesday' or 'next week' on its own gives nothing."""
    found_dates = set()

    for year_text, month_text, day_text in ISO_DATE_PATTERN.findall(text):
        found_dates.add((int(month_text), int(day_text)))

    tokens = claim_tokens(text)
    for index in range(len(tokens)):
        if tokens[index] not in MONTH_NUMBERS:
            continue
        month_number = MONTH_NUMBERS[tokens[index]]
        if index + 1 < len(tokens) and tokens[index + 1].isdigit():
            day = int(tokens[index + 1])
            if 1 <= day <= 31:
                found_dates.add((month_number, day))
                continue
        if index >= 1 and tokens[index - 1].isdigit():
            day = int(tokens[index - 1])
            if 1 <= day <= 31:
                found_dates.add((month_number, day))
                continue
        if index >= 2 and tokens[index - 1] == "of" and tokens[index - 2].isdigit():
            day = int(tokens[index - 2])
            if 1 <= day <= 31:
                found_dates.add((month_number, day))
    return found_dates


def extract_times(text: str) -> tuple[set, set]:
    """Clock times in the text, as (marked, unmarked).

    marked: (hour on a 24-hour clock, minute), from ISO times or from a time followed by am or pm.
    unmarked: (hour, minute) from a bare '10:30', which could be morning or evening.
    """
    marked_times = set()
    unmarked_times = set()

    for hour_text, minute_text in ISO_TIME_PATTERN.findall(text):
        marked_times.add((int(hour_text), int(minute_text)))
    for hour_text, minute_text in COLON_TIME_PATTERN.findall(text):
        unmarked_times.add((int(hour_text), int(minute_text)))

    tokens = claim_tokens(text)
    for index in range(len(tokens)):
        if tokens[index] != "am" and tokens[index] != "pm":
            continue
        if index < 1 or not tokens[index - 1].isdigit():
            continue
        hour = int(tokens[index - 1])
        minute = 0
        # "ten thirty am": the last token is the minutes only if it is two digits and the one before is a
        # plausible hour that is not the day of a month ("October 11 10 am" is 10:00, not 11:10).
        if index >= 2 and tokens[index - 2].isdigit() and len(tokens[index - 1]) == 2:
            hour_candidate = int(tokens[index - 2])
            minute_candidate = int(tokens[index - 1])
            preceded_by_month = index >= 3 and tokens[index - 3] in MONTH_NUMBERS
            if 1 <= hour_candidate <= 12 and minute_candidate <= 59 and not preceded_by_month:
                hour = hour_candidate
                minute = minute_candidate
        if hour < 1 or hour > 12:
            continue
        hour_on_twenty_four_hour_clock = hour % 12
        if tokens[index] == "pm":
            hour_on_twenty_four_hour_clock = hour_on_twenty_four_hour_clock + 12
        marked_times.add((hour_on_twenty_four_hour_clock, minute))
    return marked_times, unmarked_times


def collect_key_value_strings(value, key: str, pairs: list) -> None:
    """Walk nested dicts and lists, collecting (key, string) for every string value."""
    if isinstance(value, dict):
        for inner_key, inner_value in value.items():
            collect_key_value_strings(inner_value, inner_key, pairs)
        return
    if isinstance(value, list):
        for item in value:
            collect_key_value_strings(item, key, pairs)
        return
    if isinstance(value, str):
        pairs.append((key, value))


def build_seen_facts(tool_results: list[dict], current_time_description: str) -> dict:
    """Collect the provider names, dates, times and text this session's tools have actually returned.

    Names and locations come from tool results only. Dates and times also count when the model or
    caller passed them to a tool as arguments, so that repeating a date the caller asked for is not flagged.
    The current date and time count too: the system prompt gives them to the model.
    """
    result_pairs = []
    argument_pairs = []
    for entry in tool_results:
        collect_key_value_strings(entry["result"], "", result_pairs)
        collect_key_value_strings(entry["arguments"], "", argument_pairs)

    full_names = set()
    surnames = set()
    first_names = set()
    result_strings = []
    date_and_time_text = []

    for key, text in result_pairs:
        result_strings.append(text)
        date_and_time_text.append(text)
        if key in NAME_KEYS:
            name_words = normalise_text(text).split()
            if len(name_words) > 0 and name_words[0] == "dr":
                name_words = name_words[1:]
            if len(name_words) >= 2:
                full_names.add(name_words[0] + " " + name_words[-1])
                first_names.add(name_words[0])
                surnames.add(name_words[-1])
        for first_word, second_word in DOCTOR_PATTERN.findall(text):
            if second_word != "":
                full_names.add(first_word.lower() + " " + second_word.lower())
                first_names.add(first_word.lower())
                surnames.add(second_word.lower())
            else:
                surnames.add(first_word.lower())

    for key, text in argument_pairs:
        date_and_time_text.append(text)
    date_and_time_text.append(current_time_description)

    seen_dates = set()
    seen_times = set()
    for text in date_and_time_text:
        seen_dates.update(extract_dates(text))
        marked_times, unmarked_times = extract_times(text)
        seen_times.update(marked_times)
        seen_times.update(unmarked_times)

    return {
        "full_names": full_names,
        "surnames": surnames,
        "first_names": first_names,
        "dates": seen_dates,
        "times": seen_times,
        "padded_result_text": " " + normalise_text(" ".join(result_strings)) + " ",
    }


def grounding_failure(rule: str, span: str, reply_text: str) -> dict:
    """Build a failing result and log it, so the false-positive rate can be read from the log."""
    logger.info("Grounding flag: rule=%s span=%r reply=%r", rule, span, reply_text)
    return {"passed": False, "rule": rule, "span": span}


def check_grounding(
    reply_text: str, tool_results: list[dict], clinic_locations: list[str], current_time_description: str
) -> dict:
    """Check a model reply against what this session's tools have returned.

    Returns {"passed": True, "rule": None, "span": None}, or {"passed": False, "rule": ..., "span": ...}
    naming the first rule that caught something. clinic_locations is the list of every clinic location
    name; without it the check could not know that a word is a location. current_time_description is the
    sentence from the system prompt giving today's date and time.
    Generic words such as 'Tuesday' or 'tomorrow' are not claims and are never flagged.
    """
    normalised_reply = normalise_text(reply_text)

    for rule_name, phrases in FORBIDDEN_PHRASES.items():
        phrase = find_phrase(normalised_reply, phrases)
        if phrase is not None:
            return grounding_failure(rule_name, phrase, reply_text)

    seen_facts = build_seen_facts(tool_results, current_time_description)

    for match in DOCTOR_PATTERN.finditer(reply_text):
        first_word = match.group(1).lower()
        second_word = match.group(2)
        name_is_known = False
        if second_word is not None:
            if first_word + " " + second_word.lower() in seen_facts["full_names"]:
                name_is_known = True
            elif first_word in seen_facts["surnames"]:
                # "Dr Kim Tuesday": the second word was not part of the name.
                name_is_known = True
        else:
            if first_word in seen_facts["surnames"] or first_word in seen_facts["first_names"]:
                name_is_known = True
        if not name_is_known:
            return grounding_failure("doctor not seen in tool results", match.group(0), reply_text)

    padded_reply = " " + normalised_reply + " "
    for location_name in clinic_locations:
        normalised_location = " " + normalise_text(location_name) + " "
        if normalised_location in padded_reply and normalised_location not in seen_facts["padded_result_text"]:
            return grounding_failure("location not seen in tool results", location_name, reply_text)

    for month_number, day in sorted(extract_dates(reply_text)):
        if (month_number, day) not in seen_facts["dates"]:
            return grounding_failure("date not seen in tool results", f"month {month_number}, day {day}", reply_text)

    marked_times, unmarked_times = extract_times(reply_text)
    for hour, minute in sorted(marked_times):
        if (hour, minute) not in seen_facts["times"]:
            return grounding_failure("time not seen in tool results", f"{hour:02d}:{minute:02d}", reply_text)
    for hour, minute in sorted(unmarked_times):
        morning_reading = (hour, minute) in seen_facts["times"]
        evening_reading = ((hour + 12) % 24, minute) in seen_facts["times"]
        if not morning_reading and not evening_reading:
            return grounding_failure("time not seen in tool results", f"{hour}:{minute:02d}", reply_text)

    return {"passed": True, "rule": None, "span": None}


# ----------------------------------------------------------------------
# Manual checks: python -m agent.safety  (from the project root)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # (what the caller says, expected tier or None)
    red_flag_cases = [
        ("I have crushing chest pain", 0),
        ("my chest feels weird", 0),
        ("there's pressure in my chest and my left arm is numb", 0),
        ("I can't breathe properly", 0),
        ("my husband's face is drooping and his speech is slurred", 0),
        ("the bleeding won't stop", 0),
        ("I want to kill myself", 0),
        ("I've had a fever since Sunday", 1),
        ("my symptoms are getting worse", 1),
        ("the wound around my stitches is red and swollen", 1),
        ("I'd like to book an appointment with Dr Kim next Tuesday", None),
        ("I need to see a cardiologist about my cholesterol", None),
        ("Can I reschedule my appointment to Thursday morning?", None),
        ("what is your address in Oak Park", None),
        ("my date of birth is March 12, 1964", None),
        ("I want to see a doctor about my chest x-ray results", None),
    ]
    mismatches = 0
    for caller_text, expected_tier in red_flag_cases:
        outcome = check_red_flags(caller_text)
        if outcome is None:
            found_tier = None
            detail = "-"
        else:
            found_tier = outcome["tier"]
            detail = outcome["reason"]
        verdict = "ok"
        if found_tier != expected_tier:
            verdict = "MISMATCH"
            mismatches = mismatches + 1
        print(f"[{verdict}] tier={found_tier} expected={expected_tier} | {caller_text!r} | {detail}")

    sample_tool_results = [
        {
            "tool": "search_providers",
            "arguments": {"specialty": "Cardiology"},
            "result": {"providers": [{"provider_id": "PRV-1001", "name": "Susan Kim", "locations": ["LAKEVIEW"]}]},
        },
        {
            "tool": "get_availability",
            "arguments": {"provider_ids": ["PRV-1001"]},
            "result": {
                "slots": [
                    {
                        "slot_id": "SLT-00002",
                        "provider_name": "Dr. Susan Kim",
                        "location": "LAKEVIEW",
                        "start": "2026-10-23T10:30",
                        "spoken_time": "Friday, October 23 at 10:30 AM",
                    }
                ]
            },
        },
    ]
    sample_locations = ["LAKEVIEW", "DOWNTOWN", "OAK PARK"]
    sample_current_time = "The current date and time is Tuesday 20 October 2026, 5:50 PM Central."
    # (reply, should it pass)
    grounding_cases = [
        ("Dr. Susan Kim has an opening on Friday, October twenty-third at ten thirty a.m. at Lakeview.", True),
        ("Dr. Alan Brooks can see you on Friday, October 23 at 10:30 AM.", False),
        ("That sounds like a migraine. You could take ibuprofen and rest.", False),
        ("I can look at Tuesday, or any other day that suits you.", True),
        ("Dr. Kim is free at 4:15 PM at our Downtown office.", False),
        ("Today is Tuesday, October twentieth.", True),
    ]
    for reply, should_pass in grounding_cases:
        outcome = check_grounding(reply, sample_tool_results, sample_locations, sample_current_time)
        verdict = "ok"
        if outcome["passed"] != should_pass:
            verdict = "MISMATCH"
            mismatches = mismatches + 1
        print(f"[{verdict}] passed={outcome['passed']} rule={outcome['rule']} span={outcome['span']!r} | {reply!r}")

    print(f"\n{mismatches} mismatches")
    assert mismatches == 0
