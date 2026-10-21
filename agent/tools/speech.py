"""Turn data into words a caller will hear: dates, times and doctor names.
Does not hold rules, lookups or anything that reads clinic data."""

from datetime import date, datetime


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
