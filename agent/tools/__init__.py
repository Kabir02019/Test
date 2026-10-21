"""The scheduling tools. ClinicTools is the one class everything else imports, with TOOL_SCHEMAS.
Holds only the combined class and its shared state; the tools themselves live in reads, eligibility and writes."""

from datetime import datetime
from typing import Callable, Optional

from adapter.scheduling_adapter import SchedulingAdapter
from agent.tools.eligibility import EligibilityTools
from agent.tools.helpers import clinic_now
from agent.tools.reads import ReadTools
from agent.tools.schemas import TOOL_SCHEMAS
from agent.tools.writes import WriteTools

__all__ = ["ClinicTools", "TOOL_SCHEMAS"]


class ClinicTools(ReadTools, EligibilityTools, WriteTools):
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
