#
# Transport, STT, TTS and VAD setup is copied unchanged from voice_smoke_test.py (verified working).
# The LLM service is replaced by AgentBridge, which hands the transcript to agent.core.handle_turn.
#
# Requires: pip install "pipecat-ai[deepgram,silero,local]==0.0.108" (anthropic is used by agent/core.py)
# Env vars: DEEPGRAM_API_KEY, ANTHROPIC_API_KEY
# Run from the project root: .venv/bin/python -m voice.pipeline
#
"""Voice front end: carries a transcript into handle_turn and the reply out to TTS.

Holds only pipecat wiring and the bookkeeping that depends on what was actually spoken.
Does not hold any scheduling, safety or model logic; agent/ never imports this module.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Callable

from dotenv import load_dotenv
from loguru import logger

# The environment must be loaded before agent.core creates its Anthropic client.
load_dotenv(override=True)

from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.services.deepgram.stt import DeepgramSTTService  # noqa: E402
from pipecat.services.deepgram.tts import DeepgramTTSService  # noqa: E402
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams  # noqa: E402

from agent.core import handle_turn, new_session  # noqa: E402

logger.remove(0)
logger.add(sys.stderr, level="INFO")

INTERRUPTED_BEFORE_SPEECH_NOTE = "(The caller interrupted before hearing any of this reply.)"
INTERRUPTED_PARTWAY_NOTE = " [The caller interrupted here.]"


def check_required_env_vars():
    """Fail early and clearly if a key is missing, rather than deep inside the pipeline."""
    required_names = ["DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY"]
    missing_names = []

    for name in required_names:
        value = os.getenv(name)
        if value is None or value.strip() == "":
            missing_names.append(name)

    if len(missing_names) > 0:
        logger.error("Missing environment variables: {}", ", ".join(missing_names))
        sys.exit(1)


def ignore_event(event: dict) -> None:
    """The default event sink: nothing is listening, so the pipeline runs exactly as it would without an operator view."""


def session_state(session: dict) -> dict:
    """The session fields the operator view shows. Built fresh so a later change cannot alter an event already sent."""
    return {
        "stage": session["stage"],
        "patient_id": session["patient_id"],
        "collected": dict(session["collected"]),
        "hold_id": session["hold_id"],
        "slot_id": session["slot_id"],
        "appointment": session["appointment"],
    }


class SafetyLogHandler(logging.Handler):
    """Turns the warnings agent/core.py already logs for red flags and grounding failures into operator events.

    This is how the pipeline sees safety outcomes without agent/ having to report them.
    """

    def __init__(self, bridge: "AgentBridge"):
        super().__init__(level=logging.WARNING)
        self.bridge = bridge
        self.red_flag_seen = False
        self.grounding_failures = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Called on the thread that logged, which is the handle_turn worker thread."""
        if record.msg.startswith("Red flag tier"):
            tier, reason, caller_text = record.args
            self.red_flag_seen = True
            self.bridge.send_event(
                {"type": "safety", "check": "red_flag", "result": "flag", "detail": f"tier {tier}: {reason}"}
            )
        if record.msg.startswith("Grounding check failed"):
            rule, span, reply_text = record.args
            self.grounding_failures = self.grounding_failures + 1
            self.bridge.send_event(
                {"type": "safety", "check": "grounding", "result": "flag", "detail": f"{rule}: {span}"}
            )


def same_words(first_text: str, second_text: str) -> bool:
    """True if two texts differ only in whitespace."""
    return " ".join(first_text.split()) == " ".join(second_text.split())


def set_message_text(message: dict, text: str) -> None:
    """Replace an assistant message in the agent's history with a single text block."""
    message["content"] = [{"type": "text", "text": text}]


class AgentBridge(FrameProcessor):
    """Sits where the LLM service would be: finished user turn in, spoken reply out.

    Also keeps the agent's history honest about barge-in. See settle_unconfirmed_reply and
    record_spoken_text for how.
    """

    def __init__(self, session: dict, publish_event: Callable[[dict], None] = ignore_event):
        super().__init__()
        self.session = session
        # Must be thread-safe and must never block: it is called from the handle_turn worker thread too.
        self.publish_event = publish_event
        self.current_turn = 0
        self.safety_log_handler = SafetyLogHandler(self)
        logging.getLogger("agent.core").addHandler(self.safety_log_handler)
        # One turn at a time: handle_turn mutates the session and must never overlap with itself.
        self.turn_lock = asyncio.Lock()
        self.latest_turn_number = 0
        # Replies pushed to TTS that the aggregator has not yet reported on, oldest first. A list, not a
        # single slot: the caller can start the next turn before the report for the previous reply arrives.
        # Each entry is {"message": the history message, "text": the reply, "turn": its turn number}.
        self.awaiting_reports = []
        # Timestamps for the reply being spoken now; the timer logs them when audio starts.
        self.pending_timing = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Turn a finished user turn into a handle_turn call; pass every other frame through."""
        await super().process_frame(frame, direction)

        # An interruption while a reply is still unconfirmed means the caller talked over the agent.
        if isinstance(frame, InterruptionFrame) and len(self.awaiting_reports) > 0:
            # The oldest unreported reply is the one that was playing.
            self.send_event({"type": "barge_in", "turn": self.awaiting_reports[0]["turn"]})

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        last_message = frame.context.get_messages()[-1]
        if last_message["role"] != "user":
            await self.push_frame(frame, direction)
            return

        self.latest_turn_number = self.latest_turn_number + 1
        # Own task, not this frame's processing task: an interruption cancels the latter, and
        # cancelling would abandon a handle_turn that is already running in a thread.
        self.create_task(self.run_turn(last_message["content"], self.latest_turn_number))

    async def run_turn(self, user_text: str, turn_number: int):
        """Run handle_turn off the event loop, then send the reply to TTS."""
        transcript_time = time.monotonic()
        logger.info("turn {}: transcript received: {!r}", turn_number, user_text)

        async with self.turn_lock:
            self.settle_unconfirmed_reply()
            self.current_turn = turn_number
            self.send_event({"type": "transcript", "text": user_text})
            self.safety_log_handler.red_flag_seen = False
            self.safety_log_handler.grounding_failures = 0
            escalation_id_before = self.session["escalation_id"]

            # handle_turn blocks on network calls; on the event loop it would freeze audio and VAD.
            reply_text, _ = await asyncio.to_thread(
                handle_turn,
                self.session,
                user_text,
                on_tool_call=self.report_tool_call,
                on_stage_change=self.report_stage_change,
            )
            reply_time = time.monotonic()
            self.report_turn_outcome(escalation_id_before)
            logger.info("turn {}: reply returned after {:.2f}s: {!r}", turn_number, reply_time - transcript_time, reply_text)

            reply_message = self.session["messages"][-1]
            caller_spoke_again = turn_number != self.latest_turn_number
            # An escalation script is spoken even if it is late: it is the one reply that must be heard.
            is_escalation = self.session["stage"] == "escalate"
            if caller_spoke_again and not is_escalation:
                logger.info("turn {}: caller spoke again before this reply; not speaking it", turn_number)
                set_message_text(reply_message, INTERRUPTED_BEFORE_SPEECH_NOTE)
                return

            self.awaiting_reports.append({"message": reply_message, "text": reply_text, "turn": turn_number})
            self.pending_timing = {"turn": turn_number, "transcript": transcript_time, "reply": reply_time}
            self.send_event({"type": "reply", "text": reply_text})

        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(reply_text))
        await self.push_frame(LLMFullResponseEndFrame())

    def send_event(self, event: dict) -> None:
        """Stamp an event with the turn it belongs to and hand it to the operator view."""
        if "turn" not in event:
            event["turn"] = self.current_turn
        self.publish_event(event)

    def report_tool_call(self, tool_name: str, arguments: dict, result: dict) -> None:
        """handle_turn callback, on the worker thread."""
        self.send_event(
            {
                "type": "tool_call",
                "name": tool_name,
                "arguments": arguments,
                "result": result,
                "state": session_state(self.session),
            }
        )

    def report_stage_change(self, old_stage: str, new_stage: str) -> None:
        """handle_turn callback, on the worker thread."""
        self.send_event(
            {"type": "stage_change", "from": old_stage, "to": new_stage, "state": session_state(self.session)}
        )

    def report_turn_outcome(self, escalation_id_before) -> None:
        """After handle_turn: report what it did that its callbacks do not, and the safety checks that passed."""
        if self.session["escalation_id"] != escalation_id_before:
            # The escalation record is made by code inside handle_turn, so no tool callback fired for it.
            escalation_entry = self.session["tool_results"][-1]
            self.send_event(
                {
                    "type": "tool_call",
                    "name": escalation_entry["tool"],
                    "arguments": escalation_entry["arguments"],
                    "result": escalation_entry["result"],
                    "state": session_state(self.session),
                }
            )
        if not self.safety_log_handler.red_flag_seen:
            self.send_event({"type": "safety", "check": "red_flag", "result": "pass", "detail": ""})
            if self.safety_log_handler.grounding_failures == 0:
                self.send_event({"type": "safety", "check": "grounding", "result": "pass", "detail": ""})

    def settle_unconfirmed_reply(self) -> None:
        """Before a new turn runs, record every reply with no report yet as unheard.

        The aggregator reports a reply only when at least one sentence finished playing, so a reply
        interrupted before its first sentence ends produces no report at all. The note is tentative:
        if a report for the reply arrives late, record_spoken_text overwrites it with what was spoken.
        """
        for entry in self.awaiting_reports:
            logger.info("turn {}: reply has no report yet; recording it as unheard for now", entry["turn"])
            set_message_text(entry["message"], INTERRUPTED_BEFORE_SPEECH_NOTE)

    def record_spoken_text(self, spoken_text: str) -> None:
        """Called when the assistant aggregator reports what was spoken, whole or cut short by barge-in.

        The report names no turn, so it is matched to the oldest awaiting reply that starts with the spoken
        words. Replies older than the match never got a report, so none of them was heard.
        """
        spoken_words = spoken_text.split()
        if len(spoken_words) == 0:
            return
        matched_index = None
        for index in range(len(self.awaiting_reports)):
            reply_words = self.awaiting_reports[index]["text"].split()
            if reply_words[: len(spoken_words)] == spoken_words:
                matched_index = index
                break
        if matched_index is None:
            logger.warning("spoken text matches no awaiting reply: {!r}", spoken_text)
            return

        for index in range(matched_index):
            set_message_text(self.awaiting_reports[index]["message"], INTERRUPTED_BEFORE_SPEECH_NOTE)

        matched_entry = self.awaiting_reports[matched_index]
        if same_words(spoken_text, matched_entry["text"]):
            # Undo the tentative "unheard" note if the next turn had already started.
            set_message_text(matched_entry["message"], matched_entry["text"])
        else:
            logger.info("turn {}: reply was cut short; history keeps only: {!r}", matched_entry["turn"], spoken_text)
            self.send_event({"type": "barge_in", "turn": matched_entry["turn"], "spoken": spoken_text})
            set_message_text(matched_entry["message"], spoken_text + INTERRUPTED_PARTWAY_NOTE)
        self.awaiting_reports = self.awaiting_reports[matched_index + 1 :]


class FirstAudioTimer(FrameProcessor):
    """Last in the pipeline. Logs the turn's timings when the bot's audio starts playing."""

    def __init__(self, bridge: AgentBridge):
        super().__init__()
        self.bridge = bridge

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Log on BotStartedSpeakingFrame, which the output transport sends once audio is going out."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if not isinstance(frame, BotStartedSpeakingFrame):
            return
        timing = self.bridge.pending_timing
        if timing is None:
            return
        self.bridge.pending_timing = None
        first_audio_time = time.monotonic()
        self.bridge.send_event(
            {
                "type": "timing",
                "turn": timing["turn"],
                "reply_ms": round((timing["reply"] - timing["transcript"]) * 1000),
                "first_audio_ms": round((first_audio_time - timing["transcript"]) * 1000),
            }
        )
        logger.info(
            "turn {}: transcript->reply {:.2f}s, reply->first audio {:.2f}s, transcript->first audio {:.2f}s",
            timing["turn"],
            timing["reply"] - timing["transcript"],
            first_audio_time - timing["reply"],
            first_audio_time - timing["transcript"],
        )


async def main(publish_event: Callable[[dict], None] = ignore_event):
    """Run one call. publish_event receives the operator-view events; by default they go nowhere."""
    check_required_env_vars()

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramTTSService.Settings(
            voice="aura-2-andromeda-en",
        ),
    )

    # One session for the whole call. The agent's own history lives in it, not in pipecat's context.
    session = new_session()
    bridge = AgentBridge(session, publish_event)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        bridge.record_spoken_text(message.content)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            bridge,
            tts,
            transport.output(),
            assistant_aggregator,
            FirstAudioTimer(bridge),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
