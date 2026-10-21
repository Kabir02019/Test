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
import os
import sys
import time

from dotenv import load_dotenv
from loguru import logger

# The environment must be loaded before agent.core creates its Anthropic client.
load_dotenv(override=True)

from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    Frame,
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

    def __init__(self, session: dict):
        super().__init__()
        self.session = session
        # One turn at a time: handle_turn mutates the session and must never overlap with itself.
        self.turn_lock = asyncio.Lock()
        self.latest_turn_number = 0
        # The assistant message we pushed to TTS and have not yet heard the aggregator confirm.
        self.unconfirmed_reply_message = None
        self.unconfirmed_reply_text = ""
        # Timestamps for the reply being spoken now; the timer logs them when audio starts.
        self.pending_timing = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Turn a finished user turn into a handle_turn call; pass every other frame through."""
        await super().process_frame(frame, direction)

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

            # handle_turn blocks on network calls; on the event loop it would freeze audio and VAD.
            reply_text, _ = await asyncio.to_thread(handle_turn, self.session, user_text)
            reply_time = time.monotonic()
            logger.info("turn {}: reply returned after {:.2f}s: {!r}", turn_number, reply_time - transcript_time, reply_text)

            reply_message = self.session["messages"][-1]
            caller_spoke_again = turn_number != self.latest_turn_number
            # An escalation script is spoken even if it is late: it is the one reply that must be heard.
            is_escalation = self.session["stage"] == "escalate"
            if caller_spoke_again and not is_escalation:
                logger.info("turn {}: caller spoke again before this reply; not speaking it", turn_number)
                set_message_text(reply_message, INTERRUPTED_BEFORE_SPEECH_NOTE)
                return

            self.unconfirmed_reply_message = reply_message
            self.unconfirmed_reply_text = reply_text
            self.pending_timing = {"turn": turn_number, "transcript": transcript_time, "reply": reply_time}

        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(reply_text))
        await self.push_frame(LLMFullResponseEndFrame())

    def settle_unconfirmed_reply(self) -> None:
        """If the last reply was never confirmed by the assistant aggregator, none of it was spoken.

        The aggregator reports a turn only when at least one sentence finished playing, so a reply
        interrupted before its first sentence ends produces no report at all.
        """
        if self.unconfirmed_reply_message is None:
            return
        logger.info("previous reply was never spoken; recording that in the history")
        set_message_text(self.unconfirmed_reply_message, INTERRUPTED_BEFORE_SPEECH_NOTE)
        self.unconfirmed_reply_message = None

    def record_spoken_text(self, spoken_text: str) -> None:
        """Called when the assistant aggregator reports what was spoken, whole or cut short by barge-in."""
        if self.unconfirmed_reply_message is None:
            return
        if not same_words(spoken_text, self.unconfirmed_reply_text):
            logger.info("reply was cut short; history keeps only: {!r}", spoken_text)
            set_message_text(self.unconfirmed_reply_message, spoken_text + INTERRUPTED_PARTWAY_NOTE)
        self.unconfirmed_reply_message = None


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
        logger.info(
            "turn {}: transcript->reply {:.2f}s, reply->first audio {:.2f}s, transcript->first audio {:.2f}s",
            timing["turn"],
            timing["reply"] - timing["transcript"],
            first_audio_time - timing["reply"],
            first_audio_time - timing["transcript"],
        )


async def main():
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
    bridge = AgentBridge(session)

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
