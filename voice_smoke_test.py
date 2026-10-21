#
# Adapted from pipecat examples/foundational/07x-interruptible-local.py
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# Changes from upstream:
#   - Cartesia TTS  -> Deepgram TTS   (one vendor for STT and TTS)
#   - OpenAI LLM    -> Anthropic LLM
#
# Requires: pip install "pipecat-ai[deepgram,anthropic,silero,local]==0.0.108"
# Env vars: DEEPGRAM_API_KEY, ANTHROPIC_API_KEY
#

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant in a voice conversation. Your responses will be "
    "spoken aloud, so avoid emojis, bullet points, or other formatting that cannot "
    "be spoken. Respond to what the user said in a creative, helpful, and brief way."
)


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

    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        settings=AnthropicLLMService.Settings(
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    context.add_message({"role": "developer", "content": "Please introduce yourself to the user."})
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
