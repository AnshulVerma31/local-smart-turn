#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from deepgram.clients.listen.v1.websocket.options import LiveOptions
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    MetricsFrame,
    OutputImageRawFrame,
    SpriteFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import SmartTurnMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIServerMessageFrame,
    RTVITextMessageData,
    RTVIUserTranscriptionMessage,
    RTVIUserTranscriptionMessageData,
    RTVIBotLLMTextMessage,
    RTVIBotTranscriptionMessage,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecatcloud.agent import DailySessionArguments

import re
import time
from collections import deque
from dataclasses import dataclass

load_dotenv(override=True)

# Check if we're in local development mode
LOCAL = os.getenv("LOCAL_RUN")

logger.remove()
logger.add(sys.stderr, level="DEBUG")

sprites = []
script_dir = os.path.dirname(__file__)

# Load sequential animation frames
for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

# Create a smooth animation by adding reversed frames
flipped = sprites[::-1]
sprites.extend(flipped)

# Define static and animated states
quiet_frame = sprites[0]  # Static frame for when bot is listening
talking_frame = SpriteFrame(images=sprites)  # Animation sequence for when bot is talking


@dataclass
class ConversationEntry:
    speaker: str
    text: str
    timestamp: float
    is_command: bool = False


class ConversationHistory:
    """Simple in-memory conversation history for logging and summaries."""

    def __init__(self, *, max_age_secs: float = 300.0, max_entries: int = 200):
        self._entries: deque[ConversationEntry] = deque(maxlen=max_entries)
        self._max_age_secs = max_age_secs

    def add(self, speaker: str, text: str, *, is_command: bool = False):
        entry = ConversationEntry(
            speaker=speaker,
            text=text.strip(),
            timestamp=time.time(),
            is_command=is_command,
        )
        if not entry.text:
            return
        self._entries.append(entry)
        self._prune()

    def _prune(self):
        cutoff = time.time() - self._max_age_secs
        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.popleft()

    def recent(self, window_secs: float, *, include_commands: bool = False) -> list[ConversationEntry]:
        cutoff = time.time() - window_secs
        return [
            entry
            for entry in self._entries
            if entry.timestamp >= cutoff and (include_commands or not entry.is_command)
        ]


SUMMARY_COMMANDS = {"summary", "summarize"}
SUMMARY_WINDOW_SECS = 10.0


def sanitize_command(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_summary(entries: list[ConversationEntry]) -> list[str]:
    if not entries:
        return []

    # Provide up to three of the most recent messages as a lightweight summary.
    recent = entries[-3:]
    bullets = []
    for entry in recent:
        prefix = "You" if entry.speaker == "user" else "Bot"
        bullets.append(f"{prefix}: {entry.text}")
    return bullets


class TalkingAnimation(FrameProcessor):
    """Manages the bot's visual animation states.

    Switches between static (listening) and animated (talking) states based on
    the bot's current speaking status.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and update animation state.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        # Switch to talking animation when bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        # Return to static frame when bot stops speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame, direction)


class SmartTurnMetricsProcessor(FrameProcessor):
    """Processes the metrics data from Smart Turn Analyzer.

    This processor is responsible for handling smart turn metrics data
    and forwarding it to the client UI via RTVI.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle Smart Turn metrics.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        # Handle Smart Turn metrics
        if isinstance(frame, MetricsFrame):
            for metrics in frame.data:
                if isinstance(metrics, SmartTurnMetricsData):
                    logger.info(f"Smart Turn metrics: {metrics}")

                    # Create a payload with the smart turn prediction data
                    smart_turn_data = {
                        "type": "smart_turn_result",
                        "is_complete": metrics.is_complete,
                        "probability": metrics.probability,
                        "inference_time_ms": metrics.inference_time_ms,
                        "server_total_time_ms": metrics.server_total_time_ms,
                        "e2e_processing_time_ms": metrics.e2e_processing_time_ms,
                    }

                    # Send the data to the client via RTVI
                    rtvi_frame = RTVIServerMessageFrame(data=smart_turn_data)
                    await self.push_frame(rtvi_frame, FrameDirection.UPSTREAM)

        await self.push_frame(frame, direction)


class TranscriptionBroadcastProcessor(FrameProcessor):
    """Broadcasts transcription updates to RTVI clients and backend logs."""

    def __init__(self, rtvi: RTVIProcessor, history: ConversationHistory):
        super().__init__()
        self._rtvi = rtvi
        self._history = history

    async def _send_user_transcription(self, frame: TranscriptionFrame | InterimTranscriptionFrame, *, final: bool):
        if not self._rtvi:
            return
        message = RTVIUserTranscriptionMessage(
            data=RTVIUserTranscriptionMessageData(
                text=frame.text,
                user_id=frame.user_id,
                timestamp=frame.timestamp,
                final=final,
            )
        )
        await self._rtvi.push_transport_message(message)

    async def _log_summary(self):
        recent_entries = self._history.recent(SUMMARY_WINDOW_SECS)
        bullets = build_summary(recent_entries)
        if not bullets:
            logger.info("Summary requested but no recent conversation to summarize.")
            return

        logger.info("Summary of the last %.0f seconds:", SUMMARY_WINDOW_SECS)
        for bullet in bullets:
            logger.info(" • %s", bullet)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            logger.debug("User interim transcript: %s", frame.text)
            await self._send_user_transcription(frame, final=False)
        elif isinstance(frame, TranscriptionFrame):
            logger.info("User transcript: %s", frame.text)
            sanitized = sanitize_command(frame.text)
            is_summary = sanitized in SUMMARY_COMMANDS if sanitized else False
            self._history.add("user", frame.text, is_command=is_summary)
            await self._send_user_transcription(frame, final=True)
            if is_summary:
                await self._log_summary()

        await self.push_frame(frame, direction)


class LLMOutputBroadcastProcessor(FrameProcessor):
    """Streams LLM text outputs to RTVI clients and logs bot responses."""

    def __init__(self, rtvi: RTVIProcessor, history: ConversationHistory):
        super().__init__()
        self._rtvi = rtvi
        self._history = history
        self._buffer: list[str] = []

    async def _send_llm_chunk(self, text: str):
        if not self._rtvi:
            return
        message = RTVIBotLLMTextMessage(data=RTVITextMessageData(text=text))
        await self._rtvi.push_transport_message(message)

    async def _flush_bot_transcript(self):
        if not self._buffer:
            return
        full_text = "".join(self._buffer).strip()
        self._buffer.clear()
        if not full_text:
            return

        logger.info("Bot response: %s", full_text)
        self._history.add("bot", full_text)
        if self._rtvi:
            message = RTVIBotTranscriptionMessage(data=RTVITextMessageData(text=full_text))
            await self._rtvi.push_transport_message(message)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer.clear()
        elif isinstance(frame, LLMTextFrame):
            if frame.text:
                logger.debug("Bot LLM chunk: %s", frame.text)
                self._buffer.append(frame.text)
                await self._send_llm_chunk(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush_bot_transcript()

        await self.push_frame(frame, direction)


async def main(transport: DailyTransport):
    # Configure your STT and LLM services here
    # Swap out different processors or properties to customize your bot
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-3-general",
            interim_results=True,
            punctuate=True,
            smart_format=True,
        ),
    )

    # Enable multilingual captioning (English + Hindi) with automatic detection.
    stt._settings.pop("language", None)
    stt._settings["detect_language"] = True
    stt._settings["languages"] = ["en", "hi"]
    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

    # Set up the initial context for the conversation
    # You can specified initial system and assistant messages here
    messages = [
        {
            "role": "system",
            "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Respond to what the user said in a creative and helpful way, but keep your responses brief and easy to read as plain text. Start by introducing yourself.",
        },
    ]

    # This sets up the LLM context by providing messages and tools
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    ta = TalkingAnimation()
    smart_turn_metrics_processor = SmartTurnMetricsProcessor()
    conversation_history = ConversationHistory()

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]), transport=transport)
    transcription_broadcaster = TranscriptionBroadcastProcessor(rtvi, conversation_history)
    llm_output_broadcaster = LLMOutputBroadcastProcessor(rtvi, conversation_history)

    # A core voice AI pipeline
    # Add additional processors to customize the bot's behavior
    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            smart_turn_metrics_processor,
            stt,
            transcription_broadcaster,
            context_aggregator.user(),
            llm,
            llm_output_broadcaster,
            ta,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.debug("Client ready event received")
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info("First participant joined: {}", participant["id"])
        # Push a static frame to show the bot is listening
        await task.queue_frame(quiet_frame)

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info("Participant left: {}", participant)
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)


async def bot(args: DailySessionArguments):
    """Main bot entry point compatible with the FastAPI route handler.

    Args:
        room_url: The Daily room URL
        token: The Daily room token
        body: The configuration object from the request body
        session_id: The session ID for logging
    """
    from pipecat.audio.filters.krisp_filter import KrispFilter

    logger.info(f"Bot process initialized {args.room_url} {args.token}")
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            args.room_url,
            args.token,
            "Smart Turn Bot",
            params=DailyParams(
                audio_in_enabled=True,
                audio_in_filter=KrispFilter(),
                audio_out_enabled=False,
                video_out_enabled=True,
                video_out_width=1024,
                video_out_height=576,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
                turn_analyzer=LocalSmartTurnAnalyzerV3(),
            ),
        )

        try:
            await main(transport)
            logger.info("Bot process completed")
        except Exception as e:
            logger.exception(f"Error in bot process: {str(e)}")
            raise


# Local development
async def local_daily():
    """Daily transport for local development."""
    from runner import configure

    try:
        async with aiohttp.ClientSession() as session:
            (room_url, token) = await configure(session)
            transport = DailyTransport(
                room_url,
                token,
                "Smart Turn Bot",
                params=DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=False,
                    video_out_enabled=True,
                    video_out_width=1024,
                    video_out_height=576,
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
                    turn_analyzer=LocalSmartTurnAnalyzerV3(),
                ),
            )

            await main(transport)
    except Exception as e:
        logger.exception(f"Error in local development mode: {e}")


# Local development entry point
if LOCAL and __name__ == "__main__":
    try:
        asyncio.run(local_daily())
    except Exception as e:
        logger.exception(f"Failed to run in local mode: {e}")
