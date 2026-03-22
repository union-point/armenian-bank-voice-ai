from __future__ import annotations

import logging
import time

import numpy as np
from faster_whisper import WhisperModel

from livekit.agents import stt, APIConnectOptions, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr

__all__ = ["FasterWhisperSTT"]

logger = logging.getLogger(__name__)


class FasterWhisperSTT(stt.STT):
    """
    LiveKit STT using FasterWhisper for local speech recognition.

    This code integrates the faster-whisper library with LiveKit's Agents
    framework, enabling fully local speech-to-text without cloud dependencies.

    Args:
        model_name_or_path: Whisper model to use. Larger = more accurate but slower.
            Recommended: "whisper-small-hy-ct2" for low latency, "whisper-medium-hy-ct2" for accuracy
        device: Processing device - "cuda" for GPU, "cpu" for CPU
        compute_type: Model quantization for memory/speed tradeoff
            "float16" - Best for GPU (fastest)
            "int8" - Best for CPU (smallest memory)
            "int8_float16" - Balanced option
        language: Language hint for recognition (e.g., "en", "es", "fr")
            Improves accuracy when you know the expected language
        beam_size: Beam search width (1-10). Higher = more accurate but slower
        vad_filter: Enable voice activity detection to filter silence

    Performance Notes:
        - GPU Memory: ~2GB VRAM for medium model
        - Enable DEBUG logging to see latency metrics


    """

    def __init__(
        self,
        model_name_or_path: str = "whisper-small-hy-ct2",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "hy",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False
            )
        )

        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter

        logger.info(f"Loading FasterWhisper model: {model_name_or_path} on {device} ({compute_type})")

        self._model = WhisperModel(
            model_size_or_path=model_name_or_path,
            device=device,
            compute_type=compute_type
        )

        logger.info(f"FasterWhisper ready - language={language}, beam_size={beam_size}")

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions
    ) -> stt.SpeechEvent:
        """
        Process audio buffer and return transcription.

        Handles both single AudioFrame and lists of AudioFrames from LiveKit.
        Audio is normalized to float32 [-1, 1] range for Whisper processing.
        """
        # Convert AudioBuffer to numpy array
        if isinstance(buffer, list):
            all_data = []
            for frame in buffer:
                frame_data = np.frombuffer(frame.data, dtype=np.int16)
                all_data.append(frame_data)
            audio_data = np.concatenate(all_data).astype(np.float32) / 32768.0
        else:
            audio_data = np.frombuffer(buffer.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Use provided language or fall back to configured default
        lang = language if language is not NOT_GIVEN else self._language

        # Run transcription with optimized settings
        start_time = time.perf_counter()
        segments, info = self._model.transcribe(
            audio_data,
            beam_size=self._beam_size,
            best_of=self._beam_size,
            temperature=0.0,  # Greedy decoding for consistency
            vad_filter=self._vad_filter,
            language=lang,
        )

        # Combine all segments into final text
        text = "".join(segment.text for segment in segments).strip()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if text:
            logger.debug(f"Transcribed ({info.language}, {info.duration:.1f}s): {text}")

        logger.debug(f"STT latency: {elapsed_ms:.0f}ms for {info.duration:.1f}s audio")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(
                text=text,
                start_time=0,
                end_time=0,
                language=lang or ""
            )],
        )
