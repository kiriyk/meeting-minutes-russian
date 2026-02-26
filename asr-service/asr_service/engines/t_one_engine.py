"""T-One streaming engine (Stage 2 MVP).

Current implementation provides real-time partial/final behavior and VAD-based
utterance boundaries. The text decoder is a fallback placeholder and can be
replaced by real T-One inference with the same interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..diarization import Diarizer, build_diarizer
from ..schemas import FinalSegmentMessage, PartialTranscriptMessage, StartSessionMessage
from ..vad import VAD
from .onnx_ctc_recognizer import OnnxCtcRecognizer

logger = logging.getLogger(__name__)


@dataclass
class _UtteranceState:
    start_ms: int
    last_speech_ms: int
    pcm: bytearray


class TOneEngine:
    name = "t_one"

    def __init__(self, config: StartSessionMessage, diarizer: Diarizer | None = None) -> None:
        self._config = config
        self._recognizer = OnnxCtcRecognizer()
        self._vad = VAD(
            mode=config.vad.mode,
            aggressiveness=config.vad.aggressiveness,
        )
        self._diarizer: Diarizer = diarizer or build_diarizer(
            enabled=config.diarization.enabled,
            mode=config.diarization.mode,
            huggingface_token=config.diarization.huggingface_token,
        )
        self._emit_ms = max(100, config.chunking.t_one_emit_ms)
        self._final_silence_ms = 600
        self._max_segment_ms = 6_000
        self._last_partial_emit_ms = 0
        self._segment_counter = 0
        self._active_utterance: _UtteranceState | None = None
        self._logged_decode_fallback = False

    async def start(self) -> None:
        if self._recognizer.is_ready:
            logger.info(
                "[t_one] started with recognizer backend=%s",
                self._recognizer.backend_name,
            )
        else:
            logger.warning(
                "[t_one] recognizer not ready, using placeholder output. reason=%s",
                self._recognizer.error,
            )
        return None

    async def stop(self) -> None:
        return None

    def process_chunk(
        self,
        *,
        session_id: str,
        chunk_pcm: bytes,
        chunk_start_ms: int,
        chunk_end_ms: int,
    ) -> list[PartialTranscriptMessage | FinalSegmentMessage]:
        events: list[PartialTranscriptMessage | FinalSegmentMessage] = []

        speech = self._vad.is_speech(chunk_pcm) if self._config.vad.enabled else True

        if speech:
            if self._active_utterance is None:
                self._active_utterance = _UtteranceState(
                    start_ms=chunk_start_ms,
                    last_speech_ms=chunk_end_ms,
                    pcm=bytearray(),
                )
            self._active_utterance.pcm.extend(chunk_pcm)
            self._active_utterance.last_speech_ms = chunk_end_ms

            if chunk_end_ms - self._last_partial_emit_ms >= self._emit_ms:
                partial_text = self._decode_partial(bytes(self._active_utterance.pcm))
                events.append(
                    PartialTranscriptMessage(
                        session_id=session_id,
                        engine=self.name,
                        time_range_ms=[self._active_utterance.start_ms, chunk_end_ms],
                        text=partial_text,
                        confidence=0.7,
                    )
                )
                logger.info(
                    "[t_one][partial] backend=%s range=%s..%s chars=%s",
                    self._recognizer.backend_name,
                    self._active_utterance.start_ms,
                    chunk_end_ms,
                    len(partial_text),
                )
                self._last_partial_emit_ms = chunk_end_ms

            # Force periodic finalization even without long silence, so captions
            # don't collapse into one huge segment for continuous speech.
            if chunk_end_ms - self._active_utterance.start_ms >= self._max_segment_ms:
                events.append(self._make_final_segment(session_id))

            return events

        # Silence branch: if utterance exists and silence gap is large enough, finalize.
        if self._active_utterance is not None:
            silence_ms = chunk_end_ms - self._active_utterance.last_speech_ms
            if silence_ms >= self._final_silence_ms:
                events.append(self._make_final_segment(session_id))

        return events

    def flush(self, session_id: str) -> list[FinalSegmentMessage]:
        if self._active_utterance is None:
            return []
        return [self._make_final_segment(session_id)]

    def _make_final_segment(self, session_id: str) -> FinalSegmentMessage:
        assert self._active_utterance is not None
        self._segment_counter += 1
        utt = self._active_utterance
        pcm_bytes = bytes(utt.pcm)

        # Assign speaker label using diarization pipeline
        speaker = self._diarizer.assign_speaker(pcm_bytes, self._config.sample_rate)

        event = FinalSegmentMessage(
            session_id=session_id,
            engine=self.name,
            segment_id=f"t-one-seg-{self._segment_counter}",
            time_range_ms=[utt.start_ms, utt.last_speech_ms],
            speaker=speaker,
            text=self._decode_final(pcm_bytes),
            tokens=None,
            confidence=0.75,
        )
        logger.info(
            "[t_one][final] backend=%s segment=%s range=%s..%s speaker=%s chars=%s",
            self._recognizer.backend_name,
            event.segment_id,
            utt.start_ms,
            utt.last_speech_ms,
            speaker,
            len(event.text),
        )

        self._active_utterance = None
        return event

    def _decode_partial(self, pcm: bytes) -> str:
        text = self._recognizer.transcribe_pcm_s16le(pcm, self._config.sample_rate)
        if text is None:
            if not self._logged_decode_fallback:
                logger.warning(
                    "[t_one] decode fallback is active. reason=%s",
                    self._recognizer.error,
                )
                self._logged_decode_fallback = True
            duration_s = self._duration_seconds(pcm)
            return f"[live] speech {duration_s:.1f}s"
        return text or ""

    def _decode_final(self, pcm: bytes) -> str:
        text = self._recognizer.transcribe_pcm_s16le(pcm, self._config.sample_rate)
        if text is None:
            if not self._logged_decode_fallback:
                logger.warning(
                    "[t_one] decode fallback is active. reason=%s",
                    self._recognizer.error,
                )
                self._logged_decode_fallback = True
            duration_s = self._duration_seconds(pcm)
            return f"[final] speech segment {duration_s:.1f}s"
        return self._apply_basic_punctuation(text or "")

    def _duration_seconds(self, pcm: bytes) -> float:
        bytes_per_sample = 2
        num_samples = len(pcm) / bytes_per_sample
        return num_samples / float(self._config.sample_rate)

    @staticmethod
    def _apply_basic_punctuation(text: str) -> str:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            return ""
        first = cleaned[0].upper()
        rest = cleaned[1:] if len(cleaned) > 1 else ""
        out = f"{first}{rest}"
        if out[-1] not in ".!?":
            out += "."
        return out
