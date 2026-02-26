"""GigaAM quasi-realtime segmented engine (Stage 3 MVP).

This implementation emits final segments in quasi-realtime using speech/VAD
buffering with target/max segment durations and small overlap carry-over.
Real GigaAM decoding can later replace the placeholder decoder while keeping
the same interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..diarization import Diarizer, build_diarizer
from ..schemas import FinalSegmentMessage, StartSessionMessage
from ..vad import VAD
from .gigaam_pytorch_recognizer import GigaAMPytorchRecognizer
from .onnx_ctc_recognizer import OnnxCtcRecognizer

logger = logging.getLogger(__name__)


@dataclass
class _SpeechBuffer:
    start_ms: int
    last_speech_ms: int
    pcm: bytearray


class GigaAMEngine:
    name = "gigaam"

    def __init__(self, config: StartSessionMessage, diarizer: Diarizer | None = None) -> None:
        self._config = config
        self._gigaam_recognizer = GigaAMPytorchRecognizer()
        self._fallback_recognizer = OnnxCtcRecognizer()
        self._vad = VAD(
            mode=config.vad.mode,
            aggressiveness=config.vad.aggressiveness,
        )
        self._diarizer: Diarizer = diarizer or build_diarizer(
            enabled=config.diarization.enabled,
            mode=config.diarization.mode,
            huggingface_token=config.diarization.huggingface_token,
        )
        self._target_ms = max(1_000, config.chunking.gigaam_segment_target_s * 1_000)
        self._max_ms = max(
            self._target_ms, config.chunking.gigaam_segment_max_s * 1_000
        )
        self._final_silence_ms = 700
        self._overlap_ms = 500
        self._segment_counter = 0
        self._active: _SpeechBuffer | None = None
        self._logged_pytorch_fallback = False
        self._logged_onnx_fallback = False

    async def start(self) -> None:
        if self._gigaam_recognizer.is_ready:
            logger.info(
                "[gigaam] started with primary backend=%s",
                self._gigaam_recognizer.backend_name,
            )
        else:
            logger.warning(
                "[gigaam] primary recognizer unavailable, fallback to ONNX/placeholder. reason=%s",
                self._gigaam_recognizer.error,
            )

        if self._fallback_recognizer.is_ready:
            logger.info(
                "[gigaam] ONNX fallback backend=%s",
                self._fallback_recognizer.backend_name,
            )
        else:
            logger.warning(
                "[gigaam] ONNX fallback unavailable. reason=%s",
                self._fallback_recognizer.error,
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
    ) -> list[FinalSegmentMessage]:
        events: list[FinalSegmentMessage] = []
        speech = self._vad.is_speech(chunk_pcm) if self._config.vad.enabled else True

        if speech:
            if self._active is None:
                self._active = _SpeechBuffer(
                    start_ms=chunk_start_ms,
                    last_speech_ms=chunk_end_ms,
                    pcm=bytearray(),
                )
            self._active.pcm.extend(chunk_pcm)
            self._active.last_speech_ms = chunk_end_ms

            duration_ms = chunk_end_ms - self._active.start_ms
            if duration_ms >= self._target_ms or duration_ms >= self._max_ms:
                events.append(
                    self._emit_segment(
                        session_id=session_id,
                        segment_end_ms=chunk_end_ms,
                        keep_overlap=True,
                    )
                )
            return events

        # Silence branch: finalize after a pause.
        if self._active is not None:
            silence_ms = chunk_end_ms - self._active.last_speech_ms
            if silence_ms >= self._final_silence_ms:
                events.append(
                    self._emit_segment(
                        session_id=session_id,
                        segment_end_ms=self._active.last_speech_ms,
                        keep_overlap=False,
                    )
                )

        return events

    def flush(self, session_id: str) -> list[FinalSegmentMessage]:
        if self._active is None:
            return []
        return [
            self._emit_segment(
                session_id=session_id,
                segment_end_ms=self._active.last_speech_ms,
                keep_overlap=False,
            )
        ]

    def _emit_segment(
        self,
        *,
        session_id: str,
        segment_end_ms: int,
        keep_overlap: bool,
    ) -> FinalSegmentMessage:
        assert self._active is not None
        active = self._active
        self._segment_counter += 1

        pcm_bytes = bytes(active.pcm)

        # Assign speaker label using diarization pipeline
        speaker = self._diarizer.assign_speaker(pcm_bytes, self._config.sample_rate)

        event = FinalSegmentMessage(
            session_id=session_id,
            engine=self.name,
            segment_id=f"gigaam-seg-{self._segment_counter}",
            time_range_ms=[active.start_ms, segment_end_ms],
            speaker=speaker,
            text=self._decode_final(pcm_bytes),
            tokens=None,
            confidence=0.85,
        )
        logger.info(
            "[gigaam][final] segment=%s range=%s..%s speaker=%s chars=%s",
            event.segment_id,
            active.start_ms,
            segment_end_ms,
            speaker,
            len(event.text),
        )

        if keep_overlap:
            overlap_pcm, overlap_duration_ms = self._tail_overlap_pcm(pcm_bytes)
            if overlap_pcm:
                self._active = _SpeechBuffer(
                    start_ms=max(0, segment_end_ms - overlap_duration_ms),
                    last_speech_ms=segment_end_ms,
                    pcm=bytearray(overlap_pcm),
                )
            else:
                self._active = None
        else:
            self._active = None

        return event

    def _tail_overlap_pcm(self, pcm: bytes) -> tuple[bytes, int]:
        if not pcm:
            return b"", 0

        bytes_per_ms = max(1, int(self._config.sample_rate * 2 / 1_000))
        target_bytes = self._overlap_ms * bytes_per_ms
        if target_bytes >= len(pcm):
            duration_ms = int(len(pcm) / bytes_per_ms)
            return pcm, duration_ms

        tail = pcm[-target_bytes:]
        duration_ms = int(len(tail) / bytes_per_ms)
        return tail, duration_ms

    def _decode_final(self, pcm: bytes) -> str:
        text = self._gigaam_recognizer.transcribe_pcm_s16le(
            pcm, self._config.sample_rate
        )
        if text is not None:
            logger.info(
                "[gigaam][decode] backend=%s",
                self._gigaam_recognizer.backend_name,
            )
            return self._apply_basic_punctuation(text or "")
        if not self._logged_pytorch_fallback:
            logger.warning(
                "[gigaam][decode] primary backend failed, falling back to ONNX. reason=%s",
                self._gigaam_recognizer.error,
            )
            self._logged_pytorch_fallback = True

        text = self._fallback_recognizer.transcribe_pcm_s16le(
            pcm, self._config.sample_rate
        )
        if text is not None:
            logger.info(
                "[gigaam][decode] backend=%s",
                self._fallback_recognizer.backend_name,
            )
            return self._apply_basic_punctuation(text or "")

        if not self._logged_onnx_fallback:
            logger.warning(
                "[gigaam][decode] ONNX fallback failed, using placeholder. reason=%s",
                self._fallback_recognizer.error,
            )
            self._logged_onnx_fallback = True
        duration_s = self._duration_seconds(pcm)
        return f"[gigaam-final] speech segment {duration_s:.1f}s"

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
