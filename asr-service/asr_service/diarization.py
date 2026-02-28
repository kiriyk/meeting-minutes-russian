from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerTurn:
    start_ms: int
    end_ms: int
    speaker: str


class Diarizer(Protocol):
    def assign_speaker(self, pcm_s16le: bytes, sample_rate: int) -> str | None:
        """Return speaker label for an audio segment or None if unavailable."""

    def runtime_status(self) -> dict[str, str]:
        """Return runtime backend details for UI diagnostics."""
        ...


class NoopDiarizer:
    def assign_speaker(self, pcm_s16le: bytes, sample_rate: int) -> str | None:
        return None

    def runtime_status(self) -> dict[str, str]:
        return {
            "mode": "disabled",
            "backend": "none",
            "acceleration": "none",
        }


class EnergyDiarizer:
    """Very lightweight fallback diarization.

    Heuristic:
    - Compute short-term average energy of the segment.
    - Map to one of two pseudo-speakers to preserve stable labels in UI.

    This is intentionally simple and dependency-free. It is used when pyannote is
    unavailable or disabled.
    """

    def __init__(self, split_threshold: float = 0.035) -> None:
        self._split_threshold = split_threshold

    def assign_speaker(self, pcm_s16le: bytes, sample_rate: int) -> str | None:
        if not pcm_s16le:
            return None
        data = np.frombuffer(pcm_s16le, dtype=np.int16)
        if data.size == 0:
            return None
        amp = np.mean(np.abs(data.astype(np.float32))) / 32768.0
        return "SPEAKER_01" if amp >= self._split_threshold else "SPEAKER_00"

    def runtime_status(self) -> dict[str, str]:
        return {
            "mode": "energy",
            "backend": "numpy-energy",
            "acceleration": "cpu",
        }


class PyAnnoteDiarizer:
    """Optional pyannote-backed diarization wrapper.

    The model is loaded lazily in __init__. If loading fails (missing dependency,
    token, model access, etc.), call sites should fallback to EnergyDiarizer.
    """

    def __init__(
        self,
        huggingface_token: str | None = None,
        local_model_dir: Path | None = None,
    ) -> None:
        warnings.filterwarnings(
            "ignore",
            message=".*torchcodec is not installed correctly.*",
            category=UserWarning,
            module=r"pyannote\.audio\.core\.io",
        )
        from pyannote.audio import Pipeline  # type: ignore

        self._source = "remote"
        if local_model_dir is not None and (local_model_dir / "config.yaml").exists():
            self._pipeline = Pipeline.from_pretrained(
                str(local_model_dir),
                token=huggingface_token,
            )
            self._source = f"local:{local_model_dir}"
        else:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=huggingface_token,
            )

    def assign_speaker(self, pcm_s16le: bytes, sample_rate: int) -> str | None:
        if not pcm_s16le:
            return None

        pipeline = self._pipeline
        if pipeline is None:
            return None

        wav = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        if wav.size == 0:
            return None

        try:
            import torch

            waveform = torch.from_numpy(wav).unsqueeze(0)
            with torch.inference_mode():
                diarization = pipeline(
                    {"waveform": waveform, "sample_rate": sample_rate}
                )
        except Exception as exc:  # pragma: no cover - depends on runtime env
            logger.warning(
                "[diarization] pyannote inference failed, speaker=None reason=%s",
                exc,
            )
            return None

        # Pick label of the longest annotated turn for this segment.
        best_label: str | None = None
        best_dur = 0.0
        annotation = self._extract_annotation(diarization)
        if annotation is None:
            logger.warning(
                "[diarization] pyannote output unsupported type=%s",
                type(diarization).__name__,
            )
            return None

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            dur = float(turn.end - turn.start)
            if dur > best_dur:
                best_dur = dur
                best_label = str(speaker)

        return best_label

    @staticmethod
    def _extract_annotation(diarization: object) -> object | None:
        # pyannote <=3.x returned an Annotation directly with itertracks().
        if hasattr(diarization, "itertracks"):
            return diarization

        # pyannote >=4 may return DiarizeOutput with a speaker_diarization field.
        for attr in ("speaker_diarization", "annotation", "diarization"):
            value = getattr(diarization, attr, None)
            if value is not None and hasattr(value, "itertracks"):
                return value

        # Fallback for dict-like return shapes.
        if isinstance(diarization, dict):
            for key in ("speaker_diarization", "annotation", "diarization"):
                value = diarization.get(key)
                if value is not None and hasattr(value, "itertracks"):
                    return value

        return None

    def runtime_status(self) -> dict[str, str]:
        device = str(getattr(self._pipeline, "device", "cpu"))
        acceleration = "cpu"
        lowered = device.lower()
        if "cuda" in lowered:
            acceleration = "gpu"
        elif "mps" in lowered:
            acceleration = "gpu"
        return {
            "mode": "pyannote",
            "backend": f"pyannote:{self._source}:{device}",
            "acceleration": acceleration,
        }


def build_diarizer(
    *,
    enabled: bool,
    mode: str,
    huggingface_token: str | None,
    models_dir: Path | None = None,
) -> Diarizer:
    if not enabled:
        return NoopDiarizer()

    normalized = mode.strip().lower()

    if normalized == "pyannote":
        try:
            local_model_dir: Path | None = None
            if models_dir is not None:
                local_model_dir = (
                    models_dir / "diarization" / "speaker-diarization-3.1"
                )
            diarizer = PyAnnoteDiarizer(
                huggingface_token=huggingface_token,
                local_model_dir=local_model_dir,
            )
            logger.info("[diarization] initialized mode=pyannote")
            return diarizer
        except Exception as exc:  # pragma: no cover - depends on runtime env
            logger.warning(
                "[diarization] pyannote unavailable, fallback=energy reason=%s",
                exc,
            )
            return EnergyDiarizer()

    if normalized == "energy":
        logger.info("[diarization] initialized mode=energy")
        return EnergyDiarizer()

    logger.warning("[diarization] unknown mode=%s, fallback=energy", mode)
    return EnergyDiarizer()
