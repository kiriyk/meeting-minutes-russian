from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GigaAMPytorchRecognizer:
    """PyTorch recognizer for ai-sage/GigaAM-v3 (trust_remote_code)."""

    def __init__(self) -> None:
        self._ready = False
        self._error: str | None = None
        self._model: Any | None = None
        self._sample_rate = 16000
        self._model_dir: Path | None = None
        self._device = "cpu"

        self._try_init()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def backend_name(self) -> str:
        return f"pytorch:{self._device}"

    def transcribe_pcm_s16le(self, pcm: bytes, sample_rate: int) -> str | None:
        if not self._ready or self._model is None:
            return None
        if not pcm:
            return ""
        # Some callers mutate PATH after recognizer init; keep ffmpeg resolvable.
        self._ensure_ffmpeg_available()

        try:
            import numpy as np
        except Exception as exc:
            self._error = f"numpy import failed: {exc}"
            return None

        audio_i16 = np.frombuffer(pcm, dtype=np.int16)
        if audio_i16.size == 0:
            return ""

        # Ensure 16k input for GigaAM-v3.
        if sample_rate > 0 and sample_rate != self._sample_rate:
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
            audio_f32 = self._resample(audio_f32, sample_rate, self._sample_rate)
            audio_i16 = np.clip(np.round(audio_f32 * 32767.0), -32768, 32767).astype(
                np.int16
            )

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            with wave.open(tmp_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self._sample_rate)
                w.writeframes(audio_i16.tobytes())

            out = self._transcribe_path(tmp_path)
            return self._normalize_transcribe_output(out)
        except Exception as exc:
            self._error = f"gigaam transcribe failed: {exc}"
            logger.warning("%s", self._error)
            return None
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _transcribe_path(self, wav_path: str) -> Any:
        try:
            return self._model.transcribe(wav_path)
        except Exception as exc:
            msg = str(exc)
            if (
                "Too long wav file" in msg
                and hasattr(self._model, "transcribe_longform")
            ):
                logger.info("GigaAM short-form rejected long chunk; using transcribe_longform")
                return self._model.transcribe_longform(wav_path)
            raise

    def _try_init(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModel  # type: ignore
        except Exception as exc:
            self._error = f"gigaam dependencies missing: {exc}"
            logger.warning("%s", self._error)
            return

        model_dir = self._resolve_model_dir()
        if model_dir is None:
            self._error = "GigaAM model directory not found"
            logger.warning("%s", self._error)
            return

        self._ensure_ffmpeg_available()

        try:
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch, "backends", None)
                and getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
            self._device = device
            self._model = AutoModel.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
            )
            if hasattr(self._model, "to"):
                self._model.to(device)
            if hasattr(self._model, "eval"):
                self._model.eval()

            self._model_dir = model_dir
            self._ready = True
            self._error = None
            logger.info(
                "Initialized GigaAM PyTorch model: path=%s backend=%s",
                model_dir,
                self.backend_name,
            )
        except Exception as exc:
            self._error = f"failed to load GigaAM model: {exc}"
            self._ready = False
            logger.warning("%s", self._error)

    def _ensure_ffmpeg_available(self) -> None:
        # GigaAM remote-code loader shells out to "ffmpeg" by executable name.
        # Make sure this name resolves even when only bundled Meetily binary exists.
        if shutil.which("ffmpeg"):
            return

        ffmpeg_bin = self._resolve_ffmpeg_binary()
        if ffmpeg_bin is None:
            logger.warning(
                "ffmpeg was not found for GigaAM live transcription (PATH and bundled binaries)"
            )
            return

        ffmpeg_path = Path(ffmpeg_bin)
        os.environ["MEETILY_FFMPEG_PATH"] = str(ffmpeg_path)

        if ffmpeg_path.name == "ffmpeg":
            os.environ["PATH"] = (
                f"{ffmpeg_path.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            )
            logger.info("Configured ffmpeg for GigaAM from %s", ffmpeg_path)
            return

        shim_dir = Path(__file__).resolve().parents[2] / "models" / "tmp" / "ffmpeg-shim"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_path = shim_dir / ("ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg")

        try:
            if shim_path.exists() or shim_path.is_symlink():
                shim_path.unlink(missing_ok=True)
            try:
                shim_path.symlink_to(ffmpeg_path)
            except Exception:
                shutil.copy2(ffmpeg_path, shim_path)
            shim_path.chmod(0o755)
            os.environ["PATH"] = f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            logger.info("Configured ffmpeg shim for GigaAM: %s -> %s", shim_path, ffmpeg_path)
        except Exception as exc:
            logger.warning(
                "failed to prepare ffmpeg shim for GigaAM (%s): %s",
                ffmpeg_path,
                exc,
            )

    def _resolve_model_dir(self) -> Path | None:
        # 1) Explicit path wins.
        explicit = os.getenv("MEETILY_GIGAAM_MODEL_DIR")
        if explicit:
            p = Path(explicit)
            if p.exists() and p.is_dir():
                return p

        # 2) Auto-discovery under asr-service/models/gigaam/*.
        root = Path(__file__).resolve().parents[2]
        gigaam_root = root / "models" / "gigaam"
        if not gigaam_root.exists():
            return None

        for child in sorted(gigaam_root.iterdir()):
            if not child.is_dir():
                continue
            required = [
                child / "config.json",
                child / "modeling_gigaam.py",
                child / "pytorch_model.bin",
                child / "tokenizer.model",
            ]
            if all(p.exists() for p in required):
                return child
        return None

    @staticmethod
    def _normalize_transcribe_output(out: Any) -> str:
        if out is None:
            return ""
        if isinstance(out, str):
            return out.strip()
        if isinstance(out, dict):
            for key in ("text", "transcription", "result"):
                if key in out and isinstance(out[key], str):
                    return out[key].strip()
        if isinstance(out, list):
            parts = []
            for item in out:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            return " ".join(p.strip() for p in parts if p).strip()
        return str(out).strip()

    @staticmethod
    def _resample(audio, src_rate: int, dst_rate: int):
        import numpy as np

        if src_rate == dst_rate or audio.size == 0:
            return audio

        src_len = audio.shape[0]
        dst_len = int(src_len * (dst_rate / float(src_rate)))
        if dst_len <= 1:
            return audio

        src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        return np.interp(dst_x, src_x, audio).astype("float32")

    @staticmethod
    def _resolve_ffmpeg_binary() -> str | None:
        # 1) Explicit override.
        env_path = os.getenv("MEETILY_FFMPEG_PATH")
        if env_path and Path(env_path).exists():
            return env_path

        # 2) PATH.
        from_path = shutil.which("ffmpeg")
        if from_path:
            return from_path

        # 3) Meetily bundled binaries.
        # engines/* lives in asr-service/asr_service/engines, while binaries live in
        # <repo>/frontend/src-tauri/binaries.
        service_root = Path(__file__).resolve().parents[2]
        candidate_dirs = [
            service_root / "frontend" / "src-tauri" / "binaries",
            service_root.parent / "frontend" / "src-tauri" / "binaries",
        ]

        binaries_dir = next((d for d in candidate_dirs if d.exists()), None)
        if binaries_dir is not None:
            target_names: list[str] = []
            if sys.platform == "darwin":
                target_names.extend(
                    ["ffmpeg-aarch64-apple-darwin", "ffmpeg-x86_64-apple-darwin"]
                )
            elif sys.platform.startswith("linux"):
                target_names.extend(
                    [
                        "ffmpeg-aarch64-unknown-linux-gnu",
                        "ffmpeg-x86_64-unknown-linux-gnu",
                    ]
                )
            elif sys.platform.startswith("win"):
                target_names.extend(
                    [
                        "ffmpeg-x86_64-pc-windows-msvc.exe",
                        "ffmpeg-aarch64-pc-windows-msvc.exe",
                    ]
                )

            for name in target_names:
                candidate = binaries_dir / name
                if candidate.exists() and candidate.is_file():
                    try:
                        candidate.chmod(0o755)
                    except Exception:
                        pass
                    return str(candidate)

            for candidate in sorted(binaries_dir.glob("ffmpeg-*")):
                if candidate.is_file():
                    try:
                        candidate.chmod(0o755)
                    except Exception:
                        pass
                    return str(candidate)

        return None
