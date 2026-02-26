from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OnnxCtcRecognizer:
    """ONNX CTC recognizer for T-One style models.

    The recognizer is optional:
    - if runtime/model is unavailable, `transcribe_pcm_s16le` returns None
    - callers can fallback to placeholder text.
    """

    def __init__(self) -> None:
        self._ready = False
        self._error: str | None = None
        self._session: Any | None = None
        self._ort: Any | None = None

        self._signal_input_name: str | None = None
        self._state_input_name: str | None = None
        self._logprobs_output_name: str | None = None
        self._state_next_output_name: str | None = None

        self._id_to_token: dict[int, str] = {}
        self._blank_id = 34
        self._sample_rate = 8000
        self._signal_len = 2400
        self._state_size = 0
        self._model_path: Path | None = None
        self._prefer_coreml = False
        self._coreml_failed = False
        self._active_provider: str | None = None

        self._try_init()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def backend_name(self) -> str:
        provider = self._active_provider or "unknown"
        return f"onnx:{provider}"

    def transcribe_pcm_s16le(self, pcm: bytes, input_sample_rate: int) -> str | None:
        if (
            not self._ready
            or self._session is None
            or self._signal_input_name is None
            or self._state_input_name is None
            or self._logprobs_output_name is None
            or self._state_next_output_name is None
            or self._state_size <= 0
        ):
            return None

        try:
            import numpy as np
        except Exception as exc:
            self._error = f"numpy import failed: {exc}"
            self._ready = False
            return None

        audio_i16 = np.frombuffer(pcm, dtype=np.int16)
        if audio_i16.size == 0:
            return ""

        # Resample to model sample rate and keep int32 amplitude input.
        if input_sample_rate > 0 and input_sample_rate != self._sample_rate:
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
            audio_f32 = self._resample(audio_f32, input_sample_rate, self._sample_rate)
            audio_i32 = np.clip(np.round(audio_f32 * 32767.0), -32768, 32767).astype(
                np.int32
            )
        else:
            audio_i32 = audio_i16.astype(np.int32)

        windows = self._chunk_fixed(audio_i32, self._signal_len)
        if not windows:
            return ""

        state = np.zeros((1, self._state_size), dtype=np.float16)
        logits_parts = []

        for win in windows:
            signal = win.reshape(1, self._signal_len, 1).astype(np.int32)
            feed = {
                self._signal_input_name: signal,
                self._state_input_name: state,
            }
            try:
                logprobs, state_next = self._session.run(
                    [self._logprobs_output_name, self._state_next_output_name],
                    feed,
                )
            except Exception as exc:
                if self._prefer_coreml and not self._coreml_failed:
                    self._coreml_failed = True
                    self._error = f"coreml inference failed, switching to CPU: {exc}"
                    logger.warning("%s", self._error)
                    if self._rebuild_cpu_session():
                        logger.info(
                            "T-One ONNX switched backend to %s",
                            self.backend_name,
                        )
                        return self.transcribe_pcm_s16le(pcm, input_sample_rate)
                self._error = f"onnx inference failed: {exc}"
                logger.warning("%s", self._error)
                return None

            logits_parts.append(logprobs)
            state = state_next

        logits = np.concatenate(logits_parts, axis=1)

        token_ids = logits.argmax(axis=-1).tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        return self._ctc_greedy_decode(token_ids)

    def _try_init(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            self._error = f"onnxruntime import failed: {exc}"
            logger.warning("%s", self._error)
            return
        self._ort = ort

        model_path = self._resolve_model_path()
        if model_path is None:
            self._error = "T-One model.onnx not found"
            logger.warning("%s", self._error)
            return

        model_dir = model_path.parent
        vocab_path = model_dir / "vocab.json"
        config_path = model_dir / "config.json"
        if not vocab_path.exists():
            self._error = f"vocab.json not found near model: {vocab_path}"
            logger.warning("%s", self._error)
            return

        try:
            vocab_obj = json.loads(vocab_path.read_text(encoding="utf-8"))
            self._id_to_token = {int(v): str(k) for k, v in vocab_obj.items()}
        except Exception as exc:
            self._error = f"failed to parse vocab.json: {exc}"
            logger.warning("%s", self._error)
            return

        if config_path.exists():
            try:
                config_obj = json.loads(config_path.read_text(encoding="utf-8"))
                self._blank_id = int(config_obj.get("pad_token_id", self._blank_id))
                fe = config_obj.get("feature_extraction_params", {})
                self._sample_rate = int(fe.get("sample_rate", self._sample_rate))
            except Exception:
                pass

        try:
            session_opts = ort.SessionOptions()
            session_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._model_path = model_path
            providers = ort.get_available_providers()
            use_coreml = os.getenv("MEETILY_ASR_USE_COREML", "").lower() in (
                "1",
                "true",
                "yes",
            )
            self._prefer_coreml = use_coreml
            logger.info(
                "T-One ONNX init: model=%s use_coreml=%s available_providers=%s tmpdir=%s",
                model_path,
                use_coreml,
                providers,
                os.getenv("TMPDIR"),
            )
            if use_coreml:
                preferred_order = ("CoreMLExecutionProvider", "CPUExecutionProvider")
            else:
                preferred_order = ("CPUExecutionProvider", "CoreMLExecutionProvider")
            if use_coreml and "CoreMLExecutionProvider" not in providers:
                logger.warning(
                    "MEETILY_ASR_USE_COREML is enabled but CoreMLExecutionProvider is not available"
                )
            preferred = [p for p in preferred_order if p in providers]
            logger.info("T-One ONNX provider preference order=%s", preferred)
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=session_opts,
                providers=preferred or ["CPUExecutionProvider"],
            )
            session_providers = self._session.get_providers()
            self._active_provider = (
                session_providers[0] if len(session_providers) > 0 else None
            )
            if not self._bind_io_names():
                return

            self._ready = True
            self._error = None
            logger.info(
                "Initialized T-One ONNX model: path=%s backend=%s sample_rate=%s signal_len=%s",
                model_path,
                self.backend_name,
                self._sample_rate,
                self._signal_len,
            )
        except Exception as exc:
            if self._prefer_coreml and self._rebuild_cpu_session():
                self._error = f"coreml init failed, switched to CPU: {exc}"
                self._ready = True
                logger.warning("%s", self._error)
                return
            self._error = f"failed to initialize ONNX session: {exc}"
            self._ready = False
            logger.warning("%s", self._error)

    def _rebuild_cpu_session(self) -> bool:
        if self._ort is None or self._model_path is None:
            return False
        try:
            session_opts = self._ort.SessionOptions()
            session_opts.graph_optimization_level = (
                self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = self._ort.InferenceSession(
                str(self._model_path),
                sess_options=session_opts,
                providers=["CPUExecutionProvider"],
            )
            session_providers = self._session.get_providers()
            self._active_provider = (
                session_providers[0] if len(session_providers) > 0 else None
            )
            logger.info(
                "T-One ONNX rebuilt CPU session: model=%s backend=%s",
                self._model_path,
                self.backend_name,
            )
            return self._bind_io_names()
        except Exception as exc:
            self._error = f"failed to rebuild CPU session: {exc}"
            logger.warning("%s", self._error)
            return False

    def _bind_io_names(self) -> bool:
        if self._session is None:
            self._error = "onnx session is not initialized"
            return False

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()

        signal_in = None
        state_in = None
        for inp in inputs:
            name = inp.name.lower()
            if "signal" in name:
                signal_in = inp
            elif "state" in name:
                state_in = inp

        logprobs_out = None
        state_next_out = None
        for out in outputs:
            name = out.name.lower()
            if "logprob" in name:
                logprobs_out = out
            elif "state_next" in name:
                state_next_out = out

        if (
            signal_in is None
            or state_in is None
            or logprobs_out is None
            or state_next_out is None
        ):
            self._error = (
                "Unsupported ONNX IO signature: expected signal/state inputs and "
                "logprobs/state_next outputs"
            )
            self._ready = False
            return False

        try:
            sig_shape = signal_in.shape
            self._signal_len = int(sig_shape[1])
            state_shape = state_in.shape
            self._state_size = int(state_shape[1])
        except Exception as exc:
            self._error = f"Failed to parse ONNX shapes: {exc}"
            self._ready = False
            return False

        self._signal_input_name = signal_in.name
        self._state_input_name = state_in.name
        self._logprobs_output_name = logprobs_out.name
        self._state_next_output_name = state_next_out.name
        logger.info(
            "T-One ONNX bound IO: signal=%s state=%s logprobs=%s state_next=%s signal_len=%s state_size=%s",
            self._signal_input_name,
            self._state_input_name,
            self._logprobs_output_name,
            self._state_next_output_name,
            self._signal_len,
            self._state_size,
        )
        return True

    def _resolve_model_path(self) -> Path | None:
        explicit_file = os.getenv("MEETILY_T_ONE_MODEL_FILE")
        if explicit_file:
            p = Path(explicit_file)
            if p.exists() and p.is_file():
                return p

        explicit_dir = os.getenv("MEETILY_T_ONE_MODEL_DIR")
        if explicit_dir:
            p = Path(explicit_dir) / "model.onnx"
            if p.exists():
                return p

        # Default: asr-service/models/t_one/<model_id>/model.onnx
        root = Path(__file__).resolve().parents[2]
        t_one_root = root / "models" / "t_one"
        if not t_one_root.exists():
            return None

        for child in sorted(t_one_root.iterdir()):
            if child.is_dir():
                candidate = child / "model.onnx"
                if candidate.exists():
                    return candidate
        return None

    def _ctc_greedy_decode(self, token_ids: list[int]) -> str:
        decoded_tokens: list[str] = []
        prev = None
        for tid in token_ids:
            if tid == prev:
                continue
            prev = tid
            if tid == self._blank_id:
                continue

            token = self._id_to_token.get(int(tid), "")
            if not token:
                continue

            if token == "|":
                token = " "
            if token in ("[PAD]", "<s>", "</s>", "<unk>"):
                continue
            decoded_tokens.append(token)

        text = "".join(decoded_tokens)
        return " ".join(text.split())

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
    def _chunk_fixed(audio_i32, frame_len: int):
        import numpy as np

        if audio_i32.size == 0:
            return []

        chunks = []
        pos = 0
        n = audio_i32.size
        while pos < n:
            end = min(pos + frame_len, n)
            chunk = audio_i32[pos:end]
            if chunk.size < frame_len:
                pad = np.zeros((frame_len - chunk.size,), dtype=np.int32)
                chunk = np.concatenate([chunk, pad], axis=0)
            chunks.append(chunk)
            pos += frame_len
        return chunks
