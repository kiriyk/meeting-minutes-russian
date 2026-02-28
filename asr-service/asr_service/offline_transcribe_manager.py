from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .diarization import Diarizer, build_diarizer
from .engines.gigaam_pytorch_recognizer import GigaAMPytorchRecognizer
from .engines.onnx_ctc_recognizer import OnnxCtcRecognizer
from .vad import VAD

JobStatus = Literal["queued", "running", "completed", "failed"]
logger = logging.getLogger(__name__)


@dataclass
class OfflineSegment:
    engine: str
    segment_id: str
    time_range_ms: list[int]
    speaker: str | None
    text: str
    confidence: float | None = None


@dataclass
class OfflineJob:
    job_id: str
    status: JobStatus
    engine: str
    file_path: str
    segment_seconds: int
    progress: float = 0.0
    queue_position: int | None = None
    result: dict | None = None
    error: str | None = None


class OfflineTranscribeManager:
    def __init__(self, max_concurrent_jobs: int | None = None) -> None:
        self._jobs: dict[str, OfflineJob] = {}
        self._lock = asyncio.Lock()
        self._recognizer_init_lock = asyncio.Lock()
        self._t_one: OnnxCtcRecognizer | None = None
        self._gigaam_pt: GigaAMPytorchRecognizer | None = None
        self._gigaam_onnx: OnnxCtcRecognizer | None = None
        self._offline_diarizer: Diarizer | None = None

        env_limit = os.getenv("MEETILY_OFFLINE_MAX_CONCURRENT", "1")
        try:
            resolved_limit = int(max_concurrent_jobs or env_limit)
        except (TypeError, ValueError):
            resolved_limit = 1
        self._max_concurrent_jobs = max(1, resolved_limit)
        self._job_semaphore = asyncio.Semaphore(self._max_concurrent_jobs)

    async def start_job(
        self,
        *,
        file_path: str,
        engine: str,
        segment_seconds: int,
    ) -> OfflineJob:
        source_path = Path(file_path)
        if not source_path.exists():
            raise ValueError(f"File not found: {source_path}")
        if engine not in ("gigaam", "t_one"):
            raise ValueError(f"Unsupported engine: {engine}")

        job = OfflineJob(
            job_id=str(uuid.uuid4()),
            status="queued",
            engine=engine,
            file_path=str(source_path),
            segment_seconds=segment_seconds,
        )

        async with self._lock:
            self._jobs[job.job_id] = job
            self._recompute_queue_positions_locked()

        asyncio.create_task(self._worker(job.job_id))
        return job

    async def get_job(self, job_id: str) -> OfflineJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def _worker(self, job_id: str) -> None:
        async with self._job_semaphore:
            async with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job.status = "running"
                job.queue_position = None
                self._recompute_queue_positions_locked()

            cleanup_tmp = False
            wav_path: Path | None = None
            try:
                source_path = Path(job.file_path)
                wav_path, cleanup_tmp = await asyncio.to_thread(
                    self._ensure_mono_16k_wav, source_path
                )
                await self._ensure_recognizer_ready(job.engine)
                logger.info(
                    "offline job=%s engine=%s source=%s wav=%s",
                    job.job_id,
                    job.engine,
                    source_path,
                    wav_path,
                )
                with wave.open(str(wav_path), "rb") as wav_reader:
                    channels = wav_reader.getnchannels()
                    sample_rate = wav_reader.getframerate()
                    sample_width = wav_reader.getsampwidth()
                    frame_count = wav_reader.getnframes()

                if channels != 1:
                    raise ValueError("MVP offline transcribe expects mono WAV")
                if sample_width != 2:
                    raise ValueError("MVP offline transcribe expects 16-bit WAV")

                total_duration_s = frame_count / float(sample_rate)
                if total_duration_s <= 0:
                    await self._set_done(job_id, [])
                    return

                requested_seg_s = max(1, int(job.segment_seconds))
                seg_s = requested_seg_s
                if job.engine == "gigaam":
                    # GigaAM short-form transcribe degrades/fails on long chunks.
                    seg_s = min(seg_s, 25)
                segments: list[OfflineSegment] = []

                with wave.open(str(wav_path), "rb") as wav_reader:
                    pcm_all = wav_reader.readframes(frame_count)

                ranges = self._build_phrase_ranges(
                    pcm_all=pcm_all,
                    sample_rate=sample_rate,
                    max_segment_seconds=seg_s,
                    engine=job.engine,
                )
                if not ranges:
                    ranges = [(0, frame_count)]
                avg_dur_s = sum((b - a) for a, b in ranges) / float(
                    sample_rate * len(ranges)
                )
                logger.info(
                    "offline segmentation engine=%s segments=%s avg_duration_s=%.2f max_segment_s=%s requested_segment_s=%s",
                    job.engine,
                    len(ranges),
                    avg_dur_s,
                    seg_s,
                    requested_seg_s,
                )
                num_segments = len(ranges)

                for idx, (start_frame, end_frame) in enumerate(ranges):
                    start_ms = int((start_frame / float(sample_rate)) * 1000)
                    end_ms = int((end_frame / float(sample_rate)) * 1000)
                    dur_s = max(0.0, (end_frame - start_frame) / float(sample_rate))
                    start_byte = start_frame * 2
                    end_byte = end_frame * 2
                    pcm = pcm_all[start_byte:end_byte]

                    text = await asyncio.to_thread(
                        self._decode_segment,
                        engine=job.engine,
                        pcm=pcm,
                        sample_rate=sample_rate,
                    )
                    if not text or not text.strip():
                        logger.info(
                            "offline decode produced empty text engine=%s segment=%s duration_s=%.2f; skipped",
                            job.engine,
                            idx + 1,
                            dur_s,
                        )
                        progress = ((idx + 1) / num_segments) * 100.0
                        await self._set_progress(job_id, progress)
                        continue
                    # pyannote diarization is expensive; skip very short chunks.
                    speaker = None
                    if dur_s >= 2.0:
                        speaker = await asyncio.to_thread(
                            self._assign_speaker,
                            pcm=pcm,
                            sample_rate=sample_rate,
                        )

                    segments.append(
                        OfflineSegment(
                            engine=job.engine,
                            segment_id=f"offline-seg-{idx + 1}",
                            time_range_ms=[start_ms, end_ms],
                            speaker=speaker,
                            text=text,
                            confidence=0.9 if job.engine == "gigaam" else 0.8,
                        )
                    )

                    progress = ((idx + 1) / num_segments) * 100.0
                    await self._set_progress(job_id, progress)

                await self._set_done(job_id, segments)
            except Exception as exc:
                await self._set_failed(job_id, str(exc))
            finally:
                if cleanup_tmp and wav_path is not None:
                    try:
                        wav_path.unlink(missing_ok=True)
                    except Exception:
                        pass

    def _decode_segment(self, *, engine: str, pcm: bytes, sample_rate: int) -> str:
        if engine == "t_one":
            t_one = self._get_t_one_recognizer()
            text = t_one.transcribe_pcm_s16le(pcm, sample_rate)
            logger.info(
                "offline decode engine=t_one backend=%s ok=%s",
                t_one.backend_name,
                text is not None,
            )
            return (text or "").strip() if text is not None else ""

        if engine == "gigaam":
            gigaam_pt = self._get_gigaam_pytorch_recognizer()
            text = gigaam_pt.transcribe_pcm_s16le(pcm, sample_rate)
            if text is not None:
                logger.info(
                    "offline decode engine=gigaam backend=%s ok=true",
                    gigaam_pt.backend_name,
                )
                return text.strip()
            gigaam_onnx = self._get_gigaam_onnx_recognizer()
            text = gigaam_onnx.transcribe_pcm_s16le(pcm, sample_rate)
            logger.info(
                "offline decode engine=gigaam backend=%s ok=%s",
                gigaam_onnx.backend_name,
                text is not None,
            )
            return (text or "").strip() if text is not None else ""

        return ""

    def _assign_speaker(self, *, pcm: bytes, sample_rate: int) -> str | None:
        diarizer = self._get_offline_diarizer()
        if diarizer is None:
            return None
        try:
            return diarizer.assign_speaker(pcm, sample_rate)
        except Exception as exc:
            logger.warning("offline diarization failed, speaker=None reason=%s", exc)
            return None

    @staticmethod
    def _ensure_mono_16k_wav(source: Path) -> tuple[Path, bool]:
        if source.suffix.lower() == ".wav":
            try:
                with wave.open(str(source), "rb") as w:
                    if (
                        w.getnchannels() == 1
                        and w.getframerate() == 16000
                        and w.getsampwidth() == 2
                    ):
                        return source, False
            except Exception:
                pass

        ffmpeg_bin = OfflineTranscribeManager._resolve_ffmpeg_binary()
        if ffmpeg_bin is None:
            raise ValueError(
                "ffmpeg is required for offline transcription conversion but was not found (PATH and bundled binaries)"
            )
        logger.info("offline conversion using ffmpeg=%s", ffmpeg_bin)

        with tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="meetily-offline-", delete=False
        ) as tmp:
            out_path = Path(tmp.name)

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            out_path.unlink(missing_ok=True)
            stderr_tail = (proc.stderr or "").strip()[-400:]
            raise ValueError(f"ffmpeg conversion failed: {stderr_tail}")
        return out_path, True

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
        repo_root = Path(__file__).resolve().parents[2]
        binaries_dir = repo_root / "frontend" / "src-tauri" / "binaries"
        if binaries_dir.exists():
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

    @staticmethod
    def _build_phrase_ranges(
        *,
        pcm_all: bytes,
        sample_rate: int,
        max_segment_seconds: int,
        engine: str,
    ) -> list[tuple[int, int]]:
        if not pcm_all or sample_rate <= 0:
            return []

        bytes_per_sample = 2
        total_samples = len(pcm_all) // bytes_per_sample
        if total_samples <= 0:
            return []

        frame_ms = 20
        frame_samples = max(1, int(sample_rate * frame_ms / 1000))

        # Engine-specific phrase shaping:
        # - gigaam: favor fewer, longer phrases for throughput.
        # - t_one: keep stricter splitting for CTC stability.
        if engine == "gigaam":
            min_phrase_ms = 1600
            max_silence_gap_ms = 1200
            pre_pad_ms = 240
            post_pad_ms = 320
            merge_gap_ms = 900
        else:
            min_phrase_ms = 900
            max_silence_gap_ms = 700
            pre_pad_ms = 180
            post_pad_ms = 260
            merge_gap_ms = 280
        min_phrase_samples = int(sample_rate * min_phrase_ms / 1000)
        max_silence_samples = int(sample_rate * max_silence_gap_ms / 1000)
        pre_pad_samples = int(sample_rate * pre_pad_ms / 1000)
        post_pad_samples = int(sample_rate * post_pad_ms / 1000)
        merge_gap_samples = int(sample_rate * merge_gap_ms / 1000)

        # Keep API contract: segment_seconds is maximum phrase length cap.
        max_phrase_samples = max(frame_samples, int(sample_rate * max_segment_seconds))

        vad = VAD(mode="silero", aggressiveness=2)
        raw_ranges: list[tuple[int, int]] = []

        in_speech = False
        speech_start = 0
        last_speech_sample = 0

        for frame_start in range(0, total_samples, frame_samples):
            frame_end = min(total_samples, frame_start + frame_samples)
            frame_pcm = pcm_all[frame_start * bytes_per_sample : frame_end * bytes_per_sample]
            has_speech = vad.is_speech(frame_pcm)

            if has_speech:
                if not in_speech:
                    in_speech = True
                    speech_start = frame_start
                last_speech_sample = frame_end

            if in_speech:
                # Finalize phrase on long pause.
                if (frame_end - last_speech_sample) >= max_silence_samples:
                    if (last_speech_sample - speech_start) >= min_phrase_samples:
                        raw_ranges.append((speech_start, last_speech_sample))
                    in_speech = False
                    continue

                # Hard cap very long phrases.
                if (frame_end - speech_start) >= max_phrase_samples:
                    raw_ranges.append((speech_start, frame_end))
                    in_speech = False

        if in_speech and (last_speech_sample - speech_start) >= min_phrase_samples:
            raw_ranges.append((speech_start, last_speech_sample))

        if not raw_ranges:
            return []

        # Apply context padding and merge near-adjacent phrases.
        padded: list[tuple[int, int]] = []
        for start, end in raw_ranges:
            s = max(0, start - pre_pad_samples)
            e = min(total_samples, end + post_pad_samples)
            if e > s:
                padded.append((s, e))

        merged: list[tuple[int, int]] = []
        for start, end in padded:
            if not merged:
                merged.append((start, end))
                continue
            prev_start, prev_end = merged[-1]
            if start <= prev_end + merge_gap_samples:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        # Enforce max phrase length after merge by splitting oversized chunks.
        final_ranges: list[tuple[int, int]] = []
        for start, end in merged:
            cur = start
            while cur < end:
                chunk_end = min(end, cur + max_phrase_samples)
                if chunk_end - cur >= frame_samples:
                    final_ranges.append((cur, chunk_end))
                cur = chunk_end

        return final_ranges

    async def _set_progress(self, job_id: str, progress: float) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.progress = max(0.0, min(progress, 99.9))

    async def _set_done(self, job_id: str, segments: list[OfflineSegment]) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "completed"
            job.progress = 100.0
            job.error = None
            result_segments = [asdict(s) for s in segments]
            merged_text = " ".join(s["text"] for s in result_segments).strip()
            job.result = {
                "engine": job.engine,
                "file_path": job.file_path,
                "segments": result_segments,
                "text": merged_text,
            }
            self._recompute_queue_positions_locked()

    async def _set_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.error = error
            self._recompute_queue_positions_locked()

    def _recompute_queue_positions_locked(self) -> None:
        queue_pos = 1
        for job in self._jobs.values():
            if job.status == "queued":
                job.queue_position = queue_pos
                queue_pos += 1
            else:
                job.queue_position = None

    def _get_t_one_recognizer(self) -> OnnxCtcRecognizer:
        if self._t_one is None:
            self._t_one = OnnxCtcRecognizer()
        return self._t_one

    def _get_gigaam_pytorch_recognizer(self) -> GigaAMPytorchRecognizer:
        if self._gigaam_pt is None:
            self._gigaam_pt = GigaAMPytorchRecognizer()
        return self._gigaam_pt

    def _get_gigaam_onnx_recognizer(self) -> OnnxCtcRecognizer:
        if self._gigaam_onnx is None:
            self._gigaam_onnx = OnnxCtcRecognizer()
        return self._gigaam_onnx

    def _get_offline_diarizer(self) -> Diarizer | None:
        if self._offline_diarizer is not None:
            return self._offline_diarizer

        def parse_bool(value: str | None, default: bool) -> bool:
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        enabled = parse_bool(os.getenv("MEETILY_OFFLINE_DIARIZATION_ENABLED"), True)
        mode = os.getenv("MEETILY_OFFLINE_DIARIZATION_MODE", "pyannote")
        hf_token = (
            os.getenv("MEETILY_HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HF_TOKEN")
        )
        models_dir = Path(__file__).resolve().parents[1] / "models"

        try:
            self._offline_diarizer = build_diarizer(
                enabled=enabled,
                mode=mode,
                huggingface_token=hf_token,
                models_dir=models_dir,
            )
            return self._offline_diarizer
        except Exception as exc:
            logger.warning("offline diarizer init failed, disabled reason=%s", exc)
            self._offline_diarizer = None
            return None

    async def _ensure_recognizer_ready(self, engine: str) -> None:
        async with self._recognizer_init_lock:
            if engine == "t_one" and self._t_one is None:
                self._t_one = await asyncio.to_thread(OnnxCtcRecognizer)
                return
            if engine == "gigaam":
                if self._gigaam_pt is None:
                    self._gigaam_pt = await asyncio.to_thread(GigaAMPytorchRecognizer)
                if self._gigaam_onnx is None:
                    self._gigaam_onnx = await asyncio.to_thread(OnnxCtcRecognizer)

    @staticmethod
    def job_to_dict(job: OfflineJob) -> dict:
        payload = asdict(job)
        payload["job_type"] = "offline_transcribe"
        return payload
