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

from .engines.gigaam_pytorch_recognizer import GigaAMPytorchRecognizer
from .engines.onnx_ctc_recognizer import OnnxCtcRecognizer

JobStatus = Literal["queued", "running", "completed", "failed"]
logger = logging.getLogger(__name__)


@dataclass
class OfflineSegment:
    engine: str
    segment_id: str
    time_range_ms: list[int]
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
    result: dict | None = None
    error: str | None = None


class OfflineTranscribeManager:
    def __init__(self) -> None:
        self._jobs: dict[str, OfflineJob] = {}
        self._lock = asyncio.Lock()
        self._t_one = OnnxCtcRecognizer()
        self._gigaam_pt = GigaAMPytorchRecognizer()
        self._gigaam_onnx = OnnxCtcRecognizer()

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

        asyncio.create_task(self._worker(job.job_id))
        return job

    async def get_job(self, job_id: str) -> OfflineJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def _worker(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "running"

        cleanup_tmp = False
        wav_path: Path | None = None
        try:
            source_path = Path(job.file_path)
            wav_path, cleanup_tmp = self._ensure_mono_16k_wav(source_path)
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

            seg_s = max(1, int(job.segment_seconds))
            num_segments = max(1, int((total_duration_s + seg_s - 1) // seg_s))
            segments: list[OfflineSegment] = []

            bytes_per_sample = 2
            with wave.open(str(wav_path), "rb") as wav_reader:
                for idx in range(num_segments):
                    start_s = idx * seg_s
                    end_s = min(total_duration_s, (idx + 1) * seg_s)
                    start_ms = int(start_s * 1000)
                    end_ms = int(end_s * 1000)
                    dur_s = max(0.0, end_s - start_s)

                    start_frame = int(start_s * sample_rate)
                    frames_to_read = max(1, int((end_s - start_s) * sample_rate))
                    wav_reader.setpos(start_frame)
                    pcm = wav_reader.readframes(frames_to_read)

                    text = self._decode_segment(
                        engine=job.engine, pcm=pcm, sample_rate=sample_rate
                    )
                    if not text:
                        text = f"[offline-{job.engine}] speech segment {dur_s:.1f}s"

                    segments.append(
                        OfflineSegment(
                            engine=job.engine,
                            segment_id=f"offline-seg-{idx + 1}",
                            time_range_ms=[start_ms, end_ms],
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
            text = self._t_one.transcribe_pcm_s16le(pcm, sample_rate)
            logger.info(
                "offline decode engine=t_one backend=%s ok=%s",
                self._t_one.backend_name,
                text is not None,
            )
            return (text or "").strip() if text is not None else ""

        if engine == "gigaam":
            text = self._gigaam_pt.transcribe_pcm_s16le(pcm, sample_rate)
            if text is not None:
                logger.info(
                    "offline decode engine=gigaam backend=%s ok=true",
                    self._gigaam_pt.backend_name,
                )
                return text.strip()
            text = self._gigaam_onnx.transcribe_pcm_s16le(pcm, sample_rate)
            logger.info(
                "offline decode engine=gigaam backend=%s ok=%s",
                self._gigaam_onnx.backend_name,
                text is not None,
            )
            return (text or "").strip() if text is not None else ""

        return ""

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

    async def _set_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.error = error

    @staticmethod
    def job_to_dict(job: OfflineJob) -> dict:
        payload = asdict(job)
        payload["job_type"] = "offline_transcribe"
        return payload
