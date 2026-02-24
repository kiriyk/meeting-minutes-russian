from __future__ import annotations

import asyncio
import wave
from pathlib import Path

from asr_service.offline_transcribe_manager import OfflineTranscribeManager


def _write_test_wav(
    path: Path, duration_s: float = 2.5, sample_rate: int = 16000
) -> None:
    nframes = int(duration_s * sample_rate)
    # simple constant-amplitude mono 16-bit signal
    sample = (1500).to_bytes(2, byteorder="little", signed=True)
    data = sample * nframes

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data)


def test_offline_transcribe_job_completes(tmp_path: Path) -> None:
    wav_path = tmp_path / "offline.wav"
    _write_test_wav(wav_path, duration_s=2.5)

    async def _run() -> None:
        manager = OfflineTranscribeManager()
        job = await manager.start_job(
            file_path=str(wav_path),
            engine="gigaam",
            segment_seconds=1,
        )

        # Wait up to 2s for completion.
        for _ in range(200):
            current = await manager.get_job(job.job_id)
            assert current is not None
            if current.status == "completed":
                break
            await asyncio.sleep(0.01)

        current = await manager.get_job(job.job_id)
        assert current is not None
        assert current.status == "completed"
        assert current.result is not None
        segments = current.result["segments"]
        assert len(segments) >= 2
        assert all(seg["engine"] == "gigaam" for seg in segments)

    asyncio.run(_run())
