from __future__ import annotations

from asr_service.engines.gigaam_engine import GigaAMEngine
from asr_service.schemas import StartSessionMessage


def _pcm_silence_20ms(sample_rate: int = 16000) -> bytes:
    samples = int(sample_rate * 0.02)
    return (b"\x00\x00") * samples


def _pcm_speech_20ms(sample_rate: int = 16000, amplitude: int = 3000) -> bytes:
    samples = int(sample_rate * 0.02)
    sample = int(amplitude).to_bytes(2, byteorder="little", signed=True)
    return sample * samples


def test_gigaam_emits_segment_near_target_duration() -> None:
    start = StartSessionMessage(
        type="start_session",
        session_id="g1",
        sample_rate=16000,
        format="pcm_s16le",
        channels=1,
        engines=["gigaam"],
    )

    engine = GigaAMEngine(start)
    t_ms = 0
    events = []
    # 11s speech => should emit at least one chunked segment around 10s.
    for _ in range(550):
        events.extend(
            engine.process_chunk(
                session_id="g1",
                chunk_pcm=_pcm_speech_20ms(),
                chunk_start_ms=t_ms,
                chunk_end_ms=t_ms + 20,
            )
        )
        t_ms += 20

    finals = [e for e in events if e.type == "final_segment"]
    assert len(finals) >= 1
    first = finals[0]
    assert first.engine == "gigaam"
    assert first.time_range_ms[1] - first.time_range_ms[0] >= 9_500


def test_gigaam_flush_emits_remaining_segment() -> None:
    start = StartSessionMessage(
        type="start_session",
        session_id="g2",
        sample_rate=16000,
        format="pcm_s16le",
        channels=1,
        engines=["gigaam"],
    )
    engine = GigaAMEngine(start)

    t_ms = 0
    for _ in range(100):  # 2 seconds, below target
        engine.process_chunk(
            session_id="g2",
            chunk_pcm=_pcm_speech_20ms(),
            chunk_start_ms=t_ms,
            chunk_end_ms=t_ms + 20,
        )
        t_ms += 20

    # Add a short silence that is not enough for auto-finalization.
    for _ in range(10):
        engine.process_chunk(
            session_id="g2",
            chunk_pcm=_pcm_silence_20ms(),
            chunk_start_ms=t_ms,
            chunk_end_ms=t_ms + 20,
        )
        t_ms += 20

    flushed = engine.flush("g2")
    assert len(flushed) == 1
    assert flushed[0].engine == "gigaam"
    assert flushed[0].time_range_ms[1] > flushed[0].time_range_ms[0]
