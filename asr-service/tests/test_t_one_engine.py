from __future__ import annotations

import base64

from asr_service.engines.t_one_engine import TOneEngine
from asr_service.schemas import AudioChunkMessage, StartSessionMessage


def _pcm_silence_20ms(sample_rate: int = 16000) -> bytes:
    # 20ms * 16000Hz = 320 samples, 16-bit mono => 640 bytes
    samples = int(sample_rate * 0.02)
    return (b"\x00\x00") * samples


def _pcm_speech_20ms(sample_rate: int = 16000, amplitude: int = 3000) -> bytes:
    samples = int(sample_rate * 0.02)
    sample = int(amplitude).to_bytes(2, byteorder="little", signed=True)
    return sample * samples


def test_t_one_emits_partial_and_final() -> None:
    start = StartSessionMessage(
        type="start_session",
        session_id="s1",
        sample_rate=16000,
        format="pcm_s16le",
        channels=1,
        engines=["t_one"],
    )

    engine = TOneEngine(start)

    # 12 speech chunks => 240ms, enough for at least one partial with default 200ms.
    partial_count = 0
    t_ms = 0
    for _ in range(12):
        events = engine.process_chunk(
            session_id="s1",
            chunk_pcm=_pcm_speech_20ms(),
            chunk_start_ms=t_ms,
            chunk_end_ms=t_ms + 20,
        )
        partial_count += sum(1 for e in events if e.type == "partial_transcript")
        t_ms += 20

    assert partial_count >= 1

    # 30 silence chunks => 600ms, should finalize utterance.
    got_final = False
    for _ in range(30):
        events = engine.process_chunk(
            session_id="s1",
            chunk_pcm=_pcm_silence_20ms(),
            chunk_start_ms=t_ms,
            chunk_end_ms=t_ms + 20,
        )
        got_final = got_final or any(e.type == "final_segment" for e in events)
        t_ms += 20

    assert got_final


def test_audio_chunk_schema_base64_roundtrip() -> None:
    raw = _pcm_speech_20ms()
    msg = AudioChunkMessage(
        type="audio_chunk",
        session_id="s2",
        seq=1,
        timestamp_ms=0,
        data_b64=base64.b64encode(raw).decode("ascii"),
    )
    assert base64.b64decode(msg.data_b64) == raw
