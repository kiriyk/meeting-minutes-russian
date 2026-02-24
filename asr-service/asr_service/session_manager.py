from __future__ import annotations

import asyncio
import base64
import wave
from dataclasses import dataclass
from pathlib import Path

from .schemas import StartSessionMessage


@dataclass
class SessionState:
    session_id: str
    sample_rate: int
    channels: int
    sample_width_bytes: int
    engines: list[str]
    wav_path: Path
    wav_writer: wave.Wave_write
    last_seq: int = -1


class SessionManager:
    def __init__(self, recordings_dir: Path) -> None:
        self._recordings_dir = recordings_dir
        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def start_session(self, msg: StartSessionMessage) -> SessionState:
        if msg.channels != 1:
            raise ValueError("MVP supports only mono input (channels=1)")
        if msg.format != "pcm_s16le":
            raise ValueError("MVP supports only pcm_s16le format")

        async with self._lock:
            if msg.session_id in self._sessions:
                raise ValueError(f"Session already exists: {msg.session_id}")

            wav_path = self._recordings_dir / f"{msg.session_id}.wav"
            wav_writer = wave.open(str(wav_path), "wb")
            wav_writer.setnchannels(msg.channels)
            wav_writer.setsampwidth(2)  # s16le
            wav_writer.setframerate(msg.sample_rate)

            state = SessionState(
                session_id=msg.session_id,
                sample_rate=msg.sample_rate,
                channels=msg.channels,
                sample_width_bytes=2,
                engines=list(msg.engines),
                wav_path=wav_path,
                wav_writer=wav_writer,
            )
            self._sessions[msg.session_id] = state
            return state

    async def append_audio_chunk(self, session_id: str, seq: int, data_b64: str) -> bytes:
        raw = base64.b64decode(data_b64)
        await self.append_audio_chunk_raw(session_id=session_id, seq=seq, raw=raw)
        return raw

    async def append_audio_chunk_raw(self, session_id: str, seq: int, raw: bytes) -> None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise ValueError(f"Session not found: {session_id}")

            if seq <= state.last_seq:
                raise ValueError(
                    f"Out-of-order chunk for session {session_id}: seq={seq}, last_seq={state.last_seq}"
                )

            state.wav_writer.writeframes(raw)
            state.last_seq = seq

    async def get_session(self, session_id: str) -> SessionState | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def end_session(self, session_id: str) -> SessionState:
        async with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is None:
                raise ValueError(f"Session not found: {session_id}")
            state.wav_writer.close()
            return state

    async def close_all(self) -> None:
        async with self._lock:
            for state in self._sessions.values():
                state.wav_writer.close()
            self._sessions.clear()
