#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import uuid
import wave

import websockets


async def replay(ws_url: str, wav_path: str, frame_ms: int) -> None:
    session_id = str(uuid.uuid4())

    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("Only mono wav is supported by MVP replay tool")

        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        if sample_width != 2:
            raise ValueError("Only 16-bit PCM wav is supported by MVP replay tool")

        samples_per_frame = int(sample_rate * frame_ms / 1000)
        bytes_per_frame = samples_per_frame * sample_width

        async with websockets.connect(ws_url) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "start_session",
                        "session_id": session_id,
                        "sample_rate": sample_rate,
                        "format": "pcm_s16le",
                        "channels": 1,
                        "engines": ["t_one", "gigaam"],
                    }
                )
            )
            print(await ws.recv())

            seq = 0
            timestamp_ms = 0
            while True:
                raw = wf.readframes(samples_per_frame)
                if not raw:
                    break

                await ws.send(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "session_id": session_id,
                            "seq": seq,
                            "timestamp_ms": timestamp_ms,
                            "data_b64": base64.b64encode(raw).decode("ascii"),
                        }
                    )
                )

                seq += 1
                timestamp_ms += frame_ms
                await asyncio.sleep(frame_ms / 1000)

            await ws.send(json.dumps({"type": "end_session", "session_id": session_id}))
            print(await ws.recv())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay WAV to ASR WS as realtime chunks")
    parser.add_argument("wav_path")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--frame-ms", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(replay(ws_url=args.ws_url, wav_path=args.wav_path, frame_ms=args.frame_ms))


if __name__ == "__main__":
    main()
