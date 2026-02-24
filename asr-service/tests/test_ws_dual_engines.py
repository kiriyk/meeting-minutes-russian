from __future__ import annotations

import base64

from asr_service.main import create_app
from fastapi.testclient import TestClient


def _pcm_speech_20ms(sample_rate: int = 16000, amplitude: int = 3000) -> bytes:
    samples = int(sample_rate * 0.02)
    sample = int(amplitude).to_bytes(2, byteorder="little", signed=True)
    return sample * samples


def test_ws_stream_emits_t_one_and_gigaam_events(tmp_path) -> None:
    app = create_app(
        recordings_dir=tmp_path / "recordings",
        models_dir=tmp_path / "models",
    )

    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "start_session",
                "session_id": "ws-1",
                "sample_rate": 16000,
                "format": "pcm_s16le",
                "channels": 1,
                "engines": ["t_one", "gigaam"],
                "vad": {"enabled": True, "mode": "silero", "aggressiveness": 2},
                "chunking": {
                    "live_frame_ms": 20,
                    "t_one_emit_ms": 200,
                    "gigaam_segment_target_s": 1,
                    "gigaam_segment_max_s": 2,
                },
            }
        )
        status = ws.receive_json()
        assert status["type"] == "status"

        speech = _pcm_speech_20ms()
        b64 = base64.b64encode(speech).decode("ascii")

        for i in range(70):  # 1.4 seconds of speech
            ws.send_json(
                {
                    "type": "audio_chunk",
                    "session_id": "ws-1",
                    "seq": i,
                    "timestamp_ms": i * 20,
                    "data_b64": b64,
                }
            )

        ws.send_json({"type": "end_session", "session_id": "ws-1"})

        got_t_one_partial = False
        got_gigaam_final = False
        got_stopped = False

        # Drain server events until final stopped status.
        for _ in range(300):
            msg = ws.receive_json()
            if msg.get("type") == "partial_transcript" and msg.get("engine") == "t_one":
                got_t_one_partial = True
            if msg.get("type") == "final_segment" and msg.get("engine") == "gigaam":
                got_gigaam_final = True
            if msg.get("type") == "status":
                engines = msg.get("engines", {})
                if all(v.get("state") == "stopped" for v in engines.values()):
                    got_stopped = True
                    break

        assert got_stopped
        assert got_t_one_partial
        assert got_gigaam_final
