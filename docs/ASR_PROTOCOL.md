# ASR Protocol (Stage 2 MVP)

## WebSocket

Endpoint: `ws://127.0.0.1:<port>/ws`

### Client -> Server

`start_session`
```json
{
  "type": "start_session",
  "session_id": "uuid",
  "sample_rate": 16000,
  "format": "pcm_s16le",
  "channels": 1,
  "engines": ["t_one", "gigaam"],
  "vad": {"enabled": true, "mode": "silero", "aggressiveness": 2},
  "chunking": {
    "live_frame_ms": 20,
    "t_one_emit_ms": 200,
    "gigaam_segment_target_s": 10,
    "gigaam_segment_max_s": 20
  }
}
```

`audio_chunk`
```json
{
  "type": "audio_chunk",
  "session_id": "uuid",
  "seq": 123,
  "timestamp_ms": 456789,
  "data_b64": "..."
}
```

`end_session`
```json
{"type": "end_session", "session_id": "uuid"}
```

### Server -> Client

`status`
```json
{
  "type": "status",
  "session_id": "uuid",
  "engines": {
    "t_one": {"state": "running", "rtf": 0.3},
    "gigaam": {"state": "running", "rtf": 0.9}
  }
}
```

`partial_transcript`
```json
{
  "type": "partial_transcript",
  "session_id": "uuid",
  "engine": "t_one",
  "time_range_ms": [456000, 456900],
  "text": "[live] speech 0.9s",
  "confidence": 0.7
}
```

`final_segment`
```json
{
  "type": "final_segment",
  "session_id": "uuid",
  "engine": "t_one",
  "segment_id": "t-one-seg-1",
  "time_range_ms": [451000, 463000],
  "speaker": null,
  "text": "[final] speech segment 1.2s",
  "tokens": null,
  "confidence": 0.75
}
```

`error`
```json
{
  "type": "error",
  "session_id": "uuid",
  "engine": null,
  "message": "..."
}
```

## HTTP

- `GET /health` -> `{ "ok": true }`
- `GET /models` -> engine metadata
- `POST /models/download` -> starts HF model download, returns `job_id`
- `GET /models/local` -> list downloaded local ASR models
- `GET /jobs/{id}` -> download job status and progress

## Notes

- Stage 2 emits live T-One timing events with placeholder text decoder.
- GigaAM realtime decoding is planned for Stage 3.
