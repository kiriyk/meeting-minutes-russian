# ASR Service (Stage 3 MVP + HF Model Download)

Local gateway service for Meetily fork.

Implemented:
- `GET /health`
- `GET /models`
- `GET /models/local`
- `POST /models/download`
- `GET /jobs/{id}`
- `POST /offline_transcribe`
- `WS /ws` for `start_session`, `audio_chunk`, `end_session`
- In-memory session state
- Writes incoming PCM stream into WAV per session
- Stage 2 live pipeline for `t_one`:
  - `partial_transcript` every ~`t_one_emit_ms`
  - `final_segment` after speech pause (default 600ms)
- Stage 3 quasi-realtime pipeline for `gigaam`:
  - speech chunking with target/max duration (`gigaam_segment_target_s` / `gigaam_segment_max_s`)
  - overlap carry-over between emitted segments (default 500ms)
  - `final_segment` during long speech and on flush/end session

## Stage-2/3 Inference Note

T-One and GigaAM use a local ONNX CTC recognizer when available (`model.onnx` +
`vocab.json` in `models/t_one/<model_id>/`). If ONNX runtime or model files are
missing, engines fall back to synthetic placeholder text while keeping timing/events.

Model path resolution order:
- `MEETILY_T_ONE_MODEL_FILE` (absolute path to `model.onnx`)
- `MEETILY_T_ONE_MODEL_DIR` (directory containing `model.onnx`)
- auto-discovery under `models/t_one/*/model.onnx`

GigaAM (PyTorch) support:
- Engine looks for `MEETILY_GIGAAM_MODEL_DIR`, otherwise auto-discovers `models/gigaam/*`.
- Install extra deps for PyTorch backend:
  ```bash
  pip install -r requirements-gigaam-pytorch.txt
  ```
- Required files in `models/gigaam/<model_id>/`:
  - `config.json`
  - `modeling_gigaam.py`
  - `pytorch_model.bin`
  - `tokenizer.model`
  - `preprocessor.py`
  - `train_config.yaml`

## Run

```bash
cd asr-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m asr_service.main --port 8765 --recordings-dir recordings --models-dir models
```

## WebSocket URL

`ws://127.0.0.1:8765/ws`

## Data expectations

- PCM `s16le`
- mono channel
- sample rate: 16000 Hz (configurable in `start_session`, but MVP is tuned for this)
- chunks are base64-encoded in `audio_chunk.data_b64`

## HTTP API

### Download model from Hugging Face

```bash
curl -X POST http://127.0.0.1:8765/models/download \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "t-one-base",
    "engine": "t_one",
    "repo_id": "org/repo",
    "filename": "model.onnx",
    "revision": "main"
  }'
```

### Check job status

```bash
curl http://127.0.0.1:8765/jobs/<job_id>
```

### Start offline transcription

```bash
curl -X POST http://127.0.0.1:8765/offline_transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/absolute/path/to/file.wav",
    "engine": "gigaam",
    "segment_seconds": 30
  }'
```

The returned `job_id` is polled via `GET /jobs/<job_id>`.

### List downloaded local models

```bash
curl http://127.0.0.1:8765/models/local
```

## Replay test stream

```bash
python scripts/replay_wav.py /path/to/audio_16k_mono.wav --ws-url ws://127.0.0.1:8765/ws
```
