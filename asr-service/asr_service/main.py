from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .diarization import Diarizer, build_diarizer
from .download_manager import DownloadManager
from .engines.gigaam_engine import GigaAMEngine
from .engines.t_one_engine import TOneEngine
from .offline_transcribe_manager import OfflineTranscribeManager
from .schemas import (
    AudioChunkMessage,
    DownloadModelRequest,
    DownloadModelResponse,
    EndSessionMessage,
    EngineStatus,
    ErrorMessage,
    FinalSegmentMessage,
    JobStatusResponse,
    LocalModelsResponse,
    ModelsResponse,
    OfflineTranscribeRequest,
    OfflineTranscribeResponse,
    PartialTranscriptMessage,
    StartSessionMessage,
    StatusMessage,
)
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


@dataclass
class SessionRuntime:
    config: StartSessionMessage
    diarizer: Diarizer | None = None
    t_one: TOneEngine | None = None
    gigaam: GigaAMEngine | None = None


def create_app(recordings_dir: Path, models_dir: Path) -> FastAPI:
    app = FastAPI(title="Meetily ASR Gateway", version="0.3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3118",
            "http://127.0.0.1:3118",
            "tauri://localhost",
            "https://tauri.localhost",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    session_manager = SessionManager(recordings_dir=recordings_dir)
    download_manager = DownloadManager(models_dir=models_dir)
    offline_manager = OfflineTranscribeManager()
    runtimes: dict[str, SessionRuntime] = {}

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await session_manager.close_all()

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/models")
    async def models() -> ModelsResponse:
        return ModelsResponse(
            engines=[
                {
                    "id": "t_one",
                    "mode": "streaming",
                    "status": "mvp-live-enabled",
                },
                {
                    "id": "gigaam",
                    "mode": "chunked",
                    "status": "mvp-quasi-realtime",
                },
            ]
        )

    @app.post("/models/download")
    async def download_model(req: DownloadModelRequest) -> DownloadModelResponse:
        job = await download_manager.start_download(
            model_id=req.model_id,
            engine=req.engine,
            repo_id=req.repo_id,
            filename=req.filename,
            revision=req.revision,
        )
        return DownloadModelResponse(job_id=job.job_id)

    @app.get("/models/local")
    async def local_models() -> LocalModelsResponse:
        models_list = await download_manager.list_local_models()
        return LocalModelsResponse(
            models=[download_manager.model_to_dict(m) for m in models_list]
        )

    @app.get("/jobs/{job_id}")
    async def job_status(job_id: str) -> JSONResponse:
        job = await download_manager.get_job(job_id)
        if job is not None:
            payload = JobStatusResponse(**download_manager.job_to_dict(job))
            return JSONResponse(payload.model_dump())

        offline_job = await offline_manager.get_job(job_id)
        if offline_job is not None:
            return JSONResponse(offline_manager.job_to_dict(offline_job))

        raise HTTPException(status_code=404, detail="job not found")

    @app.post("/offline_transcribe")
    async def offline_transcribe(
        req: OfflineTranscribeRequest,
    ) -> OfflineTranscribeResponse:
        try:
            job = await offline_manager.start_job(
                file_path=req.file_path,
                engine=req.engine,
                segment_seconds=req.segment_seconds,
            )
            return OfflineTranscribeResponse(job_id=job.job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def send_error(
        ws: WebSocket, session_id: str, message: str, engine: str | None = None
    ) -> None:
        payload = ErrorMessage(session_id=session_id, engine=engine, message=message)
        await ws.send_json(payload.model_dump())

    async def send_event(
        ws: WebSocket,
        event: PartialTranscriptMessage | FinalSegmentMessage,
    ) -> None:
        await ws.send_json(event.model_dump())

    def chunk_duration_ms(raw: bytes, sample_rate: int, channels: int) -> int:
        if sample_rate <= 0 or channels <= 0:
            return 0
        bytes_per_sample = 2
        samples = len(raw) / float(bytes_per_sample * channels)
        return int((samples / float(sample_rate)) * 1000)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg: dict[str, Any] = json.loads(raw)
                except json.JSONDecodeError:
                    await send_error(ws, "unknown", "Invalid JSON payload")
                    continue

                msg_type = msg.get("type")

                if msg_type == "start_session":
                    try:
                        parsed = StartSessionMessage.model_validate(msg)
                        await session_manager.start_session(parsed)
                        logger.info(
                            "ASR start_session session_id=%s engines=%s sample_rate=%s",
                            parsed.session_id,
                            parsed.engines,
                            parsed.sample_rate,
                        )

                        runtime = SessionRuntime(
                            config=parsed,
                            diarizer=build_diarizer(
                                enabled=parsed.diarization.enabled,
                                mode=parsed.diarization.mode,
                                huggingface_token=parsed.diarization.huggingface_token,
                            ),
                        )
                        if "t_one" in parsed.engines:
                            runtime.t_one = TOneEngine(parsed, diarizer=runtime.diarizer)
                            await runtime.t_one.start()
                        if "gigaam" in parsed.engines:
                            runtime.gigaam = GigaAMEngine(
                                parsed,
                                diarizer=runtime.diarizer,
                            )
                            await runtime.gigaam.start()
                        runtimes[parsed.session_id] = runtime

                        status = StatusMessage(
                            session_id=parsed.session_id,
                            engines={
                                engine: EngineStatus(
                                    state="running",
                                    rtf=0.3 if engine == "t_one" else 0.9,
                                )
                                for engine in parsed.engines
                            },
                        )
                        await ws.send_json(status.model_dump())
                    except (ValidationError, ValueError) as exc:
                        await send_error(ws, msg.get("session_id", "unknown"), str(exc))

                elif msg_type == "audio_chunk":
                    try:
                        parsed = AudioChunkMessage.model_validate(msg)
                        runtime = runtimes.get(parsed.session_id)
                        if runtime is None:
                            raise ValueError(
                                f"Session runtime not found: {parsed.session_id}"
                            )

                        pcm = await session_manager.append_audio_chunk(
                            session_id=parsed.session_id,
                            seq=parsed.seq,
                            data_b64=parsed.data_b64,
                        )

                        if runtime.t_one is not None:
                            dur_ms = chunk_duration_ms(
                                pcm,
                                sample_rate=runtime.config.sample_rate,
                                channels=runtime.config.channels,
                            )
                            chunk_start_ms = parsed.timestamp_ms
                            chunk_end_ms = parsed.timestamp_ms + dur_ms
                            events = runtime.t_one.process_chunk(
                                session_id=parsed.session_id,
                                chunk_pcm=pcm,
                                chunk_start_ms=chunk_start_ms,
                                chunk_end_ms=chunk_end_ms,
                            )
                            for event in events:
                                await send_event(ws, event)
                                if event.type == "final_segment":
                                    logger.info(
                                        "ASR event final engine=%s range=%s",
                                        event.engine,
                                        event.time_range_ms,
                                    )
                        if runtime.gigaam is not None:
                            dur_ms = chunk_duration_ms(
                                pcm,
                                sample_rate=runtime.config.sample_rate,
                                channels=runtime.config.channels,
                            )
                            chunk_start_ms = parsed.timestamp_ms
                            chunk_end_ms = parsed.timestamp_ms + dur_ms
                            events = runtime.gigaam.process_chunk(
                                session_id=parsed.session_id,
                                chunk_pcm=pcm,
                                chunk_start_ms=chunk_start_ms,
                                chunk_end_ms=chunk_end_ms,
                            )
                            for event in events:
                                await send_event(ws, event)
                                logger.info(
                                    "ASR event final engine=%s range=%s",
                                    event.engine,
                                    event.time_range_ms,
                                )

                    except (ValidationError, ValueError) as exc:
                        await send_error(ws, msg.get("session_id", "unknown"), str(exc))

                elif msg_type == "end_session":
                    try:
                        parsed = EndSessionMessage.model_validate(msg)
                        runtime = runtimes.get(parsed.session_id)
                        logger.info("ASR end_session session_id=%s", parsed.session_id)

                        if runtime and runtime.t_one is not None:
                            for event in runtime.t_one.flush(parsed.session_id):
                                await send_event(ws, event)
                            await runtime.t_one.stop()
                        if runtime and runtime.gigaam is not None:
                            for event in runtime.gigaam.flush(parsed.session_id):
                                await send_event(ws, event)
                            await runtime.gigaam.stop()

                        state = await session_manager.end_session(parsed.session_id)
                        runtimes.pop(parsed.session_id, None)

                        status = StatusMessage(
                            session_id=parsed.session_id,
                            engines={
                                engine: EngineStatus(state="stopped")
                                for engine in state.engines
                            },
                        )
                        await ws.send_json(status.model_dump())
                        logger.info(
                            "ASR session closed session_id=%s", parsed.session_id
                        )
                    except (ValidationError, ValueError) as exc:
                        await send_error(ws, msg.get("session_id", "unknown"), str(exc))

                else:
                    await send_error(
                        ws,
                        msg.get("session_id", "unknown"),
                        f"Unknown message type: {msg_type}",
                    )

        except WebSocketDisconnect:
            return

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meetily ASR Gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--recordings-dir", default="recordings")
    parser.add_argument("--models-dir", default="models")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    app = create_app(
        recordings_dir=Path(args.recordings_dir),
        models_dir=Path(args.models_dir),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
