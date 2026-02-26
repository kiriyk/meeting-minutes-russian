from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EngineName = Literal["t_one", "gigaam"]


class VADConfig(BaseModel):
    enabled: bool = True
    mode: str = "silero"
    aggressiveness: int = 2


class DiarizationConfig(BaseModel):
    enabled: bool = False
    mode: str = "energy"  # "energy" | "pyannote"
    huggingface_token: str | None = None


class ChunkingConfig(BaseModel):
    live_frame_ms: int = 20
    t_one_emit_ms: int = 200
    gigaam_segment_target_s: int = 10
    gigaam_segment_max_s: int = 20


class StartSessionMessage(BaseModel):
    type: Literal["start_session"]
    session_id: str
    sample_rate: int = 16000
    format: Literal["pcm_s16le"] = "pcm_s16le"
    channels: int = 1
    engines: list[EngineName] = Field(default_factory=lambda: ["t_one"])
    vad: VADConfig = Field(default_factory=VADConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class AudioChunkMessage(BaseModel):
    type: Literal["audio_chunk"]
    session_id: str
    seq: int
    timestamp_ms: int
    data_b64: str


class EndSessionMessage(BaseModel):
    type: Literal["end_session"]
    session_id: str


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    session_id: str
    engine: str | None = None
    message: str


class EngineStatus(BaseModel):
    state: Literal["idle", "running", "stopped", "error"]
    rtf: float | None = None


class StatusMessage(BaseModel):
    type: Literal["status"] = "status"
    session_id: str
    engines: dict[str, EngineStatus]


class PartialTranscriptMessage(BaseModel):
    type: Literal["partial_transcript"] = "partial_transcript"
    session_id: str
    engine: EngineName
    time_range_ms: list[int]
    text: str
    confidence: float | None = None


class FinalSegmentMessage(BaseModel):
    type: Literal["final_segment"] = "final_segment"
    session_id: str
    engine: EngineName
    segment_id: str
    time_range_ms: list[int]
    speaker: str | None = None
    text: str
    tokens: list[str] | None = None
    confidence: float | None = None


class ModelsResponse(BaseModel):
    engines: list[dict[str, str]]


class DownloadModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    engine: EngineName
    repo_id: str
    filename: str
    revision: str = "main"


class DownloadModelResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    model_id: str
    engine: str
    repo_id: str
    filename: str
    revision: str
    target_path: str
    progress: float
    error: str | None = None


class LocalModelsResponse(BaseModel):
    models: list[dict[str, str]]


class OfflineTranscribeRequest(BaseModel):
    file_path: str
    engine: EngineName = "gigaam"
    segment_seconds: int = Field(default=30, ge=1, le=300)


class OfflineTranscribeResponse(BaseModel):
    job_id: str
