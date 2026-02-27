from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx

JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class DownloadJob:
    job_id: str
    status: JobStatus
    model_id: str
    engine: str
    repo_id: str
    filename: str
    revision: str
    target_path: str
    hf_token: str | None = None
    progress: float = 0.0
    error: str | None = None


@dataclass
class LocalModelInfo:
    model_id: str
    engine: str
    repo_id: str
    filename: str
    revision: str
    local_path: str


class DownloadManager:
    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, DownloadJob] = {}
        self._lock = asyncio.Lock()

    async def start_download(
        self,
        *,
        model_id: str,
        engine: str,
        repo_id: str,
        filename: str,
        revision: str = "main",
        hf_token: str | None = None,
    ) -> DownloadJob:
        safe_model_id = self._sanitize_path_component(model_id)
        target_dir = (
            self._models_dir / self._sanitize_path_component(engine) / safe_model_id
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / Path(filename).name

        job = DownloadJob(
            job_id=str(uuid.uuid4()),
            status="queued",
            model_id=model_id,
            engine=engine,
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            target_path=str(target_file),
            hf_token=hf_token,
        )

        async with self._lock:
            self._jobs[job.job_id] = job

        asyncio.create_task(self._download_worker(job.job_id))
        return job

    async def get_job(self, job_id: str) -> DownloadJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_local_models(self) -> list[LocalModelInfo]:
        models: list[LocalModelInfo] = []

        for engine_dir in sorted(self._models_dir.iterdir()):
            if not engine_dir.is_dir():
                continue
            engine = engine_dir.name

            for model_dir in sorted(engine_dir.iterdir()):
                if not model_dir.is_dir():
                    continue

                meta = model_dir / "meta.json"
                if not meta.exists():
                    continue

                try:
                    import json

                    meta_obj = json.loads(meta.read_text(encoding="utf-8"))
                    models.append(
                        LocalModelInfo(
                            model_id=meta_obj["model_id"],
                            engine=meta_obj["engine"],
                            repo_id=meta_obj["repo_id"],
                            filename=meta_obj["filename"],
                            revision=meta_obj.get("revision", "main"),
                            local_path=meta_obj["local_path"],
                        )
                    )
                except Exception:
                    continue

        return models

    async def _download_worker(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "running"

        try:
            target_path = Path(job.target_path)
            if target_path.exists() and target_path.stat().st_size > 0:
                await self._set_job_done(job_id)
                await self._write_meta(job)
                return

            url = self._hf_resolve_url(job.repo_id, job.revision, job.filename)
            temp_path = target_path.with_suffix(target_path.suffix + ".part")

            headers: dict[str, str] = {}
            if job.hf_token:
                headers["Authorization"] = f"Bearer {job.hf_token}"

            async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
                async with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", "0"))
                    downloaded = 0

                    with temp_path.open("wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = (
                                (downloaded / total * 100.0) if total > 0 else 0.0
                            )
                            await self._set_job_progress(job_id, min(progress, 99.5))

            temp_path.replace(target_path)
            await self._set_job_done(job_id)
            await self._write_meta(job)
        except Exception as exc:
            await self._set_job_failed(job_id, str(exc))

    async def _write_meta(self, job: DownloadJob) -> None:
        target_path = Path(job.target_path)
        model_dir = target_path.parent
        meta_path = model_dir / "meta.json"

        import json

        meta = {
            "model_id": job.model_id,
            "engine": job.engine,
            "repo_id": job.repo_id,
            "filename": job.filename,
            "revision": job.revision,
            "local_path": str(target_path),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    async def _set_job_progress(self, job_id: str, progress: float) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.progress = progress

    async def _set_job_done(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "completed"
            job.progress = 100.0
            job.error = None

    async def _set_job_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.error = error

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        return (
            "".join(c for c in value if c.isalnum() or c in ("-", "_", ".")).strip()
            or "model"
        )

    @staticmethod
    def _hf_resolve_url(repo_id: str, revision: str, filename: str) -> str:
        # Public URL format used by Whisper/Parakeet downloads in this repository.
        return (
            f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/"
            f"{quote(revision, safe='/')}/{quote(filename, safe='/')}"
        )

    @staticmethod
    def job_to_dict(job: DownloadJob) -> dict:
        payload = asdict(job)
        payload.pop("hf_token", None)
        return payload

    @staticmethod
    def model_to_dict(model: LocalModelInfo) -> dict:
        return asdict(model)
