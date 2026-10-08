//! Offline, whole-recording speaker clustering. No Python service is involved.
mod community;
mod native;
pub mod turns;

use anyhow::{anyhow, bail, Result};
use futures_util::StreamExt;
use native::Diarizer;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicU32, Ordering},
};
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tokio_util::sync::CancellationToken;
use turns::{SpeakerTurn, SpeechRegion};

pub struct DiarizationOutput {
    pub turns: Vec<SpeakerTurn>,
    // Some(empty) means successful Community-1 silence, not unavailable models.
    pub speech_regions: Option<Vec<SpeechRegion>>,
}

const SEGMENTATION: &str = "segmentation.onnx";
const EMBEDDING: &str = "embedding.onnx";
const SEGMENTATION_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";
const EMBEDDING_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx";
static DOWNLOADING: AtomicBool = AtomicBool::new(false);
static CANCEL_DOWNLOAD: AtomicBool = AtomicBool::new(false);
static DOWNLOAD_PROGRESS: AtomicU32 = AtomicU32::new(0);
static DOWNLOAD_MODEL: Lazy<std::sync::Mutex<Option<DiarizationModel>>> =
    Lazy::new(|| std::sync::Mutex::new(None));
static DOWNLOAD_TOKEN: Lazy<std::sync::Mutex<Option<CancellationToken>>> =
    Lazy::new(|| std::sync::Mutex::new(None));

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiarizationModel {
    #[default]
    #[serde(rename = "legacy")]
    Legacy,
    #[serde(rename = "community-1")]
    Community1,
}

impl DiarizationModel {
    fn directory_name(self) -> &'static str {
        match self {
            Self::Legacy => "diarization-v1",
            Self::Community1 => "diarization-community-1",
        }
    }
}

pub fn models_dir<R: Runtime>(app: &AppHandle<R>, model: DiarizationModel) -> Result<PathBuf> {
    Ok(app
        .path()
        .app_data_dir()?
        .join("models")
        .join(model.directory_name()))
}

pub fn models_available_for(dir: &Path, model: DiarizationModel) -> bool {
    match model {
        DiarizationModel::Legacy => models_available(dir),
        DiarizationModel::Community1 => community::models_available(dir),
    }
}

pub fn models_available(dir: &Path) -> bool {
    // Fixed release artifacts: a truncated or modified file must be repairable
    // via Download, rather than being advertised as ready by its filename.
    [
        (
            SEGMENTATION,
            "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
        ),
        (
            EMBEDDING,
            "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
        ),
    ]
    .iter()
    .all(|(name, expected)| {
        file_digest(&dir.join(name))
            .map(|digest| digest == *expected)
            .unwrap_or(false)
    })
}

fn file_digest(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let len = file.read(&mut buffer)?;
        if len == 0 {
            break;
        }
        hasher.update(&buffer[..len]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Serialize)]
pub struct ModelStatus {
    model: DiarizationModel,
    available: bool,
    downloading: bool,
    progress: u32,
    active_download: Option<DiarizationModel>,
}

#[tauri::command]
pub async fn diarization_get_model_status<R: Runtime>(
    app: AppHandle<R>,
    model: Option<DiarizationModel>,
) -> Result<ModelStatus, String> {
    let model = model.unwrap_or_default();
    let dir = models_dir(&app, model).map_err(|e| e.to_string())?;
    let available = tokio::task::spawn_blocking(move || models_available_for(&dir, model))
        .await
        .map_err(|e| e.to_string())?;
    let active_download = *DOWNLOAD_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    let downloading = DOWNLOADING.load(Ordering::SeqCst) && active_download == Some(model);
    Ok(ModelStatus {
        model,
        available,
        downloading,
        progress: if downloading {
            DOWNLOAD_PROGRESS.load(Ordering::SeqCst)
        } else {
            0
        },
        active_download,
    })
}

struct DownloadGuard;
impl Drop for DownloadGuard {
    fn drop(&mut self) {
        *DOWNLOAD_TOKEN.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *DOWNLOAD_MODEL.lock().unwrap_or_else(|e| e.into_inner()) = None;
        DOWNLOADING.store(false, Ordering::SeqCst);
    }
}

#[tauri::command]
pub async fn diarization_download_models<R: Runtime>(
    app: AppHandle<R>,
    model: Option<DiarizationModel>,
) -> Result<(), String> {
    let model = model.unwrap_or_default();
    let dir = models_dir(&app, model).map_err(|e| e.to_string())?;
    let checked_dir = dir.clone();
    if tokio::task::spawn_blocking(move || models_available_for(&checked_dir, model))
        .await
        .map_err(|e| e.to_string())?
    {
        return Ok(());
    }
    DOWNLOADING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .map_err(|_| "Diarization download already in progress".to_string())?;
    CANCEL_DOWNLOAD.store(false, Ordering::SeqCst);
    *DOWNLOAD_MODEL.lock().unwrap_or_else(|e| e.into_inner()) = Some(model);
    let token = CancellationToken::new();
    *DOWNLOAD_TOKEN.lock().unwrap_or_else(|e| e.into_inner()) = Some(token.clone());
    DOWNLOAD_PROGRESS.store(0, Ordering::SeqCst);
    // Keep the guard inside the blocking worker even if its caller is dropped.
    let app_for_download = app.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _guard = DownloadGuard;
        download_bundle_for(&dir, model, &token, |progress| {
            DOWNLOAD_PROGRESS.store(progress, Ordering::SeqCst);
            let _ = app_for_download.emit(
                "diarization-model-download-progress",
                serde_json::json!({"model": model, "progress": progress}),
            );
        })
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r.map_err(|e| e.to_string()));
    match &result {
        Ok(()) => {
            let _ = app.emit(
                "diarization-model-download-complete",
                serde_json::json!({"model": model}),
            );
        }
        Err(error) => {
            let _ = app.emit(
                "diarization-model-download-error",
                serde_json::json!({"model": model, "error": error}),
            );
        }
    }
    result
}

#[tauri::command]
pub fn diarization_cancel_download(model: Option<DiarizationModel>) {
    if let Some(model) = model {
        if *DOWNLOAD_MODEL.lock().unwrap_or_else(|e| e.into_inner()) != Some(model) {
            return;
        }
    }
    CANCEL_DOWNLOAD.store(true, Ordering::SeqCst);
    if let Some(token) = DOWNLOAD_TOKEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
    {
        token.cancel();
    }
}

fn download_bundle_for(
    dir: &Path,
    model: DiarizationModel,
    token: &CancellationToken,
    progress: impl Fn(u32),
) -> Result<()> {
    match model {
        DiarizationModel::Legacy => download_bundle(dir, token, progress),
        DiarizationModel::Community1 => download_community_bundle(dir, token, progress),
    }
}

fn download_community_bundle(
    dir: &Path,
    token: &CancellationToken,
    progress: impl Fn(u32),
) -> Result<()> {
    let parent = dir
        .parent()
        .ok_or_else(|| anyhow!("Missing model directory"))?;
    std::fs::create_dir_all(parent)?;
    let staging = tempfile::tempdir_in(parent)?;
    let bundle = staging.path().join("bundle");
    std::fs::create_dir(&bundle)?;
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(1800))
        .build()?;
    let total: u64 = community::ARTIFACTS.iter().map(|a| a.size).sum();
    let mut downloaded = 0u64;
    for artifact in community::ARTIFACTS {
        let url = format!(
            "https://huggingface.co/avencera/speakrs-models/resolve/{}/{}",
            community::REVISION,
            artifact.name
        );
        let base = (downloaded * 99 / total) as u32;
        let end = ((downloaded + artifact.size) * 99 / total) as u32;
        download_file(
            &client,
            &url,
            &bundle.join(artifact.name),
            base,
            end - base,
            token,
            &progress,
        )?;
        downloaded += artifact.size;
    }
    check_download_cancelled()?;
    let _ = community::create_pipeline(&bundle)?;
    check_download_cancelled()?;
    std::fs::write(bundle.join("MODEL-SOURCE.txt"), format!(
        "Community-1 weights (CC-BY-4.0): https://huggingface.co/pyannote/speaker-diarization-community-1\nONNX conversion: https://huggingface.co/avencera/speakrs-models/tree/{}\nImplementation: https://github.com/avencera/speakrs (Apache-2.0)\n", community::REVISION
    ))?;
    publish_bundle(&bundle, dir, staging.path())?;
    progress(100);
    Ok(())
}

fn publish_bundle(bundle: &Path, dir: &Path, staging: &Path) -> Result<()> {
    // Preserve the previous installation if publishing the repaired bundle fails.
    let backup = staging.join("previous");
    let had_previous = dir.exists();
    if had_previous {
        std::fs::rename(dir, &backup)?;
    }
    if let Err(error) = std::fs::rename(bundle, dir) {
        if had_previous {
            std::fs::rename(&backup, dir)?;
        }
        return Err(error.into());
    }
    Ok(())
}

fn download_bundle(dir: &Path, token: &CancellationToken, progress: impl Fn(u32)) -> Result<()> {
    let parent = dir
        .parent()
        .ok_or_else(|| anyhow!("Missing model directory"))?;
    std::fs::create_dir_all(parent)?;
    let staging = tempfile::tempdir_in(parent)?;
    let bundle = staging.path().join("bundle");
    std::fs::create_dir(&bundle)?;
    let archive = staging.path().join("segmentation.tar.bz2");
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(1800))
        .build()?;
    download_file(&client, SEGMENTATION_URL, &archive, 0, 30, token, &progress)?;
    let decoder = bzip2::read::BzDecoder::new(std::fs::File::open(&archive)?);
    // Extract only named files; never trust paths from the downloaded archive.
    for entry in tar::Archive::new(decoder).entries()? {
        check_download_cancelled()?;
        let mut entry = entry?;
        let name = entry.path()?.file_name().map(|s| s.to_owned());
        let destination = match name.as_deref().and_then(|s| s.to_str()) {
            Some("model.onnx") => Some(bundle.join(SEGMENTATION)),
            Some("LICENSE") => Some(bundle.join("segmentation-LICENSE")),
            Some("README.md") => Some(bundle.join("segmentation-README.md")),
            _ => None,
        };
        if let Some(path) = destination {
            if entry.header().entry_type().is_file() {
                std::io::copy(&mut entry, &mut std::fs::File::create(path)?)?;
            }
        }
    }
    download_file(
        &client,
        EMBEDDING_URL,
        &bundle.join(EMBEDDING),
        30,
        69,
        token,
        &progress,
    )?;
    check_download_cancelled()?;
    if !models_available(&bundle) {
        bail!("Downloaded diarization bundle is incomplete");
    }
    // Validate both ONNX models before publishing the directory as available.
    let _ = create_diarizer(&bundle)?;
    check_download_cancelled()?;
    publish_bundle(&bundle, dir, staging.path())?;
    progress(100);
    Ok(())
}

fn check_download_cancelled() -> Result<()> {
    if CANCEL_DOWNLOAD.load(Ordering::SeqCst) {
        bail!("Diarization download cancelled");
    }
    Ok(())
}

fn download_file(
    client: &reqwest::Client,
    url: &str,
    path: &Path,
    base: u32,
    span: u32,
    token: &CancellationToken,
    progress: &impl Fn(u32),
) -> Result<()> {
    check_download_cancelled()?;
    tauri::async_runtime::block_on(async {
        let response = tokio::select! {
            biased;
            _ = token.cancelled() => bail!("Diarization download cancelled"),
            response = client.get(url).send() => response?.error_for_status()?,
        };
        let total = response.content_length().unwrap_or(0);
        let mut file = std::fs::File::create(path)?;
        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;
        let mut previous = u32::MAX;
        loop {
            check_download_cancelled()?;
            let chunk = tokio::select! {
                biased;
                _ = token.cancelled() => bail!("Diarization download cancelled"),
                chunk = stream.next() => chunk,
            };
            let Some(chunk) = chunk else {
                break;
            };
            let chunk = chunk?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            let percent = base
                + if total > 0 {
                    (downloaded.saturating_mul(span as u64) / total).min(span as u64) as u32
                } else {
                    0
                };
            if percent != previous {
                progress(percent);
                previous = percent;
            }
        }
        file.sync_all()?;
        if downloaded == 0 {
            bail!("Empty model download");
        }
        Ok(())
    })
}

fn create_diarizer(dir: &Path) -> Result<Diarizer> {
    if !models_available(dir) {
        bail!("Speaker models are not downloaded");
    }
    Diarizer::create(dir)
}

/// Called in spawn_blocking. The native API processes the whole file synchronously.
pub fn diarize(samples: &[f32], dir: &Path) -> Result<Vec<SpeakerTurn>> {
    if samples.is_empty() || samples.len() > i32::MAX as usize {
        bail!("Unsupported waveform length");
    }
    let started = std::time::Instant::now();
    log::info!(
        "Starting speaker identification for {:.2}s of audio",
        samples.len() as f64 / 16000.0
    );
    let diarizer = create_diarizer(dir)?;
    let result = diarizer.process(samples)?;
    let duration_ms = samples.len() as f64 / 16.0;
    let mut labels = HashMap::new();
    let mut turns = Vec::new();
    for segment in result {
        if !segment.start.is_finite() || !segment.end.is_finite() {
            continue;
        }
        let start_ms = (segment.start as f64 * 1000.0).max(0.0);
        let end_ms = (segment.end as f64 * 1000.0).min(duration_ms);
        if segment.speaker < 0 || !start_ms.is_finite() || !end_ms.is_finite() || end_ms <= start_ms
        {
            continue;
        }
        let next_label = format!("SPEAKER_{:02}", labels.len());
        let speaker = labels.entry(segment.speaker).or_insert(next_label).clone();
        turns.push(SpeakerTurn {
            start_ms,
            end_ms,
            speaker,
        });
    }
    if turns.is_empty() {
        bail!("Speaker inference produced no speaker turns");
    }
    log::info!(
        "Speaker identification complete: {} turns in {:.2}s",
        turns.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(turns)
}

pub fn diarize_with_model(
    samples: &[f32],
    dir: &Path,
    model: DiarizationModel,
    mut progress: impl FnMut(u32) -> bool,
) -> Result<DiarizationOutput> {
    if !progress(0) {
        bail!("Diarization cancelled");
    }
    match model {
        DiarizationModel::Legacy => Ok(DiarizationOutput {
            turns: diarize(samples, dir)?,
            speech_regions: None,
        }),
        DiarizationModel::Community1 => community::diarize(samples, dir, progress),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_choice_keeps_legacy_default_and_separate_model_directories() {
        assert_eq!(DiarizationModel::default(), DiarizationModel::Legacy);
        assert_ne!(
            DiarizationModel::Legacy.directory_name(),
            DiarizationModel::Community1.directory_name()
        );
        assert_eq!(
            serde_json::from_str::<DiarizationModel>("\"community-1\"").unwrap(),
            DiarizationModel::Community1
        );
        assert!(serde_json::from_str::<DiarizationModel>("\"unknown\"").is_err());
    }

    #[test]
    fn publishing_failure_preserves_previous_installation() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("installed");
        let staging = root.path().join("staging");
        std::fs::create_dir(&dir).unwrap();
        std::fs::create_dir(&staging).unwrap();
        std::fs::write(dir.join("previous-model"), b"previous").unwrap();
        assert!(publish_bundle(&staging.join("missing-bundle"), &dir, &staging).is_err());
        assert_eq!(
            std::fs::read(dir.join("previous-model")).unwrap(),
            b"previous"
        );
    }

    #[test]
    fn stalled_model_transfer_can_be_cancelled_and_retried() {
        use std::{net::TcpListener, sync::mpsc, time::Duration};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (headers_sent, wait_for_headers) = mpsc::channel();
        let (release_server, wait_for_release) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            let mut request = [0; 4096];
            connection.read(&mut request).unwrap();
            connection
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\npart")
                .unwrap();
            headers_sent.send(()).unwrap();
            wait_for_release
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
            drop(connection);
            let (mut connection, _) = listener.accept().unwrap();
            connection.read(&mut request).unwrap();
            connection
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nfull")
                .unwrap();
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("download");
        let worker_path = path.clone();
        let worker_url = url.clone();
        let token = CancellationToken::new();
        let worker_token = token.clone();
        let (finished, wait_for_finish) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let client = reqwest::Client::new();
            finished
                .send(download_file(
                    &client,
                    &worker_url,
                    &worker_path,
                    0,
                    100,
                    &worker_token,
                    &|_| {},
                ))
                .unwrap();
        });
        wait_for_headers
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        token.cancel();
        let result = wait_for_finish
            .recv_timeout(Duration::from_secs(2))
            .expect("Cancellation must interrupt a stalled response");
        assert!(result.unwrap_err().to_string().contains("cancelled"));
        worker.join().unwrap();
        release_server.send(()).unwrap();
        download_file(
            &reqwest::Client::new(),
            &url,
            &path,
            0,
            100,
            &CancellationToken::new(),
            &|_| {},
        )
        .unwrap();
        assert_eq!(std::fs::read(path).unwrap(), b"full");
        server.join().unwrap();
    }

    #[test]
    fn partial_model_bundle_is_not_available() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(SEGMENTATION), b"model").unwrap();
        assert!(!models_available(dir.path()));
        std::fs::write(dir.path().join(EMBEDDING), b"").unwrap();
        assert!(!models_available(dir.path()));
        std::fs::write(dir.path().join(EMBEDDING), b"model").unwrap();
        assert!(
            !models_available(dir.path()),
            "Nonempty corrupt files must allow repair download"
        );
    }

    /// Run with DIARIZATION_MODELS_DIR and DIARIZATION_AUDIO_FILE pointing to real fixtures.
    #[test]
    #[ignore = "requires downloaded models and a multi-speaker audio fixture"]
    fn native_model_smoke() {
        let dir = PathBuf::from(std::env::var("DIARIZATION_MODELS_DIR").unwrap());
        let audio = PathBuf::from(std::env::var("DIARIZATION_AUDIO_FILE").unwrap());
        if !models_available(&dir) {
            download_bundle(&dir, &CancellationToken::new(), |_| {}).unwrap();
        }
        let decoded = crate::audio::decoder::decode_audio_file(&audio).unwrap();
        let turns = diarize(&decoded.to_whisper_format(), &dir).unwrap();
        assert!(turns.iter().any(|t| t.speaker == "SPEAKER_00"));
        assert!(turns.iter().any(|t| t.speaker == "SPEAKER_01"));
        assert!(turns.windows(2).all(|w| w[0].start_ms <= w[1].start_ms));
        println!("{} speaker turns", turns.len());
    }
}
