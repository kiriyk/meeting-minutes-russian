use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::StatusCode;
use serde::Serialize;
use std::fs::OpenOptions;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

#[derive(Default)]
struct ManagedAsrService {
    child: Option<Child>,
    port: Option<u16>,
}

#[derive(Clone, Default)]
pub struct AsrGatewayServiceManager {
    inner: Arc<Mutex<ManagedAsrService>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AsrGatewayServiceStatus {
    pub port: u16,
    pub managed: bool,
    pub healthy: bool,
    pub mode: String,
}

impl AsrGatewayServiceManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn ensure_running(&self, port: u16) -> Result<()> {
        if Self::is_healthy(port).await {
            return Ok(());
        }

        let mut state = self.inner.lock().await;
        if state.child.is_some() && state.port != Some(port) {
            Self::stop_locked(&mut state).await?;
        } else if state.child.is_some() && state.port == Some(port) {
            if Self::is_healthy(port).await {
                return Ok(());
            }
            Self::stop_locked(&mut state).await?;
        }

        let asr_service_dir = resolve_asr_service_dir()?;
        let python_bin = resolve_python_binary(&asr_service_dir)
            .ok_or_else(|| anyhow!("Python interpreter not found for asr-service"))?;
        let log_file_path = resolve_log_file_path(&asr_service_dir);
        if let Some(parent) = log_file_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create ASR log directory {:?}", parent))?;
        }
        let hf_home = asr_service_dir.join("models").join("hf-cache");
        let tmp_dir = asr_service_dir.join("models").join("tmp");
        std::fs::create_dir_all(&hf_home)
            .with_context(|| format!("Failed to create HF cache dir {:?}", hf_home))?;
        std::fs::create_dir_all(&tmp_dir)
            .with_context(|| format!("Failed to create ASR tmp dir {:?}", tmp_dir))?;
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file_path)
            .with_context(|| format!("Failed to open ASR log file {:?}", log_file_path))?;
        let log_file_err = log_file
            .try_clone()
            .with_context(|| format!("Failed to clone ASR log file handle {:?}", log_file_path))?;

        let mut command = Command::new(python_bin);
        command
            .arg("-m")
            .arg("asr_service.main")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--recordings-dir")
            .arg("recordings")
            .arg("--models-dir")
            .arg("models")
            .current_dir(&asr_service_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(log_file_err))
            .kill_on_drop(true);
        command.env("HF_HOME", &hf_home);
        command.env("TMPDIR", &tmp_dir);
        if cfg!(target_os = "macos") && std::env::var("MEETILY_ASR_USE_COREML").is_err() {
            command.env("MEETILY_ASR_USE_COREML", "1");
        }
        if std::env::var("MEETILY_GIGAAM_MODEL_DIR").is_err() {
            if let Some(gigaam_dir) = resolve_gigaam_model_dir(&asr_service_dir) {
                command.env("MEETILY_GIGAAM_MODEL_DIR", gigaam_dir);
            }
        }
        log::info!(
            "Spawning managed ASR service on port {} (logs: {})",
            port,
            log_file_path.display()
        );

        let child = command
            .spawn()
            .with_context(|| format!("Failed to spawn ASR service in {:?}", asr_service_dir))?;
        state.child = Some(child);
        state.port = Some(port);
        drop(state);

        if let Err(err) = wait_for_health(port, Duration::from_secs(8)).await {
            let mut state = self.inner.lock().await;
            let _ = Self::stop_locked(&mut state).await;
            return Err(err);
        }
        Ok(())
    }

    pub async fn stop_managed(&self) -> Result<()> {
        let mut state = self.inner.lock().await;
        Self::stop_locked(&mut state).await
    }

    pub async fn shutdown_for_exit(&self) {
        if let Err(err) = self.stop_managed().await {
            log::warn!("Failed to stop managed ASR service: {}", err);
        }
    }

    pub async fn status(&self, port: u16) -> AsrGatewayServiceStatus {
        let mut managed = false;
        {
            let mut state = self.inner.lock().await;
            if let Some(child) = state.child.as_mut() {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        state.child = None;
                        state.port = None;
                    }
                    Ok(None) => {
                        managed = state.port == Some(port);
                    }
                    Err(_) => {
                        state.child = None;
                        state.port = None;
                    }
                }
            }
        }

        let healthy = Self::is_healthy(port).await;
        let mode = if managed && healthy {
            "managed_running"
        } else if managed && !healthy {
            "managed_starting"
        } else if !managed && healthy {
            "external_running"
        } else {
            "stopped"
        }
        .to_string();

        AsrGatewayServiceStatus {
            port,
            managed,
            healthy,
            mode,
        }
    }

    async fn stop_locked(state: &mut ManagedAsrService) -> Result<()> {
        if let Some(mut child) = state.child.take() {
            let _ = child.start_kill();
            let _ = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
        }
        state.port = None;
        Ok(())
    }

    async fn is_healthy(port: u16) -> bool {
        let url = format!("http://127.0.0.1:{}/health", port);
        match reqwest::get(url).await {
            Ok(resp) => resp.status() == StatusCode::OK,
            Err(_) => false,
        }
    }
}

fn resolve_asr_service_dir() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("MEETILY_ASR_SERVICE_DIR") {
        if !path.is_empty() {
            let candidate = PathBuf::from(path);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow!("Unable to resolve repository root from CARGO_MANIFEST_DIR"))?;

    let candidate = repo_root.join("asr-service");
    if candidate.exists() {
        Ok(candidate)
    } else {
        Err(anyhow!(
            "asr-service directory not found (expected at {})",
            candidate.display()
        ))
    }
}

fn resolve_python_binary(asr_service_dir: &Path) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if cfg!(target_os = "windows") {
        candidates.push(asr_service_dir.join(".venv").join("Scripts").join("python.exe"));
    } else {
        candidates.push(asr_service_dir.join(".venv").join("bin").join("python"));
    }

    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }

    if let Some(path) = which::which("python3").ok() {
        return Some(path);
    }
    which::which("python").ok()
}

fn resolve_log_file_path(asr_service_dir: &Path) -> PathBuf {
    if let Ok(path) = std::env::var("MEETILY_ASR_LOG_FILE") {
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }
    asr_service_dir.join("recordings").join("asr-service.log")
}

fn resolve_gigaam_model_dir(asr_service_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        asr_service_dir.join("models").join("gigaam"),
        asr_service_dir.join("models").join("t_one").join("GigaAM-v3"),
    ];
    for candidate in candidates {
        if candidate.is_dir() {
            let required = [
                candidate.join("config.json"),
                candidate.join("modeling_gigaam.py"),
                candidate.join("pytorch_model.bin"),
                candidate.join("tokenizer.model"),
            ];
            if required.iter().all(|p| p.exists()) {
                return Some(candidate);
            }
        }
    }
    None
}

async fn wait_for_health(port: u16, timeout: Duration) -> Result<()> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if AsrGatewayServiceManager::is_healthy(port).await {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(anyhow!(
                "ASR service did not become healthy on port {} within {:?}",
                port,
                timeout
            ));
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}
