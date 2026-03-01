use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;

use crate::tone_engine::model::ToneModel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToneModelStatus {
    Available,
    Missing,
    Downloading(u8),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_mb: u32,
    pub status: ToneModelStatus,
    pub description: String,
}

/// HuggingFace base URL for T-One models
const HF_TONE_BASE_URL: &str =
    "https://huggingface.co/kiriyk/T-one-onnx-ctc/resolve/main";

/// Files required for T-One CTC model
const TONE_FILES: &[(&str, u64)] = &[
    ("model.onnx", 138_000_000),   // ~138 MB
    ("vocab.json", 100_000),       // ~100 KB
];

const TONE_MODELS: &[(&str, u32, &str)] = &[(
    "t-one",
    138,
    "T-Tech T-One — lightweight streaming Russian ASR (CTC)",
)];

pub struct ToneEngine {
    models_dir: PathBuf,
    current_model: Arc<RwLock<Option<ToneModel>>>,
    current_model_name: Arc<RwLock<Option<String>>>,
    available_models: Arc<RwLock<HashMap<String, ToneModelInfo>>>,
    cancel_download_flag: Arc<RwLock<Option<String>>>,
    pub(crate) active_downloads: Arc<RwLock<HashSet<String>>>,
}

impl ToneEngine {
    pub fn new_with_models_dir(models_dir: Option<PathBuf>) -> Result<Self> {
        let models_dir = if let Some(dir) = models_dir {
            dir.join("russianAsr")
        } else {
            let current = std::env::current_dir()
                .map_err(|e| anyhow!("Failed to get cwd: {}", e))?;
            if cfg!(debug_assertions) {
                current.join("models").join("russianAsr")
            } else {
                dirs::data_dir()
                    .or_else(|| dirs::home_dir())
                    .ok_or_else(|| anyhow!("Cannot find system data dir"))?
                    .join("Meetily")
                    .join("models")
                    .join("russianAsr")
            }
        };

        log::info!("ToneEngine models dir: {}", models_dir.display());
        if !models_dir.exists() {
            std::fs::create_dir_all(&models_dir)?;
        }

        Ok(Self {
            models_dir,
            current_model: Arc::new(RwLock::new(None)),
            current_model_name: Arc::new(RwLock::new(None)),
            available_models: Arc::new(RwLock::new(HashMap::new())),
            cancel_download_flag: Arc::new(RwLock::new(None)),
            active_downloads: Arc::new(RwLock::new(HashSet::new())),
        })
    }

    pub async fn discover_models(&self) -> Result<Vec<ToneModelInfo>> {
        let mut result = Vec::new();
        let active_downloads = self.active_downloads.read().await;

        for (name, size_mb, desc) in TONE_MODELS {
            let model_path = self.models_dir.join(name);

            let status = if active_downloads.contains(*name) {
                ToneModelStatus::Downloading(0)
            } else if model_path.join("model.onnx").exists()
                && model_path.join("vocab.json").exists()
            {
                ToneModelStatus::Available
            } else {
                ToneModelStatus::Missing
            };

            let info = ToneModelInfo {
                name: name.to_string(),
                path: model_path,
                size_mb: *size_mb,
                status,
                description: desc.to_string(),
            };
            result.push(info.clone());
        }

        let mut cache = self.available_models.write().await;
        cache.clear();
        for m in &result {
            cache.insert(m.name.clone(), m.clone());
        }

        Ok(result)
    }

    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        if let Some(name) = self.current_model_name.read().await.as_ref() {
            if name == model_name {
                log::info!("T-One model '{}' already loaded", model_name);
                return Ok(());
            }
        }

        let path = {
            let cache = self.available_models.read().await;
            cache
                .get(model_name)
                .map(|m| m.path.clone())
                .ok_or_else(|| anyhow!("T-One model '{}' not found", model_name))?
        };

        log::info!("Loading T-One model '{}' from {}", model_name, path.display());
        let model = ToneModel::new(&path)
            .map_err(|e| anyhow!("Failed to load T-One model: {}", e))?;

        *self.current_model.write().await = Some(model);
        *self.current_model_name.write().await = Some(model_name.to_string());
        log::info!("T-One model '{}' loaded successfully", model_name);
        Ok(())
    }

    pub async fn unload_model(&self) {
        self.current_model.write().await.take();
        self.current_model_name.write().await.take();
    }

    pub async fn get_current_model(&self) -> Option<String> {
        self.current_model_name.read().await.clone()
    }

    pub async fn is_model_loaded(&self) -> bool {
        self.current_model.read().await.is_some()
    }

    pub async fn transcribe_audio(&self, samples: Vec<f32>) -> Result<String> {
        let mut guard = self.current_model.write().await;
        let model = guard
            .as_mut()
            .ok_or_else(|| anyhow!("No T-One model loaded"))?;
        model
            .transcribe_samples(samples)
            .map_err(|e| anyhow!("T-One transcription failed: {}", e))
    }

    pub async fn get_models_directory(&self) -> PathBuf {
        self.models_dir.clone()
    }

    /// Cancel an in-progress download.
    pub async fn cancel_download(&self, model_name: &str) {
        let mut flag = self.cancel_download_flag.write().await;
        *flag = Some(model_name.to_string());
        log::info!("T-One: cancellation requested for '{}'", model_name);
    }

    /// Delete a downloaded (or partial) model directory.
    pub async fn delete_model(&self, model_name: &str) -> Result<String> {
        log::info!("Deleting T-One model: {}", model_name);

        let model_info = {
            let cache = self.available_models.read().await;
            cache.get(model_name).cloned()
        };

        let model_info =
            model_info.ok_or_else(|| anyhow!("T-One model '{}' not found", model_name))?;

        if model_info.path.exists() {
            fs::remove_dir_all(&model_info.path).await.map_err(|e| {
                anyhow!("Failed to delete '{}': {}", model_info.path.display(), e)
            })?;
            log::info!("Deleted T-One model dir: {}", model_info.path.display());
        }

        {
            let mut cache = self.available_models.write().await;
            if let Some(m) = cache.get_mut(model_name) {
                m.status = ToneModelStatus::Missing;
            }
        }

        if self.current_model_name.read().await.as_deref() == Some(model_name) {
            self.unload_model().await;
        }

        Ok(format!("Deleted T-One model '{}'", model_name))
    }

    /// Download a T-One model from HuggingFace with progress callback.
    pub async fn download_model(
        &self,
        model_name: &str,
        progress_callback: Option<Box<dyn Fn(u8) + Send + Sync>>,
    ) -> Result<()> {
        log::info!("Starting download for T-One model: {}", model_name);

        {
            let active = self.active_downloads.read().await;
            if active.contains(model_name) {
                return Err(anyhow!("Download already in progress for '{}'", model_name));
            }
        }
        {
            let mut active = self.active_downloads.write().await;
            active.insert(model_name.to_string());
        }

        {
            let mut flag = self.cancel_download_flag.write().await;
            *flag = None;
        }

        let model_dir = {
            let cache = self.available_models.read().await;
            cache
                .get(model_name)
                .map(|m| m.path.clone())
                .unwrap_or_else(|| self.models_dir.join(model_name))
        };

        {
            let mut cache = self.available_models.write().await;
            if let Some(m) = cache.get_mut(model_name) {
                m.status = ToneModelStatus::Downloading(0);
            }
        }

        if !model_dir.exists() {
            fs::create_dir_all(&model_dir).await.map_err(|e| {
                anyhow!("Failed to create model directory: {}", e)
            })?;
        }

        let client = reqwest::Client::builder()
            .tcp_nodelay(true)
            .pool_max_idle_per_host(1)
            .timeout(Duration::from_secs(3600))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        let total_size_bytes: u64 = TONE_FILES.iter().map(|(_, s)| s).sum();
        let mut total_downloaded: u64 = 0;

        // Count already-downloaded bytes
        for (filename, expected_size) in TONE_FILES {
            let file_path = model_dir.join(filename);
            if file_path.exists() {
                if let Ok(meta) = fs::metadata(&file_path).await {
                    total_downloaded += meta.len().min(*expected_size);
                }
            }
        }

        let mut last_report = Instant::now();
        let mut last_reported_pct: u8 = 0;

        let result: Result<()> = async {
            for (filename, expected_size) in TONE_FILES {
                // Check cancel
                {
                    let flag = self.cancel_download_flag.read().await;
                    if flag.as_deref() == Some(model_name) {
                        return Err(anyhow!("Download cancelled by user"));
                    }
                }

                let file_path = model_dir.join(filename);
                let existing_size: u64 = if file_path.exists() {
                    fs::metadata(&file_path).await.map(|m| m.len()).unwrap_or(0)
                } else {
                    0
                };

                let tolerance = (*expected_size as f64 * 0.99) as u64;
                if existing_size >= tolerance && *expected_size > 0 {
                    log::info!("Skipping complete file: {}", filename);
                    continue;
                }

                let file_url = format!("{}/{}", HF_TONE_BASE_URL, filename);
                log::info!("Downloading T-One file: {} (resuming from {} bytes)", filename, existing_size);

                let mut request = client.get(&file_url);
                if existing_size > 0 {
                    request = request.header("Range", format!("bytes={}-", existing_size));
                }

                let response = request.send().await.map_err(|e| {
                    anyhow!("Failed to start download for {}: {}", filename, e)
                })?;

                let (resume_offset, mut stream_response) = if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
                    (existing_size, response)
                } else if response.status().is_success() {
                    (0u64, response)
                } else {
                    return Err(anyhow!("HTTP {} for {}", response.status(), filename));
                };

                let file = if resume_offset > 0 {
                    tokio::fs::OpenOptions::new()
                        .append(true)
                        .open(&file_path)
                        .await
                        .map_err(|e| anyhow!("Failed to open {}: {}", filename, e))?
                } else {
                    tokio::fs::File::create(&file_path)
                        .await
                        .map_err(|e| anyhow!("Failed to create {}: {}", filename, e))?
                };

                let mut writer = tokio::io::BufWriter::new(file);

                while let Some(chunk) = stream_response.chunk().await.map_err(|e| {
                    anyhow!("Download error for {}: {}", filename, e)
                })? {
                    {
                        let flag = self.cancel_download_flag.read().await;
                        if flag.as_deref() == Some(model_name) {
                            let _ = writer.flush().await;
                            return Err(anyhow!("Download cancelled by user"));
                        }
                    }

                    writer.write_all(&chunk).await.map_err(|e| {
                        anyhow!("Write error for {}: {}", filename, e)
                    })?;

                    total_downloaded += chunk.len() as u64;

                    let now = Instant::now();
                    if now.duration_since(last_report) > Duration::from_millis(300) || {
                        let pct = ((total_downloaded as f64 / total_size_bytes as f64) * 100.0)
                            .min(100.0) as u8;
                        pct.saturating_sub(last_reported_pct) >= 5
                    } {
                        let pct = ((total_downloaded as f64 / total_size_bytes as f64) * 100.0)
                            .min(100.0) as u8;
                        last_report = now;
                        last_reported_pct = pct;

                        {
                            let mut cache = self.available_models.write().await;
                            if let Some(m) = cache.get_mut(model_name) {
                                m.status = ToneModelStatus::Downloading(pct);
                            }
                        }

                        if let Some(ref cb) = progress_callback {
                            cb(pct);
                        }
                    }
                }

                writer.flush().await.map_err(|e| {
                    anyhow!("Flush error for {}: {}", filename, e)
                })?;
                log::info!("Downloaded T-One file: {}", filename);
            }
            Ok(())
        }
        .await;

        {
            let mut active = self.active_downloads.write().await;
            active.remove(model_name);
        }

        match result {
            Ok(()) => {
                {
                    let mut cache = self.available_models.write().await;
                    if let Some(m) = cache.get_mut(model_name) {
                        m.status = ToneModelStatus::Available;
                    }
                }
                log::info!("T-One model '{}' downloaded successfully", model_name);
                Ok(())
            }
            Err(e) => {
                {
                    let mut cache = self.available_models.write().await;
                    if let Some(m) = cache.get_mut(model_name) {
                        m.status = ToneModelStatus::Error(e.to_string());
                    }
                }
                Err(e)
            }
        }
    }
}
