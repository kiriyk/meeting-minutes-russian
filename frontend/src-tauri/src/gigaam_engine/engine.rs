use crate::gigaam_engine::model::GigaAmModel;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GigaAmModelStatus {
    Available,
    Missing,
    Downloading(u8),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GigaAmModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_mb: u32,
    pub status: GigaAmModelStatus,
    pub description: String,
}

/// HuggingFace base URL for GigaAM models
const HF_GIGAAM_BASE_URL: &str =
    "https://huggingface.co/kiriyk/GigaAM-v3-onnx-rnnt-e2e/resolve/main";

/// Files required for GigaAM v3 E2E RNN-T
const GIGAAM_FILES: &[(&str, u64)] = &[
    ("encoder.onnx", 350_000_000),     // ~350 MB
    ("decoder.onnx", 350_000_000),     // ~350 MB
    ("joint.onnx", 10_000_000),        // ~10 MB
    ("vocab.json", 100_000),           // ~100 KB
    ("tokenizer.model", 200_000),      // ~200 KB
];

/// Known GigaAM model configurations (currently only one).
const GIGAAM_MODELS: &[(&str, u32, &str)] = &[(
    "gigaam-v3-e2e-rnnt",
    851,
    "GigaAM v3 E2E RNN-T — high-quality Russian ASR",
)];

pub struct GigaAmEngine {
    models_dir: PathBuf,
    current_model: Arc<RwLock<Option<GigaAmModel>>>,
    current_model_name: Arc<RwLock<Option<String>>>,
    available_models: Arc<RwLock<HashMap<String, GigaAmModelInfo>>>,
    cancel_download_flag: Arc<RwLock<Option<String>>>, // model name being cancelled
    pub(crate) active_downloads: Arc<RwLock<HashSet<String>>>,
}

impl GigaAmEngine {
    pub fn new_with_models_dir(models_dir: Option<PathBuf>) -> Result<Self> {
        let models_dir = if let Some(dir) = models_dir {
            dir.join("gigaam")
        } else {
            let current = std::env::current_dir()
                .map_err(|e| anyhow!("Failed to get cwd: {}", e))?;
            if cfg!(debug_assertions) {
                current.join("models").join("gigaam")
            } else {
                dirs::data_dir()
                    .or_else(|| dirs::home_dir())
                    .ok_or_else(|| anyhow!("Cannot find system data dir"))?
                    .join("Meetily")
                    .join("models")
                    .join("gigaam")
            }
        };

        log::info!("GigaAmEngine models dir: {}", models_dir.display());

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

    /// Scan models_dir for known model subdirectories.
    pub async fn discover_models(&self) -> Result<Vec<GigaAmModelInfo>> {
        let mut result = Vec::new();
        let active_downloads = self.active_downloads.read().await;

        for (name, size_mb, desc) in GIGAAM_MODELS {
            let model_path = self.models_dir.join(name);

            let status = if active_downloads.contains(*name) {
                GigaAmModelStatus::Downloading(0)
            } else if Self::model_files_present(&model_path) {
                GigaAmModelStatus::Available
            } else {
                GigaAmModelStatus::Missing
            };

            let info = GigaAmModelInfo {
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

    fn model_files_present(dir: &PathBuf) -> bool {
        ["encoder.onnx", "decoder.onnx", "joint.onnx", "vocab.json"]
            .iter()
            .all(|f| dir.join(f).exists())
    }

    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        // Return early if already loaded
        if let Some(name) = self.current_model_name.read().await.as_ref() {
            if name == model_name {
                log::info!("GigaAM model '{}' already loaded", model_name);
                return Ok(());
            }
        }

        let path = {
            let cache = self.available_models.read().await;
            cache
                .get(model_name)
                .map(|m| m.path.clone())
                .ok_or_else(|| anyhow!("GigaAM model '{}' not found", model_name))?
        };

        log::info!("Loading GigaAM model '{}' from {}", model_name, path.display());
        let model =
            GigaAmModel::new(&path).map_err(|e| anyhow!("Failed to load GigaAM: {}", e))?;

        *self.current_model.write().await = Some(model);
        *self.current_model_name.write().await = Some(model_name.to_string());
        log::info!("GigaAM model '{}' loaded successfully", model_name);
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
            .ok_or_else(|| anyhow!("No GigaAM model loaded"))?;

        model
            .transcribe_samples(samples)
            .map_err(|e| anyhow!("GigaAM transcription failed: {}", e))
    }

    pub async fn get_models_directory(&self) -> PathBuf {
        self.models_dir.clone()
    }

    /// Cancel an in-progress download for `model_name`.
    pub async fn cancel_download(&self, model_name: &str) {
        let mut flag = self.cancel_download_flag.write().await;
        *flag = Some(model_name.to_string());
        log::info!("GigaAM: cancellation requested for '{}'", model_name);
    }

    /// Delete a downloaded (or partially downloaded) model directory.
    pub async fn delete_model(&self, model_name: &str) -> Result<String> {
        log::info!("Deleting GigaAM model: {}", model_name);

        let model_info = {
            let cache = self.available_models.read().await;
            cache.get(model_name).cloned()
        };

        let model_info =
            model_info.ok_or_else(|| anyhow!("GigaAM model '{}' not found", model_name))?;

        if model_info.path.exists() {
            fs::remove_dir_all(&model_info.path).await.map_err(|e| {
                anyhow!(
                    "Failed to delete '{}': {}",
                    model_info.path.display(),
                    e
                )
            })?;
            log::info!("Deleted GigaAM model dir: {}", model_info.path.display());
        }

        // Update cache
        {
            let mut cache = self.available_models.write().await;
            if let Some(m) = cache.get_mut(model_name) {
                m.status = GigaAmModelStatus::Missing;
            }
        }

        // If this was the loaded model, unload it
        if self.current_model_name.read().await.as_deref() == Some(model_name) {
            self.unload_model().await;
        }

        Ok(format!("Deleted GigaAM model '{}'", model_name))
    }

    /// Download a GigaAM model from HuggingFace with progress callback.
    pub async fn download_model(
        &self,
        model_name: &str,
        progress_callback: Option<Box<dyn Fn(u8) + Send + Sync>>,
    ) -> Result<()> {
        log::info!("Starting download for GigaAM model: {}", model_name);

        // Guard against concurrent downloads
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

        // Clear any previous cancel flag
        {
            let mut flag = self.cancel_download_flag.write().await;
            *flag = None;
        }

        // Get model path from cache (or construct it)
        let model_dir = {
            let cache = self.available_models.read().await;
            cache
                .get(model_name)
                .map(|m| m.path.clone())
                .unwrap_or_else(|| self.models_dir.join(model_name))
        };

        // Update status to Downloading
        {
            let mut cache = self.available_models.write().await;
            if let Some(m) = cache.get_mut(model_name) {
                m.status = GigaAmModelStatus::Downloading(0);
            }
        }

        // Create model directory
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

        // Calculate total expected size for aggregate progress
        let total_size_bytes: u64 = GIGAAM_FILES.iter().map(|(_, s)| s).sum();
        let mut total_downloaded: u64 = 0;

        // Tally already-complete files
        for (filename, expected_size) in GIGAAM_FILES {
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
            for (filename, expected_size) in GIGAAM_FILES {
                // Check cancel flag
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

                // Skip already-complete files (99% tolerance)
                let tolerance = (*expected_size as f64 * 0.99) as u64;
                if existing_size >= tolerance && *expected_size > 0 {
                    log::info!("Skipping complete file: {}", filename);
                    continue;
                }

                let file_url = format!("{}/{}", HF_GIGAAM_BASE_URL, filename);
                log::info!("Downloading GigaAM file: {} (resuming from {} bytes)", filename, existing_size);

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
                    return Err(anyhow!(
                        "HTTP {} for {}",
                        response.status(),
                        filename
                    ));
                };

                // Open file in append or create mode
                let file = if resume_offset > 0 {
                    tokio::fs::OpenOptions::new()
                        .append(true)
                        .open(&file_path)
                        .await
                        .map_err(|e| anyhow!("Failed to open file {}: {}", filename, e))?
                } else {
                    tokio::fs::File::create(&file_path)
                        .await
                        .map_err(|e| anyhow!("Failed to create file {}: {}", filename, e))?
                };

                let mut writer = tokio::io::BufWriter::new(file);

                while let Some(chunk) = stream_response.chunk().await.map_err(|e| {
                    anyhow!("Download error for {}: {}", filename, e)
                })? {
                    // Check cancel
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

                    // Throttle progress: update every 300ms or 5% jump
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

                        // Update cache status
                        {
                            let mut cache = self.available_models.write().await;
                            if let Some(m) = cache.get_mut(model_name) {
                                m.status = GigaAmModelStatus::Downloading(pct);
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
                log::info!("Downloaded GigaAM file: {}", filename);
            }
            Ok(())
        }
        .await;

        // Remove from active downloads
        {
            let mut active = self.active_downloads.write().await;
            active.remove(model_name);
        }

        match result {
            Ok(()) => {
                // Update to Available
                {
                    let mut cache = self.available_models.write().await;
                    if let Some(m) = cache.get_mut(model_name) {
                        m.status = GigaAmModelStatus::Available;
                    }
                }
                log::info!("GigaAM model '{}' downloaded successfully", model_name);
                Ok(())
            }
            Err(e) => {
                // Update to Error
                {
                    let mut cache = self.available_models.write().await;
                    if let Some(m) = cache.get_mut(model_name) {
                        m.status = GigaAmModelStatus::Error(e.to_string());
                    }
                }
                Err(e)
            }
        }
    }
}
