use crate::gigaam_engine::model::{GigaAmError, GigaAmModel};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GigaAmModelStatus {
    Available,
    Missing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GigaAmModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_mb: u32,
    pub status: GigaAmModelStatus,
    pub description: String,
}

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
        })
    }

    /// Scan models_dir for known model subdirectories.
    pub async fn discover_models(&self) -> Result<Vec<GigaAmModelInfo>> {
        let mut result = Vec::new();

        for (name, size_mb, desc) in GIGAAM_MODELS {
            let model_path = self.models_dir.join(name);
            let status = if Self::model_files_present(&model_path) {
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
}
