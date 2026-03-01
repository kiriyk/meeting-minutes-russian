use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::tone_engine::model::ToneModel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToneModelStatus {
    Available,
    Missing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_mb: u32,
    pub status: ToneModelStatus,
    pub description: String,
}

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
        })
    }

    pub async fn discover_models(&self) -> Result<Vec<ToneModelInfo>> {
        let mut result = Vec::new();

        for (name, size_mb, desc) in TONE_MODELS {
            let model_path = self.models_dir.join(name);
            let status = if model_path.join("model.onnx").exists()
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
}
