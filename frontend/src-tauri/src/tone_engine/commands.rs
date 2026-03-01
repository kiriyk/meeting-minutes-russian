use crate::tone_engine::engine::{ToneEngine, ToneModelInfo};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{command, Emitter, AppHandle, Manager, Runtime};

pub static TONE_ENGINE: Mutex<Option<Arc<ToneEngine>>> = Mutex::new(None);
static MODELS_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

pub fn set_models_directory<R: Runtime>(app: &AppHandle<R>) {
    let models_dir = app
        .path()
        .app_data_dir()
        .expect("Failed to get app data dir")
        .join("models");

    if !models_dir.exists() {
        let _ = std::fs::create_dir_all(&models_dir);
    }

    log::info!("T-One models directory: {}", models_dir.display());
    *MODELS_DIR.lock().unwrap() = Some(models_dir);
}

fn get_models_directory() -> Option<PathBuf> {
    MODELS_DIR.lock().unwrap().clone()
}

#[command]
pub async fn tone_init() -> Result<(), String> {
    let mut guard = TONE_ENGINE.lock().unwrap();
    if guard.is_some() {
        return Ok(());
    }
    let engine = ToneEngine::new_with_models_dir(get_models_directory())
        .map_err(|e| format!("Failed to init T-One engine: {}", e))?;
    *guard = Some(Arc::new(engine));
    Ok(())
}

#[command]
pub async fn tone_get_available_models() -> Result<Vec<ToneModelInfo>, String> {
    let engine = { TONE_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => e
            .discover_models()
            .await
            .map_err(|e| format!("Failed to discover T-One models: {}", e)),
        None => Err("T-One engine not initialized".into()),
    }
}

#[command]
pub async fn tone_load_model<R: Runtime>(
    app_handle: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let engine = { TONE_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => {
            let _ = app_handle.emit(
                "tone-model-loading-started",
                serde_json::json!({ "modelName": model_name }),
            );
            let result = e
                .load_model(&model_name)
                .await
                .map_err(|e| format!("Failed to load T-One model: {}", e));

            if result.is_ok() {
                let _ = app_handle.emit(
                    "tone-model-loading-completed",
                    serde_json::json!({ "modelName": model_name }),
                );
            } else if let Err(ref err) = result {
                let _ = app_handle.emit(
                    "tone-model-loading-failed",
                    serde_json::json!({ "modelName": model_name, "error": err }),
                );
            }
            result
        }
        None => Err("T-One engine not initialized".into()),
    }
}

#[command]
pub async fn tone_get_current_model() -> Result<Option<String>, String> {
    let engine = { TONE_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => Ok(e.get_current_model().await),
        None => Err("T-One engine not initialized".into()),
    }
}

#[command]
pub async fn tone_is_model_loaded() -> Result<bool, String> {
    let engine = { TONE_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => Ok(e.is_model_loaded().await),
        None => Ok(false),
    }
}

#[command]
pub async fn tone_has_available_models() -> Result<bool, String> {
    let engine = { TONE_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => {
            let models = e
                .discover_models()
                .await
                .map_err(|e| format!("Failed to discover: {}", e))?;
            Ok(models
                .iter()
                .any(|m| matches!(m.status, crate::tone_engine::ToneModelStatus::Available)))
        }
        None => Ok(false),
    }
}

#[command]
pub async fn tone_transcribe_audio(audio_data: Vec<f32>) -> Result<String, String> {
    let engine = { TONE_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => e
            .transcribe_audio(audio_data)
            .await
            .map_err(|e| format!("T-One transcription failed: {}", e)),
        None => Err("T-One engine not initialized".into()),
    }
}

#[command]
pub async fn tone_get_models_directory() -> Result<String, String> {
    let engine = { TONE_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => Ok(e.get_models_directory().await.to_string_lossy().to_string()),
        None => Err("T-One engine not initialized".into()),
    }
}
