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

/// Download a T-One model from HuggingFace, emitting progress events.
#[command]
pub async fn tone_download_model<R: Runtime>(
    app_handle: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let engine = { TONE_ENGINE.lock().unwrap().as_ref().cloned() };
    let engine = engine.ok_or("T-One engine not initialized")?;

    let app_for_progress = app_handle.clone();
    let model_name_for_progress = model_name.clone();

    let progress_cb: Box<dyn Fn(u8) + Send + Sync> = Box::new(move |pct: u8| {
        let _ = app_for_progress.emit(
            "tone-model-download-progress",
            serde_json::json!({ "modelName": model_name_for_progress, "progress": pct }),
        );
    });

    let result = engine
        .download_model(&model_name, Some(progress_cb))
        .await
        .map_err(|e| e.to_string());

    match &result {
        Ok(()) => {
            let _ = app_handle.emit(
                "tone-model-download-complete",
                serde_json::json!({ "modelName": model_name }),
            );
        }
        Err(err) => {
            let _ = app_handle.emit(
                "tone-model-download-error",
                serde_json::json!({ "modelName": model_name, "error": err }),
            );
        }
    }

    result
}

/// Cancel an in-progress T-One model download.
#[command]
pub async fn tone_cancel_download(model_name: String) -> Result<(), String> {
    let engine = { TONE_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => {
            e.cancel_download(&model_name).await;
            Ok(())
        }
        None => Err("T-One engine not initialized".into()),
    }
}

/// Delete a downloaded T-One model to free up disk space.
#[command]
pub async fn tone_delete_model(model_name: String) -> Result<String, String> {
    let engine = { TONE_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => e
            .delete_model(&model_name)
            .await
            .map_err(|e| format!("Failed to delete T-One model: {}", e)),
        None => Err("T-One engine not initialized".into()),
    }
}

/// Auto-select and load an available T-One model. Returns loaded model name.
/// If `preferred_model_name` is provided and available, it is loaded first.
pub async fn tone_validate_model_ready_with_config(
    preferred_model_name: Option<&str>,
) -> Result<String, String> {
    let engine = { TONE_ENGINE.lock().unwrap().as_ref().cloned() };
    let engine = engine.ok_or("T-One engine not initialized")?;

    if engine.is_model_loaded().await {
        if let Some(name) = engine.get_current_model().await {
            return Ok(name);
        }
    }

    let models = engine
        .discover_models()
        .await
        .map_err(|e| format!("Discover failed: {}", e))?;

    let available_models: Vec<&ToneModelInfo> = models
        .iter()
        .filter(|m| matches!(m.status, crate::tone_engine::ToneModelStatus::Available))
        .collect();

    let chosen = if let Some(preferred) = preferred_model_name {
        available_models
            .iter()
            .find(|m| m.name == preferred)
            .copied()
            .or_else(|| available_models.first().copied())
    } else {
        available_models.first().copied()
    }
    .ok_or("No T-One models available. Place model files in the tone models directory.")?;

    engine
        .load_model(&chosen.name)
        .await
        .map_err(|e| format!("Failed to load {}: {}", chosen.name, e))?;

    Ok(chosen.name.clone())
}
