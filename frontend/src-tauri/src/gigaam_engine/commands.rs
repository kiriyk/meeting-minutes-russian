use crate::gigaam_engine::engine::{GigaAmEngine, GigaAmModelInfo};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{command, Emitter, AppHandle, Manager, Runtime};

pub static GIGAAM_ENGINE: Mutex<Option<Arc<GigaAmEngine>>> = Mutex::new(None);
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

    log::info!("GigaAM models directory: {}", models_dir.display());
    *MODELS_DIR.lock().unwrap() = Some(models_dir);
}

fn get_models_directory() -> Option<PathBuf> {
    MODELS_DIR.lock().unwrap().clone()
}

#[command]
pub async fn gigaam_init() -> Result<(), String> {
    let mut guard = GIGAAM_ENGINE.lock().unwrap();
    if guard.is_some() {
        return Ok(());
    }
    let engine = GigaAmEngine::new_with_models_dir(get_models_directory())
        .map_err(|e| format!("Failed to init GigaAM engine: {}", e))?;
    *guard = Some(Arc::new(engine));
    Ok(())
}

#[command]
pub async fn gigaam_get_available_models() -> Result<Vec<GigaAmModelInfo>, String> {
    let engine = { GIGAAM_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => e
            .discover_models()
            .await
            .map_err(|e| format!("Failed to discover GigaAM models: {}", e)),
        None => Err("GigaAM engine not initialized".into()),
    }
}

#[command]
pub async fn gigaam_load_model<R: Runtime>(
    app_handle: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let engine = { GIGAAM_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => {
            let _ = app_handle.emit(
                "gigaam-model-loading-started",
                serde_json::json!({ "modelName": model_name }),
            );
            let result = e
                .load_model(&model_name)
                .await
                .map_err(|e| format!("Failed to load GigaAM model: {}", e));

            if result.is_ok() {
                let _ = app_handle.emit(
                    "gigaam-model-loading-completed",
                    serde_json::json!({ "modelName": model_name }),
                );
            } else if let Err(ref err) = result {
                let _ = app_handle.emit(
                    "gigaam-model-loading-failed",
                    serde_json::json!({ "modelName": model_name, "error": err }),
                );
            }
            result
        }
        None => Err("GigaAM engine not initialized".into()),
    }
}

#[command]
pub async fn gigaam_get_current_model() -> Result<Option<String>, String> {
    let engine = { GIGAAM_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => Ok(e.get_current_model().await),
        None => Err("GigaAM engine not initialized".into()),
    }
}

#[command]
pub async fn gigaam_is_model_loaded() -> Result<bool, String> {
    let engine = { GIGAAM_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => Ok(e.is_model_loaded().await),
        None => Ok(false),
    }
}

#[command]
pub async fn gigaam_has_available_models() -> Result<bool, String> {
    let engine = { GIGAAM_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => {
            let models = e
                .discover_models()
                .await
                .map_err(|e| format!("Failed to discover: {}", e))?;
            Ok(models
                .iter()
                .any(|m| matches!(m.status, crate::gigaam_engine::GigaAmModelStatus::Available)))
        }
        None => Ok(false),
    }
}

#[command]
pub async fn gigaam_transcribe_audio(audio_data: Vec<f32>) -> Result<String, String> {
    let engine = { GIGAAM_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => e
            .transcribe_audio(audio_data)
            .await
            .map_err(|e| format!("GigaAM transcription failed: {}", e)),
        None => Err("GigaAM engine not initialized".into()),
    }
}

#[command]
pub async fn gigaam_get_models_directory() -> Result<String, String> {
    let engine = { GIGAAM_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => Ok(e.get_models_directory().await.to_string_lossy().to_string()),
        None => Err("GigaAM engine not initialized".into()),
    }
}

/// Download a GigaAM model from HuggingFace, emitting progress events.
#[command]
pub async fn gigaam_download_model<R: Runtime>(
    app_handle: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let engine = { GIGAAM_ENGINE.lock().unwrap().as_ref().cloned() };
    let engine = engine.ok_or("GigaAM engine not initialized")?;

    let app_for_progress = app_handle.clone();
    let model_name_for_progress = model_name.clone();

    let progress_cb: Box<dyn Fn(u8) + Send + Sync> = Box::new(move |pct: u8| {
        let _ = app_for_progress.emit(
            "gigaam-model-download-progress",
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
                "gigaam-model-download-complete",
                serde_json::json!({ "modelName": model_name }),
            );
        }
        Err(err) => {
            let _ = app_handle.emit(
                "gigaam-model-download-error",
                serde_json::json!({ "modelName": model_name, "error": err }),
            );
        }
    }

    result
}

/// Cancel an in-progress GigaAM model download.
#[command]
pub async fn gigaam_cancel_download(model_name: String) -> Result<(), String> {
    let engine = { GIGAAM_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => {
            e.cancel_download(&model_name).await;
            Ok(())
        }
        None => Err("GigaAM engine not initialized".into()),
    }
}

/// Delete a downloaded GigaAM model to free up disk space.
#[command]
pub async fn gigaam_delete_model(model_name: String) -> Result<String, String> {
    let engine = { GIGAAM_ENGINE.lock().unwrap().as_ref().cloned() };
    match engine {
        Some(e) => e
            .delete_model(&model_name)
            .await
            .map_err(|e| format!("Failed to delete GigaAM model: {}", e)),
        None => Err("GigaAM engine not initialized".into()),
    }
}

/// Auto-select and load first available GigaAM model. Returns model name.
pub async fn gigaam_validate_model_ready_with_config<R: tauri::Runtime>(
    _app: &tauri::AppHandle<R>,
) -> Result<String, String> {
    let engine = { GIGAAM_ENGINE.lock().unwrap().as_ref().cloned() };
    let engine = engine.ok_or("GigaAM engine not initialized")?;

    if engine.is_model_loaded().await {
        if let Some(name) = engine.get_current_model().await {
            return Ok(name);
        }
    }

    let models = engine
        .discover_models()
        .await
        .map_err(|e| format!("Discover failed: {}", e))?;

    let first_available = models
        .iter()
        .find(|m| matches!(m.status, crate::gigaam_engine::GigaAmModelStatus::Available))
        .ok_or("No GigaAM models available. Place model files in the gigaam models directory.")?;

    engine
        .load_model(&first_available.name)
        .await
        .map_err(|e| format!("Failed to load {}: {}", first_available.name, e))?;

    Ok(first_available.name.clone())
}
