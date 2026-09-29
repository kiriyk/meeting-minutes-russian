use anyhow::{bail, Result};
use std::sync::Arc;
use tauri::{AppHandle, Runtime};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Provider {
    Whisper,
    Parakeet,
    GigaAm,
    Tone,
}

impl Provider {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("whisper") {
            "whisper" | "localWhisper" => Ok(Self::Whisper),
            "parakeet" => Ok(Self::Parakeet),
            "gigaam" => Ok(Self::GigaAm),
            "tone" | "t_one" => Ok(Self::Tone),
            other => bail!("Unsupported retranscription provider: {other}"),
        }
    }

    pub async fn unload(self) {
        match self {
            Self::Whisper | Self::Parakeet => {
                super::common::unload_engine_after_batch(self == Self::Parakeet).await
            }
            Self::GigaAm | Self::Tone => {
                let _guard = super::common::acquire_engine_lifecycle_lock().await;
                if super::recording_commands::is_recording().await {
                    return;
                }
                if self == Self::GigaAm {
                    let engine = crate::gigaam_engine::commands::GIGAAM_ENGINE
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone();
                    if let Some(engine) = engine {
                        engine.unload_model().await;
                    }
                } else {
                    let engine = crate::tone_engine::commands::TONE_ENGINE
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone();
                    if let Some(engine) = engine {
                        engine.unload_model().await;
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
pub(super) enum BatchEngine {
    Whisper(Arc<crate::whisper_engine::WhisperEngine>),
    Parakeet(Arc<crate::parakeet_engine::ParakeetEngine>),
    GigaAm(Arc<crate::gigaam_engine::GigaAmEngine>),
    Tone(Arc<crate::tone_engine::ToneEngine>),
}

impl BatchEngine {
    pub async fn load<R: Runtime>(
        app: &AppHandle<R>,
        provider: Provider,
        model: Option<&str>,
    ) -> Result<Self> {
        Ok(match provider {
            Provider::Whisper => {
                Self::Whisper(super::retranscription::get_or_init_whisper(app, model).await?)
            }
            Provider::Parakeet => {
                Self::Parakeet(super::retranscription::get_or_init_parakeet(app, model).await?)
            }
            Provider::GigaAm => {
                crate::gigaam_engine::commands::gigaam_init()
                    .await
                    .map_err(anyhow::Error::msg)?;
                let engine = crate::gigaam_engine::commands::GIGAAM_ENGINE
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("GigaAM engine is not initialized"))?;
                engine.discover_models().await?;
                engine
                    .load_model(model.unwrap_or("gigaam-v3-e2e-rnnt"))
                    .await?;
                Self::GigaAm(engine)
            }
            Provider::Tone => {
                crate::tone_engine::commands::tone_init()
                    .await
                    .map_err(anyhow::Error::msg)?;
                let engine = crate::tone_engine::commands::TONE_ENGINE
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("T-one engine is not initialized"))?;
                engine.discover_models().await?;
                engine.load_model(model.unwrap_or("t-one")).await?;
                Self::Tone(engine)
            }
        })
    }

    /// Invoke inside a blocking worker: the engine's async API runs CPU inference synchronously.
    pub async fn transcribe(
        &self,
        samples: Vec<f32>,
        language: Option<String>,
    ) -> Result<(String, f32)> {
        Ok(match self {
            Self::Whisper(engine) => {
                let (text, confidence, _) = engine
                    .transcribe_audio_with_confidence(samples, language)
                    .await?;
                (text, confidence)
            }
            Self::Parakeet(engine) => (engine.transcribe_audio(samples).await?, 0.9),
            Self::GigaAm(engine) => (engine.transcribe_audio(samples).await?, 0.9),
            Self::Tone(engine) => (engine.transcribe_audio(samples).await?, 0.9),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    #[ignore = "requires downloaded Russian ASR models and a speech fixture"]
    async fn russian_engines_smoke() {
        let dir = std::path::PathBuf::from(std::env::var("RUSSIAN_ASR_MODELS_DIR").unwrap());
        let audio = std::path::PathBuf::from(std::env::var("RUSSIAN_ASR_AUDIO_FILE").unwrap());
        tokio::task::spawn_blocking(move || {
            tauri::async_runtime::block_on(async move {
                let samples = crate::audio::decoder::decode_audio_file(&audio)
                    .unwrap()
                    .to_whisper_format();
                let gigaam = Arc::new(
                    crate::gigaam_engine::GigaAmEngine::new_with_models_dir(Some(dir.clone()))
                        .unwrap(),
                );
                gigaam.discover_models().await.unwrap();
                gigaam.load_model("gigaam-v3-e2e-rnnt").await.unwrap();
                let (text, _) = BatchEngine::GigaAm(gigaam.clone())
                    .transcribe(samples.clone(), Some("ru".into()))
                    .await
                    .unwrap();
                println!("GigaAM: {text:?}");
                let gigaam_text = text;
                gigaam.unload_model().await;
                assert!(!gigaam.is_model_loaded().await);
                let tone = Arc::new(
                    crate::tone_engine::ToneEngine::new_with_models_dir(Some(dir)).unwrap(),
                );
                tone.discover_models().await.unwrap();
                tone.load_model("t-one").await.unwrap();
                let (text, _) = BatchEngine::Tone(tone.clone())
                    .transcribe(samples, Some("ru".into()))
                    .await
                    .unwrap();
                println!("T-one: {text:?}");
                assert!(text.chars().any(|ch| ('а'..='я').contains(&ch)));
                assert!(gigaam_text.chars().any(|ch| ('а'..='я').contains(&ch)));
                tone.unload_model().await;
                assert!(!tone.is_model_loaded().await);
            })
        })
        .await
        .unwrap();
    }
    #[test]
    fn providers_route_explicitly_without_whisper_fallback() {
        assert_eq!(Provider::parse(Some("gigaam")).unwrap(), Provider::GigaAm);
        assert_eq!(Provider::parse(Some("tone")).unwrap(), Provider::Tone);
        assert_eq!(Provider::parse(None).unwrap(), Provider::Whisper);
        assert!(Provider::parse(Some("russianAsr")).is_err());
    }
}
