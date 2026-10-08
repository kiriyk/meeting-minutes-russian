//! Global VAD engine selection shared by live recording, import and retranscription.

use super::classifier::{EarshotClassifier, FrameClassifier, SileroV6Classifier};
use anyhow::Result;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VadEngine {
    #[default]
    SileroV6,
    Earshot,
}

static SELECTED: AtomicU8 = AtomicU8::new(0);

fn to_bits(engine: VadEngine) -> u8 {
    match engine {
        VadEngine::SileroV6 => 0,
        VadEngine::Earshot => 1,
    }
}

fn from_bits(bits: u8) -> VadEngine {
    match bits {
        1 => VadEngine::Earshot,
        _ => VadEngine::SileroV6,
    }
}

pub fn selected_vad_engine() -> VadEngine {
    from_bits(SELECTED.load(Ordering::SeqCst))
}

pub fn set_selected_vad_engine(engine: VadEngine) {
    SELECTED.store(to_bits(engine), Ordering::SeqCst);
}

/// Builds the requested classifier; Silero failures fall back to Earshot, which needs no ONNX Runtime.
pub fn create_classifier(engine: VadEngine) -> (Box<dyn FrameClassifier>, VadEngine) {
    resolve_classifier(engine, || Ok(Box::new(SileroV6Classifier::new()?)))
}

fn resolve_classifier(
    engine: VadEngine,
    silero: impl FnOnce() -> Result<Box<dyn FrameClassifier>>,
) -> (Box<dyn FrameClassifier>, VadEngine) {
    if engine == VadEngine::SileroV6 {
        match silero() {
            Ok(classifier) => return (classifier, VadEngine::SileroV6),
            Err(e) => log::warn!("Silero VAD v6 unavailable, using Earshot: {e:#}"),
        }
    }
    (Box::new(EarshotClassifier::new()), VadEngine::Earshot)
}

#[tauri::command]
pub async fn set_vad_engine(engine: VadEngine) -> Result<(), String> {
    log::info!("VAD engine set to {engine:?} (applies to the next recording or batch job)");
    set_selected_vad_engine(engine);
    Ok(())
}

#[tauri::command]
pub async fn get_vad_engine() -> VadEngine {
    selected_vad_engine()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_names_are_stable() {
        assert_eq!(serde_json::to_string(&VadEngine::SileroV6).unwrap(), "\"silero_v6\"");
        assert_eq!(serde_json::from_str::<VadEngine>("\"earshot\"").unwrap(), VadEngine::Earshot);
        assert!(serde_json::from_str::<VadEngine>("\"silero_v4\"").is_err());
    }

    // Does not touch the global: other tests build processors in parallel.
    #[test]
    fn stored_bits_round_trip_and_default_to_silero() {
        assert_eq!(VadEngine::default(), VadEngine::SileroV6);
        for engine in [VadEngine::SileroV6, VadEngine::Earshot] {
            assert_eq!(from_bits(to_bits(engine)), engine);
        }
        assert_eq!(from_bits(0), VadEngine::SileroV6);
        assert_eq!(from_bits(200), VadEngine::SileroV6);
    }

    #[test]
    fn earshot_selection_builds_earshot() {
        let (classifier, used) = create_classifier(VadEngine::Earshot);
        assert_eq!((used, classifier.frame_len()), (VadEngine::Earshot, 256));
    }

    #[test]
    fn unavailable_silero_falls_back_to_earshot() {
        let (classifier, used) = resolve_classifier(VadEngine::SileroV6, || anyhow::bail!("ONNX Runtime missing"));
        assert_eq!((used, classifier.frame_len()), (VadEngine::Earshot, 256));
    }
}
