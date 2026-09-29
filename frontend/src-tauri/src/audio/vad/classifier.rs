//! Frame-level speech classifiers. Each returns a speech probability per fixed-size 16 kHz frame.

use anyhow::{bail, Result};

pub trait FrameClassifier: Send {
    /// Exact frame length in 16 kHz samples that `predict` accepts.
    fn frame_len(&self) -> usize;
    /// Speech probability in [0, 1] for one frame of exactly `frame_len()` samples.
    fn predict(&mut self, frame: &[f32]) -> Result<f32>;
}

pub const EARSHOT_FRAME: usize = 256;

pub struct EarshotClassifier {
    detector: Box<earshot::Detector>,
    scratch: Vec<f32>,
}

impl EarshotClassifier {
    pub fn new() -> Self {
        Self {
            detector: earshot::Detector::default_boxed(),
            scratch: Vec::with_capacity(EARSHOT_FRAME),
        }
    }
}

impl FrameClassifier for EarshotClassifier {
    fn frame_len(&self) -> usize {
        EARSHOT_FRAME
    }

    fn predict(&mut self, frame: &[f32]) -> Result<f32> {
        if frame.len() != EARSHOT_FRAME {
            bail!("Earshot expects {EARSHOT_FRAME} samples, got {}", frame.len());
        }
        // Earshot requires [-1, 1]; mixed recordings can slightly exceed it.
        self.scratch.clear();
        self.scratch.extend(frame.iter().map(|s| s.clamp(-1.0, 1.0)));
        Ok(self.detector.predict_f32(&self.scratch).clamp(0.0, 1.0))
    }
}

pub const SILERO_FRAME: usize = 512;

/// Silero VAD v6.2 via the `silero` crate (bundled, verified model).
pub struct SileroV6Classifier {
    session: silero::Session,
    stream: silero::StreamState,
}

fn silero_ort_session() -> ort::Result<ort::session::Session> {
    use ort::session::builder::GraphOptimizationLevel;
    // One thread: the model is tiny and runs next to ASR inference.
    Ok(ort::session::Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_memory(silero::BUNDLED_MODEL)?)
}

impl SileroV6Classifier {
    pub fn new() -> Result<Self> {
        crate::ensure_onnx_runtime_available()?;
        Ok(Self {
            session: silero::Session::from_ort_session(silero_ort_session()?),
            stream: silero::StreamState::new(silero::SampleRate::Rate16k),
        })
    }
}

impl FrameClassifier for SileroV6Classifier {
    fn frame_len(&self) -> usize {
        SILERO_FRAME
    }

    fn predict(&mut self, frame: &[f32]) -> Result<f32> {
        Ok(self.session.infer_chunk(&mut self.stream, frame)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earshot_frame_is_16ms() {
        assert_eq!(EarshotClassifier::new().frame_len(), 256);
    }

    #[test]
    fn earshot_scores_silence_low() {
        let mut vad = EarshotClassifier::new();
        for _ in 0..200 {
            let p = vad.predict(&[0.0; 256]).unwrap();
            assert!((0.0..0.5).contains(&p), "silence scored {p}");
        }
    }

    #[test]
    fn earshot_rejects_wrong_frame_length() {
        assert!(EarshotClassifier::new().predict(&[0.0; 255]).is_err());
    }

    #[test]
    fn earshot_clamps_out_of_range_samples() {
        let p = EarshotClassifier::new().predict(&[1.5; 256]).unwrap();
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn silero_v6_frame_is_32ms() {
        assert_eq!(SileroV6Classifier::new().unwrap().frame_len(), 512);
    }

    #[test]
    fn silero_v6_scores_silence_low_and_rejects_bad_frames() {
        let mut vad = SileroV6Classifier::new().unwrap();
        for _ in 0..100 {
            let p = vad.predict(&[0.0; 512]).unwrap();
            assert!((0.0..0.5).contains(&p), "silence scored {p}");
        }
        assert!(vad.predict(&[0.0; 480]).is_err());
    }

    #[test]
    fn bundled_silero_model_is_v6_2() {
        use sha2::{Digest, Sha256};
        let digest = format!("{:x}", Sha256::digest(silero::BUNDLED_MODEL));
        assert_eq!(digest, "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3");
    }

    #[test]
    fn onnx_runtime_reports_build_info() {
        // Two static ONNX Runtimes (ort + sherpa-onnx) share one binary on macOS/Linux.
        // Print which one answers so a mismatch is visible in CI logs.
        let _ = SileroV6Classifier::new().unwrap();
        let info = ort::info();
        println!("ONNX Runtime: {info}");
        assert!(!info.is_empty());
    }

    /// Real speech check for both engines; run with a WAV/MP3/MP4 containing speech.
    #[test]
    #[ignore]
    fn vad_engines_real_audio_smoke() {
        let path = std::env::var("MEETILY_VAD_AUDIO_FILE").unwrap();
        let audio = crate::audio::decoder::decode_audio_file(std::path::Path::new(&path)).unwrap().to_whisper_format();
        for (name, classifier) in [
            ("silero_v6", Box::new(SileroV6Classifier::new().unwrap()) as Box<dyn FrameClassifier>),
            ("earshot", Box::new(EarshotClassifier::new())),
        ] {
            let started = std::time::Instant::now();
            let mut p = crate::audio::vad::ContinuousVadProcessor::with_classifier(16_000, 2000, classifier);
            let mut segments = p.process_audio(&audio).unwrap();
            segments.extend(p.flush().unwrap());
            let speech: f64 = segments.iter().map(|s| s.end_timestamp_ms - s.start_timestamp_ms).sum();
            println!("{name}: {} segments, {:.1}s speech of {:.1}s, {:.2}s", segments.len(), speech / 1000.0, audio.len() as f64 / 16_000.0, started.elapsed().as_secs_f64());
            assert!(!segments.is_empty(), "{name} found no speech");
        }
    }
}
