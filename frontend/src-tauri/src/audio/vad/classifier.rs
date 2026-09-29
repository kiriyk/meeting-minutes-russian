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

/// Temporary adapter kept only until the ort rc.12 switch (Task 4 deletes it).
pub struct SileroV4Classifier {
    session: silero_rs::VadSession,
}

impl SileroV4Classifier {
    pub fn new() -> Result<Self> {
        crate::ensure_onnx_runtime_available()?;
        let config = silero_rs::VadConfig { sample_rate: 16_000, ..Default::default() };
        Ok(Self { session: silero_rs::VadSession::new(config)? })
    }
}

impl FrameClassifier for SileroV4Classifier {
    fn frame_len(&self) -> usize {
        512
    }

    fn predict(&mut self, frame: &[f32]) -> Result<f32> {
        let output = self.session.forward(frame.to_vec())?;
        let probs = output.try_extract_array::<f32>()?;
        probs.first().copied().ok_or_else(|| anyhow::anyhow!("Silero returned no probability"))
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
}
