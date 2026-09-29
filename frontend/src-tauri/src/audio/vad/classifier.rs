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
