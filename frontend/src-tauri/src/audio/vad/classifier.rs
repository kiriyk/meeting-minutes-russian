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

    /// Minimal RIFF/WAV parser for the committed 16 kHz mono PCM16 fixture.
    /// Scans chunks rather than assuming a fixed header layout, but only
    /// understands the PCM16 case the fixture is committed in.
    fn parse_16k_mono_pcm16_wav(bytes: &[u8]) -> Vec<f32> {
        assert_eq!(&bytes[0..4], b"RIFF", "not a RIFF file");
        assert_eq!(&bytes[8..12], b"WAVE", "not a WAVE file");
        let mut pos = 12;
        let mut channels = 0u16;
        let mut sample_rate = 0u32;
        let mut bits_per_sample = 0u16;
        let mut data: Option<&[u8]> = None;
        while pos + 8 <= bytes.len() {
            let chunk_id = &bytes[pos..pos + 4];
            let chunk_len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            let chunk_start = pos + 8;
            let chunk_end = chunk_start + chunk_len;
            match chunk_id {
                b"fmt " => {
                    channels = u16::from_le_bytes(bytes[chunk_start + 2..chunk_start + 4].try_into().unwrap());
                    sample_rate = u32::from_le_bytes(bytes[chunk_start + 4..chunk_start + 8].try_into().unwrap());
                    bits_per_sample = u16::from_le_bytes(bytes[chunk_start + 14..chunk_start + 16].try_into().unwrap());
                }
                b"data" => data = Some(&bytes[chunk_start..chunk_end]),
                _ => {}
            }
            pos = chunk_end + (chunk_len % 2); // chunks are word-aligned
        }
        assert_eq!(channels, 1, "fixture must be mono");
        assert_eq!(sample_rate, 16_000, "fixture must be 16 kHz");
        assert_eq!(bits_per_sample, 16, "fixture must be PCM16");
        let data = data.expect("no data chunk found");
        data.chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / i16::MAX as f32)
            .collect()
    }

    /// Both selectable VAD engines must find real speech in a real recording.
    /// This is the regression guard the ignored smoke test below can't provide:
    /// it runs on every `cargo test` and would fail if a default-path regression
    /// silently produced empty transcripts.
    #[test]
    fn both_engines_find_speech_in_real_recording() {
        let audio = parse_16k_mono_pcm16_wav(include_bytes!("testdata/jfk-speech-4s.wav"));
        let audio_duration_ms = audio.len() as f64 * 1000.0 / 16_000.0;
        assert!((3_999.0..=4_001.0).contains(&audio_duration_ms), "fixture duration {audio_duration_ms}ms");

        for (name, classifier) in [
            ("silero_v6", Box::new(SileroV6Classifier::new().unwrap()) as Box<dyn FrameClassifier>),
            ("earshot", Box::new(EarshotClassifier::new())),
        ] {
            let mut p = crate::audio::vad::ContinuousVadProcessor::with_classifier(16_000, 2000, classifier);
            let mut segments = p.process_audio(&audio).unwrap();
            segments.extend(p.flush().unwrap());
            let speech_ms: f64 = segments.iter().map(|s| s.end_timestamp_ms - s.start_timestamp_ms).sum();
            println!(
                "{name}: {} segments, {:.1}s speech of {:.1}s",
                segments.len(),
                speech_ms / 1000.0,
                audio_duration_ms / 1000.0
            );
            assert!(!segments.is_empty(), "{name} found no speech in real recording");
            assert!(speech_ms >= 1_500.0, "{name} only found {speech_ms:.0}ms of speech, expected >= 1500ms");
            for s in &segments {
                assert!(
                    s.start_timestamp_ms >= 0.0 && s.end_timestamp_ms <= audio_duration_ms,
                    "{name} segment [{:.0}, {:.0}]ms outside audio bounds [0, {audio_duration_ms:.0}]ms",
                    s.start_timestamp_ms,
                    s.end_timestamp_ms
                );
            }
        }
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
