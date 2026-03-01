use ndarray::{Array2, Array3, ArrayD};
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::fs;
use std::path::Path;

// T-One CTC streaming constants
// Input: signal [batch, CHUNK_SAMPLES, 1] as int32 (8 kHz raw PCM scaled to int32 range)
// State: [batch, STATE_SIZE] as float16
// Output: logprobs [batch, FRAMES_PER_CHUNK, VOCAB_SIZE], state_next
const CHUNK_SAMPLES: usize = 2400; // 300ms @ 8 kHz
const STATE_SIZE: usize = 219729;
const FRAMES_PER_CHUNK: usize = 10;
const BLANK_IDX: usize = 34; // [PAD] token = CTC blank
// T-One records at 8 kHz; Meetily records at 16 kHz → downsample by 2
const DOWNSAMPLE_FACTOR: usize = 2;

#[derive(thiserror::Error, Debug)]
pub enum ToneError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
}

pub struct ToneModel {
    session: Session,
    /// vocab[id] = char string; blank = vocab[BLANK_IDX] = "[PAD]"
    vocab: Vec<String>,
}

impl ToneModel {
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self, ToneError> {
        let dir = model_dir.as_ref();
        let providers = vec![CPUExecutionProvider::default().build()];

        let onnx_path = dir.join("model.onnx");
        log::info!("Loading T-One ONNX: {}", onnx_path.display());
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_parallel_execution(true)?
            .commit_from_file(onnx_path)?;

        let vocab_path = dir.join("vocab.json");
        log::info!("Loading T-One vocab: {}", vocab_path.display());
        let raw = fs::read_to_string(&vocab_path)?;
        let vocab: Vec<String> = serde_json::from_str(&raw)?;
        log::info!("T-One vocab size={} blank_idx={}", vocab.len(), BLANK_IDX);

        Ok(Self { session, vocab })
    }

    // ----- audio preprocessing -----------------------------------------------

    /// Downsample from 16 kHz to 8 kHz by taking every 2nd sample, then return
    /// chunks of CHUNK_SAMPLES each (zero-padded for last chunk).
    fn to_8khz_chunks(samples_16khz: &[f32]) -> Vec<Vec<f32>> {
        let samples_8khz: Vec<f32> = samples_16khz
            .iter()
            .step_by(DOWNSAMPLE_FACTOR)
            .cloned()
            .collect();

        let mut chunks = Vec::new();
        let mut offset = 0;
        loop {
            if offset >= samples_8khz.len() {
                break;
            }
            let end = (offset + CHUNK_SAMPLES).min(samples_8khz.len());
            let mut chunk = samples_8khz[offset..end].to_vec();
            if chunk.len() < CHUNK_SAMPLES {
                chunk.resize(CHUNK_SAMPLES, 0.0);
            }
            chunks.push(chunk);
            offset += CHUNK_SAMPLES;
        }
        chunks
    }

    /// Convert float32 samples [-1.0, 1.0] to int32 PCM range.
    /// T-One expects raw int32 PCM values (16-bit range scaled to i32).
    fn float_to_int32_pcm(samples: &[f32]) -> Vec<i32> {
        samples
            .iter()
            .map(|&s| (s * i16::MAX as f32) as i32)
            .collect()
    }

    // ----- inference ----------------------------------------------------------

    fn run_chunk(
        &mut self,
        chunk_f32: &[f32],
        state: &Array2<half::f16>,
    ) -> Result<(Array2<f32>, Array2<half::f16>), ToneError> {
        // signal: [1, CHUNK_SAMPLES, 1] int32
        let pcm: Vec<i32> = Self::float_to_int32_pcm(chunk_f32);
        let signal =
            Array3::<i32>::from_shape_vec((1, CHUNK_SAMPLES, 1), pcm)?.into_dyn();

        let inputs = inputs![
            "signal" => TensorRef::from_array_view(signal.view())?,
            "state"  => TensorRef::from_array_view(state.view().into_dyn())?,
        ];
        let outputs = self.session.run(inputs)?;

        // logprobs: [1, FRAMES_PER_CHUNK, VOCAB_SIZE]
        let logprobs: ArrayD<f32> = outputs
            .get("logprobs")
            .ok_or_else(|| ToneError::OutputNotFound("logprobs".into()))?
            .try_extract_array()?
            .to_owned();

        let state_next_raw: ArrayD<half::f16> = outputs
            .get("state_next")
            .ok_or_else(|| ToneError::OutputNotFound("state_next".into()))?
            .try_extract_array()?
            .to_owned();

        // Reshape to [FRAMES_PER_CHUNK, VOCAB_SIZE] and [1, STATE_SIZE]
        let frames = logprobs.into_dimensionality::<ndarray::Ix3>()?;
        let frames_2d = frames
            .slice(ndarray::s![0, .., ..])
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()?;

        let state_2d = state_next_raw
            .into_dimensionality::<ndarray::Ix2>()?;

        Ok((frames_2d, state_2d))
    }

    // ----- CTC greedy decode --------------------------------------------------

    /// argmax per frame → collapse adjacent same → remove blank
    fn ctc_decode(&self, frame_logprobs: &[ndarray::Array2<f32>]) -> String {
        let mut token_ids: Vec<usize> = Vec::new();
        for frame in frame_logprobs {
            let best = frame
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(BLANK_IDX);
            token_ids.push(best);
        }

        // Collapse repeats & remove blank
        let mut prev = BLANK_IDX + 1; // impossible start
        let mut result = String::new();
        for id in token_ids {
            if id != prev && id != BLANK_IDX {
                if id == 33 {
                    // '|' = word boundary → space
                    result.push(' ');
                } else if id < self.vocab.len() {
                    result.push_str(&self.vocab[id]);
                }
            }
            prev = id;
        }
        result.trim().to_string()
    }

    // ----- public API ---------------------------------------------------------

    pub fn transcribe_samples(&mut self, samples_16khz: Vec<f32>) -> Result<String, ToneError> {
        let chunks = Self::to_8khz_chunks(&samples_16khz);
        if chunks.is_empty() {
            return Ok(String::new());
        }

        log::debug!("T-One: {} chunks to process", chunks.len());

        // Initial state: zeros (float16)
        let mut state =
            Array2::<half::f16>::zeros((1, STATE_SIZE));

        let mut all_frames: Vec<ndarray::Array2<f32>> = Vec::new();

        for chunk in &chunks {
            let (frames, next_state) = self.run_chunk(chunk, &state)?;
            all_frames.push(frames);
            state = next_state;
        }

        let text = self.ctc_decode(&all_frames);
        log::debug!("T-One decoded: {:?}", text);
        Ok(text)
    }
}
