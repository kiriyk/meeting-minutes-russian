use ndarray::{s, Array1, Array2, Array3, ArrayD};
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::fs;
use std::path::Path;

// GigaAM v3 e2e RNNT constants
// Vocabulary: ids 0..1023 are real tokens, id 1024 = blank
const BLANK_IDX: i32 = 1024;
const MAX_TOKENS_PER_STEP: usize = 5;
const PRED_HIDDEN: usize = 320;
const ENC_HIDDEN: usize = 768;

/// LSTM state for GigaAM RNN-T decoder: (h, c) both shape [1, 1, PRED_HIDDEN]
pub type GigaAmDecoderState = (Array3<f32>, Array3<f32>);

#[derive(thiserror::Error, Debug)]
pub enum GigaAmError {
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

pub struct GigaAmModel {
    encoder: Session,
    decoder: Session,
    joint: Session,
    /// vocab[id] = piece string (SentencePiece ▁ → space replacement done at decode time)
    vocab: Vec<String>,
}

impl GigaAmModel {
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self, GigaAmError> {
        let dir = model_dir.as_ref();
        let providers = vec![CPUExecutionProvider::default().build()];

        let make_session = |name: &str| -> Result<Session, GigaAmError> {
            let path = dir.join(format!("{}.onnx", name));
            log::info!("Loading GigaAM ONNX: {}", path.display());
            Ok(Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers(providers.clone())?
                .with_parallel_execution(true)?
                .commit_from_file(path)?)
        };

        let encoder = make_session("encoder")?;
        let decoder = make_session("decoder")?;
        let joint = make_session("joint")?;

        // Load vocab.json — a JSON array of piece strings (index = token id)
        let vocab_path = dir.join("vocab.json");
        log::info!("Loading GigaAM vocab: {}", vocab_path.display());
        let raw = fs::read_to_string(&vocab_path)?;
        let vocab: Vec<String> = serde_json::from_str(&raw)?;
        log::info!("GigaAM vocab size={} (blank={})", vocab.len(), BLANK_IDX);

        Ok(Self { encoder, decoder, joint, vocab })
    }

    // ----- encoder ---------------------------------------------------------

    /// Run encoder on raw waveform samples.
    /// inputs:  audio_signal [1, N] (f32), length [1] (i64)
    /// outputs: encoded [1, ENC_HIDDEN, T], encoded_len [1]
    fn encode(&mut self, samples: &[f32]) -> Result<(ArrayD<f32>, usize), GigaAmError> {
        let n = samples.len();
        let audio = Array2::from_shape_vec((1, n), samples.to_vec())?.into_dyn();
        let length = Array1::from_vec(vec![n as i64]).into_dyn();

        let inputs = inputs![
            "audio_signal" => TensorRef::from_array_view(audio.view())?,
            "length"       => TensorRef::from_array_view(length.view())?,
        ];
        let outputs = self.encoder.run(inputs)?;

        let encoded: ArrayD<f32> = outputs
            .get("encoded")
            .ok_or_else(|| GigaAmError::OutputNotFound("encoded".into()))?
            .try_extract_array()?
            .to_owned();
        let encoded_len: ArrayD<i64> = outputs
            .get("encoded_len")
            .ok_or_else(|| GigaAmError::OutputNotFound("encoded_len".into()))?
            .try_extract_array()?
            .to_owned();

        let t = encoded_len.as_slice().unwrap_or(&[0])[0] as usize;
        Ok((encoded, t))
    }

    // ----- decoder ---------------------------------------------------------

    pub fn init_decoder_state() -> GigaAmDecoderState {
        (
            Array3::zeros((1, 1, PRED_HIDDEN)),
            Array3::zeros((1, 1, PRED_HIDDEN)),
        )
    }

    /// inputs: x [1,1] (i64), h.1 [1,1,320], c.1 [1,1,320]
    /// outputs: dec [1,1,320], h [1,1,320], c [1,1,320]
    fn decoder_step(
        &mut self,
        token: i32,
        state: &GigaAmDecoderState,
    ) -> Result<(ArrayD<f32>, GigaAmDecoderState), GigaAmError> {
        let x = Array2::from_shape_vec((1, 1), vec![token as i64])?;

        let inputs = inputs![
            "x"   => TensorRef::from_array_view(x.view())?,
            "h.1" => TensorRef::from_array_view(state.0.view())?,
            "c.1" => TensorRef::from_array_view(state.1.view())?,
        ];
        let outputs = self.decoder.run(inputs)?;

        let dec: ArrayD<f32> = outputs
            .get("dec")
            .ok_or_else(|| GigaAmError::OutputNotFound("dec".into()))?
            .try_extract_array()?
            .to_owned();
        let h: ArrayD<f32> = outputs
            .get("h")
            .ok_or_else(|| GigaAmError::OutputNotFound("h".into()))?
            .try_extract_array()?
            .to_owned();
        let c: ArrayD<f32> = outputs
            .get("c")
            .ok_or_else(|| GigaAmError::OutputNotFound("c".into()))?
            .try_extract_array()?
            .to_owned();

        let h3 = h.into_dimensionality::<ndarray::Ix3>()?;
        let c3 = c.into_dimensionality::<ndarray::Ix3>()?;
        Ok((dec, (h3, c3)))
    }

    // ----- joint -----------------------------------------------------------

    /// inputs: enc [1, ENC_HIDDEN, 1], dec [1, PRED_HIDDEN, 1]
    /// outputs: joint [..., vocab+1] — argmax over last dim
    fn joint_step(
        &mut self,
        enc_frame: &ArrayD<f32>, // shape [1, ENC_HIDDEN, T]; we take frame t
        t: usize,
        dec_out: &ArrayD<f32>,   // shape [1, 1, PRED_HIDDEN]
    ) -> Result<i32, GigaAmError> {
        // Slice enc frame: [ENC_HIDDEN] → reshape to [1, ENC_HIDDEN, 1]
        let enc_slice = enc_frame.slice(s![0, .., t]).to_owned(); // [ENC_HIDDEN]
        let enc_3d = enc_slice
            .into_shape_with_order((1, ENC_HIDDEN, 1))?
            .into_dyn();

        // dec_out [1, 1, PRED_HIDDEN] → [1, PRED_HIDDEN, 1]
        let dec_slice = dec_out.slice(s![0, 0, ..]).to_owned(); // [PRED_HIDDEN]
        let dec_3d = dec_slice
            .into_shape_with_order((1, PRED_HIDDEN, 1))?
            .into_dyn();

        let inputs = inputs![
            "enc" => TensorRef::from_array_view(enc_3d.view())?,
            "dec" => TensorRef::from_array_view(dec_3d.view())?,
        ];
        let outputs = self.joint.run(inputs)?;

        let logits: ArrayD<f32> = outputs
            .get("joint")
            .ok_or_else(|| GigaAmError::OutputNotFound("joint".into()))?
            .try_extract_array()?
            .to_owned();

        let flat = logits.as_slice().unwrap_or(&[]);
        let token = flat
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i32)
            .unwrap_or(BLANK_IDX);

        Ok(token)
    }

    // ----- greedy RNN-T decode --------------------------------------------

    fn decode(&mut self, encoded: &ArrayD<f32>, t_len: usize) -> Result<Vec<i32>, GigaAmError> {
        let mut state = Self::init_decoder_state();
        let mut prev_token = BLANK_IDX;
        let mut tokens: Vec<i32> = Vec::new();

        // Compute initial decoder output
        let (mut dec_out, mut new_state) = self.decoder_step(prev_token, &state)?;

        for t in 0..t_len {
            let mut emitted = 0;
            loop {
                let token = self.joint_step(encoded, t, &dec_out)?;
                if token == BLANK_IDX || emitted >= MAX_TOKENS_PER_STEP {
                    break;
                }
                tokens.push(token);
                prev_token = token;
                emitted += 1;
                state = new_state;
                (dec_out, new_state) = self.decoder_step(prev_token, &state)?;
            }
        }

        Ok(tokens)
    }

    // ----- token detokenization -------------------------------------------

    /// Convert token ids to text.
    /// SentencePiece uses ▁ (U+2581) as a word-boundary prefix; we replace it with a space.
    fn detokenize(&self, ids: &[i32]) -> String {
        let pieces: Vec<&str> = ids
            .iter()
            .filter_map(|&id| {
                let idx = id as usize;
                if idx < self.vocab.len() {
                    Some(self.vocab[idx].as_str())
                } else {
                    None
                }
            })
            .collect();

        pieces
            .join("")
            .replace('\u{2581}', " ")
            .trim()
            .to_string()
    }

    // ----- public API ------------------------------------------------------

    pub fn transcribe_samples(&mut self, samples: Vec<f32>) -> Result<String, GigaAmError> {
        let (encoded, t_len) = self.encode(&samples)?;

        if t_len == 0 {
            log::debug!("GigaAM: encoder returned 0 frames");
            return Ok(String::new());
        }

        let token_ids = self.decode(&encoded, t_len)?;
        log::debug!("GigaAM: decoded {} tokens", token_ids.len());

        Ok(self.detokenize(&token_ids))
    }
}
