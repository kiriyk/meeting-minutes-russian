//! RAII boundary around sherpa's C API. Its C++ exceptions must be caught on
//! the C++ side, before returning to Rust (catch_unwind cannot catch them).
use anyhow::{anyhow, bail, Context, Result};
use sherpa_onnx_sys as sys;
use std::{
    ffi::{CStr, CString},
    path::Path,
    ptr::NonNull,
};

extern "C" {
    fn meetily_diarization_create(
        config: *const sys::OfflineSpeakerDiarizationConfig,
    ) -> *mut sys::OfflineSpeakerDiarization;
    fn meetily_diarization_process(
        diarizer: *const sys::OfflineSpeakerDiarization,
        samples: *const f32,
        length: i32,
    ) -> *mut sys::OfflineSpeakerDiarizationResult;
    fn meetily_diarization_sort(
        result: *const sys::OfflineSpeakerDiarizationResult,
    ) -> *mut sys::OfflineSpeakerDiarizationSegment;
    fn meetily_diarization_error() -> *const std::ffi::c_char;
}

fn native_error(operation: &str) -> anyhow::Error {
    // SAFETY: the shim returns a NUL-terminated, thread-local buffer that remains
    // valid until the next native call on this thread. Copy it immediately.
    let detail = unsafe { CStr::from_ptr(meetily_diarization_error()) }.to_string_lossy();
    anyhow!("{operation}: {detail}")
}

pub struct Diarizer(NonNull<sys::OfflineSpeakerDiarization>);
impl Diarizer {
    pub fn create(dir: &Path) -> Result<Self> {
        let segmentation = CString::new(
            dir.join(super::SEGMENTATION)
                .to_str()
                .context("Invalid segmentation path")?,
        )?;
        let embedding = CString::new(
            dir.join(super::EMBEDDING)
                .to_str()
                .context("Invalid embedding path")?,
        )?;
        let cpu = CString::new("cpu")?;
        let config = sys::OfflineSpeakerDiarizationConfig {
            segmentation: sys::OfflineSpeakerSegmentationModelConfig {
                pyannote: sys::OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: segmentation.as_ptr(),
                    window_shift_ratio: 0.1,
                },
                num_threads: 2,
                debug: 0,
                provider: cpu.as_ptr(),
            },
            embedding: sys::SpeakerEmbeddingExtractorConfig {
                model: embedding.as_ptr(),
                num_threads: 2,
                debug: 0,
                provider: cpu.as_ptr(),
            },
            clustering: sys::FastClusteringConfig {
                num_clusters: -1,
                threshold: 0.5,
                compute_confidence: 0,
            },
            min_duration_on: 0.3,
            min_duration_off: 0.5,
        };
        // SAFETY: config and all CStrings outlive the synchronous create call;
        // sherpa copies its configuration. The shim catches C++ exceptions.
        let ptr = unsafe { meetily_diarization_create(&config) };
        let diarizer = Self(
            NonNull::new(ptr).ok_or_else(|| native_error("Cannot initialize speaker models"))?,
        );
        // SAFETY: non-null pointer is owned and remains live until Drop.
        if unsafe { sys::SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(diarizer.0.as_ptr()) }
            != 16000
        {
            bail!("Speaker model requires an unsupported sample rate");
        }
        Ok(diarizer)
    }

    pub fn process(&self, samples: &[f32]) -> Result<Vec<sys::OfflineSpeakerDiarizationSegment>> {
        let length = i32::try_from(samples.len()).context("Unsupported waveform length")?;
        // SAFETY: samples and self remain live for this synchronous call. The
        // result is exclusively owned by the RAII guard below.
        let ptr = unsafe { meetily_diarization_process(self.0.as_ptr(), samples.as_ptr(), length) };
        let result = NativeResult(
            NonNull::new(ptr).ok_or_else(|| native_error("Speaker inference failed"))?,
        );
        // SAFETY: result is a valid, live native result.
        let count = unsafe {
            sys::SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result.0.as_ptr())
        };
        if count <= 0 {
            return Ok(Vec::new());
        }
        // SAFETY: the shim returns exactly count entries allocated by sherpa;
        // copying happens before its matching destructor frees the allocation.
        let sorted = unsafe { meetily_diarization_sort(result.0.as_ptr()) };
        let sorted =
            NonNull::new(sorted).ok_or_else(|| native_error("Cannot sort speaker turns"))?;
        let segments =
            unsafe { std::slice::from_raw_parts(sorted.as_ptr(), count as usize) }.to_vec();
        unsafe { sys::SherpaOnnxOfflineSpeakerDiarizationDestroySegment(sorted.as_ptr()) };
        Ok(segments)
    }
}
impl Drop for Diarizer {
    fn drop(&mut self) {
        // SAFETY: this guard exclusively owns the allocation; destructor called once.
        unsafe { sys::SherpaOnnxDestroyOfflineSpeakerDiarization(self.0.as_ptr()) };
    }
}
struct NativeResult(NonNull<sys::OfflineSpeakerDiarizationResult>);
impl Drop for NativeResult {
    fn drop(&mut self) {
        // SAFETY: this guard exclusively owns the allocation; destructor called once.
        unsafe { sys::SherpaOnnxOfflineSpeakerDiarizationDestroyResult(self.0.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corrupt_onnx_returns_error_without_aborting_app() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(super::super::SEGMENTATION),
            b"not an ONNX model",
        )
        .unwrap();
        std::fs::write(
            dir.path().join(super::super::EMBEDDING),
            b"not an ONNX model",
        )
        .unwrap();
        assert!(Diarizer::create(dir.path()).is_err());
    }
}
