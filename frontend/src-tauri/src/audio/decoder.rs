// Audio file decoder for retranscription feature
// Uses Symphonia to decode MP4/AAC audio files, with ffmpeg fallback for
// formats Symphonia can't handle (MKV, WebM, WMA)

use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use std::borrow::Cow;
use std::path::Path;
use std::process::{Command, Stdio};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::audio_processing::audio_to_mono;
use super::offline_resampling::resample as resample_offline;
use super::ffmpeg::find_ffmpeg_path;

/// Extensions requiring ffmpeg pre-conversion (Symphonia lacks these demuxers/codecs)
const FFMPEG_ONLY_EXTENSIONS: &[&str] = &["mkv", "webm", "wma"];

/// Progress callback for long-running operations
/// Returns current progress (0-100) and a message
pub type ProgressCallback = Box<dyn Fn(u32, &str) + Send>;

/// Decoded audio data from a file
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Raw audio samples (interleaved if stereo)
    pub samples: Vec<f32>,
    /// Sample rate of the decoded audio
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Duration in seconds
    pub duration_seconds: f64,
}

impl DecodedAudio {
    /// Convert to normalized 16kHz mono audio, preserving duration and the final tail.
    pub fn to_whisper_format(&self) -> Vec<f32> {
        self.to_whisper_format_with_progress(None)
    }

    pub fn to_whisper_format_with_progress(&self, progress: Option<ProgressCallback>) -> Vec<f32> {
        self.to_whisper_format_cancellable(|percent, message| {
            if let Some(callback) = &progress {
                callback(percent, message);
            }
            true
        }).unwrap_or_else(|error| {
            warn!("Audio conversion failed: {}", error);
            Vec::new()
        })
    }

    /// Returns false from the callback to stop between bounded FFT blocks.
    /// Errors are propagated so callers never interpret unconverted audio as 16kHz.
    pub fn to_whisper_format_cancellable(
        &self,
        mut progress: impl FnMut(u32, &str) -> bool,
    ) -> Result<Vec<f32>> {
        let mono = if self.channels > 1 {
            audio_to_mono(&self.samples, self.channels)
        } else {
            self.samples.clone()
        };
        let mono = normalize_audio_samples(mono);
        let mut output = resample_offline(&mono, self.sample_rate, 16000, |percent| {
            progress(percent, &format!("Converting audio format... {percent}%"))
        })?;
        for sample in &mut output {
            *sample = sample.clamp(-1.0, 1.0);
        }
        Ok(output)
    }
}

/// Normalize audio samples to the valid range (-1.0 to 1.0)
/// This handles audio files that may have samples slightly outside the expected range
fn normalize_audio_samples(mut samples: Vec<f32>) -> Vec<f32> {
    // First, find the maximum absolute value
    let max_abs = samples
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b));

    if max_abs > 1.0 {
        // Audio exceeds valid range - normalize by scaling
        info!(
            "Audio samples exceed valid range (max: {:.3}), normalizing...",
            max_abs
        );
        let scale = 1.0 / max_abs;
        for sample in &mut samples {
            *sample *= scale;
        }
    }

    // Also clamp any remaining edge cases (NaN, infinity, etc.)
    for sample in &mut samples {
        if !sample.is_finite() {
            *sample = 0.0;
        } else {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }

    samples
}

/// Check if a file extension requires ffmpeg pre-conversion
fn needs_ffmpeg_conversion(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| FFMPEG_ONLY_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Convert an audio file to WAV using ffmpeg for formats Symphonia can't decode.
///
/// Returns a `TempPath` that auto-deletes the temporary WAV file when dropped.
/// The caller must keep the `TempPath` alive until decoding of the WAV is complete.
fn convert_to_wav_with_ffmpeg(
    input_path: &Path,
    progress_callback: Option<&ProgressCallback>,
) -> Result<tempfile::TempPath> {
    let ffmpeg_path = find_ffmpeg_path().ok_or_else(|| {
        anyhow!(
            "FFmpeg not found. FFmpeg is required to decode .{} files. \
             It will be downloaded automatically on next launch, or install it manually.",
            input_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("this format")
        )
    })?;

    // Create temp file in the same directory as the input to avoid cross-device issues
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new("."));
    let temp_file = tempfile::Builder::new()
        .prefix(".meetily_decode_")
        .suffix(".wav")
        .tempfile_in(parent_dir)
        .map_err(|e| anyhow!("Failed to create temporary WAV file: {}", e))?;

    let temp_path = temp_file.into_temp_path();

    info!(
        "Converting .{} to temporary WAV via ffmpeg: {} -> {}",
        input_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown"),
        input_path.display(),
        temp_path.display()
    );

    if let Some(cb) = progress_callback {
        cb(0, "Converting audio format with FFmpeg...");
    }

    let input_str = input_path
        .to_str()
        .ok_or_else(|| anyhow!("Invalid input path (non-UTF8)"))?;
    let output_str = temp_path
        .to_str()
        .ok_or_else(|| anyhow!("Invalid temp path (non-UTF8)"))?;

    let mut command = Command::new(&ffmpeg_path);
    command
        .args([
            "-i", input_str,
            "-vn",                  // Strip video tracks
            "-acodec", "pcm_s16le", // Output PCM WAV (Symphonia handles natively)
            "-y",                   // Overwrite without prompt
            output_str,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Hide console window on Windows
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    debug!("FFmpeg conversion command: {:?}", command);

    #[allow(clippy::zombie_processes)]
    let child = command
        .spawn()
        .map_err(|e| anyhow!("Failed to spawn ffmpeg process: {}", e))?;

    let output = child
        .wait_with_output()
        .map_err(|e| anyhow!("Failed to wait for ffmpeg process: {}", e))?;

    let stderr_text = String::from_utf8_lossy(&output.stderr);
    debug!("FFmpeg stderr: {}", stderr_text);

    if !output.status.success() {
        error!(
            "FFmpeg conversion failed (exit code: {}): {}",
            output.status, stderr_text
        );
        return Err(anyhow!(
            "FFmpeg conversion failed with exit code: {}. \
             The file may be corrupted or in an unsupported format.",
            output.status
        ));
    }

    // Verify output file exists and has content
    let output_meta = std::fs::metadata(&temp_path)
        .map_err(|e| anyhow!("FFmpeg output file not found: {}", e))?;

    if output_meta.len() == 0 {
        return Err(anyhow!(
            "FFmpeg produced an empty output file. The input may contain no audio."
        ));
    }

    if let Some(cb) = progress_callback {
        cb(100, "FFmpeg conversion complete");
    }

    info!(
        "FFmpeg conversion complete: {} bytes output",
        output_meta.len()
    );

    Ok(temp_path)
}

/// Decode an audio file (MP4, M4A, WAV, etc.) to raw samples
pub fn decode_audio_file(path: &Path) -> Result<DecodedAudio> {
    decode_audio_file_with_progress(path, None)
}

/// Decode an audio file with optional progress callback
pub fn decode_audio_file_with_progress(
    path: &Path,
    progress_callback: Option<ProgressCallback>,
) -> Result<DecodedAudio> {
    info!("Decoding audio file: {}", path.display());

    // FFmpeg pre-conversion for unsupported formats (MKV, WebM, WMA).
    // If the file is in a format Symphonia can't decode, use ffmpeg to convert
    // it to a temporary WAV file first, then decode the WAV with Symphonia.
    // The _temp_wav_guard keeps the temp file alive until decoding completes,
    // then auto-deletes it when dropped (even on error/panic).
    let (_temp_wav_guard, decode_path): (Option<tempfile::TempPath>, Cow<'_, Path>) =
        if needs_ffmpeg_conversion(path) {
            info!(
                "Format requires ffmpeg pre-conversion: .{}",
                path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown")
            );
            let temp_path = convert_to_wav_with_ffmpeg(path, progress_callback.as_ref())?;
            let wav_path = temp_path.to_path_buf();
            (Some(temp_path), Cow::Owned(wav_path))
        } else {
            (None, Cow::Borrowed(path))
        };

    // Open the file (use decode_path which may be the temp WAV)
    let file = std::fs::File::open(decode_path.as_ref())
        .map_err(|e| anyhow!("Failed to open audio file '{}': {}", decode_path.display(), e))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Set up format hint based on file extension
    let mut hint = Hint::new();
    if let Some(ext) = decode_path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // Probe the file format
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| anyhow!("Failed to probe audio format: {}", e))?;

    let mut format = probed.format;

    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("No audio track found in file"))?;

    let track_id = track.id;

    // Get audio parameters
    let mut sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("Unknown sample rate"))?;

    let mut channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(1);

    debug!(
        "Audio track: {}Hz, {} channels (from metadata)",
        sample_rate, channels
    );

    // Create the decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| anyhow!("Failed to create decoder: {}", e))?;

    // Decode all packets
    let mut all_samples: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    // Calculate expected samples for progress tracking
    let expected_duration = track.codec_params.n_frames
        .map(|frames| frames as f64 / sample_rate as f64);
    let expected_samples = expected_duration
        .map(|dur| (dur * sample_rate as f64 * channels as f64) as usize);

    let mut last_progress = 0u32;

    loop {
        // Get the next packet
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                // End of file
                break;
            }
            Err(e) => {
                warn!("Error reading packet: {}", e);
                break;
            }
        };

        // Skip packets from other tracks
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet
        match decoder.decode(&packet) {
            Ok(decoded) => {
                // Initialize sample buffer if needed
                if sample_buf.is_none() {
                    let spec = *decoded.spec();
                    let duration = decoded.capacity() as u64;
                    // Detect actual channel count from decoded audio (metadata may be wrong/missing)
                    let actual_channels = spec.channels.count() as u16;
                    if actual_channels != channels {
                        info!(
                            "Channel count corrected: metadata={} actual={} (using actual)",
                            channels, actual_channels
                        );
                        channels = actual_channels;
                    }
                    // Detect actual sample rate from decoded audio. The container can
                    // declare a different rate than the decoder produces: HE-AAC (SBR)
                    // files declare e.g. 48000 Hz, but Symphonia has no SBR support and
                    // decodes the AAC-LC core at half that rate. Trusting the container
                    // rate makes the audio play at 2x speed and halves the duration.
                    let actual_rate = spec.rate;
                    if actual_rate != sample_rate {
                        info!(
                            "Sample rate corrected: metadata={} actual={} (using actual)",
                            sample_rate, actual_rate
                        );
                        sample_rate = actual_rate;
                    }
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }

                // Copy samples to buffer
                if let Some(ref mut buf) = sample_buf {
                    buf.copy_interleaved_ref(decoded);
                    all_samples.extend_from_slice(buf.samples());
                }

                // Emit progress updates (every 10%)
                if let (Some(callback), Some(expected)) = (&progress_callback, expected_samples) {
                    let current_progress = ((all_samples.len() as f64 / expected as f64) * 100.0) as u32;
                    if current_progress >= last_progress + 10 && current_progress <= 100 {
                        last_progress = current_progress;
                        callback(current_progress, &format!("Decoding audio: {}%", current_progress));
                    }
                }
            }
            Err(e) => {
                warn!("Error decoding packet: {}", e);
                continue;
            }
        }
    }

    // Ensure we report 100% completion
    if let Some(callback) = &progress_callback {
        callback(100, "Decoding complete");
    }

    if all_samples.is_empty() {
        return Err(anyhow!("No audio samples decoded from file"));
    }

    let total_frames = all_samples.len() / channels as usize;
    let duration_seconds = total_frames as f64 / sample_rate as f64;

    info!(
        "Decoded {} samples ({:.2}s) at {}Hz, {} channels",
        all_samples.len(),
        duration_seconds,
        sample_rate,
        channels
    );

    Ok(DecodedAudio {
        samples: all_samples,
        sample_rate,
        channels,
        duration_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_he_aac_uses_decoded_rate_not_container_rate() {
        // HE-AAC (SBR) fixture: the container declares 48 kHz stereo, but
        // Symphonia has no SBR support and decodes only the AAC-LC core at
        // 24 kHz. Trusting the container rate halves the duration and makes
        // the audio play at 2x speed, which truncated long imported
        // recordings to half their length.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/he_aac_48k_5s.m4a");
        let decoded = decode_audio_file(&path).expect("failed to decode HE-AAC fixture");

        assert_eq!(
            decoded.sample_rate, 24000,
            "sample_rate must come from the decoder output, not container metadata"
        );
        assert!(
            (decoded.duration_seconds - 5.16).abs() < 0.5,
            "duration must be ~5s (was reported as ~2.6s before the fix), got {:.2}s",
            decoded.duration_seconds
        );

        // The 16 kHz conversion must preserve the real duration as well.
        let whisper_samples = decoded.to_whisper_format();
        let duration_16k = whisper_samples.len() as f64 / 16000.0;
        assert!(
            (duration_16k - 5.16).abs() < 0.5,
            "whisper-format duration must be ~5s, got {:.2}s",
            duration_16k
        );
    }

    #[test]
    fn test_to_whisper_format_mono_16k() {
        // Already in correct format
        let audio = DecodedAudio {
            samples: vec![0.1, 0.2, 0.3],
            sample_rate: 16000,
            channels: 1,
            duration_seconds: 0.0001875,
        };

        let result = audio.to_whisper_format();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_to_whisper_format_stereo_to_mono() {
        // Stereo input
        let audio = DecodedAudio {
            samples: vec![0.2, 0.4, 0.6, 0.8], // 2 stereo frames
            sample_rate: 16000,
            channels: 2,
            duration_seconds: 0.000125,
        };

        let result = audio.to_whisper_format();
        assert_eq!(result.len(), 2); // Should be mono now
        // Average of (0.2, 0.4) = 0.3 and (0.6, 0.8) = 0.7
        assert!((result[0] - 0.3).abs() < 0.001);
        assert!((result[1] - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_to_whisper_format_resamples_48k_to_16k() {
        // 48kHz mono input - should be downsampled to 16kHz
        // Use a larger sample to ensure resampler works correctly
        // 48000 samples at 48kHz = 1 second → 16000 samples at 16kHz
        let audio = DecodedAudio {
            samples: vec![0.5; 4800], // 0.1 seconds at 48kHz
            sample_rate: 48000,
            channels: 1,
            duration_seconds: 4800.0 / 48000.0,
        };

        let result = audio.to_whisper_format();
        // Output length should be approximately input_len / 3 (16000/48000 ratio)
        // 4800 / 3 = 1600
        assert!(!result.is_empty(), "Result should not be empty");
        assert!(result.len() > 1000 && result.len() < 2000,
            "Expected ~1600 samples, got {}", result.len());
    }

    #[test]
    fn test_offline_conversion_preserves_duration_and_tail() {
        let mut samples = vec![0.0; 8000];
        // A final impulse must survive the resampler's latency and final partial block.
        samples[7990] = 0.5;
        let audio = DecodedAudio {
            samples,
            sample_rate: 8000,
            channels: 1,
            duration_seconds: 1.0,
        };
        let result = audio.to_whisper_format();
        assert_eq!(result.len(), 16000);
        assert!(result[15970..].iter().any(|s| s.abs() > 0.1));
    }

    #[test]
    fn test_offline_conversion_reports_progress_for_8khz_audio() {
        let audio = DecodedAudio {
            samples: vec![0.1; 16000],
            sample_rate: 8000,
            channels: 1,
            duration_seconds: 2.0,
        };
        let updates = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = updates.clone();
        audio.to_whisper_format_with_progress(Some(Box::new(move |percent, _| {
            observed.lock().unwrap().push(percent);
        })));
        let updates = updates.lock().unwrap();
        assert!(updates.iter().any(|&p| p > 0 && p < 100), "No intermediate progress: {updates:?}");
        assert_eq!(updates.last(), Some(&100));
    }

    #[test]
    fn test_offline_conversion_cancellation_propagates() {
        let audio = DecodedAudio {
            samples: vec![0.1; 80000],
            sample_rate: 8000,
            channels: 1,
            duration_seconds: 10.0,
        };
        let error = audio.to_whisper_format_cancellable(|percent, _| percent < 10).unwrap_err();
        assert_eq!(error.to_string(), "Retranscription cancelled");
    }

    #[test]
    #[ignore = "Set MEETILY_RESAMPLING_AUDIO to a WAV/MP4 file; reads audio without modifying the meeting"]
    fn offline_conversion_real_audio_smoke() {
        let path = std::env::var("MEETILY_RESAMPLING_AUDIO").unwrap();
        let decoded = decode_audio_file(Path::new(&path)).unwrap();
        let started = std::time::Instant::now();
        let mut updates = Vec::new();
        let result = decoded.to_whisper_format_cancellable(|percent, _| {
            updates.push(percent);
            true
        }).unwrap();
        assert_eq!(result.len(), (decoded.samples.len() as f64 / decoded.channels as f64 * 16000.0 / decoded.sample_rate as f64).round() as usize);
        assert!(result.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
        assert_eq!(updates.last(), Some(&100));
        assert!(updates.windows(2).all(|p| p[0] <= p[1]));
        println!("Converted {:.2}s of {}Hz audio in {:.2}s; {} samples, {} progress updates", decoded.duration_seconds, decoded.sample_rate, started.elapsed().as_secs_f64(), result.len(), updates.len());
    }

    #[test]
    fn test_block_resample_same_rate() {
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = resample_offline(&input, 16000, 16000, |_| true).unwrap();
        assert_eq!(result.len(), input.len());
        for (i, &sample) in result.iter().enumerate() {
            assert!((sample - input[i]).abs() < 0.001);
        }
    }

    #[test]
    fn test_block_resample_empty_input() {
        let input: Vec<f32> = vec![];
        let result = resample_offline(&input, 48000, 16000, |_| true).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_block_resample_downsamples_correctly() {
        // 48kHz to 16kHz = 3x downsampling with a 2-second signal
        let input: Vec<f32> = (0..96000).map(|i| (i as f32 / 96000.0)).collect();
        let result = resample_offline(&input, 48000, 16000, |_| true).unwrap();

        // Output should be approximately 1/3 the length
        let expected_len = 96000.0 * (16000.0 / 48000.0);
        assert!(
            (result.len() as f64 - expected_len).abs() < 200.0,
            "Expected ~{} samples, got {}",
            expected_len,
            result.len()
        );
    }

    #[test]
    fn test_block_resample_preserves_signal_range() {
        // 1 second of sine wave at 44100Hz
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let result = resample_offline(&input, 44100, 16000, |_| true).unwrap();

        for sample in &result {
            assert!(
                *sample >= -1.1 && *sample <= 1.1,
                "Sample {} out of expected range",
                sample
            );
        }
    }

    #[test]
    fn test_decoded_audio_duration_calculation() {
        let audio = DecodedAudio {
            samples: vec![0.0; 48000], // 1 second at 48kHz mono
            sample_rate: 48000,
            channels: 1,
            duration_seconds: 1.0,
        };

        // Duration should be samples / sample_rate for mono
        let calculated_duration = audio.samples.len() as f64 / audio.sample_rate as f64;
        assert!((calculated_duration - audio.duration_seconds).abs() < 0.001);
    }

    #[test]
    fn test_decoded_audio_stereo_duration() {
        let audio = DecodedAudio {
            samples: vec![0.0; 96000], // 1 second at 48kHz stereo (2 channels)
            sample_rate: 48000,
            channels: 2,
            duration_seconds: 1.0,
        };

        // Duration should be samples / (sample_rate * channels) for stereo
        let frames = audio.samples.len() / audio.channels as usize;
        let calculated_duration = frames as f64 / audio.sample_rate as f64;
        assert!((calculated_duration - audio.duration_seconds).abs() < 0.001);
    }

    #[test]
    fn test_to_whisper_format_handles_short_file() {
        // Short files use the same filtered conversion as long recordings.
        let audio = DecodedAudio {
            samples: vec![0.5; 1000], // Small file
            sample_rate: 48000,
            channels: 1,
            duration_seconds: 1000.0 / 48000.0,
        };

        let result = audio.to_whisper_format();
        // Should complete without error and produce valid output
        assert!(!result.is_empty());
        assert!(result.len() < 1000); // Downsampled
    }

    #[test]
    fn test_normalize_audio_samples_already_normalized() {
        let samples = vec![0.5, -0.5, 0.0, 0.9, -0.9];
        let result = normalize_audio_samples(samples.clone());
        // Should be unchanged (already in range)
        for (i, &s) in result.iter().enumerate() {
            assert!((s - samples[i]).abs() < 0.001);
        }
    }

    #[test]
    fn test_normalize_audio_samples_exceeds_range() {
        let samples = vec![0.5, -0.5, 2.0, -1.5]; // max_abs = 2.0
        let result = normalize_audio_samples(samples);
        // All samples should be scaled by 0.5 (1.0 / 2.0)
        assert!((result[0] - 0.25).abs() < 0.001);
        assert!((result[1] - -0.25).abs() < 0.001);
        assert!((result[2] - 1.0).abs() < 0.001);
        assert!((result[3] - -0.75).abs() < 0.001);
    }

    #[test]
    fn test_normalize_audio_samples_handles_nan() {
        let samples = vec![0.5, f32::NAN, 0.3];
        let result = normalize_audio_samples(samples);
        assert!((result[0] - 0.5).abs() < 0.001);
        assert_eq!(result[1], 0.0); // NaN replaced with 0
        assert!((result[2] - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_normalize_audio_samples_handles_infinity() {
        let samples = vec![0.5, f32::INFINITY, -0.3];
        let result = normalize_audio_samples(samples);
        assert!((result[0] - 0.5).abs() < 0.001); // preserved
        assert_eq!(result[1], 0.0); // infinity → 0
        assert!((result[2] - (-0.3)).abs() < 0.001); // preserved
    }

    #[test]
    fn test_needs_ffmpeg_conversion() {
        assert!(needs_ffmpeg_conversion(Path::new("video.mkv")));
        assert!(needs_ffmpeg_conversion(Path::new("audio.webm")));
        assert!(needs_ffmpeg_conversion(Path::new("audio.wma")));
        // Case insensitive
        assert!(needs_ffmpeg_conversion(Path::new("meeting.MKV")));
        assert!(needs_ffmpeg_conversion(Path::new("audio.WMA")));
        assert!(needs_ffmpeg_conversion(Path::new("audio.WebM")));
        // Symphonia-native formats should NOT need ffmpeg
        assert!(!needs_ffmpeg_conversion(Path::new("audio.mp4")));
        assert!(!needs_ffmpeg_conversion(Path::new("audio.wav")));
        assert!(!needs_ffmpeg_conversion(Path::new("audio.mp3")));
        assert!(!needs_ffmpeg_conversion(Path::new("audio.flac")));
        assert!(!needs_ffmpeg_conversion(Path::new("audio.ogg")));
        assert!(!needs_ffmpeg_conversion(Path::new("audio.aac")));
        assert!(!needs_ffmpeg_conversion(Path::new("audio.m4a")));
        // No extension
        assert!(!needs_ffmpeg_conversion(Path::new("noext")));
    }
}
