use anyhow::{anyhow, Result};
use log::{debug, info, warn};

pub mod classifier;
pub mod segmenter;

use classifier::FrameClassifier;
use segmenter::{RawSegment, SegmenterConfig, SpeechSegmenter, VAD_SAMPLE_RATE as VAD_RATE};

/// Every sample count and timestamp inside this module is at 16 kHz.
const VAD_SAMPLE_RATE: u32 = VAD_RATE as u32;

/// Represents a complete speech segment detected by VAD
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub samples: Vec<f32>,
    pub start_timestamp_ms: f64,
    pub end_timestamp_ms: f64,
    pub confidence: f32,
}

impl From<RawSegment> for SpeechSegment {
    fn from(raw: RawSegment) -> Self {
        let to_ms = |s: usize| s as f64 * 1000.0 / VAD_SAMPLE_RATE as f64;
        Self {
            start_timestamp_ms: to_ms(raw.start_sample),
            end_timestamp_ms: to_ms(raw.end_sample),
            samples: raw.samples,
            confidence: 0.9,
        }
    }
}

/// Frames 16 kHz audio for a classifier and turns probabilities into speech segments.
pub struct ContinuousVadProcessor {
    classifier: Box<dyn FrameClassifier>,
    segmenter: SpeechSegmenter,
    input_sample_rate: u32,
    buffer: Vec<f32>,
    /// 16 kHz samples already classified (excludes flush padding).
    fed_samples: usize,
}

impl ContinuousVadProcessor {
    pub fn new(input_sample_rate: u32, redemption_time_ms: u32) -> Result<Self> {
        let classifier = classifier::SileroV6Classifier::new()
            .map_err(|e| anyhow!("Failed to create VAD session: {e:?}"))?;
        Ok(Self::with_classifier(input_sample_rate, redemption_time_ms, Box::new(classifier)))
    }

    pub(crate) fn with_classifier(
        input_sample_rate: u32,
        redemption_time_ms: u32,
        classifier: Box<dyn FrameClassifier>,
    ) -> Self {
        let frame = classifier.frame_len();
        info!(
            "VAD processor created: input={}Hz, frame={} samples, redemption={}ms",
            input_sample_rate, frame, redemption_time_ms
        );
        Self {
            classifier,
            segmenter: SpeechSegmenter::new(SegmenterConfig::meetily(redemption_time_ms)),
            input_sample_rate,
            buffer: Vec::with_capacity(frame * 2),
            fed_samples: 0,
        }
    }

    pub fn is_speaking(&self) -> bool {
        self.segmenter.is_speaking()
    }

    pub fn frame_len(&self) -> usize {
        self.classifier.frame_len()
    }

    /// Process incoming audio samples and return any complete speech segments
    pub fn process_audio(&mut self, samples: &[f32]) -> Result<Vec<SpeechSegment>> {
        let resampled = if self.input_sample_rate == VAD_SAMPLE_RATE {
            samples.to_vec()
        } else {
            self.resample_to_16k(samples)?
        };
        self.buffer.extend_from_slice(&resampled);

        let frame_len = self.frame_len();
        let mut completed = Vec::new();
        while self.buffer.len() >= frame_len {
            let frame: Vec<f32> = self.buffer.drain(..frame_len).collect();
            if let Some(raw) = self.classify(&frame)? {
                completed.push(SpeechSegment::from(raw));
            }
            self.fed_samples += frame_len;
        }
        Ok(completed)
    }

    /// Improved resampling from input sample rate to 16kHz with anti-aliasing
    /// Uses linear interpolation and basic low-pass filtering for better quality
    fn resample_to_16k(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if self.input_sample_rate == 16000 {
            return Ok(samples.to_vec());
        }

        // Calculate downsampling ratio
        let ratio = self.input_sample_rate as f64 / 16000.0;
        let output_len = (samples.len() as f64 / ratio) as usize;
        let mut resampled = Vec::with_capacity(output_len);

        // Apply simple low-pass filter before downsampling to reduce aliasing
        let cutoff_freq = 0.4; // Normalized frequency (0.4 * Nyquist)
        let mut filtered_samples = Vec::with_capacity(samples.len());

        // Simple moving average filter (basic low-pass)
        let filter_size = (self.input_sample_rate as f64 / (cutoff_freq * self.input_sample_rate as f64)) as usize;
        let filter_size = std::cmp::max(1, std::cmp::min(filter_size, 5)); // Limit filter size

        for i in 0..samples.len() {
            let start = if i >= filter_size { i - filter_size } else { 0 };
            let end = std::cmp::min(i + filter_size + 1, samples.len());
            let sum: f32 = samples[start..end].iter().sum();
            filtered_samples.push(sum / (end - start) as f32);
        }

        // Linear interpolation downsampling
        for i in 0..output_len {
            let source_pos = i as f64 * ratio;
            let source_index = source_pos as usize;
            let fraction = source_pos - source_index as f64;

            if source_index + 1 < filtered_samples.len() {
                // Linear interpolation
                let sample1 = filtered_samples[source_index];
                let sample2 = filtered_samples[source_index + 1];
                let interpolated = sample1 + (sample2 - sample1) * fraction as f32;
                resampled.push(interpolated);
            } else if source_index < filtered_samples.len() {
                resampled.push(filtered_samples[source_index]);
            }
        }

        debug!("Resampled from {} samples ({}Hz) to {} samples (16kHz) with anti-aliasing",
               samples.len(), self.input_sample_rate, resampled.len());

        Ok(resampled)
    }

    /// Flush remaining audio; an utterance in progress ends at the real audio end.
    pub fn flush(&mut self) -> Result<Vec<SpeechSegment>> {
        let real_end = self.fed_samples + self.buffer.len();
        let mut completed = Vec::new();
        if !self.buffer.is_empty() {
            let mut frame = std::mem::take(&mut self.buffer);
            frame.resize(self.frame_len(), 0.0);
            if let Some(mut raw) = self.classify(&frame)? {
                raw.truncate_end(real_end);
                completed.push(SpeechSegment::from(raw));
            }
            self.fed_samples = real_end;
        }
        if let Some(raw) = self.segmenter.finish(real_end) {
            debug!("VAD flush: force-ending speech at {}ms", real_end * 1000 / VAD_RATE);
            completed.push(SpeechSegment::from(raw));
        }
        Ok(completed)
    }

    fn classify(&mut self, frame: &[f32]) -> Result<Option<RawSegment>> {
        let probability = self
            .classifier
            .predict(frame)
            .map_err(|e| anyhow!("VAD processing failed: {e}"))?;
        let segment = self.segmenter.push_frame(frame, probability);
        if let Some(s) = &segment {
            info!(
                "VAD: Completed speech segment: {:.1}ms duration, {} samples",
                (s.end_sample - s.start_sample) as f64 * 1000.0 / VAD_RATE as f64,
                s.samples.len()
            );
        }
        if self.segmenter.retained_samples() > 1_000_000 {
            warn!("VAD: retaining {} samples of active speech", self.segmenter.retained_samples());
        }
        Ok(segment)
    }
}

/// Legacy function for backward compatibility - now uses the optimized approach
pub fn extract_speech_16k(samples_mono_16k: &[f32]) -> Result<Vec<f32>> {
    let mut processor = ContinuousVadProcessor::new(16000, 400)?;

    // Process all audio
    let mut all_segments = processor.process_audio(samples_mono_16k)?;
    let final_segments = processor.flush()?;
    all_segments.extend(final_segments);

    // Concatenate all speech segments
    let mut result = Vec::new();
    let num_segments = all_segments.len();
    for segment in &all_segments {
        result.extend_from_slice(&segment.samples);
    }

    // Apply balanced energy filtering for very short segments
    if result.len() < 1600 { // Less than 100ms at 16kHz
        let input_energy: f32 = samples_mono_16k.iter().map(|&x| x * x).sum::<f32>() / samples_mono_16k.len() as f32;
        let rms = input_energy.sqrt();
        let peak = samples_mono_16k.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        // BALANCED FIX: Lowered thresholds to preserve quiet speech while still filtering silence
        // Previous aggressive values (0.08/0.15) were discarding valid quiet speech
        // New values (0.03/0.08) are more balanced - catch quiet speech, reject pure silence
        if rms < 0.2 || peak < 0.20 {
            info!("-----VAD detected silence/noise (RMS: {:.6}, Peak: {:.6}), skipping to prevent hallucinations-----", rms, peak);
            return Ok(Vec::new());
        } else {
            info!("VAD detected speech with sufficient energy (RMS: {:.6}, Peak: {:.6})", rms, peak);
            return Ok(samples_mono_16k.to_vec());
        }
    }

    debug!("VAD: Processed {} samples, extracted {} speech samples from {} segments",
           samples_mono_16k.len(), result.len(), num_segments);

    Ok(result)
}

/// Simple convenience function to get speech chunks from audio
/// Uses the optimized ContinuousVadProcessor with configurable redemption time
pub fn get_speech_chunks(samples_mono_16k: &[f32], redemption_time_ms: u32) -> Result<Vec<SpeechSegment>> {
    get_speech_chunks_with_progress(samples_mono_16k, redemption_time_ms, |_, _| true)
}

/// Get speech chunks with progress callback and cancellation support
/// The callback receives (progress_percent, segments_found) and returns false to cancel
pub fn get_speech_chunks_with_progress<F>(
    samples_mono_16k: &[f32],
    redemption_time_ms: u32,
    progress_callback: F,
) -> Result<Vec<SpeechSegment>>
where
    F: FnMut(u32, usize) -> bool,
{
    let processor = ContinuousVadProcessor::new(16000, redemption_time_ms)?;
    speech_chunks_with_processor(processor, samples_mono_16k, progress_callback)
}

fn speech_chunks_with_processor<F>(
    mut processor: ContinuousVadProcessor,
    samples_mono_16k: &[f32],
    mut progress_callback: F,
) -> Result<Vec<SpeechSegment>>
where
    F: FnMut(u32, usize) -> bool,
{
    let total_samples = samples_mono_16k.len();

    // For large files (>1 minute at 16kHz = 960,000 samples), process in chunks with progress logging
    const LARGE_FILE_THRESHOLD: usize = 960_000;
    const CHUNK_SIZE: usize = 160_000; // 10 seconds at 16kHz

    let mut all_segments = Vec::new();

    if total_samples > LARGE_FILE_THRESHOLD {
        info!("VAD: Processing large file ({} samples = {:.1}s), will log progress...",
              total_samples, total_samples as f64 / 16000.0);

        let mut processed = 0;
        let mut last_progress = 0u32;
        let mut chunk_count = 0;
        let total_chunks = (total_samples + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for chunk in samples_mono_16k.chunks(CHUNK_SIZE) {
            chunk_count += 1;

            let start_time = std::time::Instant::now();
            let segments = processor.process_audio(chunk)?;
            let elapsed = start_time.elapsed();

            // Debug log for chunk processing details
            debug!("VAD: Chunk {}/{} processed in {:?}, found {} segments",
                  chunk_count, total_chunks, elapsed, segments.len());

            // Warn if chunk processing took too long (>1 second)
            if elapsed.as_secs() > 1 {
                warn!("VAD: Chunk {} took {:?} - possible performance issue", chunk_count, elapsed);
            }

            all_segments.extend(segments);

            processed += chunk.len();
            let progress = ((processed * 100) / total_samples) as u32;

            // Call progress callback every 5%
            if progress >= last_progress + 5 {
                debug!("VAD: Progress {}% ({} segments found so far)", progress, all_segments.len());

                // Check for cancellation
                if !progress_callback(progress, all_segments.len()) {
                    info!("VAD: Cancelled by callback at {}%", progress);
                    return Err(anyhow!("VAD processing cancelled"));
                }

                last_progress = progress;
            }
        }

        let final_segments = processor.flush()?;
        all_segments.extend(final_segments);

        info!("VAD: Complete! Found {} speech segments", all_segments.len());
    } else {
        // Small file - process all at once
        all_segments = processor.process_audio(samples_mono_16k)?;
        let final_segments = processor.flush()?;
        all_segments.extend(final_segments);
    }

    Ok(all_segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic stand-in for a neural VAD: loud frames are speech.
    struct EnergyClassifier(usize);
    impl crate::audio::vad::classifier::FrameClassifier for EnergyClassifier {
        fn frame_len(&self) -> usize { self.0 }
        fn predict(&mut self, frame: &[f32]) -> Result<f32> {
            let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
            Ok(if rms > 0.05 { 0.9 } else { 0.05 })
        }
    }

    /// Synthetic tones are not speech to a neural VAD, so framing/progress/cancellation
    /// tests use the deterministic energy classifier (Silero v6 frame size).
    fn energy_processor(redemption_time_ms: u32) -> ContinuousVadProcessor {
        ContinuousVadProcessor::with_classifier(16_000, redemption_time_ms, Box::new(EnergyClassifier(512)))
    }

    fn energy_chunks(audio: &[f32], redemption_time_ms: u32) -> Result<Vec<SpeechSegment>> {
        speech_chunks_with_processor(energy_processor(redemption_time_ms), audio, |_, _| true)
    }

    fn energy_chunks_with_progress<F: FnMut(u32, usize) -> bool>(
        audio: &[f32],
        redemption_time_ms: u32,
        progress: F,
    ) -> Result<Vec<SpeechSegment>> {
        speech_chunks_with_processor(energy_processor(redemption_time_ms), audio, progress)
    }

    /// 1 s silence, 2 s tone, 3 s silence, 1.5 s tone, 0.7 s silence.
    fn two_utterances() -> Vec<f32> {
        let mut audio = vec![0.0f32; 16_000];
        audio.extend((0..32_000).map(|i| 0.3 * (i as f32 * 0.07).sin()));
        audio.extend(vec![0.0f32; 48_000]);
        audio.extend((0..24_000).map(|i| 0.3 * (i as f32 * 0.05).sin()));
        audio.extend(vec![0.0f32; 11_200]);
        audio
    }

    fn collect(p: &mut ContinuousVadProcessor, audio: &[f32], chunk: usize) -> Vec<SpeechSegment> {
        let mut out = Vec::new();
        for piece in audio.chunks(chunk) {
            out.extend(p.process_audio(piece).unwrap());
        }
        out.extend(p.flush().unwrap());
        out
    }

    #[test]
    fn processor_segments_do_not_depend_on_chunking() {
        for frame in [256usize, 512] {
            let audio = two_utterances();
            let whole = collect(&mut ContinuousVadProcessor::with_classifier(16_000, 500, Box::new(EnergyClassifier(frame))), &audio, audio.len());
            let odd = collect(&mut ContinuousVadProcessor::with_classifier(16_000, 500, Box::new(EnergyClassifier(frame))), &audio, 777);
            assert_eq!(whole.len(), 2, "frame {frame}");
            let key = |s: &[SpeechSegment]| s.iter().map(|x| (x.start_timestamp_ms, x.end_timestamp_ms, x.samples.len())).collect::<Vec<_>>();
            assert_eq!(key(&whole), key(&odd), "frame {frame}");
        }
    }

    #[test]
    fn flush_emits_tail_at_real_end_once() {
        let mut audio = vec![0.0f32; 16_000];
        audio.extend((0..16_000 + 123).map(|i| 0.3 * (i as f32 * 0.07).sin()));
        let mut p = ContinuousVadProcessor::with_classifier(16_000, 2000, Box::new(EnergyClassifier(512)));
        assert!(p.process_audio(&audio).unwrap().is_empty());
        assert!(p.is_speaking());
        let tail = p.flush().unwrap();
        assert_eq!(tail.len(), 1);
        let s = &tail[0];
        let real_end_ms = audio.len() as f64 * 1000.0 / 16_000.0;
        assert!((s.end_timestamp_ms - real_end_ms).abs() < 1e-9, "end {} != {}", s.end_timestamp_ms, real_end_ms);
        let start = (s.start_timestamp_ms * 16.0).round() as usize;
        assert_eq!(s.samples.as_slice(), &audio[start..]);
        assert!(p.flush().unwrap().is_empty());
    }

    #[test]
    fn processor_resamples_48k_input_timestamps_to_real_time() {
        let audio16 = two_utterances();
        let audio48: Vec<f32> = audio16.iter().flat_map(|s| [*s, *s, *s]).collect();
        let segments = collect(&mut ContinuousVadProcessor::with_classifier(48_000, 500, Box::new(EnergyClassifier(512))), &audio48, 4800);
        assert_eq!(segments.len(), 2);
        assert!((segments[0].start_timestamp_ms - 700.0).abs() < 40.0, "start {}", segments[0].start_timestamp_ms);
    }

    /// Generate synthetic speech-like audio with alternating speech/silence
    fn generate_test_audio_with_speech(duration_seconds: f32, sample_rate: u32) -> Vec<f32> {
        let total_samples = (duration_seconds * sample_rate as f32) as usize;
        let mut samples = vec![0.0f32; total_samples];

        // Create speech-like patterns: bursts of sine waves with varying amplitude
        // Speech every 10 seconds for 5 seconds
        let speech_interval = 10.0; // seconds between speech starts
        let speech_duration = 5.0;  // seconds of speech

        for i in 0..total_samples {
            let time = i as f32 / sample_rate as f32;
            let cycle_time = time % speech_interval;

            // Speech occurs in the first `speech_duration` seconds of each cycle
            if cycle_time < speech_duration {
                // Generate speech-like signal: multiple frequencies with amplitude modulation
                let freq1 = 200.0 + (time * 50.0).sin() * 100.0; // Varying fundamental
                let freq2 = freq1 * 2.0; // Harmonic
                let freq3 = freq1 * 3.0; // Another harmonic

                let amplitude = 0.3 + 0.1 * (time * 5.0).sin(); // Amplitude modulation
                samples[i] = amplitude * (
                    0.5 * (2.0 * std::f32::consts::PI * freq1 * time).sin() +
                    0.3 * (2.0 * std::f32::consts::PI * freq2 * time).sin() +
                    0.2 * (2.0 * std::f32::consts::PI * freq3 * time).sin()
                );
            }
            // else: silence (already 0.0)
        }

        samples
    }

    #[test]
    fn test_vad_chunked_vs_single_processing() {
        // Generate 60 seconds of audio with speech patterns at 16kHz
        let audio = generate_test_audio_with_speech(60.0, 16000);
        println!("Generated {} samples ({:.1}s)", audio.len(), audio.len() as f32 / 16000.0);

        // Process all at once (like small files)
        let segments_single = energy_chunks(&audio, 2000).expect("Single processing failed");
        println!("Single processing found {} segments", segments_single.len());

        // Process in chunks (like large files)
        let segments_chunked = energy_chunks_with_progress(&audio, 2000, |progress, segments| {
            println!("Chunked progress: {}%, {} segments", progress, segments);
            true // Don't cancel
        }).expect("Chunked processing failed");
        println!("Chunked processing found {} segments", segments_chunked.len());

        // Both should find the same number of segments (approximately)
        // Allow some variance due to chunk boundary effects
        let diff = (segments_single.len() as i32 - segments_chunked.len() as i32).abs();
        assert!(diff <= 1,
            "Chunked and single processing found different segment counts: {} vs {} (diff: {})",
            segments_single.len(), segments_chunked.len(), diff);
    }

    #[test]
    fn test_vad_large_file_progress() {
        // Generate 120 seconds (2 minutes) of audio - triggers large file threshold
        let audio = generate_test_audio_with_speech(120.0, 16000);
        let total_samples = audio.len();
        println!("Generated {} samples ({:.1}s)", total_samples, total_samples as f32 / 16000.0);

        // This should trigger the large file path (>960,000 samples)
        assert!(total_samples > 960_000, "Audio should be large enough to trigger chunked processing");

        let mut progress_updates = Vec::new();
        let segments = energy_chunks_with_progress(&audio, 2000, |progress, segments| {
            progress_updates.push((progress, segments));
            true // Don't cancel
        }).expect("Processing failed");

        println!("Found {} segments with {} progress updates", segments.len(), progress_updates.len());

        // The synthetic signal is not real speech, so Silero may merge it into
        // one long segment. This test is specifically for the large-file path:
        // it must still emit speech and report monotonic progress through 100%.
        assert!(!segments.is_empty(), "Expected at least one speech segment");
        assert!(
            segments.iter().all(|segment| !segment.samples.is_empty()
                && segment.end_timestamp_ms > segment.start_timestamp_ms),
            "Expected all speech segments to contain audio with positive duration"
        );

        // Should have received progress updates
        assert!(!progress_updates.is_empty(), "Expected progress updates for large file");
        assert_eq!(
            progress_updates.last().map(|(progress, _)| *progress),
            Some(100),
            "Expected progress to reach 100%"
        );
        assert!(
            progress_updates
                .windows(2)
                .all(|pair| pair[0].0 < pair[1].0),
            "Expected progress updates to increase monotonically: {:?}",
            progress_updates
        );
    }

    #[test]
    fn test_vad_cancellation() {
        let audio = generate_test_audio_with_speech(120.0, 16000);

        // Cancel at 50%
        let result = energy_chunks_with_progress(&audio, 2000, |progress, _| {
            progress < 50 // Cancel when reaching 50%
        });

        // Should return error due to cancellation
        assert!(result.is_err(), "Expected cancellation error");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("cancelled"), "Error should mention cancellation: {}", err_msg);
    }

    #[test]
    fn test_vad_continuous_processor_state_across_chunks() {
        // Test that VAD state is correctly maintained across chunk boundaries
        let mut processor = energy_processor(2000);

        // Generate audio with a speech segment that spans a chunk boundary
        let chunk_size = 160_000; // 10 seconds
        let audio = generate_test_audio_with_speech(30.0, 16000); // 30 seconds

        // Process in 10-second chunks
        let mut all_segments = Vec::new();
        for (i, chunk) in audio.chunks(chunk_size).enumerate() {
            let segments = processor.process_audio(chunk).expect("Processing failed");
            println!("Chunk {}: processed {} samples, found {} segments", i, chunk.len(), segments.len());
            all_segments.extend(segments);
        }

        // Flush remaining
        let final_segments = processor.flush().expect("Flush failed");
        all_segments.extend(final_segments);

        println!("Total segments found: {}", all_segments.len());

        // Should find speech segments
        assert!(all_segments.len() >= 1, "Expected at least 1 speech segment");
    }

    #[test]
    fn test_vad_400ms_vs_2000ms_segmentation() {
        // Demonstrates why 2000ms redemption is needed for batch processing:
        // 400ms creates excessive fragmentation, 2000ms bridges natural pauses.
        //
        // Audio pattern: 60s with 5s speech / 5s silence cycles
        // Natural pauses within speech (sentence gaps) are 500ms-1.5s
        let audio = generate_test_audio_with_speech(60.0, 16000);

        let segments_400 = energy_chunks(&audio, 400).expect("400ms processing failed");
        let segments_2000 = energy_chunks(&audio, 2000).expect("2000ms processing failed");

        println!(
            "400ms redemption: {} segments, 2000ms redemption: {} segments",
            segments_400.len(),
            segments_2000.len()
        );

        // 2000ms should produce fewer or equal segments (bridges more pauses)
        assert!(
            segments_2000.len() <= segments_400.len(),
            "2000ms redemption ({} segments) should not produce more segments than 400ms ({} segments)",
            segments_2000.len(),
            segments_400.len()
        );

        // Verify segments have reasonable durations with 2000ms
        for (i, seg) in segments_2000.iter().enumerate() {
            let duration_ms = seg.end_timestamp_ms - seg.start_timestamp_ms;
            println!("2000ms segment {}: {:.0}ms duration", i, duration_ms);
            // Each segment should be at least 250ms (min_speech_time)
            assert!(duration_ms >= 200.0, "Segment {} too short: {:.0}ms", i, duration_ms);
        }
    }
    /// Leading silence, then speech that runs to the end of the buffer.
    ///
    /// This is the shape that matters for the flush path: an utterance that begins
    /// late in a long session and is still in progress when recording stops.
    fn generate_late_speech_audio(
        silence_seconds: f32,
        speech_seconds: f32,
        sample_rate: u32,
    ) -> Vec<f32> {
        let silence_samples = (silence_seconds * sample_rate as f32) as usize;
        let speech = generate_test_audio_with_speech(speech_seconds, sample_rate);

        let mut samples = vec![0.0f32; silence_samples];
        samples.extend_from_slice(&speech);
        samples
    }

    /// A speech segment's `start_timestamp_ms` must never point past the audio the VAD
    /// has actually been fed: it must be non-negative, no later than the segment's own
    /// `end_timestamp_ms`, and no later than the duration of audio supplied so far.
    ///
    /// This used to be violated because `speech_start_sample` was computed as
    /// `processed_samples + timestamp_ms` where silero's `timestamp_ms` is ALREADY
    /// session-absolute (`processed_duration() - pre_speech_pad`), which doubled the
    /// position. The only reader was the force-end branch in `flush()`, so in
    /// production the corruption escaped as one phantom segment per recording,
    /// timestamped past the end of the audio. The error grows with how late the
    /// utterance starts, which is why it took a long recording to surface. The private
    /// field that carried the bug is gone, so this now checks the same invariant on the
    /// emitted segment's public timestamps instead.
    #[test]
    fn speech_start_never_exceeds_audio_fed() {
        // 20s of silence, then 3s of speech still running when the buffer ends.
        let audio = generate_late_speech_audio(20.0, 3.0, 16000);
        let audio_duration_ms = (audio.len() as f64 / VAD_SAMPLE_RATE as f64) * 1000.0;

        let mut processor =
            energy_processor(2000);
        let segments = processor
            .process_audio(&audio)
            .expect("process_audio failed");
        assert!(
            segments.is_empty(),
            "process_audio completed a segment, so flush() would not exercise force-end"
        );

        assert!(
            processor.is_speaking(),
            "expected to still be mid-speech at the end of the buffer; the invariant \
             below would not be exercised otherwise"
        );

        let flushed = processor.flush().expect("flush failed");
        assert_eq!(flushed.len(), 1, "force-end must emit exactly one segment");

        let segment = &flushed[0];
        assert!(
            segment.start_timestamp_ms >= 0.0,
            "segment starts before the beginning of the audio: {:.0}ms",
            segment.start_timestamp_ms
        );
        assert!(
            segment.start_timestamp_ms <= segment.end_timestamp_ms,
            "segment starts after it ends: {:.0}ms -> {:.0}ms",
            segment.start_timestamp_ms,
            segment.end_timestamp_ms
        );
        assert!(
            segment.start_timestamp_ms <= audio_duration_ms,
            "segment starts at {:.0}ms, beyond the {:.0}ms of audio fed so far",
            segment.start_timestamp_ms,
            audio_duration_ms
        );
    }

    /// A forced segment's timestamps and samples must describe the same real audio interval.
    #[test]
    fn test_flush_segment_timestamps_stay_within_audio_duration() {
        let audio = generate_late_speech_audio(20.0, 3.0, 16000);
        let audio_duration_ms = (audio.len() as f64 / VAD_SAMPLE_RATE as f64) * 1000.0;

        assert_eq!(audio.len(), 368_000);

        let mut processor =
            energy_processor(2000);

        assert!(
            audio.len() % processor.frame_len() != 0,
            "fixture must require terminal VAD padding"
        );

        let segments = processor
            .process_audio(&audio)
            .expect("process_audio failed");
        assert!(
            segments.is_empty(),
            "process_audio completed a segment, so flush() would not exercise force-end"
        );
        assert!(
            processor.is_speaking(),
            "expected to still be mid-speech before flush()"
        );

        let flushed = processor.flush().expect("flush failed");
        assert_eq!(
            flushed.len(),
            1,
            "force-end must emit exactly one segment"
        );

        let segment = &flushed[0];
        let start_sample = ((segment.start_timestamp_ms / 1000.0)
            * VAD_SAMPLE_RATE as f64)
            .round() as usize;

        assert!(
            segment.start_timestamp_ms <= audio_duration_ms,
            "segment starts at {:.0}ms, beyond the {:.0}ms of audio supplied",
            segment.start_timestamp_ms,
            audio_duration_ms
        );
        assert_eq!(
            segment.end_timestamp_ms, 23_000.0,
            "forced segment must end at the real audio endpoint"
        );
        assert!(
            segment.end_timestamp_ms <= audio_duration_ms,
            "segment ends at {:.0}ms, beyond the {:.0}ms of audio supplied",
            segment.end_timestamp_ms,
            audio_duration_ms
        );
        assert!(
            segment.end_timestamp_ms >= segment.start_timestamp_ms,
            "segment ends before it starts: {:.0}ms -> {:.0}ms",
            segment.start_timestamp_ms,
            segment.end_timestamp_ms
        );
        assert_eq!(
            segment.samples.as_slice(),
            &audio[start_sample..],
            "forced payload must contain the exact real audio interval named by its timestamps"
        );
        assert_eq!(segment.samples.len(), audio.len() - start_sample);

        let timestamp_sample_count = (((segment.end_timestamp_ms
            - segment.start_timestamp_ms)
            / 1000.0)
            * VAD_SAMPLE_RATE as f64)
            .round() as usize;
        assert_eq!(
            timestamp_sample_count,
            segment.samples.len(),
            "timestamp duration and payload length must describe the same sample interval"
        );

        let payload_end_ms = segment.start_timestamp_ms
            + (segment.samples.len() as f64 / VAD_SAMPLE_RATE as f64) * 1000.0;
        assert!(
            (payload_end_ms - segment.end_timestamp_ms).abs() < 0.001,
            "payload ends at {payload_end_ms:.3}ms, timestamp ends at {:.3}ms",
            segment.end_timestamp_ms
        );

        assert!(
            processor.flush().expect("second flush failed").is_empty(),
            "flush() must not emit the same forced segment twice"
        );
    }
}
