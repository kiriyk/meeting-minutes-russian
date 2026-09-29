// Retranscription module - allows re-processing stored audio with different settings

use super::common::{create_transcript_segments, split_segment_at_silence, write_transcripts_json};
use super::constants::AUDIO_EXTENSIONS;
use super::retranscription_engine::{BatchEngine, Provider};
use crate::audio::decoder::decode_audio_file;
use crate::audio::vad::get_speech_chunks_with_progress;
use crate::config::{DEFAULT_PARAKEET_MODEL, DEFAULT_WHISPER_MODEL};
use crate::parakeet_engine::ParakeetEngine;
use crate::state::AppState;
use crate::whisper_engine::WhisperEngine;
use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, Runtime};

/// Global flag to track if retranscription is in progress
static RETRANSCRIPTION_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Global flag to signal cancellation
static RETRANSCRIPTION_CANCELLED: AtomicBool = AtomicBool::new(false);

/// RAII guard for RETRANSCRIPTION_IN_PROGRESS flag
/// Ensures flag is cleared even if retranscription panics or returns early
struct RetranscriptionGuard;

impl RetranscriptionGuard {
    /// Create guard and set flag atomically
    fn acquire() -> Result<Self, String> {
        if RETRANSCRIPTION_IN_PROGRESS
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err("Retranscription already in progress".to_string());
        }
        Ok(RetranscriptionGuard)
    }
}

impl Drop for RetranscriptionGuard {
    fn drop(&mut self) {
        RETRANSCRIPTION_IN_PROGRESS.store(false, Ordering::SeqCst);
    }
}

/// VAD redemption time in milliseconds - bridges natural pauses in speech
/// Batch processing needs longer redemption (2000ms) than the live pipeline
/// (500ms) because the entire file is processed at once by VAD with no
/// latency requirement, and short redemption fragments speech at every
/// natural sentence/topic pause (500ms-2s)
const VAD_REDEMPTION_TIME_MS: u32 = 2000;

/// Progress update emitted during retranscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionProgress {
    pub meeting_id: String,
    pub stage: String, // "decoding", "transcribing", "saving"
    pub progress_percentage: u32,
    pub message: String,
}

/// Result of retranscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionResult {
    pub meeting_id: String,
    pub segments_count: usize,
    pub duration_seconds: f64,
    pub language: Option<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Error during retranscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionError {
    pub meeting_id: String,
    pub error: String,
}

/// Check if retranscription is currently in progress
pub fn is_retranscription_in_progress() -> bool {
    RETRANSCRIPTION_IN_PROGRESS.load(Ordering::SeqCst)
}

/// Cancel ongoing retranscription
pub fn cancel_retranscription() {
    RETRANSCRIPTION_CANCELLED.store(true, Ordering::SeqCst);
}

/// Start retranscription of a meeting's audio
async fn start_retranscription<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    meeting_folder_path: String,
    language: Option<String>,
    model: Option<String>,
    provider: Provider,
    diarization_enabled: bool,
    diarization_model: crate::diarization::DiarizationModel,
    guard: RetranscriptionGuard,
) -> Result<RetranscriptionResult> {
    let result = run_retranscription(
        app.clone(),
        meeting_id.clone(),
        meeting_folder_path,
        language,
        model,
        provider,
        diarization_enabled,
        diarization_model,
    )
    .await;
    provider.unload().await;
    // Terminal events mean a new job can start immediately.
    drop(guard);

    match &result {
        Ok(res) => {
            let _ = app.emit(
                "retranscription-complete",
                serde_json::json!({
                    "meeting_id": res.meeting_id,
                    "segments_count": res.segments_count,
                    "duration_seconds": res.duration_seconds,
                    "language": res.language,
                    "warnings": res.warnings
                }),
            );
        }
        Err(e) => {
            let _ = app.emit(
                "retranscription-error",
                RetranscriptionError {
                    meeting_id: meeting_id.clone(),
                    error: e.to_string(),
                },
            );
        }
    }

    result
}

/// Find audio file in meeting folder
/// Tries common names first, then scans for any file with an audio extension
fn find_audio_file(folder: &Path) -> Result<PathBuf> {
    let candidates = [
        "audio.mp4",
        "audio.m4a",
        "audio.wav",
        "audio.mp3",
        "audio.flac",
        "audio.ogg",
        "recording.mp4",
        "audio.mkv",
        "audio.webm",
        "audio.wma",
    ];

    for name in candidates {
        let path = folder.join(name);
        if path.exists() {
            return Ok(path);
        }
    }

    // Fallback: scan folder for any file with an audio extension
    if let Ok(entries) = std::fs::read_dir(folder) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if AUDIO_EXTENSIONS.contains(&ext.as_str()) {
                    return Ok(path);
                }
            }
        }
    }

    Err(anyhow!("No audio file found in: {}", folder.display()))
}

/// Internal function to run retranscription
async fn run_retranscription<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    meeting_folder_path: String,
    language: Option<String>,
    model: Option<String>,
    provider: Provider,
    diarization_enabled: bool,
    diarization_model: crate::diarization::DiarizationModel,
) -> Result<RetranscriptionResult> {
    let folder_path = PathBuf::from(&meeting_folder_path);
    let audio_path = find_audio_file(&folder_path)?;
    let language = match provider {
        Provider::GigaAm | Provider::Tone => Some("ru".to_string()),
        Provider::Parakeet => None,
        Provider::Whisper => language,
    };
    let mut warnings = Vec::new();

    info!(
        "Starting retranscription for meeting {} with language {:?}, model {:?}, provider {:?}",
        meeting_id, language, model, provider
    );

    // Emit progress: decoding
    emit_progress(&app, &meeting_id, "decoding", 5, "Decoding audio file...");

    // Check for cancellation
    if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
        return Err(anyhow!("Retranscription cancelled"));
    }

    // Decode the audio file (CPU-intensive, run in blocking task)
    let path_for_decode = audio_path.clone();
    let decoded = tokio::task::spawn_blocking(move || decode_audio_file(&path_for_decode))
        .await
        .map_err(|e| anyhow!("Decode task panicked: {}", e))??;
    let duration_seconds = decoded.duration_seconds;

    info!(
        "Decoded audio: {:.2}s, {}Hz, {} channels",
        duration_seconds, decoded.sample_rate, decoded.channels
    );

    emit_progress(
        &app,
        &meeting_id,
        "decoding",
        15,
        "Converting audio format...",
    );

    // Check for cancellation
    if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
        return Err(anyhow!("Retranscription cancelled"));
    }

    // Convert to 16kHz mono format (CPU-intensive, run in blocking task)
    let app_for_resample = app.clone();
    let meeting_for_resample = meeting_id.clone();
    let audio_samples = tokio::task::spawn_blocking(move || {
        decoded.to_whisper_format_cancellable(|percent, message| {
            emit_progress(
                &app_for_resample,
                &meeting_for_resample,
                "decoding",
                15 + percent * 5 / 100,
                message,
            );
            if percent % 10 == 0 {
                info!("Audio conversion progress: {}%", percent);
            }
            !RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst)
        })
    })
    .await
    .map_err(|e| anyhow!("Resample task panicked: {}", e))??;
    info!(
        "Converted to 16kHz mono format: {} samples",
        audio_samples.len()
    );

    emit_progress(&app, &meeting_id, "vad", 20, "Detecting speech segments...");

    // Check for cancellation
    if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
        return Err(anyhow!("Retranscription cancelled"));
    }

    let audio_samples = Arc::new(audio_samples);
    let mut community_speech = None;
    check_cancelled()?;
    let turns = if diarization_enabled {
        emit_progress(
            &app,
            &meeting_id,
            "diarizing",
            20,
            "Identifying speakers across the recording...",
        );
        let dir = crate::diarization::models_dir(&app, diarization_model)?;
        let samples = audio_samples.clone();
        let app_for_diarization = app.clone();
        let meeting_for_diarization = meeting_id.clone();
        let result = tokio::task::spawn_blocking(move || {
            let mut previous = u32::MAX;
            crate::diarization::diarize_with_model(
                &samples,
                &dir,
                diarization_model,
                |percentage| {
                    if percentage != previous {
                        previous = percentage;
                        emit_progress(
                            &app_for_diarization,
                            &meeting_for_diarization,
                            "diarizing",
                            20 + percentage * 15 / 100,
                            &format!("Identifying speakers across the recording... {percentage}%"),
                        );
                    }
                    !RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst)
                },
            )
        })
        .await;
        // Community-1 checks between windows; legacy waits for its native call.
        // Both keep the job busy until cleanup has actually finished.
        check_cancelled()?;
        match result {
            Ok(Ok(output)) => {
                community_speech = output
                    .speech_regions
                    .map(|regions| speech_from_regions(&audio_samples, &regions));
                output.turns
            }
            error => {
                let detail = match error {
                    Ok(Err(e)) => e.to_string(),
                    Err(e) => e.to_string(),
                    _ => unreachable!(),
                };
                warn!("Diarization unavailable: {}", detail);
                warnings.push(format!("Speaker identification was unavailable ({detail}). Text and timestamps were transcribed without speaker labels."));
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    let speech_segments = if let Some(speech) = community_speech {
        info!(
            "Community-1 detected {} speech regions; skipping Silero VAD",
            speech.len()
        );
        speech
    } else {
        let samples_for_vad = audio_samples.clone();
        let app_for_vad = app.clone();
        let meeting_id_for_vad = meeting_id.clone();

        let vad_result = tokio::task::spawn_blocking(move || {
            get_speech_chunks_with_progress(
                &samples_for_vad,
                VAD_REDEMPTION_TIME_MS,
                |vad_progress, segments_found| {
                    // Map VAD progress to the remaining segmentation stage (30-35)
                    let overall_progress = 30 + (vad_progress as f32 * 0.05) as u32;
                    emit_progress(
                        &app_for_vad,
                        &meeting_id_for_vad,
                        "vad",
                        overall_progress,
                        &format!(
                            "Detecting speech segments... {}% ({} found)",
                            vad_progress, segments_found
                        ),
                    );

                    // Return false to cancel if cancellation requested
                    !RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst)
                },
            )
        })
        .await
        .map_err(|e| anyhow!("VAD task panicked: {}", e))?;
        check_cancelled()?;
        let speech_segments = vad_result.map_err(|e| anyhow!("VAD processing failed: {}", e))?;

        let total_segments = speech_segments.len();
        info!(
            "VAD detected {} speech segments (redemption_time={}ms)",
            total_segments, VAD_REDEMPTION_TIME_MS
        );

        // Diagnostic: log segment duration distribution
        if !speech_segments.is_empty() {
            let durations_ms: Vec<f64> = speech_segments
                .iter()
                .map(|s| s.end_timestamp_ms - s.start_timestamp_ms)
                .collect();
            let total_speech_ms: f64 = durations_ms.iter().sum();
            let avg_duration = total_speech_ms / durations_ms.len() as f64;
            let min_duration = durations_ms.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_duration = durations_ms
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            info!(
            "VAD segment stats: avg={:.0}ms, min={:.0}ms, max={:.0}ms, total_speech={:.1}s/{:.1}s ({:.0}%)",
            avg_duration, min_duration, max_duration,
            total_speech_ms / 1000.0, duration_seconds,
            (total_speech_ms / 1000.0 / duration_seconds) * 100.0
        );
            // Log first 10 segments for detailed inspection
            for (i, seg) in speech_segments.iter().take(10).enumerate() {
                let dur = seg.end_timestamp_ms - seg.start_timestamp_ms;
                debug!(
                    "  Segment {}: {:.0}ms-{:.0}ms ({:.0}ms, {} samples)",
                    i,
                    seg.start_timestamp_ms,
                    seg.end_timestamp_ms,
                    dur,
                    seg.samples.len()
                );
            }
            if total_segments > 10 {
                debug!("  ... and {} more segments", total_segments - 10);
            }
        }

        if total_segments == 0 {
            warn!("No speech detected in audio");
            return Err(anyhow!("No speech detected in audio file"));
        }

        speech_segments
    };
    if speech_segments.is_empty() {
        return Err(anyhow!("No speech detected in audio file"));
    }
    check_cancelled()?;
    emit_progress(
        &app,
        &meeting_id,
        "transcribing",
        35,
        "Loading transcription engine...",
    );
    check_cancelled()?;
    let app_for_engine = app.clone();
    let engine = tokio::task::spawn_blocking(move || {
        tauri::async_runtime::block_on(BatchEngine::load(
            &app_for_engine,
            provider,
            model.as_deref(),
        ))
    })
    .await
    .map_err(|e| anyhow!("Engine loading task failed: {e}"))??;
    check_cancelled()?;
    let processable_segments = prepare_segments(&speech_segments, &turns);

    let processable_count = processable_segments.len();
    info!(
        "Processing {} segments (after splitting)",
        processable_count
    );

    // Process each speech segment with progress updates
    let mut all_transcripts: Vec<(String, f64, f64)> = Vec::new(); // (text, start_ms, end_ms)
    let mut speakers = Vec::new();
    let mut total_confidence = 0.0f32;

    for (i, (segment, speaker)) in processable_segments.iter().enumerate() {
        // Check for cancellation before each segment
        if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
            return Err(anyhow!("Retranscription cancelled"));
        }

        // Calculate progress (25% to 80% range for transcription)
        let progress = 35 + ((i as f32 / processable_count as f32) * 45.0) as u32;
        let segment_duration_sec = (segment.end_timestamp_ms - segment.start_timestamp_ms) / 1000.0;
        emit_progress(
            &app,
            &meeting_id,
            "transcribing",
            progress,
            &format!(
                "Transcribing segment {} of {} ({:.1}s)...",
                i + 1,
                processable_count,
                segment_duration_sec
            ),
        );

        let segment_engine = engine.clone();
        // Speaker changes can produce tiny fragments. Pad inference input without
        // dropping the fragment or changing its original audio timestamps.
        let mut samples = segment.samples.clone();
        if samples.len() < 1600 {
            samples.resize(1600, 0.0);
        }
        let segment_language = language.clone();
        let (text, conf) = tokio::task::spawn_blocking(move || {
            tauri::async_runtime::block_on(segment_engine.transcribe(samples, segment_language))
        })
        .await
        .map_err(|e| anyhow!("Transcription task failed: {e}"))?
        .map_err(|e| anyhow!("Transcription failed on segment {i}: {e}"))?;
        check_cancelled()?;

        // Skip empty transcripts
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            debug!(
                "Segment {}/{}: {:.1}s, conf={:.2}, text='{}'",
                i + 1,
                processable_count,
                segment_duration_sec,
                conf,
                if trimmed.len() > 80 {
                    let mut end = 80;
                    while !trimmed.is_char_boundary(end) {
                        end -= 1;
                    }
                    &trimmed[..end]
                } else {
                    trimmed
                }
            );
            all_transcripts.push((text, segment.start_timestamp_ms, segment.end_timestamp_ms));
            speakers.push(speaker.clone());
            total_confidence += conf;
        } else {
            debug!(
                "Segment {}/{}: {:.1}s — empty transcription",
                i + 1,
                processable_count,
                segment_duration_sec
            );
        }
    }

    let transcribed_count = all_transcripts.len();
    let avg_confidence = if transcribed_count > 0 {
        total_confidence / transcribed_count as f32
    } else {
        0.0
    };

    info!(
        "Transcription complete: {} segments transcribed out of {}, avg confidence: {:.2}",
        transcribed_count, processable_count, avg_confidence
    );

    // Check for cancellation
    if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
        return Err(anyhow!("Retranscription cancelled"));
    }

    emit_progress(&app, &meeting_id, "saving", 80, "Saving transcripts...");

    let mut segments = create_transcript_segments(&all_transcripts);
    for (segment, speaker) in segments.iter_mut().zip(speakers) {
        segment.speaker = speaker;
    }
    let app_state = app
        .try_state::<AppState>()
        .ok_or_else(|| anyhow!("App state not available"))?;
    save_segments(app_state.db_manager.pool(), &meeting_id, &segments).await?;

    // Write updated transcripts.json and metadata.json to the meeting folder
    emit_progress(
        &app,
        &meeting_id,
        "saving",
        90,
        "Writing transcript files...",
    );

    if let Err(e) = write_transcripts_json(&folder_path, &segments) {
        warn!("Failed to write transcripts.json: {}", e);
        warnings.push(format!(
            "Transcript saved in the app, but transcripts.json could not be updated: {e}"
        ));
    }

    // Find audio filename for metadata
    let audio_filename = audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio.mp4")
        .to_string();

    if let Err(e) =
        write_retranscription_metadata(&folder_path, &meeting_id, duration_seconds, &audio_filename)
    {
        warn!("Failed to update metadata.json: {}", e);
        warnings.push(format!(
            "Transcript saved, but metadata.json could not be updated: {e}"
        ));
    }

    emit_progress(
        &app,
        &meeting_id,
        "complete",
        100,
        "Retranscription complete",
    );

    Ok(RetranscriptionResult {
        meeting_id,
        segments_count: segments.len(),
        duration_seconds,
        language,
        warnings,
    })
}

fn check_cancelled() -> Result<()> {
    if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
        return Err(anyhow!("Retranscription cancelled"));
    }
    Ok(())
}

/// Community-1 already detects speech. Union overlapping speaker activity before
/// extracting audio, so a simultaneous conversation is never transcribed twice.
fn speech_from_regions(
    samples: &[f32],
    turns: &[crate::diarization::turns::SpeechRegion],
) -> Vec<crate::audio::vad::SpeechSegment> {
    use crate::audio::vad::SpeechSegment;
    let mut regions: Vec<_> = turns
        .iter()
        .filter_map(|turn| {
            if !turn.start_ms.is_finite()
                || !turn.end_ms.is_finite()
                || turn.end_ms <= turn.start_ms
            {
                return None;
            }
            let start = (turn.start_ms.max(0.0) * 16.0).round() as usize;
            let end = (turn.end_ms.max(0.0) * 16.0).round() as usize;
            let start = start.min(samples.len());
            let end = end.min(samples.len());
            (end > start).then_some((start, end))
        })
        .collect();
    regions.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in regions {
        if let Some(previous) = merged.last_mut() {
            if start <= previous.1 {
                previous.1 = previous.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
        .into_iter()
        .map(|(start, end)| SpeechSegment {
            samples: samples[start..end].to_vec(),
            start_timestamp_ms: start as f64 / 16.0,
            end_timestamp_ms: end as f64 / 16.0,
            confidence: 1.0,
        })
        .collect()
}

fn prepare_segments(
    segments: &[crate::audio::vad::SpeechSegment],
    turns: &[crate::diarization::turns::SpeakerTurn],
) -> Vec<(crate::audio::vad::SpeechSegment, Option<String>)> {
    use crate::audio::vad::SpeechSegment;
    let mut output = Vec::new();
    for segment in segments {
        let duration_ms = segment.end_timestamp_ms - segment.start_timestamp_ms;
        for range in crate::diarization::turns::split_at_speakers(
            segment.start_timestamp_ms,
            segment.end_timestamp_ms,
            turns,
        ) {
            let index = |ms: f64| {
                (((ms - segment.start_timestamp_ms) / duration_ms * segment.samples.len() as f64)
                    .round() as usize)
                    .min(segment.samples.len())
            };
            let start = index(range.start_ms);
            let end = index(range.end_ms);
            if end <= start {
                continue;
            }
            let piece = SpeechSegment {
                samples: segment.samples[start..end].to_vec(),
                start_timestamp_ms: segment.start_timestamp_ms + start as f64 / 16.0,
                end_timestamp_ms: segment.start_timestamp_ms + end as f64 / 16.0,
                confidence: segment.confidence,
            };
            for chunk in split_segment_at_silence(&piece, 25 * 16000) {
                output.push((chunk, range.speaker.clone()));
            }
        }
    }
    output
}

async fn save_segments(
    pool: &sqlx::SqlitePool,
    meeting_id: &str,
    segments: &[crate::api::TranscriptSegment],
) -> Result<()> {
    if segments.is_empty() {
        return Err(anyhow!(
            "No text was recognized. Existing transcript was preserved."
        ));
    }
    check_cancelled()?;
    let mut tx = pool.begin().await?;
    let exists: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM meetings WHERE id = ?)")
        .bind(meeting_id)
        .fetch_one(&mut *tx)
        .await?;
    if !exists {
        return Err(anyhow!("Meeting no longer exists"));
    }
    sqlx::query("DELETE FROM transcripts WHERE meeting_id = ?")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
    for segment in segments {
        sqlx::query("INSERT INTO transcripts (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time, duration, speaker) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
            .bind(&segment.id).bind(meeting_id).bind(&segment.text).bind(&segment.timestamp)
            .bind(segment.audio_start_time).bind(segment.audio_end_time).bind(segment.duration).bind(&segment.speaker)
            .execute(&mut *tx).await?;
    }
    sqlx::query("UPDATE meetings SET updated_at = ? WHERE id = ?")
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
    check_cancelled()?;
    tx.commit().await?;
    Ok(())
}

/// Emit progress event
fn emit_progress<R: Runtime>(
    app: &AppHandle<R>,
    meeting_id: &str,
    stage: &str,
    progress: u32,
    message: &str,
) {
    let _ = app.emit(
        "retranscription-progress",
        RetranscriptionProgress {
            meeting_id: meeting_id.to_string(),
            stage: stage.to_string(),
            progress_percentage: progress,
            message: message.to_string(),
        },
    );
}

/// Get or initialize the Whisper engine, auto-loading the model if needed
/// If `requested_model` is provided, ensures that specific model is loaded
pub(super) async fn get_or_init_whisper<R: Runtime>(
    app: &AppHandle<R>,
    requested_model: Option<&str>,
) -> Result<Arc<WhisperEngine>> {
    use crate::whisper_engine::commands::WHISPER_ENGINE;

    let engine = {
        let guard = WHISPER_ENGINE.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().cloned()
    };

    match engine {
        Some(e) => {
            // Determine which model to use
            let target_model = match requested_model {
                Some(model) => model.to_string(),
                None => get_configured_whisper_model(app).await?,
            };

            // Check if the correct model is already loaded
            let current_model = e.get_current_model().await;
            let needs_load = match &current_model {
                Some(loaded) => loaded != &target_model,
                None => true,
            };

            if needs_load {
                info!(
                    "Loading Whisper model '{}' (current: {:?})",
                    target_model, current_model
                );

                // Discover available models first (populates the internal cache)
                info!("Discovering available Whisper models...");
                if let Err(discover_err) = e.discover_models().await {
                    warn!(
                        "Error during model discovery (continuing anyway): {}",
                        discover_err
                    );
                }

                match e.load_model(&target_model).await {
                    Ok(_) => {
                        info!("Whisper model '{}' loaded successfully", target_model);
                        Ok(e)
                    }
                    Err(load_err) => {
                        error!(
                            "Failed to load Whisper model '{}': {}",
                            target_model, load_err
                        );
                        Err(anyhow!(
                            "Failed to load Whisper model '{}': {}",
                            target_model,
                            load_err
                        ))
                    }
                }
            } else {
                info!("Whisper model '{}' already loaded", target_model);
                Ok(e)
            }
        }
        None => Err(anyhow!("Whisper engine not initialized")),
    }
}

/// Get the configured Whisper model name from the database
async fn get_configured_whisper_model<R: Runtime>(app: &AppHandle<R>) -> Result<String> {
    debug!("Getting configured Whisper model from database...");

    let app_state = app.try_state::<AppState>().ok_or_else(|| {
        error!("App state not available");
        anyhow!("App state not available")
    })?;

    debug!("Querying transcript_settings table...");

    // Query the transcript settings from the database - get both provider and model
    let result: Option<(String, String)> =
        sqlx::query_as("SELECT provider, model FROM transcript_settings WHERE id = '1'")
            .fetch_optional(app_state.db_manager.pool())
            .await
            .map_err(|e| {
                error!("Failed to query transcript config: {}", e);
                anyhow!("Failed to query transcript config: {}", e)
            })?;

    match result {
        Some((provider, model)) => {
            info!(
                "Found transcript config: provider={}, model={}",
                provider, model
            );

            // Check if provider is Whisper-based
            if provider == "localWhisper" || provider == "whisper" {
                Ok(model)
            } else {
                error!(
                    "Retranscription requires Whisper provider, but configured provider is: {}",
                    provider
                );
                Err(anyhow!("Retranscription requires Whisper. Current provider '{}' does not support retranscription with language selection.", provider))
            }
        }
        None => {
            // Default to configured Whisper model if no config exists
            warn!(
                "No transcript config found, using default model '{}'",
                DEFAULT_WHISPER_MODEL
            );
            Ok(DEFAULT_WHISPER_MODEL.to_string())
        }
    }
}

/// Get or initialize the Parakeet engine, auto-loading the model if needed
pub(super) async fn get_or_init_parakeet<R: Runtime>(
    app: &AppHandle<R>,
    requested_model: Option<&str>,
) -> Result<Arc<ParakeetEngine>> {
    use crate::parakeet_engine::commands::PARAKEET_ENGINE;

    let engine = {
        let guard = PARAKEET_ENGINE.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().cloned()
    };

    match engine {
        Some(e) => {
            // Determine which model to use
            let target_model = match requested_model {
                Some(model) => model.to_string(),
                None => get_configured_parakeet_model(app).await?,
            };

            // Check if the correct model is already loaded
            let current_model = e.get_current_model().await;
            let needs_load = match &current_model {
                Some(loaded) => loaded != &target_model,
                None => true,
            };

            if needs_load {
                info!(
                    "Loading Parakeet model '{}' (current: {:?})",
                    target_model, current_model
                );

                // Discover available models first
                info!("Discovering available Parakeet models...");
                if let Err(discover_err) = e.discover_models().await {
                    warn!(
                        "Error during Parakeet model discovery (continuing anyway): {}",
                        discover_err
                    );
                }

                match e.load_model(&target_model).await {
                    Ok(_) => {
                        info!("Parakeet model '{}' loaded successfully", target_model);
                        Ok(e)
                    }
                    Err(load_err) => {
                        error!(
                            "Failed to load Parakeet model '{}': {}",
                            target_model, load_err
                        );
                        Err(anyhow!(
                            "Failed to load Parakeet model '{}': {}",
                            target_model,
                            load_err
                        ))
                    }
                }
            } else {
                info!("Parakeet model '{}' already loaded", target_model);
                Ok(e)
            }
        }
        None => Err(anyhow!("Parakeet engine not initialized")),
    }
}

/// Get the configured Parakeet model name from the database
async fn get_configured_parakeet_model<R: Runtime>(app: &AppHandle<R>) -> Result<String> {
    debug!("Getting configured Parakeet model from database...");

    let app_state = app.try_state::<AppState>().ok_or_else(|| {
        error!("App state not available");
        anyhow!("App state not available")
    })?;

    // Query the transcript settings from the database
    let result: Option<(String, String)> =
        sqlx::query_as("SELECT provider, model FROM transcript_settings WHERE id = '1'")
            .fetch_optional(app_state.db_manager.pool())
            .await
            .map_err(|e| {
                error!("Failed to query transcript config: {}", e);
                anyhow!("Failed to query transcript config: {}", e)
            })?;

    match result {
        Some((provider, model)) => {
            info!(
                "Found transcript config: provider={}, model={}",
                provider, model
            );

            if provider == "parakeet" {
                Ok(model)
            } else {
                // Default to configured Parakeet model
                warn!("Configured provider is not Parakeet, using default model");
                Ok(DEFAULT_PARAKEET_MODEL.to_string())
            }
        }
        None => {
            // Default to configured Parakeet model if no config exists
            warn!("No transcript config found, using default Parakeet model");
            Ok(DEFAULT_PARAKEET_MODEL.to_string())
        }
    }
}

/// Write or update metadata.json for retranscription (preserves existing fields, adds retranscribed_at)
fn write_retranscription_metadata(
    folder: &Path,
    meeting_id: &str,
    duration_seconds: f64,
    audio_filename: &str,
) -> Result<()> {
    let metadata_path = folder.join("metadata.json");
    let temp_path = folder.join(".metadata.json.tmp");
    let now = chrono::Utc::now().to_rfc3339();

    // Try to read existing metadata and update it
    let json = if metadata_path.exists() {
        let existing = std::fs::read_to_string(&metadata_path)?;
        let mut value: serde_json::Value = serde_json::from_str(&existing)?;
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "duration_seconds".to_string(),
                serde_json::json!(duration_seconds),
            );
            obj.insert("retranscribed_at".to_string(), serde_json::json!(now));
            obj.insert("status".to_string(), serde_json::json!("completed"));
            obj.insert(
                "transcript_file".to_string(),
                serde_json::json!("transcripts.json"),
            );
            obj.remove("detected_summary_language");
        }
        value
    } else {
        serde_json::json!({
            "version": "1.0",
            "meeting_id": meeting_id,
            "created_at": now,
            "completed_at": now,
            "retranscribed_at": now,
            "duration_seconds": duration_seconds,
            "audio_file": audio_filename,
            "transcript_file": "transcripts.json",
            "status": "completed",
            "source": "retranscription"
        })
    };

    let json_string = serde_json::to_string_pretty(&json)?;
    std::fs::write(&temp_path, &json_string)?;
    std::fs::rename(&temp_path, &metadata_path)?;

    info!("Wrote metadata.json to {}", metadata_path.display());
    Ok(())
}

// Tauri commands

/// Response when retranscription is started
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionStarted {
    pub meeting_id: String,
    pub message: String,
}

// Start retranscription (Beta gated using configContext.betaFeatures)
#[tauri::command]
pub async fn start_retranscription_command<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    meeting_folder_path: String,
    language: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    diarization_enabled: Option<bool>,
    diarization_model: Option<crate::diarization::DiarizationModel>,
) -> Result<RetranscriptionStarted, String> {
    let provider = Provider::parse(provider.as_deref()).map_err(|e| e.to_string())?;
    // Reserve before returning to the UI; a queued task must not lose a cancel request.
    let _lifecycle_guard = super::common::acquire_engine_lifecycle_lock().await;
    if super::recording_commands::is_recording().await {
        return Err("Stop recording before retranscribing".into());
    }
    let guard = RetranscriptionGuard::acquire()?;
    RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);

    // Clone values for the spawned task
    let meeting_id_clone = meeting_id.clone();

    // Spawn the retranscription in a background task
    tauri::async_runtime::spawn(async move {
        let result = start_retranscription(
            app,
            meeting_id_clone,
            meeting_folder_path,
            language,
            model,
            provider,
            diarization_enabled.unwrap_or(false),
            diarization_model.unwrap_or_default(),
            guard,
        )
        .await;

        // Errors are already emitted as events in start_retranscription
        // so we just log here for debugging
        if let Err(e) = result {
            error!("Retranscription failed: {}", e);
        }
    });

    Ok(RetranscriptionStarted {
        meeting_id,
        message: "Retranscription started".to_string(),
    })
}

#[tauri::command]
pub async fn cancel_retranscription_command() -> Result<(), String> {
    if !is_retranscription_in_progress() {
        return Err("No retranscription in progress".to_string());
    }
    cancel_retranscription();
    Ok(())
}

#[tauri::command]
pub async fn is_retranscription_in_progress_command() -> bool {
    is_retranscription_in_progress()
}

#[cfg(test)]
mod tests {
    use super::*;
    static TEST_JOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn community_speech_coverage_preserves_short_turns_and_deduplicates_overlap() {
        use crate::diarization::turns::SpeakerTurn;
        let samples: Vec<_> = (0..6 * 16000).map(|i| i as f32).collect();
        let turns = vec![
            SpeakerTurn {
                start_ms: 1500.0,
                end_ms: 2200.0,
                speaker: "B".into(),
            },
            SpeakerTurn {
                start_ms: 1000.0,
                end_ms: 1800.0,
                speaker: "A".into(),
            },
            SpeakerTurn {
                start_ms: 4000.0,
                end_ms: 4040.0,
                speaker: "A".into(),
            },
            SpeakerTurn {
                start_ms: f64::NAN,
                end_ms: 5000.0,
                speaker: "C".into(),
            },
        ];
        let regions = turns
            .iter()
            .map(|t| crate::diarization::turns::SpeechRegion {
                start_ms: t.start_ms,
                end_ms: t.end_ms,
            })
            .collect::<Vec<_>>();
        let speech = speech_from_regions(&samples, &regions);
        assert_eq!(speech.len(), 2);
        assert_eq!(
            (speech[0].start_timestamp_ms, speech[0].end_timestamp_ms),
            (1000.0, 2200.0)
        );
        assert_eq!(speech[0].samples, samples[16000..35200]);
        let pieces = prepare_segments(&speech, &turns);
        assert_eq!(
            pieces.iter().map(|(s, _)| s.samples.len()).sum::<usize>(),
            speech.iter().map(|s| s.samples.len()).sum::<usize>()
        );
        assert!(pieces.iter().any(|(s, label)| label.as_deref() == Some("A")
            && s.start_timestamp_ms == 4000.0
            && s.samples.len() == 640));
        assert!(pieces.iter().any(|(s, label)| label.is_none()
            && s.start_timestamp_ms == 1500.0
            && s.end_timestamp_ms == 1800.0));
    }

    #[test]
    fn community_speech_coverage_clips_to_audio_and_keeps_silence_empty() {
        use crate::diarization::turns::SpeechRegion;
        let samples = vec![0.0; 16000];
        assert!(speech_from_regions(&samples, &[]).is_empty());
        let speech = speech_from_regions(
            &samples,
            &[SpeechRegion {
                start_ms: -100.0,
                end_ms: 2000.0,
            }],
        );
        assert_eq!(speech.len(), 1);
        assert_eq!(speech[0].samples.len(), samples.len());
        assert_eq!(
            (speech[0].start_timestamp_ms, speech[0].end_timestamp_ms),
            (0.0, 1000.0)
        );
    }

    #[test]
    fn cancelled_worker_keeps_its_reservation_until_it_exits() {
        let _test_lock = TEST_JOB_LOCK.lock().unwrap();
        RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);
        let guard = RetranscriptionGuard::acquire().unwrap();
        let release = Arc::new(std::sync::Barrier::new(2));
        let worker_release = release.clone();
        let worker = std::thread::spawn(move || {
            let _guard = guard;
            worker_release.wait();
        });
        cancel_retranscription();
        assert!(is_retranscription_in_progress());
        assert!(RetranscriptionGuard::acquire().is_err());
        release.wait();
        worker.join().unwrap();
        assert!(!is_retranscription_in_progress());
        RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);
    }

    #[tokio::test]
    async fn empty_retranscription_preserves_existing_database_transcript() {
        let _test_lock = TEST_JOB_LOCK.lock().unwrap();
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::query("CREATE TABLE meetings (id TEXT PRIMARY KEY, updated_at TEXT)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("CREATE TABLE transcripts (id TEXT PRIMARY KEY, meeting_id TEXT, transcript TEXT, timestamp TEXT, audio_start_time REAL, audio_end_time REAL, duration REAL, speaker TEXT)").execute(&pool).await.unwrap();
        sqlx::query("INSERT INTO meetings VALUES ('meeting', 'original')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO transcripts (id, meeting_id, transcript) VALUES ('old', 'meeting', 'original text')").execute(&pool).await.unwrap();
        assert!(save_segments(&pool, "meeting", &[]).await.is_err());
        let text: String =
            sqlx::query_scalar("SELECT transcript FROM transcripts WHERE id = 'old'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(text, "original text");
        let mut segments = create_transcript_segments(&[("Новый текст".into(), 0.0, 1000.0)]);
        segments[0].speaker = Some("SPEAKER_00".into());
        cancel_retranscription();
        assert!(save_segments(&pool, "meeting", &segments).await.is_err());
        RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);
        assert_eq!(
            sqlx::query_scalar::<_, String>("SELECT transcript FROM transcripts WHERE id = 'old'")
                .fetch_one(&pool)
                .await
                .unwrap(),
            "original text"
        );
        save_segments(&pool, "meeting", &segments).await.unwrap();
        let row: (String, String, f64) =
            sqlx::query_as("SELECT id, speaker, audio_end_time FROM transcripts")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(row, (segments[0].id.clone(), "SPEAKER_00".into(), 1.0));
        // A failed insertion rolls back the preceding deletion.
        let mut duplicates = create_transcript_segments(&[
            ("First".into(), 0.0, 1000.0),
            ("Second".into(), 1000.0, 2000.0),
        ]);
        duplicates[1].id = duplicates[0].id.clone();
        assert!(save_segments(&pool, "meeting", &duplicates).await.is_err());
        assert_eq!(
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM transcripts")
                .fetch_one(&pool)
                .await
                .unwrap(),
            1
        );
        assert!(save_segments(&pool, "deleted-meeting", &segments)
            .await
            .is_err());
    }

    #[test]
    fn speaker_split_preserves_samples_and_inherits_labels_through_silence_split() {
        use crate::diarization::turns::SpeakerTurn;
        let segment = crate::audio::vad::SpeechSegment {
            samples: vec![0.1; 640_000],
            start_timestamp_ms: 1000.0,
            end_timestamp_ms: 41000.0,
            confidence: 0.9,
        };
        let turns = vec![
            SpeakerTurn {
                start_ms: 1000.0,
                end_ms: 31000.0,
                speaker: "SPEAKER_00".into(),
            },
            SpeakerTurn {
                start_ms: 31000.0,
                end_ms: 41000.0,
                speaker: "SPEAKER_01".into(),
            },
        ];
        let pieces = prepare_segments(&[segment], &turns);
        assert_eq!(
            pieces.iter().map(|(s, _)| s.samples.len()).sum::<usize>(),
            640_000
        );
        assert!(pieces.len() >= 3);
        assert_eq!(pieces.last().unwrap().1.as_deref(), Some("SPEAKER_01"));
        assert!(pieces[..pieces.len() - 1]
            .iter()
            .all(|(_, label)| label.as_deref() == Some("SPEAKER_00")));
        assert!(pieces
            .windows(2)
            .all(|w| w[0].0.end_timestamp_ms == w[1].0.start_timestamp_ms));
    }

    #[test]
    fn test_create_transcript_segments_empty() {
        let transcripts: Vec<(String, f64, f64)> = vec![];
        let segments = create_transcript_segments(&transcripts);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_create_transcript_segments_single() {
        let transcripts = vec![
            ("Hello world".to_string(), 0.0, 1500.0), // 0-1.5 seconds
        ];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello world");
        assert_eq!(segments[0].audio_start_time, Some(0.0));
        assert_eq!(segments[0].audio_end_time, Some(1.5));
        assert_eq!(segments[0].duration, Some(1.5));
    }

    #[test]
    fn test_create_transcript_segments_multiple() {
        let transcripts = vec![
            ("First segment".to_string(), 0.0, 2000.0), // 0-2 seconds
            ("Second segment".to_string(), 3000.0, 5000.0), // 3-5 seconds
            ("Third segment".to_string(), 6500.0, 8000.0), // 6.5-8 seconds
        ];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 3);

        // First segment
        assert_eq!(segments[0].text, "First segment");
        assert_eq!(segments[0].audio_start_time, Some(0.0));
        assert_eq!(segments[0].audio_end_time, Some(2.0));
        assert_eq!(segments[0].duration, Some(2.0));

        // Second segment
        assert_eq!(segments[1].text, "Second segment");
        assert_eq!(segments[1].audio_start_time, Some(3.0));
        assert_eq!(segments[1].audio_end_time, Some(5.0));
        assert_eq!(segments[1].duration, Some(2.0));

        // Third segment
        assert_eq!(segments[2].text, "Third segment");
        assert_eq!(segments[2].audio_start_time, Some(6.5));
        assert_eq!(segments[2].audio_end_time, Some(8.0));
        assert_eq!(segments[2].duration, Some(1.5));
    }

    #[test]
    fn test_create_transcript_segments_trims_whitespace() {
        let transcripts = vec![("  Hello with spaces  ".to_string(), 0.0, 1000.0)];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello with spaces");
    }

    #[test]
    fn test_create_transcript_segments_generates_unique_ids() {
        let transcripts = vec![
            ("Segment one".to_string(), 0.0, 1000.0),
            ("Segment two".to_string(), 1000.0, 2000.0),
        ];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 2);
        assert_ne!(segments[0].id, segments[1].id);
        assert!(segments[0].id.starts_with("transcript-"));
        assert!(segments[1].id.starts_with("transcript-"));
    }

    #[test]
    fn test_cancellation_flag() {
        let _test_lock = TEST_JOB_LOCK.lock().unwrap();
        // Reset flag to known state
        RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);
        RETRANSCRIPTION_IN_PROGRESS.store(false, Ordering::SeqCst);

        assert!(!is_retranscription_in_progress());

        // Test cancellation
        cancel_retranscription();
        assert!(RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst));

        // Reset for other tests
        RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_vad_redemption_time_constant() {
        // Batch processing uses 2000ms to bridge natural pauses in full-file VAD
        assert_eq!(VAD_REDEMPTION_TIME_MS, 2000);
    }

    #[test]
    fn test_find_audio_file_common_candidates() {
        let dir = tempfile::tempdir().unwrap();

        // No audio file → error
        assert!(find_audio_file(dir.path()).is_err());

        // Create audio.mp4 — should be found first
        std::fs::write(dir.path().join("audio.mp4"), b"fake").unwrap();
        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "audio.mp4");
    }

    #[test]
    fn test_find_audio_file_non_mp4_extensions() {
        let dir = tempfile::tempdir().unwrap();

        // Create audio.wav (imported as .wav, not .mp4)
        std::fs::write(dir.path().join("audio.wav"), b"fake").unwrap();
        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "audio.wav");
    }

    #[test]
    fn test_find_audio_file_fallback_scan() {
        let dir = tempfile::tempdir().unwrap();

        // Create a file with an audio extension but non-standard name
        std::fs::write(dir.path().join("my_recording.flac"), b"fake").unwrap();
        // Also add a non-audio file that should be ignored
        std::fs::write(dir.path().join("notes.txt"), b"text").unwrap();

        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "my_recording.flac");
    }

    #[test]
    fn test_find_audio_file_priority_order() {
        let dir = tempfile::tempdir().unwrap();

        // Create both audio.m4a and audio.mp4 — mp4 should win (listed first in candidates)
        std::fs::write(dir.path().join("audio.m4a"), b"fake").unwrap();
        std::fs::write(dir.path().join("audio.mp4"), b"fake").unwrap();
        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "audio.mp4");
    }

    #[test]
    fn test_find_audio_file_empty_folder() {
        let dir = tempfile::tempdir().unwrap();
        let result = find_audio_file(dir.path());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No audio file found"));
    }

    #[test]
    fn test_find_audio_file_nonexistent_folder() {
        let result = find_audio_file(Path::new("/nonexistent/path/12345"));
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_extensions_constant() {
        // Verify all expected formats are covered
        assert!(AUDIO_EXTENSIONS.contains(&"mp4"));
        assert!(AUDIO_EXTENSIONS.contains(&"m4a"));
        assert!(AUDIO_EXTENSIONS.contains(&"wav"));
        assert!(AUDIO_EXTENSIONS.contains(&"mp3"));
        assert!(AUDIO_EXTENSIONS.contains(&"flac"));
        assert!(AUDIO_EXTENSIONS.contains(&"ogg"));
        assert!(AUDIO_EXTENSIONS.contains(&"aac"));
        // FFmpeg-backed formats
        assert!(AUDIO_EXTENSIONS.contains(&"mkv"));
        assert!(AUDIO_EXTENSIONS.contains(&"webm"));
        assert!(AUDIO_EXTENSIONS.contains(&"wma"));
        // Non-audio formats
        assert!(!AUDIO_EXTENSIONS.contains(&"txt"));
        assert!(!AUDIO_EXTENSIONS.contains(&"pdf"));
    }

    #[test]
    fn retranscription_metadata_replaces_stale_duration_without_losing_existing_fields() {
        let dir = tempfile::tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.json");
        std::fs::write(
            &metadata_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "meeting_id": "existing-meeting",
                "duration_seconds": 2523.2,
                "audio_file": "recording.m4a",
                "summary_language": "fr",
                "detected_summary_language": "en",
                "custom_field": "preserve me"
            }))
            .unwrap(),
        )
        .unwrap();

        write_retranscription_metadata(
            dir.path(),
            "ignored-for-existing-metadata",
            5046.4,
            "recording.m4a",
        )
        .unwrap();

        let metadata: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(metadata_path).unwrap()).unwrap();
        assert_eq!(metadata["duration_seconds"], 5046.4);
        assert_eq!(metadata["meeting_id"], "existing-meeting");
        assert_eq!(metadata["audio_file"], "recording.m4a");
        assert_eq!(metadata["summary_language"], "fr");
        assert_eq!(metadata["custom_field"], "preserve me");
        assert!(metadata.get("detected_summary_language").is_none());
    }
}
