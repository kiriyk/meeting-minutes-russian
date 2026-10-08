//! Full Community-1 pipeline; ONNX inference and PLDA/VBx remain in Rust.
use super::{
    file_digest,
    turns::{SpeakerTurn, SpeechRegion},
    DiarizationOutput,
};
use anyhow::{bail, Result};
use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
use std::{collections::HashMap, path::Path};

pub const REVISION: &str = "5d24ffee75f13fb061fa6d10944a64e2dc1d5e6f";
pub struct Artifact {
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}
pub const ARTIFACTS: &[Artifact] = &[
    Artifact {
        name: "segmentation-3.0.onnx",
        size: 5916308,
        sha256: "038b971741ed623af9773ecafdefa4b7bc523520099c2a68f8568b24189e8ad9",
    },
    Artifact {
        name: "wespeaker-voxceleb-resnet34.onnx",
        size: 26894815,
        sha256: "203a4c67112167580ab1fcb62f4568c633499fb283805890aebe1c48564fcc0f",
    },
    Artifact {
        name: "wespeaker-voxceleb-resnet34.onnx.data",
        size: 26673152,
        sha256: "dc105e7857156611381b95cc961b277d8e1e098e7af1c919a77c68c7257ce956",
    },
    Artifact {
        name: "wespeaker-voxceleb-resnet34.min_num_samples.txt",
        size: 4,
        sha256: "e4df891c484d7abb985dadf539fa1883a646dab6337af5cae4159c587b7050cc",
    },
    Artifact {
        name: "plda_lda.npy",
        size: 131200,
        sha256: "e20c9b012bebd1aabda5a38a127e63a43cf35debdc502715fc143e2fb6bc3c4b",
    },
    Artifact {
        name: "plda_mean1.npy",
        size: 2176,
        sha256: "e424c0c352182aa8e0f555dec1f3b30e29a20b9ed6b25d339f112af92e51e36f",
    },
    Artifact {
        name: "plda_mean2.npy",
        size: 640,
        sha256: "6f6fb708a2037197b5b84ffeaa8f140cb878088fbecd6ab042ad26a7691bd2cf",
    },
    Artifact {
        name: "plda_mu.npy",
        size: 1152,
        sha256: "d286d48acf99bbc1ed1502fed0a3e361ae5626ce1870c8be9f7397c5e47886c6",
    },
    Artifact {
        name: "plda_psi.npy",
        size: 1152,
        sha256: "d7128c9ed2f28a9781971805131129f077c04f948e2df12e52dcdb99f2b4e5f5",
    },
    Artifact {
        name: "plda_tr.npy",
        size: 131200,
        sha256: "e700b68cb319de3fafb5fa093eb9222c23c447084741f8d3a533640d425510ee",
    },
];

pub fn models_available(dir: &Path) -> bool {
    ARTIFACTS.iter().all(|artifact| {
        let path = dir.join(artifact.name);
        std::fs::metadata(&path)
            .map(|m| m.len() == artifact.size)
            .unwrap_or(false)
            && file_digest(&path)
                .map(|digest| digest == artifact.sha256)
                .unwrap_or(false)
    })
}

pub fn create_pipeline(dir: &Path) -> Result<OwnedDiarizationPipeline> {
    if !models_available(dir) {
        bail!("Community-1 models are not downloaded or are incomplete");
    }
    crate::ensure_onnx_runtime_available()?;
    Ok(OwnedDiarizationPipeline::from_dir(dir, ExecutionMode::Cpu)?)
}

pub fn diarize(
    samples: &[f32],
    dir: &Path,
    mut progress: impl FnMut(u32) -> bool,
) -> Result<DiarizationOutput> {
    if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
        bail!("Unsupported waveform for Community-1");
    }
    let started = std::time::Instant::now();
    log::info!(
        "Starting Community-1 on CPU for {:.2}s of audio",
        samples.len() as f64 / 16000.0
    );
    let mut pipeline = create_pipeline(dir)?;
    // Upstream speakrs has no in-run progress hook: cancellation is honoured
    // before inference and after it returns (the result is then discarded).
    if !progress(0) {
        bail!("Diarization cancelled");
    }
    let result = pipeline.run(samples)?;
    if !progress(100) {
        bail!("Diarization cancelled");
    }
    // Silence is a successful empty result; it must not trigger another VAD pass.
    let speech_regions = speech_regions_from_counts(&result.speaker_count, samples.len());
    let turns = normalize_turns(result.segments, samples.len());
    log::info!(
        "Community-1 complete: {} turns in {:.2}s",
        turns.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(DiarizationOutput {
        turns,
        speech_regions: Some(speech_regions),
    })
}

fn normalize_turns(segments: Vec<speakrs::Segment>, sample_count: usize) -> Vec<SpeakerTurn> {
    let duration_ms = sample_count as f64 / 16.0;
    let mut turns: Vec<_> = segments
        .into_iter()
        .filter_map(|s| {
            if !s.start.is_finite() || !s.end.is_finite() || s.speaker.is_empty() {
                return None;
            }
            let start_ms = (s.start * 1000.0).max(0.0);
            let end_ms = (s.end * 1000.0).min(duration_ms);
            (end_ms > start_ms).then_some(SpeakerTurn {
                start_ms,
                end_ms,
                speaker: s.speaker,
            })
        })
        .collect();
    turns.sort_by(|a, b| {
        a.start_ms
            .total_cmp(&b.start_ms)
            .then(a.end_ms.total_cmp(&b.end_ms))
            .then(a.speaker.cmp(&b.speaker))
    });
    let mut labels = HashMap::new();
    for turn in &mut turns {
        let next = format!("SPEAKER_{:02}", labels.len());
        turn.speaker = labels.entry(turn.speaker.clone()).or_insert(next).clone();
    }
    turns
}

fn speech_regions_from_counts(counts: &[usize], sample_count: usize) -> Vec<SpeechRegion> {
    let duration_ms = sample_count as f64 / 16.0;
    let timestamp = |i: usize| {
        (i as f64 * speakrs::pipeline::FRAME_STEP_SECONDS
            + 0.5 * speakrs::pipeline::FRAME_DURATION_SECONDS)
            * 1000.0
    };
    let mut start = None;
    let mut speech = Vec::new();
    // A synthetic inactive frame closes final activity with nonzero duration,
    // including a single short frame whose voice has no usable embedding.
    for (i, &count) in counts.iter().chain(std::iter::once(&0)).enumerate() {
        if count > 0 && start.is_none() {
            start = Some(if i == 0 { 0.0 } else { timestamp(i) });
        } else if count == 0 {
            if let Some(start_ms) = start.take() {
                let end_ms = timestamp(i).min(duration_ms);
                if end_ms > start_ms {
                    speech.push(SpeechRegion { start_ms, end_ms });
                }
            }
        }
    }
    speech
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn speech_coverage_keeps_activity_without_a_usable_speaker_embedding() {
        // Even a single active frame must be retained when clustering cannot
        // assign its voice. Coverage is independent of assigned speaker turns.
        let speech = speech_regions_from_counts(&[0, 1, 1, 0, 0, 1, 0], 16000);
        assert_eq!(speech.len(), 2);
        assert!((speech[0].start_ms - 47.84375).abs() < 1e-6);
        assert!((speech[0].end_ms - 81.59375).abs() < 1e-6);
        assert!((speech[1].end_ms - speech[1].start_ms - 16.875).abs() < 1e-6);
        assert!(speech_regions_from_counts(&[0, 0], 16000).is_empty());
        let tail = speech_regions_from_counts(&[0, 0, 1], 16000);
        assert_eq!(tail.len(), 1);
        assert!((tail[0].end_ms - tail[0].start_ms - 16.875).abs() < 1e-6);
    }
    #[test]
    fn labels_follow_first_appearance_and_preserve_overlapping_turns() {
        let turns = normalize_turns(
            vec![
                speakrs::Segment::new(2.0, 8.0, "B"),
                speakrs::Segment::new(-0.1, 3.0, "A"),
                speakrs::Segment::new(4.0, 5.0, "A"),
                speakrs::Segment::new(f64::NAN, 2.0, "C"),
                speakrs::Segment::new(8.0, 9.0, "C"),
            ],
            6 * 16000,
        );
        assert_eq!(turns.len(), 3);
        assert_eq!(
            turns.iter().map(|s| s.speaker.as_str()).collect::<Vec<_>>(),
            ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
        );
        assert_eq!(turns[0].start_ms, 0.0);
        assert_eq!(turns[1].end_ms, 6000.0);
        assert!(turns[1].start_ms < turns[0].end_ms);
    }
    #[test]
    fn corrupt_and_partial_community_bundles_remain_downloadable() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!models_available(dir.path()));
        for artifact in ARTIFACTS {
            std::fs::File::create(dir.path().join(artifact.name))
                .unwrap()
                .set_len(artifact.size)
                .unwrap();
        }
        assert!(!models_available(dir.path()));
    }
    #[test]
    #[ignore = "requires Community-1 models and a multi-speaker fixture"]
    fn community_model_smoke() {
        let dir = std::path::PathBuf::from(std::env::var("COMMUNITY_MODELS_DIR").unwrap());
        let audio = std::path::PathBuf::from(std::env::var("DIARIZATION_AUDIO_FILE").unwrap());
        if !models_available(&dir) {
            super::super::download_community_bundle(
                &dir,
                &tokio_util::sync::CancellationToken::new(),
                |p| println!("Community-1 download: {p}%"),
            )
            .unwrap();
        }
        let decoded = crate::audio::decoder::decode_audio_file(&audio).unwrap();
        let samples = decoded.to_whisper_format();
        let started = std::time::Instant::now();
        let mut percentages = Vec::new();
        let output = diarize(&samples, &dir, |p| {
            percentages.push(p);
            true
        })
        .unwrap();
        let turns = output.turns;
        assert_eq!(percentages.last(), Some(&100));
        assert!(percentages.windows(2).all(|w| w[0] <= w[1]));
        assert!(turns.iter().any(|t| t.speaker == "SPEAKER_00"));
        assert!(turns.iter().any(|t| t.speaker == "SPEAKER_01"));
        assert!(turns.windows(2).all(|w| w[0].start_ms <= w[1].start_ms));
        assert!(turns
            .iter()
            .all(|t| t.end_ms <= samples.len() as f64 / 16.0));
        println!(
            "Community-1: {} turns, {:.2}s audio, {:.2}s inference",
            turns.len(),
            samples.len() as f64 / 16000.0,
            started.elapsed().as_secs_f64()
        );
        let silence = diarize(&vec![0.0; 16000], &dir, |_| true).unwrap();
        assert!(
            silence.turns.is_empty(),
            "silence must not trigger fallback VAD"
        );
        assert!(silence.speech_regions.unwrap().is_empty());
        // Upstream speakrs has no in-run hook: cancellation is checked before (0)
        // and after (100) inference.
        for stop_at in [0, 100] {
            let error = diarize(&samples, &dir, |p| p < stop_at)
                .err()
                .expect("cancellation must discard the diarization");
            assert!(error.to_string().contains("cancelled"));
        }
        let mut pipeline = create_pipeline(&dir).unwrap();
        let retry = pipeline.run(&vec![0.0; 16000]).unwrap();
        assert!(
            retry.segments.is_empty(),
            "cancellation must not poison a reusable pipeline"
        );
    }
}
