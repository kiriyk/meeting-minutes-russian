#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerTurn {
    pub start_ms: f64,
    pub end_ms: f64,
    pub speaker: String,
}
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechRegion {
    pub start_ms: f64,
    pub end_ms: f64,
}
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerRange {
    pub start_ms: f64,
    pub end_ms: f64,
    pub speaker: Option<String>,
}

/// Partition a VAD phrase at speaker boundaries without dropping any audio.
pub fn split_at_speakers(start_ms: f64, end_ms: f64, turns: &[SpeakerTurn]) -> Vec<SpeakerRange> {
    if !start_ms.is_finite() || !end_ms.is_finite() || end_ms <= start_ms {
        return Vec::new();
    }
    let valid: Vec<_> = turns
        .iter()
        .filter(|t| {
            t.start_ms.is_finite()
                && t.end_ms.is_finite()
                && t.end_ms > t.start_ms
                && t.end_ms > start_ms
                && t.start_ms < end_ms
                && !t.speaker.is_empty()
        })
        .collect();
    let mut boundaries = vec![start_ms, end_ms];
    for turn in &valid {
        boundaries.push(turn.start_ms.max(start_ms));
        boundaries.push(turn.end_ms.min(end_ms));
    }
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup();
    let mut ranges: Vec<SpeakerRange> = Vec::new();
    for pair in boundaries.windows(2) {
        // Count each speaker once, including overlapping turns of the same speaker.
        let mut speakers: Vec<&str> = valid
            .iter()
            .filter(|t| t.start_ms < pair[1] && t.end_ms > pair[0])
            .map(|t| t.speaker.as_str())
            .collect();
        speakers.sort_unstable();
        speakers.dedup();
        let speaker = if speakers.len() == 1 {
            Some(speakers[0].to_owned())
        } else {
            None
        };
        if let Some(previous) = ranges.last_mut() {
            if previous.speaker == speaker {
                previous.end_ms = pair[1];
                continue;
            }
        }
        ranges.push(SpeakerRange {
            start_ms: pair[0],
            end_ms: pair[1],
            speaker,
        });
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    fn turn(start: f64, end: f64, speaker: &str) -> SpeakerTurn {
        SpeakerTurn {
            start_ms: start,
            end_ms: end,
            speaker: speaker.into(),
        }
    }
    #[test]
    fn speaker_changes_split_a_phrase_without_losing_audio() {
        let ranges = split_at_speakers(
            0.0,
            6000.0,
            &[turn(0.0, 3000.0, "A"), turn(3000.0, 6000.0, "B")],
        );
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].speaker.as_deref(), Some("A"));
        assert_eq!(ranges[0].end_ms, ranges[1].start_ms);
        assert_eq!(ranges[1].speaker.as_deref(), Some("B"));
        assert_eq!(ranges[1].end_ms, 6000.0);
    }
    #[test]
    fn overlapping_speakers_with_equal_coverage_remain_unknown() {
        let ranges = split_at_speakers(
            0.0,
            6000.0,
            &[turn(0.0, 4000.0, "A"), turn(2000.0, 6000.0, "B")],
        );
        assert_eq!(
            ranges
                .iter()
                .map(|r| r.speaker.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("A"), None, Some("B")]
        );
        assert_eq!(ranges[1].start_ms, 2000.0);
        assert_eq!(ranges[1].end_ms, 4000.0);
    }
    #[test]
    fn no_diarization_keeps_the_original_range() {
        assert_eq!(
            split_at_speakers(1000.0, 2000.0, &[]),
            vec![SpeakerRange {
                start_ms: 1000.0,
                end_ms: 2000.0,
                speaker: None
            }]
        );
    }
    #[test]
    fn adjacent_turns_of_one_speaker_merge_and_outside_turns_are_clipped() {
        let ranges = split_at_speakers(
            1000.0,
            4000.0,
            &[turn(0.0, 2000.0, "A"), turn(2000.0, 5000.0, "A")],
        );
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].speaker.as_deref(), Some("A"));
        assert_eq!((ranges[0].start_ms, ranges[0].end_ms), (1000.0, 4000.0));
    }
    #[test]
    fn invalid_turns_do_not_corrupt_timing() {
        let ranges = split_at_speakers(
            1000.0,
            2000.0,
            &[turn(f64::NAN, 3000.0, "A"), turn(4000.0, 1000.0, "B")],
        );
        assert_eq!(ranges[0].speaker, None);
        assert!(split_at_speakers(2000.0, 1000.0, &[]).is_empty());
    }
}
