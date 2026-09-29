//! Engine-independent speech segmentation over per-frame speech probabilities.

pub const VAD_SAMPLE_RATE: usize = 16_000;

fn ms_to_samples(ms: u32) -> usize {
    ms as usize * VAD_SAMPLE_RATE / 1000
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmenterConfig {
    pub positive_threshold: f32,
    pub negative_threshold: f32,
    pub pre_speech_pad_ms: u32,
    pub post_speech_pad_ms: u32,
    pub redemption_ms: u32,
    pub min_speech_ms: u32,
}

impl SegmenterConfig {
    /// Values tuned for Whisper-sized utterances; unchanged from the silero-rs era.
    pub fn meetily(redemption_ms: u32) -> Self {
        Self {
            positive_threshold: 0.50,
            negative_threshold: 0.35,
            pre_speech_pad_ms: 300,
            post_speech_pad_ms: 400,
            redemption_ms,
            min_speech_ms: 250,
        }
    }
}

/// A sample-exact speech interval `[start_sample, end_sample)` at 16 kHz.
#[derive(Clone, Debug, PartialEq)]
pub struct RawSegment {
    pub start_sample: usize,
    pub end_sample: usize,
    pub samples: Vec<f32>,
}

impl RawSegment {
    pub fn truncate_end(&mut self, max_end: usize) {
        if max_end < self.end_sample {
            let keep = max_end.saturating_sub(self.start_sample);
            self.samples.truncate(keep);
            self.end_sample = self.start_sample + keep;
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum State {
    Silence,
    Pending { start: usize, speech: usize },
    Speech { start: usize },
}

pub struct SpeechSegmenter {
    pre_pad: usize,
    post_pad: usize,
    redemption: usize,
    min_speech: usize,
    positive: f32,
    negative: f32,
    /// Retained audio; `audio[0]` is absolute sample `origin`.
    audio: Vec<f32>,
    origin: usize,
    processed: usize,
    silent: usize,
    /// End of the last emitted segment; later segments never start before it.
    floor: usize,
    state: State,
}

impl SpeechSegmenter {
    pub fn new(config: SegmenterConfig) -> Self {
        Self {
            pre_pad: ms_to_samples(config.pre_speech_pad_ms),
            post_pad: ms_to_samples(config.post_speech_pad_ms),
            redemption: ms_to_samples(config.redemption_ms),
            min_speech: ms_to_samples(config.min_speech_ms),
            positive: config.positive_threshold,
            negative: config.negative_threshold,
            audio: Vec::new(),
            origin: 0,
            processed: 0,
            silent: 0,
            floor: 0,
            state: State::Silence,
        }
    }

    pub fn push_frame(&mut self, frame: &[f32], probability: f32) -> Option<RawSegment> {
        let before = self.processed;
        let after = before + frame.len();
        self.audio.extend_from_slice(frame);
        if probability < self.negative {
            self.silent += frame.len();
        } else {
            self.silent = 0;
        }

        let mut emitted = None;
        match self.state {
            State::Silence => {
                if probability > self.positive {
                    let start = before.saturating_sub(self.pre_pad).max(self.floor);
                    self.state = State::Pending { start, speech: 0 };
                }
            }
            State::Pending { start, speech } => {
                let speech = speech + frame.len();
                self.state = if speech > self.min_speech {
                    State::Speech { start }
                } else if probability < self.negative {
                    State::Silence
                } else {
                    State::Pending { start, speech }
                };
            }
            State::Speech { .. } => {}
        }

        if let State::Speech { start } = self.state {
            if probability < self.negative && self.silent > self.redemption {
                let speech_end = after - self.silent;
                let end = (speech_end + self.post_pad).min(after);
                emitted = Some(self.slice(start, end));
                self.floor = end;
                self.state = State::Silence;
            }
        }

        self.processed = after;
        if matches!(self.state, State::Silence) {
            self.trim();
        }
        emitted
    }

    pub fn finish(&mut self, real_end_sample: usize) -> Option<RawSegment> {
        let state = std::mem::replace(&mut self.state, State::Silence);
        let State::Speech { start } = state else {
            self.trim();
            return None;
        };
        let end = real_end_sample.min(self.origin + self.audio.len()).max(start);
        if end == start {
            return None;
        }
        let segment = self.slice(start, end);
        self.floor = end;
        self.trim();
        Some(segment)
    }

    pub fn is_speaking(&self) -> bool {
        matches!(self.state, State::Speech { .. })
    }

    pub fn retained_samples(&self) -> usize {
        self.audio.len()
    }

    fn slice(&self, start: usize, end: usize) -> RawSegment {
        RawSegment {
            start_sample: start,
            end_sample: end,
            samples: self.audio[start - self.origin..end - self.origin].to_vec(),
        }
    }

    fn trim(&mut self) {
        let keep_from = self.processed.saturating_sub(self.pre_pad).max(self.floor.min(self.processed));
        if keep_from > self.origin {
            let drop = (keep_from - self.origin).min(self.audio.len());
            self.audio.drain(..drop);
            self.origin += drop;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: usize = 160; // 10 ms at 16 kHz keeps the arithmetic readable

    fn config() -> SegmenterConfig {
        SegmenterConfig {
            positive_threshold: 0.5,
            negative_threshold: 0.35,
            pre_speech_pad_ms: 100,  // 1600 samples
            post_speech_pad_ms: 50,  // 800 samples
            redemption_ms: 200,      // 3200 samples
            min_speech_ms: 100,      // 1600 samples
        }
    }

    /// Each sample holds its own absolute index so emitted payloads can be checked exactly.
    fn frame_at(index: usize) -> Vec<f32> {
        (index * FRAME..(index + 1) * FRAME).map(|s| s as f32).collect()
    }

    /// `pattern` is one bool per frame: true = speech probability 0.9, false = 0.1.
    fn run(seg: &mut SpeechSegmenter, pattern: &[bool], first_frame: usize) -> Vec<RawSegment> {
        pattern
            .iter()
            .enumerate()
            .filter_map(|(i, speech)| {
                seg.push_frame(&frame_at(first_frame + i), if *speech { 0.9 } else { 0.1 })
            })
            .collect()
    }

    fn pattern(parts: &[(bool, usize)]) -> Vec<bool> {
        parts.iter().flat_map(|(v, n)| std::iter::repeat(*v).take(*n)).collect()
    }

    fn assert_exact_payload(segment: &RawSegment) {
        let expected: Vec<f32> = (segment.start_sample..segment.end_sample).map(|s| s as f32).collect();
        assert_eq!(segment.samples, expected, "payload must be exactly [start, end)");
    }

    #[test]
    fn silence_emits_nothing() {
        let mut seg = SpeechSegmenter::new(config());
        assert!(run(&mut seg, &pattern(&[(false, 500)]), 0).is_empty());
        assert_eq!(seg.finish(500 * FRAME), None);
    }

    #[test]
    fn utterance_is_padded_and_ends_after_redemption() {
        let mut seg = SpeechSegmenter::new(config());
        let segments = run(&mut seg, &pattern(&[(false, 30), (true, 50), (false, 40)]), 0);
        assert_eq!(segments.len(), 1);
        let s = &segments[0];
        assert_eq!(s.start_sample, 30 * FRAME - 1600, "pre-pad before first speech frame");
        assert_eq!(s.end_sample, 80 * FRAME + 800, "post-pad after last speech frame");
        assert_exact_payload(s);
    }

    #[test]
    fn blip_shorter_than_min_speech_is_dropped() {
        let mut seg = SpeechSegmenter::new(config());
        assert!(run(&mut seg, &pattern(&[(false, 30), (true, 5), (false, 60)]), 0).is_empty());
        assert!(!seg.is_speaking());
    }

    #[test]
    fn pause_shorter_than_redemption_keeps_one_segment() {
        let mut seg = SpeechSegmenter::new(config());
        let p = pattern(&[(false, 30), (true, 30), (false, 10), (true, 30), (false, 40)]);
        let segments = run(&mut seg, &p, 0);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].end_sample, 100 * FRAME + 800);
        assert_exact_payload(&segments[0]);
    }

    #[test]
    fn finish_emits_active_speech_up_to_real_end() {
        let mut seg = SpeechSegmenter::new(config());
        assert!(run(&mut seg, &pattern(&[(false, 10), (true, 50)]), 0).is_empty());
        assert!(seg.is_speaking());
        let real_end = 60 * FRAME - 37;
        let tail = seg.finish(real_end).expect("active speech must be emitted");
        assert_eq!(tail.start_sample, 0, "pre-pad saturates at session start");
        assert_eq!(tail.end_sample, real_end);
        assert_exact_payload(&tail);
        assert_eq!(seg.finish(real_end), None, "finish must not emit twice");
    }

    #[test]
    fn finish_drops_unconfirmed_speech() {
        let mut seg = SpeechSegmenter::new(config());
        run(&mut seg, &pattern(&[(false, 10), (true, 5)]), 0);
        assert_eq!(seg.finish(15 * FRAME), None);
    }

    #[test]
    fn next_segment_never_starts_before_previous_end() {
        let mut cfg = config();
        cfg.pre_speech_pad_ms = 150;  // 2400
        cfg.post_speech_pad_ms = 150; // 2400
        let mut seg = SpeechSegmenter::new(cfg);
        let p = pattern(&[(false, 20), (true, 30), (false, 25), (true, 30), (false, 40)]);
        let segments = run(&mut seg, &p, 0);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[1].start_sample, segments[0].end_sample);
        segments.iter().for_each(assert_exact_payload);
    }

    #[test]
    fn post_pad_longer_than_redemption_is_clamped_to_processed_audio() {
        let mut cfg = config();
        cfg.redemption_ms = 50;      // 800
        cfg.post_speech_pad_ms = 400; // 6400
        let mut seg = SpeechSegmenter::new(cfg);
        let segments = run(&mut seg, &pattern(&[(false, 10), (true, 30), (false, 20)]), 0);
        assert_eq!(segments.len(), 1);
        assert!(segments[0].end_sample <= 46 * FRAME, "end {} beyond processed audio", segments[0].end_sample);
        assert_exact_payload(&segments[0]);
    }

    #[test]
    fn silence_keeps_memory_bounded() {
        let mut seg = SpeechSegmenter::new(config());
        run(&mut seg, &pattern(&[(false, 10_000)]), 0);
        assert!(seg.retained_samples() <= 1600 + FRAME, "retained {}", seg.retained_samples());
        run(&mut seg, &pattern(&[(true, 30), (false, 10_000)]), 10_000);
        assert!(seg.retained_samples() <= 1600 + FRAME, "retained {}", seg.retained_samples());
    }

    #[test]
    fn meetily_defaults_match_previous_silero_settings() {
        let c = SegmenterConfig::meetily(2000);
        assert_eq!(
            (c.positive_threshold, c.negative_threshold, c.pre_speech_pad_ms, c.post_speech_pad_ms, c.redemption_ms, c.min_speech_ms),
            (0.50, 0.35, 300, 400, 2000, 250)
        );
    }

    #[test]
    fn truncate_end_cuts_payload_and_end() {
        let mut s = RawSegment { start_sample: 10, end_sample: 20, samples: (10..20).map(|v| v as f32).collect() };
        s.truncate_end(15);
        assert_eq!((s.end_sample, s.samples.len()), (15, 5));
        s.truncate_end(99);
        assert_eq!(s.end_sample, 15);
    }
}
