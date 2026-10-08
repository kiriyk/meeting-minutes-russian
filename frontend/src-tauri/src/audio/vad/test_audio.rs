//! Real-speech fixtures shared by VAD and pipeline tests.

/// Four seconds of real speech (JFK), 16 kHz mono PCM16.
pub(crate) fn jfk_speech_16k() -> Vec<f32> {
    parse_16k_mono_pcm16_wav(include_bytes!("testdata/jfk-speech-4s.wav"))
}

/// Minimal RIFF/WAV parser for the committed 16 kHz mono PCM16 fixture.
/// Scans chunks rather than assuming a fixed header layout, but only
/// understands the PCM16 case the fixture is committed in.
pub(crate) fn parse_16k_mono_pcm16_wav(bytes: &[u8]) -> Vec<f32> {
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
