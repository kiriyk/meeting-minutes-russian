//! Filtered, fixed-rate resampling for files. One stateful FFT resampler keeps
//! block boundaries continuous; flushing and removing its delay preserves time.
use anyhow::{anyhow, bail, Result};
use log::info;
use rubato::{FftFixedInOut, Resampler};

pub(super) fn resample(
    input: &[f32],
    from_rate: u32,
    to_rate: u32,
    mut progress: impl FnMut(u32) -> bool,
) -> Result<Vec<f32>> {
    if from_rate == 0 || to_rate == 0 {
        bail!("Invalid audio sample rate");
    }
    if !progress(0) {
        bail!("Retranscription cancelled");
    }
    if input.is_empty() || from_rate == to_rate {
        if !progress(100) {
            bail!("Retranscription cancelled");
        }
        return Ok(input.to_vec());
    }

    let started = std::time::Instant::now();
    info!(
        "Resampling {} samples from {}Hz to {}Hz in FFT blocks",
        input.len(),
        from_rate,
        to_rate
    );
    let output_len = usize::try_from(
        (input.len() as u128 * to_rate as u128 + from_rate as u128 / 2) / from_rate as u128,
    )?;
    let mut resampler = FftFixedInOut::<f32>::new(from_rate as usize, to_rate as usize, 1024, 1)?;
    // Odd FFT block lengths give the filter a fractional-sample delay, which
    // output_delay() rounds down. Even input/output blocks keep timestamps exact.
    if resampler.input_frames_next() % 2 != 0 || resampler.output_frames_next() % 2 != 0 {
        resampler = FftFixedInOut::<f32>::new(
            from_rate as usize,
            to_rate as usize,
            resampler.input_frames_next() * 2,
            1,
        )?;
    }
    let delay = resampler.output_delay();
    let required = output_len
        .checked_add(delay)
        .ok_or_else(|| anyhow!("Audio is too long"))?;
    let mut output = Vec::with_capacity(required);
    let mut block = resampler.output_buffer_allocate(true);
    let mut position = 0;
    let mut last_percent = 0;
    while position < input.len() {
        let end = (position + resampler.input_frames_next()).min(input.len());
        let (_, written) = resampler.process_partial_into_buffer(
            Some(&[&input[position..end]]),
            &mut block,
            None,
        )?;
        output.extend_from_slice(&block[0][..written]);
        position = end;
        // Reserve 100% until the delayed tail has also been flushed.
        let percent = ((position as u128 * 100 / input.len() as u128) as u32).min(99);
        if percent > last_percent {
            if !progress(percent) {
                bail!("Retranscription cancelled");
            }
            last_percent = percent;
        }
    }
    while output.len() < required {
        if !progress(99) {
            bail!("Retranscription cancelled");
        }
        let (_, written) =
            resampler.process_partial_into_buffer::<&[f32], Vec<f32>>(None, &mut block, None)?;
        output.extend_from_slice(&block[0][..written]);
    }
    output.drain(..delay);
    output.truncate(output_len);
    if !progress(100) {
        bail!("Retranscription cancelled");
    }
    info!(
        "Resampling complete: {} -> {} samples in {:.2}s",
        input.len(),
        output.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_timing_and_signal_across_blocks_at_common_rates() {
        for from_rate in [8000, 16000, 24000, 44100, 48000] {
            let count = from_rate as usize * 2 + 123;
            let input: Vec<_> = (0..count)
                .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / from_rate as f32).sin() * 0.5)
                .collect();
            let output = resample(&input, from_rate, 16000, |_| true).unwrap();
            assert_eq!(
                output.len(),
                (count as f64 * 16000.0 / from_rate as f64).round() as usize
            );
            for (i, sample) in output
                .iter()
                .enumerate()
                .skip(1000)
                .take(output.len() - 2000)
            {
                let expected = (std::f32::consts::TAU * 440.0 * i as f32 / 16000.0).sin() * 0.5;
                assert!(
                    (sample - expected).abs() < 0.002,
                    "{from_rate}Hz: discontinuity at {i}: {sample} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn downsampling_suppresses_aliasing() {
        let input: Vec<_> = (0..96000)
            .map(|i| (std::f32::consts::TAU * 10000.0 * i as f32 / 48000.0).sin() * 0.5)
            .collect();
        let output = resample(&input, 48000, 16000, |_| true).unwrap();
        let middle = &output[1000..output.len() - 1000];
        let rms = (middle.iter().map(|x| x * x).sum::<f32>() / middle.len() as f32).sqrt();
        assert!(rms < 0.002, "Aliased energy: {rms}");
    }

    #[test]
    fn cancellation_stops_before_completion() {
        let mut updates = Vec::new();
        let error = resample(&vec![0.1; 80000], 8000, 16000, |p| {
            updates.push(p);
            p < 10
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "Retranscription cancelled");
        assert!(!updates.contains(&100));
        assert!(updates.windows(2).all(|p| p[0] <= p[1]));
    }

    #[test]
    fn short_and_empty_inputs_preserve_their_duration() {
        for count in [0, 1, 17, 1024, 1025] {
            let output = resample(&vec![0.1; count], 8000, 16000, |_| true).unwrap();
            assert_eq!(output.len(), count * 2);
            assert!(output.iter().all(|s| s.is_finite()));
        }
        assert!(resample(&[0.1], 0, 16000, |_| true).is_err());
    }
}
