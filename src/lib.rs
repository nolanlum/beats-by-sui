extern crate apodize;
extern crate ringbuf;
extern crate rustfft;

#[cfg(test)]
#[macro_use]
extern crate approx;

pub mod onsets;
pub mod tempo_track;

use std::collections::HashMap;

use onsets::DetectionFunction::DetectionFunction;
use ringbuf::{storage::Heap, traits::*, LocalRb};
use tempo_track::TempoTrackV2;
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub struct BpmInfo {
    pub bpm: f64,

    pub range_lower: f64,
    pub range_upper: f64,
}

pub struct BpmDetector {
    sample_rate: u32,
    window_size: usize,
    step_size_frames: usize,

    window_buf: LocalRb<Heap<f64>>,

    detection_function: DetectionFunction,
    detection_results: Vec<f64>,
}

impl BpmDetector {
    // This determines the resolution of the resulting BeatMap.
    // ~12 ms (86 Hz) is a fair compromise between accuracy and analysis speed,
    // also matching the preferred window/step sizes from BeatTrack VAMP.
    const STEP_SECS: f64 = 0.01161;
    // results in 43 Hz @ 44.1 kHz / 47 Hz @ 48 kHz / 47 Hz @ 96 kHz
    const MAX_BIN_SIZE_HZ: u32 = 10;

    pub fn new(sample_rate: u32) -> Self {
        let step_size_frames = (sample_rate as f64 * Self::STEP_SECS).floor() as usize;
        let window_size = (sample_rate / Self::MAX_BIN_SIZE_HZ).next_power_of_two() as usize;
        let detection_function =
            DetectionFunction::new_with_defaults(step_size_frames, window_size);

        // println!(
        //     "sample rate {}Hz, window size {} frames, step size {} frames",
        //     sample_rate, window_size, step_size_frames
        // );

        // make sure the first frame is centered into the fft window. This makes sure
        // that the result is significant starting from the first step.
        let mut window_buf = LocalRb::new(window_size);
        let frames_pushed = window_buf.push_iter((0..window_size / 2).map(|_| 0.0));
        assert!(frames_pushed == window_size / 2);

        BpmDetector {
            sample_rate,
            window_size,
            step_size_frames,
            window_buf,
            detection_function,

            detection_results: vec![0.0; 0],
        }
    }

    pub fn process_samples(&mut self, samples: &[f32]) {
        let mut samples_read = 0;
        while samples_read < samples.len() {
            let samples_available = samples.len() - samples_read;
            let write_available = self.window_buf.vacant_len();
            assert!(write_available > 0);

            let read_size = samples_available.min(write_available);
            assert!(samples_read + read_size <= samples.len());

            let samples_pushed = self.window_buf.push_iter(
                samples[samples_read..samples_read + read_size]
                    .iter()
                    .map(|x| *x as f64),
            );
            assert!(samples_pushed == read_size);
            samples_read += read_size;

            if self.window_buf.is_full() {
                self.detection_results.push(
                    self.detection_function
                        .process_time_domain(self.window_buf.iter(), self.window_size),
                );

                let samples_dropped = self.window_buf.skip(self.step_size_frames);
                assert!(samples_dropped == self.step_size_frames);
            }
        }
    }

    pub fn finalize(&mut self) -> BpmInfo {
        // Process the last remaining samples by appending silence.
        let frames_to_fill_window = self.window_buf.vacant_len();
        if frames_to_fill_window < self.step_size_frames {
            let frames_pushed = self
                .window_buf
                .push_iter((0..frames_to_fill_window).map(|_| 0.0));
            assert!(frames_pushed == frames_to_fill_window);

            self.detection_results.push(
                self.detection_function
                    .process_time_domain(self.window_buf.iter(), self.window_size),
            )
        }

        let non_zero_count = self.detection_results.len()
            - self
                .detection_results
                .iter()
                .rev()
                .map_while(|x| if *x <= 0.0 { Some(1) } else { None })
                .sum::<usize>();

        // skip first 2 results as it might have detect noise as onset
        // that's how vamp does and seems works best this way
        let df = &self.detection_results[2.min(self.detection_results.len())..non_zero_count];
        let mut beat_period = vec![0; df.len()];

        let tt = TempoTrackV2::new(self.sample_rate, self.step_size_frames);
        tt.calculate_beat_period(df, &mut beat_period, 120.0, false);

        let mut period_counts = HashMap::new();
        let likely_period = beat_period
            .iter()
            .copied()
            .max_by_key(|&n| {
                let count = period_counts.entry(n).or_insert(0);
                *count += 1;
                *count
            })
            .unwrap();
        let bpm =
            60.0 * self.sample_rate as f64 / self.step_size_frames as f64 / likely_period as f64;

        // Take the mode of the tempi of the beat periods of the entire track
        // and call it a day.
        // let mut tempo_counts = HashMap::new();
        // let tempo = tempi
        //     .iter()
        //     .copied()
        //     .max_by_key(|&n| {
        //         let count = tempo_counts
        //             .entry((n * 10000.0).floor() as u64)
        //             .or_insert(0);
        //         *count += 1;
        //         *count
        //     })
        //     .unwrap();
        //
        // println!(
        //     "Detected tempo: {:.2} ({:.2}~{:.2}) bpm",
        //     bpm,
        //     60.0 * self.sample_rate as f64
        //         / self.step_size_frames as f64
        //         / (likely_period as f64 + 0.5),
        //     60.0 * self.sample_rate as f64
        //         / self.step_size_frames as f64
        //         / (likely_period as f64 - 0.5),
        // );

        BpmInfo {
            bpm,
            range_lower: 60.0 * self.sample_rate as f64
                / self.step_size_frames as f64
                / (likely_period as f64 + 0.5),
            range_upper: 60.0 * self.sample_rate as f64
                / self.step_size_frames as f64
                / (likely_period as f64 - 0.5),
        }
    }
}

#[wasm_bindgen]
pub fn detect_bpm(samples: &[f32], sample_rate: u32) -> BpmInfo {
    let mut bpm_machine = BpmDetector::new(sample_rate);
    bpm_machine.process_samples(samples);
    bpm_machine.finalize()
}
