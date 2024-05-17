use std::collections::HashMap;
use std::time::Duration;

use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::qm_dsp::keydetection;

/// This enum is in chromatic order (not circle of fifths ordering) since
/// the Chromagram returns key information in chromatic ordering.
#[wasm_bindgen]
#[derive(FromPrimitive, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ChromaticKey {
    // Major keys
    CMajor = 1,
    DFlatMajor = 2,
    DMajor = 3,
    EFlatMajor = 4,
    EMajor = 5,
    FMajor = 6,
    GFlatMajor = 7,
    GMajor = 8,
    AFlatMajor = 9,
    AMajor = 10,
    BFlatMajor = 11,
    BMajor = 12,

    // Minor keys
    CMinor = 13,
    CSharpMinor = 14,
    DMinor = 15,
    EFlatMinor = 16,
    EMinor = 17,
    FMinor = 18,
    FSharpMinor = 19,
    GMinor = 20,
    GSharpMinor = 21,
    AMinor = 22,
    BFlatMinor = 23,
    BMinor = 24,
}

impl ChromaticKey {}

pub struct KeyDetector {
    sample_rate: u32,
    window_size: usize,
    step_size_frames: usize,

    window_buf: Vec<f64>,
    processed_frame_count: u64,

    key_mode: keydetection::GetKeyMode,
    result_counts_by_key: HashMap<i32, usize>,
}

impl KeyDetector {
    pub fn new(sample_rate: u32) -> Self {
        let key_mode = keydetection::GetKeyMode::new_with_defaults(sample_rate, 440.0);
        let window_size = key_mode.get_block_size();
        let step_size_frames = key_mode.get_hop_size();
        assert_eq!(window_size, step_size_frames);

        let window_buf = Vec::with_capacity(window_size);
        let result_counts_by_key = HashMap::new();

        KeyDetector {
            sample_rate,
            window_size,
            step_size_frames,

            window_buf,
            processed_frame_count: 0,

            key_mode,
            result_counts_by_key,
        }
    }

    pub fn process_samples(&mut self, samples: &[f32]) {
        let mut samples_read = 0;
        while samples_read < samples.len() {
            let samples_available = samples.len() - samples_read;
            let write_available = self.window_size - self.window_buf.len();
            assert!(write_available > 0);

            let read_size = samples_available.min(write_available);
            assert!(samples_read + read_size <= samples.len());

            samples[samples_read..samples_read + read_size]
                .iter()
                .map(|x| *x as f64)
                .for_each(|x| self.window_buf.push(x));
            samples_read += read_size;

            if self.window_buf.len() == self.window_size {
                let result_key = self.key_mode.process(&self.window_buf);
                *self.result_counts_by_key.entry(result_key).or_default() += 1;

                self.window_buf.clear();
            }
        }
        self.processed_frame_count += samples_read as u64
    }

    pub fn finalize(&mut self) -> Vec<(ChromaticKey, usize)> {
        // Process the last remaining samples by appending silence.
        let frames_to_fill_window = self.window_size - self.window_buf.len();
        if frames_to_fill_window < self.step_size_frames {
            (0..frames_to_fill_window).for_each(|_| self.window_buf.push(0.0));

            let result_key = self.key_mode.process(&self.window_buf);
            *self.result_counts_by_key.entry(result_key).or_default() += 1;
        }

        let mut results_vec = self
            .result_counts_by_key
            .clone()
            .into_iter()
            .map(|(k, v)| (ChromaticKey::from_i32(k).unwrap(), v))
            .collect::<Vec<_>>();
        results_vec.sort_by_key(|x| x.1);
        results_vec.reverse();
        results_vec
    }

    pub fn processed_frames(&self) -> u64 {
        self.processed_frame_count
    }

    pub fn processed_frames_duration(&self) -> Duration {
        Duration::from_secs_f64(self.processed_frame_count as f64 / self.sample_rate as f64)
    }
}
