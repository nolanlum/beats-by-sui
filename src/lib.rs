extern crate apodize;
extern crate ringbuf;
extern crate rustfft;

#[cfg(test)]
#[macro_use]
extern crate approx;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

pub mod analyzer;
mod qm_dsp;

use analyzer::{Beatmania, ChromaticKey, CoarseBeatDetector, KeyDetector};
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub struct AnalysisResult {
    pub bpm: f64,
    pub key: Option<ChromaticKey>,
}

#[wasm_bindgen]
pub fn analyze_track(samples: &[f32], sample_rate: u32) -> AnalysisResult {
    let mut bpm_machine = CoarseBeatDetector::new(sample_rate);
    let mut key_machine = KeyDetector::new(sample_rate);

    bpm_machine.process_samples(samples);
    key_machine.process_samples(samples);

    let beats = bpm_machine.finalize();
    let iidx = Beatmania::new(sample_rate);
    let bpm = iidx.calculate_bpm(&beats);

    let keys = key_machine.finalize();

    AnalysisResult {
        bpm,
        key: keys.first().map(|x| x.0),
    }
}
