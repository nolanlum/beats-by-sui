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

use analyzer::{Beatmania, CoarseBeatDetector};
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub fn detect_bpm(samples: &[f32], sample_rate: u32) -> f64 {
    let mut bpm_machine = CoarseBeatDetector::new(sample_rate);
    bpm_machine.process_samples(samples);
    let beats = bpm_machine.finalize();
    let iidx = Beatmania::new(sample_rate);
    iidx.calculate_bpm(&beats)
}
