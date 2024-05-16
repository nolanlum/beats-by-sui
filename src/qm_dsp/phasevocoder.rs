/*
This code is based on the QM DSP Library, badly ported to Rust.

The original code can be found at:
    - https://github.com/c4dm/qm-dsp/blob/master/dsp/phasevocoder/PhaseVocoder.cpp
*/

use super::math;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

/// As far as I can tell, this is not a "true" phase vocoder, it only extracts phase information
/// and does not resynthesize them into time-domain samples.
///
/// Nevertheless, we faithfully recreate it as the QM DSP library expects.
pub struct PhaseVocoder {
    frame_size: usize,
    half_size: usize,
    hop: usize,

    freq: Vec<Complex64>,
    phase: Vec<f64>,
    unwrapped: Vec<f64>,

    fft: Arc<dyn Fft<f64>>,
    fft_scratch: Vec<Complex64>,
}

#[allow(non_snake_case)]
impl PhaseVocoder {
    pub fn new(frame_size: usize, hop: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame_size);
        let fft_scratch = vec![Complex64::default(); fft.get_inplace_scratch_len()];

        let half_size = frame_size / 2 + 1;

        // phase stores the "previous" phase, so set to one step
        // behind so that a signal with initial phase at zero matches
        // the expected values. This is completely unnecessary for any
        // analytical purpose, it's just tidier.
        let phase = (0..half_size)
            .map(|i| (-2.0 * PI * hop as f64 * i as f64) / frame_size as f64)
            .collect::<Vec<f64>>();
        let unwrapped = phase.clone();

        PhaseVocoder {
            frame_size,
            half_size,
            hop,

            freq: vec![Complex64::new(0.0, 0.0); frame_size],
            phase,
            unwrapped,

            fft,
            fft_scratch,
        }
    }

    pub fn process_time_domain(
        &mut self,
        frame: &[f64],
        magnitudes: &mut [f64],
        theta: &mut [f64],
        unwrapped: &mut [f64],
    ) {
        assert!(frame.len() >= self.frame_size);

        // I do not quite understand why, but the vocoder places the "zero" of the phases in the
        // center of the (windowed) samples, so we shift the values over here.
        for i in 0..self.frame_size {
            self.freq[i] = Complex64::new(frame[(i + self.frame_size / 2) % self.frame_size], 0.0)
        }

        self.fft
            .process_with_scratch(&mut self.freq, &mut self.fft_scratch);
        self.get_magnitudes(magnitudes);
        self.get_phases(theta);
        self.unwrap_phases(theta, unwrapped);
    }

    pub fn get_magnitudes(&self, magnitudes: &mut [f64]) {
        assert!(magnitudes.len() >= self.half_size);
        magnitudes
            .iter_mut()
            .zip(self.freq.iter())
            .for_each(|(mag, freq)| *mag = freq.norm());
    }

    pub fn get_phases(&self, theta: &mut [f64]) {
        assert!(theta.len() >= self.half_size);
        theta
            .iter_mut()
            .zip(self.freq.iter())
            .for_each(|(theta, freq)| *theta = freq.arg());
    }

    pub fn unwrap_phases(&mut self, theta: &[f64], unwrapped: &mut [f64]) {
        assert!(theta.len() >= self.half_size);
        assert!(unwrapped.len() >= self.half_size);

        for i in 0..self.half_size {
            let omega = (2.0 * PI * self.hop as f64 * i as f64) / self.frame_size as f64;
            let expected = self.phase[i] + omega;
            let error = math::princarg(theta[i] - expected);

            unwrapped[i] = self.unwrapped[i] + omega + error;

            self.phase[i] = theta[i];
            self.unwrapped[i] = unwrapped[i];
        }
    }
}
