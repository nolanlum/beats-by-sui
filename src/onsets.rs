/*
This code is based on the QM DSP Library, ported to WebAssembly-suitable Rust.

The original code can be found at https://github.com/c4dm/qm-dsp/blob/master/dsp/onsets/DetectionFunction.cpp.
*/

#![allow(non_snake_case, non_camel_case_types)]

use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

type c64 = Complex64;

/// As far as I can tell, this is not a "true" phase vocoder, it only extracts phase information
/// and does not resynthesize them into time-domain samples.
///
/// Nevertheless, we faithfully recreate it as the QM DSP library expects.
struct PhaseVocoder {
    frame_size: usize,
    half_size: usize,
    hop: usize,

    freq: Vec<c64>,
    phase: Vec<f64>,
    unwrapped: Vec<f64>,

    fft: Arc<dyn Fft<f64>>,
    fft_scratch: Vec<c64>,
}

impl PhaseVocoder {
    fn new(frame_size: usize, hop: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame_size);
        let fft_scratch = vec![Complex64::new(0.0, 0.0); fft.get_inplace_scratch_len()];

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

    fn process_time_domain(
        &mut self,
        frame: &[f64],
        magnitudes: &mut [f64],
        theta: &mut [f64],
        unwrapped: &mut [f64],
    ) {
        assert!(frame.len() >= self.frame_size);

        for i in 0..self.frame_size {
            self.freq[i] = Complex64::new(frame[(i + self.frame_size / 2) % self.frame_size], 0.0)
        }

        self.fft
            .process_with_scratch(&mut self.freq, &mut self.fft_scratch);
        self.get_magnitudes(magnitudes);
        self.get_phases(theta);
        self.unwrap_phases(theta, unwrapped);
    }

    fn get_magnitudes(&self, magnitudes: &mut [f64]) {
        for i in 0..self.half_size {
            magnitudes[i] = self.freq[i].norm();
        }
    }

    fn get_phases(&self, theta: &mut [f64]) {
        for i in 0..self.half_size {
            theta[i] = self.freq[i].arg();
        }
    }

    fn unwrap_phases(&mut self, theta: &[f64], unwrapped: &mut [f64]) {
        assert!(theta.len() >= self.half_size);
        assert!(unwrapped.len() >= self.half_size);

        for i in 0..self.half_size {
            let omega = (2.0 * PI * self.hop as f64 * i as f64) / self.frame_size as f64;
            let expected = self.phase[i] + omega;
            let error = PhaseVocoder::princarg(theta[i] - expected);

            unwrapped[i] = self.unwrapped[i] + omega + error;

            self.phase[i] = theta[i];
            self.unwrapped[i] = unwrapped[i];
        }
    }

    fn modulus(x: f64, y: f64) -> f64 {
        let a = f64::floor(x / y);
        x - (y * a)
    }

    fn princarg(ang: f64) -> f64 {
        PhaseVocoder::modulus(ang + PI, -2.0 * PI) + PI
    }
}

mod DetectionFunction {
    pub enum Kind {
        HFC,
        SPECDIFF,
        PHASEDEV,
        COMPLEXSD,
        BROADBAND,
    }

    pub struct DetectionFunction {
        dfType: Kind,

        dataLength: usize,
        halfLength: usize,
        stepSize: usize,
        dbRise: f64,
        whiten: bool,
        whitenRelaxCoeff: f64,
        whitenFloor: f64,

        phaseVoc: super::PhaseVocoder,

        magnitude: Vec<f64>,
        thetaAngle: Vec<f64>,
        unwrapped: Vec<f64>,

        window: Vec<f64>,
        windowed: Vec<f64>,
    }

    impl DetectionFunction {
        pub fn new(
            step_size: usize,           // DF step in samples
            frame_length: usize,        // DF analysis window - usually 2*step. Must be even!
            df_type: Kind,              // Type of detection function
            db_rise: f64,               // Only used for broadband df (and required for it)
            adaptive_whitening: bool,   // Perform adaptive whitening
            whitening_relax_coeff: f64, // If < 0, a sensible default will be used
            whitening_floor: f64,       // If < 0, a sensible default will be used
        ) -> Self {
            assert!(frame_length % 2 == 0);
            let halfLength = frame_length / 2 + 1;

            DetectionFunction {
                dfType: df_type,
                dataLength: frame_length,
                halfLength,
                stepSize: step_size,
                dbRise: db_rise,

                whiten: adaptive_whitening,
                whitenRelaxCoeff: if whitening_relax_coeff < 0.0 {
                    0.9997
                } else {
                    whitening_relax_coeff
                },
                whitenFloor: if whitening_floor < 0.0 {
                    0.01
                } else {
                    whitening_floor
                },

                phaseVoc: super::PhaseVocoder::new(frame_length, step_size),

                magnitude: vec![0.0; halfLength],
                thetaAngle: vec![0.0; halfLength],
                unwrapped: vec![0.0; halfLength],

                window: apodize::hanning_iter(frame_length).collect(),
                windowed: vec![0.0; frame_length],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use super::*;

    #[test]
    fn fullcycle() {
        // Cosine with one cycle exactly equal to pvoc hopsize. This is
        // pretty much the most trivial case -- in fact it's
        // indistinguishable from totally silent input (in the phase
        // values) because the measured phases are zero throughout.

        // We aren't windowing the input frame because (for once) it
        // actually *is* just a short part of a continuous infinite
        // sinusoid.
        let frame = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let mut pvoc = PhaseVocoder::new(8, 4);

        let mut mag = vec![999.0; 5];
        let mut phase = vec![999.0; 5];
        let mut unwrapped = vec![999.0; 5];

        pvoc.process_time_domain(&frame, &mut mag, &mut phase, &mut unwrapped);

        assert_eq!(mag, vec![0.0, 0.0, 4.0, 0.0, 0.0]);
        assert_ulps_eq!(
            phase.as_slice(),
            [0.0, 0.0, 0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-7
        );
        assert_ulps_eq!(
            unwrapped.as_slice(),
            [0.0, 0.0, 0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-7
        );

        pvoc.process_time_domain(&frame, &mut mag, &mut phase, &mut unwrapped);

        assert_eq!(mag, vec![0.0, 0.0, 4.0, 0.0, 0.0]);
        assert_ulps_eq!(
            phase.as_slice(),
            [0.0, 0.0, 0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-7
        );

        // Derivation of unwrapped values:
        //
        // * Bin 0 (DC) always has phase 0 and expected phase 0
        //
        // * Bin 1 has expected phase pi (the hop size is half a cycle at
        //   its frequency), but measured phase 0 (because there is no
        //   signal in that bin). So it has phase error -pi, which is
        //   mapped into (-pi,pi] range as pi, giving an unwrapped phase
        //   of 2*pi.
        //
        // * Bin 2 has expected phase 2*pi, measured phase 0, hence error
        //   0 and unwrapped phase 2*pi.
        //
        // * Bin 3 is like bin 1: it has expected phase 3*pi, measured
        //   phase 0, so phase error -pi and unwrapped phase 4*pi.
        //
        // * Bin 4 (Nyquist) has expected phase 4*pi, measured phase 0,
        //   hence error 0 and unwrapped phase 4*pi.
        assert_ulps_eq!(
            unwrapped.as_slice(),
            [0.0, 2.0 * PI, 2.0 * PI, 4.0 * PI, 4.0 * PI].as_slice(),
            epsilon = 1e-7
        );

        pvoc.process_time_domain(&frame, &mut mag, &mut phase, &mut unwrapped);

        assert_eq!(mag, vec![0.0, 0.0, 4.0, 0.0, 0.0]);
        assert_ulps_eq!(
            phase.as_slice(),
            [0.0, 0.0, 0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-7
        );
        assert_ulps_eq!(
            unwrapped.as_slice(),
            [0.0, 4.0 * PI, 4.0 * PI, 8.0 * PI, 8.0 * PI].as_slice(),
            epsilon = 1e-7
        );
    }

    #[test]
    fn overlapping() {
        // Sine (i.e. cosine starting at phase -pi/2) starting with the
        // first sample, introducing a cosine of half the frequency
        // starting at the fourth sample, i.e. the second hop. The cosine
        // is introduced "by magic", i.e. it doesn't appear in the second
        // half of the first frame (it would have quite strange effects on
        // the first frame if it did).

        // 3 x 8-sample frames which we pretend are overlapping
        let data = [
            0.0,
            1.0,
            0.0,
            -1.0,
            0.0,
            1.0,
            0.0,
            -1.0,
            1.0,
            1.70710678,
            0.0,
            -1.70710678,
            -1.0,
            0.29289322,
            0.0,
            -0.29289322,
            -1.0,
            0.29289322,
            0.0,
            -0.29289322,
            1.0,
            1.70710678,
            0.0,
            -1.70710678,
        ];
        let mut pvoc = PhaseVocoder::new(8, 4);

        let mut mag = vec![999.0; 5];
        let mut phase = vec![999.0; 5];
        let mut unwrapped = vec![999.0; 5];

        pvoc.process_time_domain(&data[0..8], &mut mag, &mut phase, &mut unwrapped);

        assert_ulps_eq!(
            mag.as_slice(),
            [0.0, 0.0, 4.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-7
        );
        assert_ulps_eq!(
            phase.as_slice(),
            [0.0, 0.0, -FRAC_PI_2, 0.0, 0.0].as_slice(),
            epsilon = 1e-7
        );
        assert_ulps_eq!(
            unwrapped.as_slice(),
            [0.0, 0.0, -FRAC_PI_2, 0.0, 0.0].as_slice(),
            epsilon = 1e-7
        );

        pvoc.process_time_domain(&data[8..16], &mut mag, &mut phase, &mut unwrapped);

        assert_ulps_eq!(
            mag.as_slice(),
            [0.0, 4.0, 4.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-7
        );
        // Derivation of unwrapped values:
        //
        // * Bin 0 (DC) always has phase 0 and expected phase 0
        //
        // * Bin 1 has a new signal, a cosine starting with phase 0. But
        //   because of the "FFT shift" which the phase vocoder carries
        //   out to place zero phase in the centre of the (usually
        //   windowed) frame, and because a single cycle at this frequency
        //   spans the whole frame, this bin actually has measured phase
        //   of either pi or -pi. (The shift doesn't affect those
        //   higher-frequency bins whose signals fit exact multiples of a
        //   cycle into a frame.) This maps into (-pi,pi] as pi, which
        //   matches the expected phase, hence unwrapped phase is also pi.
        //
        // * Bin 2 has expected phase 3pi/2 (being the previous measured
        //   phase of -pi/2 plus advance of 2pi). It has the same measured
        //   phase as last time around, -pi/2, which is consistent with
        //   the expected phase, so the unwrapped phase is 3pi/2.
        //
        // * Bin 3 I don't really know about -- the magnitude here is 0,
        //   but we get non-zero measured phase whose sign is
        //   implementation-dependent
        //
        // * Bin 4 (Nyquist) has expected phase 4*pi, measured phase 0,
        //   hence error 0 and unwrapped phase 4*pi.
        phase[3] = 0.0; // Because we aren't testing for this one
        assert_ulps_eq!(
            phase.as_slice(),
            [0.0, PI, -FRAC_PI_2, 0.0, 0.0].as_slice(),
            epsilon = 1e-7
        );
        assert_ulps_eq!(
            unwrapped.as_slice(),
            [0.0, PI, 3.0 * FRAC_PI_2, 3.0 * PI, 4.0 * PI].as_slice(),
            epsilon = 1e-7
        );

        pvoc.process_time_domain(&data[16..24], &mut mag, &mut phase, &mut unwrapped);

        assert_ulps_eq!(
            mag.as_slice(),
            [0.0, 4.0, 4.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-7
        );
        assert_ulps_eq!(
            phase.as_slice(),
            [0.0, 0.0, -FRAC_PI_2, 0.0, 0.0].as_slice(),
            epsilon = 1e-7
        );
        assert_ulps_eq!(
            unwrapped.as_slice(),
            [0.0, 2.0 * PI, 7.0 * FRAC_PI_2, 6.0 * PI, 8.0 * PI].as_slice(),
            epsilon = 1e-7
        );
    }
}
