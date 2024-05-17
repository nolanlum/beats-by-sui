/*
This code is based on the QM DSP Library, badly ported to Rust.

The original code can be found at:
    - https://github.com/c4dm/qm-dsp/blob/master/dsp/onsets/DetectionFunction.cpp
*/
#![allow(non_snake_case)]

use rustfft::num_complex::Complex64;

use super::phasevocoder::PhaseVocoder;

#[allow(unused)]
pub enum Type {
    Hfc,
    SpecDiff,
    PhaseDev,
    ComplexSD,
    Broadband,
}

pub struct DetectionFunction {
    dfType: Type,

    dataLength: usize,
    dbRise: f64,
    whiten: bool,
    whitenRelaxCoeff: f64,
    whitenFloor: f64,

    magHistory: Vec<f64>,
    phaseHistory: Vec<f64>,
    phaseHistoryOld: Vec<f64>,
    magPeaks: Vec<f64>,

    phaseVoc: PhaseVocoder,

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
        df_type: Type,              // Type of detection function
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

            magHistory: vec![0.0; halfLength],
            phaseHistory: vec![0.0; halfLength],
            phaseHistoryOld: vec![0.0; halfLength],
            magPeaks: vec![0.0; halfLength],

            phaseVoc: PhaseVocoder::new(frame_length, step_size),

            magnitude: vec![0.0; halfLength],
            thetaAngle: vec![0.0; halfLength],
            unwrapped: vec![0.0; halfLength],

            window: apodize::hanning_iter(frame_length).collect(),
            windowed: vec![0.0; frame_length],
        }
    }

    pub fn new_with_defaults(
        step_size: usize,    // DF step in samples
        frame_length: usize, // DF analysis window - usually 2*step. Must be even!
    ) -> Self {
        Self::new(
            step_size,
            frame_length,
            Type::ComplexSD,
            3.0,
            false,
            -1.0,
            -1.0,
        )
    }

    pub fn process_time_domain<'a>(
        &mut self,
        samples: impl IntoIterator<Item = &'a f64>,
        sample_count: usize,
    ) -> f64 {
        assert!(sample_count >= self.dataLength);

        self.windowed
            .iter_mut()
            .zip(
                samples
                    .into_iter()
                    .zip(self.window.iter())
                    .map(|(x, y)| x * y),
            )
            .for_each(|(win, sample)| *win = sample);

        self.phaseVoc.process_time_domain(
            &self.windowed,
            &mut self.magnitude,
            &mut self.thetaAngle,
            &mut self.unwrapped,
        );

        if self.whiten {
            self.whiten()
        }

        self.run_df()
    }

    fn whiten(&mut self) {
        for (mag, peak) in self.magnitude.iter_mut().zip(self.magPeaks.iter_mut()) {
            let m = (if *mag < *peak {
                *mag + (*peak - *mag) * self.whitenRelaxCoeff
            } else {
                *mag
            })
            .max(self.whitenFloor);

            *peak = m;
            *mag /= m;
        }
    }

    fn run_df(&mut self) -> f64 {
        match self.dfType {
            Type::Hfc => self
                .magnitude
                .iter()
                .enumerate()
                .map(|(i, val)| val * (i + 1) as f64)
                .sum(),
            Type::SpecDiff => self
                .magnitude
                .iter()
                .zip(self.magHistory.iter_mut())
                .map(|(mag, magHistory)| {
                    let diff = (mag.powi(2) - magHistory.powi(2)).abs().sqrt();

                    // (See note in phaseDev below.)

                    *magHistory = *mag;
                    diff
                })
                .sum(),
            Type::PhaseDev => self
                .thetaAngle
                .iter()
                .zip(
                    self.phaseHistory
                        .iter_mut()
                        .zip(self.phaseHistoryOld.iter_mut()),
                )
                .map(|(theta, (history, historyOld))| {
                    let dev = super::math::princarg(theta - 2.0 * *history + *historyOld).abs();

                    // A previous version of this code only counted the value here
                    // if the magnitude exceeded 0.1.  My impression is that
                    // doesn't greatly improve the results for "loud" music (so
                    // long as the peak picker is reasonably sophisticated), but
                    // does significantly damage its ability to work with quieter
                    // music, so I'm removing it and counting the result always.
                    // Same goes for the spectral difference measure above.

                    *historyOld = *history;
                    *history = *theta;
                    dev
                })
                .sum(),
            Type::ComplexSD => self
                .thetaAngle
                .iter()
                .zip(self.magnitude.iter().zip(self.magHistory.iter_mut()))
                .zip(
                    self.phaseHistory
                        .iter_mut()
                        .zip(self.phaseHistoryOld.iter_mut()),
                )
                .map(
                    |((theta, (mag, magHistory)), (phaseHistory, phaseHistoryOld))| {
                        let dev =
                            super::math::princarg(theta - 2.0 * *phaseHistory + *phaseHistoryOld);
                        let meas = *magHistory - (*mag * (Complex64::i() * dev).exp());
                        let val = meas.norm();

                        *phaseHistoryOld = *phaseHistory;
                        *phaseHistory = *theta;
                        *magHistory = *mag;

                        val
                    },
                )
                .sum(),
            Type::Broadband => self
                .magnitude
                .iter()
                .zip(self.magHistory.iter_mut())
                .map(|(mag, magHistory)| {
                    let sqrMag = mag.powi(2);
                    let val = if *magHistory > 0.0
                        && (10.0 * (sqrMag / *magHistory).log10() > self.dbRise)
                    {
                        1.0
                    } else {
                        0.0
                    };

                    *magHistory = sqrMag;

                    val
                })
                .sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

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
