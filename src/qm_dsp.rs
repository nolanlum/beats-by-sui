/*
This code is based on the QM DSP Library, badly ported to Rust.

The original code can be found at:
    - https://github.com/c4dm/qm-dsp/blob/master/dsp/phasevocoder/PhaseVocoder.cpp
    - https://github.com/c4dm/qm-dsp/blob/master/dsp/onsets/DetectionFunction.cpp
    - https://github.com/c4dm/qm-dsp/blob/master/dsp/tempotracking/TempoTrackV2.cpp
*/

use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

/// Just some arbitrary small number
const EPS: f64 = 8e-7;

mod math {
    use std::f64::consts::PI;

    fn modulus(x: f64, y: f64) -> f64 {
        let a = f64::floor(x / y);
        x - (y * a)
    }

    pub fn princarg(ang: f64) -> f64 {
        modulus(ang + PI, -2.0 * PI) + PI
    }

    pub fn adaptive_threshold(data: &mut [f64]) {
        if data.is_empty() {
            return;
        }

        let pre = 8;
        let post = 7;

        let smoothed: Vec<f64> = (0..data.len())
            .map(|i| {
                let first = i.saturating_sub(pre);
                let last = (data.len() - 1).min(i + post);
                data[first..last].iter().sum::<f64>() / (last - first) as f64
            })
            .collect();

        data.iter_mut()
            .zip(smoothed.iter())
            .for_each(|(d, s)| *d = (*d - s).max(0.0));
    }

    pub fn max(data: &[f64]) -> (usize, f64) {
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, x)| (i, *x))
            .expect("data was empty?")
    }
}

/// As far as I can tell, this is not a "true" phase vocoder, it only extracts phase information
/// and does not resynthesize them into time-domain samples.
///
/// Nevertheless, we faithfully recreate it as the QM DSP library expects.
struct PhaseVocoder {
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

    fn get_magnitudes(&self, magnitudes: &mut [f64]) {
        assert!(magnitudes.len() >= self.half_size);
        magnitudes
            .iter_mut()
            .zip(self.freq.iter())
            .for_each(|(mag, freq)| *mag = freq.norm());
    }

    fn get_phases(&self, theta: &mut [f64]) {
        assert!(theta.len() >= self.half_size);
        theta
            .iter_mut()
            .zip(self.freq.iter())
            .for_each(|(theta, freq)| *theta = freq.arg());
    }

    fn unwrap_phases(&mut self, theta: &[f64], unwrapped: &mut [f64]) {
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

#[allow(non_snake_case)]
pub mod DetectionFunction {
    use rustfft::num_complex::Complex64;

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

                phaseVoc: super::PhaseVocoder::new(frame_length, step_size),

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
                            let dev = super::math::princarg(
                                theta - 2.0 * *phaseHistory + *phaseHistoryOld,
                            );
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
        use std::f64::consts::FRAC_PI_2;

        use super::super::*;

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
}

/// A tempo tracker that will operate on beat detection function data calculated from
/// audio at the given sample rate with the given frame increment.
///
/// Currently the sample rate and increment are used only for the conversion from
/// beat frame location to bpm in the tempo array.
pub struct TempoTrackV2 {
    rate: u32,
    increment: usize,
}

/// !!! Question: how far is this actually sample rate dependent?  I
/// think it does produce plausible results for e.g. 48000 as well as
/// 44100, but surely the fixed window sizes and comb filtering will
/// make it prefer double or half time when run at e.g. 96000?
impl TempoTrackV2 {
    pub fn new(rate: u32, increment: usize) -> Self {
        TempoTrackV2 { rate, increment }
    }

    /// Returned beat periods are given in df increment units; inputtempo and tempi in bpm
    ///
    /// MEPD 28/11/12
    /// This function now allows for a user to specify an inputtempo (in BPM)
    /// and a flag "constraintempo" which replaces the general rayleigh weighting for periodicities
    /// with a gaussian which is centered around the input tempo
    /// Note, if inputtempo = 120 and constraintempo = false, then functionality is
    /// as it was before
    pub fn calculate_beat_period(
        &self,
        df: &[f64],
        beat_period: &mut [usize],
        input_tempo: f64,
        constrain_tempo: bool,
    ) -> Vec<f64> {
        // to follow matlab.. split into 512 sample frames with a 128 hop size
        // calculate the acf,
        // then the rcf.. and then stick the rcfs as columns of a matrix
        // then call viterbi decoding with weight vector and transition matrix
        // and get best path
        let wv_len = 128;

        // MEPD 28/11/12
        // the default value of inputtempo in the beat tracking plugin is 120
        // so if the user specifies a different inputtempo, the rayparam will be updated
        // accordingly.
        // note: 60*44100/512 is a magic number
        // this might (will?) break if a user specifies a different frame rate for the onset detection function
        let rayparam = (60.0 * 44100.0 / 512.0) / input_tempo;

        // check whether or not to use rayleigh weighting (if constraintempo is false)
        // or use gaussian weighting it (constraintempo is true)
        let wv: Vec<_> = if constrain_tempo {
            // MEPD 28/11/12
            // do a gaussian weighting instead of rayleigh
            (0..wv_len)
                .map(|i| {
                    ((-1.0 * ((i as f64) - rayparam).powi(2)) / (2.0 * (rayparam / 4.0).powi(2)))
                        .exp()
                })
                .collect()
        } else {
            // MEPD 28/11/12
            // standard rayleigh weighting over periodicities
            (0..wv_len)
                .map(|i| {
                    ((i as f64) / rayparam.powi(2))
                        * ((-1.0 * (i as f64).powi(2)) / (2.0 * rayparam.powi(2))).exp()
                })
                .collect()
        };

        // beat tracking frame size (roughly 6 seconds) and hop (1.5 seconds)
        let winlen = 512;
        let step = 128;

        // matrix to store output of comb filter bank, increment column of matrix at each frame
        let mut rcfmat = Vec::<Vec<f64>>::new();
        let df_len = df.len();

        // main loop for beat period calculation
        let mut i = 0;
        while i + winlen < df_len {
            // get dfframe
            let df_frame = &df[i..i + winlen];

            // get rcf vector for current frame
            let rcf = self.get_rcf(df_frame, &wv);

            // add a new colume to rcfmat
            rcfmat.push(rcf);

            i += step
        }

        // now call viterbi decoding function
        self.viterbi_decode(&rcfmat, &wv, beat_period)
    }

    fn get_rcf(&self, df_frame: &[f64], wv: &[f64]) -> Vec<f64> {
        // calculate autocorrelation function
        // then rcf
        // just hard code for now... don't really need separate functions to do this

        // make acf
        let mut df_frame = df_frame.to_vec();
        math::adaptive_threshold(&mut df_frame);

        let df_frame_len = df_frame.len();
        let rcf_len = wv.len();
        let acf = (0..df_frame_len)
            .map(|lag| {
                (0..(df_frame_len - lag))
                    .map(|n| df_frame[n] * df_frame[n + lag])
                    .sum::<f64>()
                    / (df_frame_len - lag) as f64
            })
            .collect::<Vec<_>>();

        // now apply comb filtering
        let mut rcf = vec![0.0; rcf_len];
        let numelem: i32 = 4;

        // max beat period
        (2..rcf_len).for_each(|i| {
            // number of comb elements
            (1..numelem).for_each(|a| {
                // general state using normalisation of comb elements
                (1 - a..=a - 1).for_each(|b| {
                    // calculate value for comb filter row
                    rcf[i - 1] += (acf[((a * i as i32 + b) - 1) as usize] * wv[i - 1])
                        / (2.0 * a as f64 - 1.0);
                })
            })
        });

        // apply adaptive threshold to rcf
        math::adaptive_threshold(&mut rcf);

        // normalise rcf to sum to unity
        let mut rcfsum = 0.0;
        for x in rcf.iter_mut() {
            *x += EPS;
            rcfsum += *x;
        }
        rcf.iter_mut().for_each(|x| *x /= rcfsum + EPS);
        rcf
    }

    #[allow(non_snake_case)]
    fn viterbi_decode(
        &self,
        rcfmat: &[Vec<f64>],
        wv: &[f64],
        beat_period: &mut [usize],
    ) -> Vec<f64> {
        // following Kevin Murphy's Viterbi decoding to get best path of
        // beat periods through rfcmat
        let wv_len = wv.len();

        // make transition matrix
        let mut tmat = vec![vec![0.0; wv_len]; wv_len];

        // variance of Gaussians in transition matrix
        // formed of Gaussians on diagonal - implies slow tempo change
        let sigma = 8f64;
        // don't want really short beat periods, or really long ones
        (20..wv_len - 20).for_each(|i| {
            (20..wv_len - 20).for_each(|j| {
                let mu = i as f64;
                tmat[i][j] = (-((j as f64 - mu).powi(2)) / (2.0 * sigma.powi(2))).exp();
            })
        });

        // parameters for Viterbi decoding... this part is taken from
        // Murphy's matlab
        let mut delta = vec![vec![0.0; rcfmat[0].len()]; rcfmat.len()];
        let mut psi = vec![vec![0; rcfmat[0].len()]; rcfmat.len()];

        let T = delta.len();

        if T < 2 {
            return vec![0.0; 0];
        }; // can't do anything at all meaningful

        let Q = delta[0].len();

        // initialize first column of delta
        (0..Q).for_each(|j| {
            delta[0][j] = wv[j] * rcfmat[0][j];
            psi[0][j] = 0;
        });

        let deltasum = delta[0].iter().sum::<f64>();
        delta[0].iter_mut().for_each(|i| *i /= deltasum + EPS);

        (1..T).for_each(|t| {
            let mut tmp_vec = vec![0.0; Q];
            (0..Q).for_each(|j| {
                tmp_vec
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, tv)| *tv = delta[t - 1][i] * tmat[j][i]);
                let (max_idx, max_val) = math::max(&tmp_vec);
                delta[t][j] = max_val;
                psi[t][j] = max_idx;
                delta[t][j] *= rcfmat[t][j];
            });

            // normalise current delta column
            let deltasum = delta[t].iter().sum::<f64>();
            delta[t].iter_mut().for_each(|i| *i /= deltasum + EPS);
        });

        let mut bestpath = vec![0; T];
        let tmp_vec = &delta[T - 1];

        // find starting point - best beat period for "last" frame
        bestpath[T - 1] = math::max(tmp_vec).0;

        // backtrace through index of maximum values in psi
        (0..T - 1).rev().for_each(|t| {
            bestpath[t] = psi[t + 1][bestpath[t + 1]];
        });

        let mut lastind = 0;
        (0..T).for_each(|i| {
            let step = 128;
            (0..step).for_each(|j| {
                lastind = i * step + j;
                beat_period[lastind] = bestpath[i];
            });
            // println!(
            //     "bestpath[{}] = {} (used for beat_periods {} to {})",
            //     i,
            //     bestpath[i],
            //     i * step,
            //     i * step + step - 1
            // );
        });

        // fill in the last values...
        (lastind..beat_period.len()).for_each(|i| {
            beat_period[i] = beat_period[lastind];
        });

        beat_period
            .iter()
            .map(|period| 60.0 * self.rate as f64 / self.increment as f64 / *period as f64)
            .collect()
    }

    pub fn calculate_beats(
        &self,
        df: &[f64],
        beat_period: &[usize],
        alpha: f64,
        tightness: f64,
    ) -> Vec<usize> {
        if df.is_empty() || beat_period.is_empty() {
            return Vec::new();
        }

        let df_len = df.len();
        let mut cumscore = vec![0.0; df_len]; // store cumulative score
        let mut backlink = vec![-1; df_len]; // backlink (stores best beat locations at each time instant)
        let localscore = df; // localscore, for now this is the same as the detection function

        //double tightness = 4.;
        //double alpha = 0.9;

        // main loop
        (0..df_len).for_each(|i| {
            let prange_min = -2 * beat_period[i] as i32;
            let prange_max = (-0.5 * beat_period[i] as f64).round() as i32;

            // transition range
            let txwt_len = (prange_max - prange_min + 1) as usize;
            let mut txwt = vec![0.0; txwt_len];
            let mut scorecands = vec![0.0; txwt_len];

            (0..txwt_len).for_each(|j| {
                let mu = beat_period[i] as f64;
                txwt[j] = f64::exp(
                    -0.5 * f64::powi(
                        tightness * f64::ln((f64::round(2.0 * mu) - j as f64) / mu),
                        2,
                    ),
                );

                // IF IN THE ALLOWED RANGE, THEN LOOK AT CUMSCORE[I+PRANGE_MIN+J
                // ELSE LEAVE AT DEFAULT VALUE FROM INITIALISATION:  D_VEC_T SCORECANDS (TXWT.SIZE());
                let cscore_ind = i as i32 + prange_min + j as i32;
                if cscore_ind >= 0 {
                    scorecands[j] = txwt[j] * cumscore[cscore_ind as usize];
                }
            });

            // find max value and index of maximum value
            let (xx, vv) = math::max(&scorecands);
            cumscore[i] = alpha * vv + (1. - alpha) * localscore[i];
            backlink[i] = i as i32 + prange_min + xx as i32;
        });

        // STARTING POINT, I.E. LAST BEAT.. PICK A STRONG POINT IN cumscore VECTOR
        let tmp_vec = cumscore[(df_len - beat_period[beat_period.len() - 1])..df_len].to_vec();
        let (max_idx, _) = math::max(&tmp_vec);
        let startpoint = max_idx + df_len - beat_period[beat_period.len() - 1];

        // can happen if no results obtained earlier (e.g. input too short)
        let startpoint = startpoint.min(backlink.len() - 1);

        // USE BACKLINK TO GET EACH NEW BEAT (TOWARDS THE BEGINNING OF THE FILE)
        //  BACKTRACKING FROM THE END TO THE BEGINNING.. MAKING SURE NOT TO GO BEFORE SAMPLE 0
        let mut ibeats = Vec::new();
        ibeats.push(startpoint);
        while backlink[ibeats[ibeats.len() - 1]] > 0 {
            let b = ibeats[ibeats.len() - 1];
            if backlink[b] == b as i32 {
                break;
            } // shouldn't happen... haha
            ibeats.push(backlink[b] as usize);
        }

        // REVERSE SEQUENCE OF IBEATS AND STORE AS BEATS
        ibeats.reverse();
        ibeats
    }
}
