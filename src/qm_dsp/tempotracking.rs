/*
This code is based on the QM DSP Library, badly ported to Rust.

The original code can be found at:
    - https://github.com/c4dm/qm-dsp/blob/master/dsp/tempotracking/TempoTrackV2.cpp
*/

use super::math;

/// Just some arbitrary small number
const EPS: f64 = 8e-7;

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
