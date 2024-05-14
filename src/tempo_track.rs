/*
This code is based on the QM DSP Library, badly ported to Rust.

The original code can be found at https://github.com/c4dm/qm-dsp/blob/master/dsp/tempotracking/TempoTrackV2.cpp.
*/

/// Just some arbitrary small number
const EPS: f64 = 8e-7;

mod math {
    pub fn adaptive_threshold(data: &mut [f64]) {
        if data.len() == 0 {
            return;
        }

        let pre = 8;
        let post = 7;

        let smoothed: Vec<f64> = (0..data.len())
            .map(|i| {
                let first = 0.max(i - pre);
                let last = (data.len() - 1).min(i + post);
                data[first..last].iter().sum::<f64>() / (last - first) as f64
            })
            .collect();

        data.iter_mut()
            .zip(smoothed.iter())
            .for_each(|(d, s)| *d = (*d - s).max(0.0));
    }

    pub fn max(data: &[f64]) -> (usize, f64) {
        data.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, x)| (i, *x)).expect("data was empty?")
    }
}

/// A tempo tracker that will operate on beat detection function data calculated from
/// audio at the given sample rate with the given frame increment.
///
/// Currently the sample rate and increment are used only for the conversion from
/// beat frame location to bpm in the tempo array.
pub struct TempoTrackV2 {
    rate: f64,
    increment: usize,
}

/// !!! Question: how far is this actually sample rate dependent?  I
/// think it does produce plausible results for e.g. 48000 as well as
/// 44100, but surely the fixed window sizes and comb filtering will
/// make it prefer double or half time when run at e.g. 96000?
impl TempoTrackV2 {
    fn filter_df(df: &mut [f64]) {
        let mut lp_df = vec![0.0; df.len()];

        // equivalent in matlab to [b,a] = butter(2,0.4);
        //
        // [b,a] = butter(n,Wn) returns the transfer function coefficients
        // of an nth-order lowpass digital Butterworth filter with
        // normalized cutoff frequency Wn.
        let a = (1.0, -0.3695, 0.1958);
        let b = (0.2066, 0.4131, 0.2066);

        // forwards filtering
        {
            let mut inp = (0.0, 0.0);
            let mut out = (0.0, 0.0);

            df.iter().zip(lp_df.iter_mut()).for_each(|(df_i, lp_df_i)| {
                *lp_df_i = b.0 * df_i + b.1 * inp.0 + b.2 * inp.1 - a.1 * out.0 - a.2 * out.1;
                inp = (*df_i, inp.0);
                out = (*lp_df_i, out.0);
            });
        }

        // copy forwards filtering...
        // but, time-reversed, ready for backwards filtering
        let backwards_df = lp_df.clone().into_iter().rev();

        // backwards filtering on time-reversed df
        {
            let mut inp = (0.0, 0.0);
            let mut out = (0.0, 0.0);

            backwards_df
                .zip(lp_df.iter_mut())
                .for_each(|(df_i, lp_df_i)| {
                    *lp_df_i = b.0 * df_i + b.1 * inp.0 + b.2 * inp.1 - a.1 * out.0 - a.2 * out.1;
                    inp = (df_i, inp.0);
                    out = (*lp_df_i, out.0);
                });
        }

        // write the re-reversed (i.e. forward) version back to df
        df.iter_mut()
            .zip(lp_df.iter())
            .for_each(|(df_i, lp_df_i)| *df_i = *lp_df_i);
    }

    /// MEPD 28/11/12
    /// This function now allows for a user to specify an inputtempo (in BPM)
    /// and a flag "constraintempo" which replaces the general rayleigh weighting for periodicities
    /// with a gaussian which is centered around the input tempo
    /// Note, if inputtempo = 120 and constraintempo = false, then functionality is
    /// as it was before
    pub fn calculate_beat_period(
        &self,
        df: &[f64],
        beat_period: &mut [f64],
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
        let rayparam = (60 * 44100 / 512) as f64 / input_tempo;

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
            let rcf = Self::get_rcf(df_frame, &wv);

            // add a new colume to rcfmat
            rcfmat.push(rcf);

            i += step
        }

        // now call viterbi decoding function
        self.viterbi_decode(&rcfmat, &wv, beat_period)
    }

    fn get_rcf(df_frame: &[f64], wv: &[f64]) -> Vec<f64> {
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
        let numelem: usize = 4;

        // max beat period
        (2..rcf_len).for_each(|i| {
            // number of comb elements
            (1..numelem).for_each(|a| {
                // general state using normalisation of comb elements
                (1 - a..=a - 1).for_each(|b| {
                    // calculate value for comb filter row
                    rcf[i - 1] += (acf[(a * i + b) - 1] * wv[i - 1]) / (2.0 * a as f64 - 1.0);
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

        return rcf;
    }

    #[allow(non_snake_case)]
    fn viterbi_decode(&self, rcfmat: &[Vec<f64>], wv: &[f64], beat_period: &mut [f64]) -> Vec<f64> {
        // following Kevin Murphy's Viterbi decoding to get best path of
        // beat periods through rfcmat
        let wv_len = wv.len();

        // make transition matrix
        let mut tmat = vec![vec![0.0; wv_len]; wv_len];

        // variance of Gaussians in transition matrix
        // formed of Gaussians on diagonal - implies slow tempo change
        let sigma = 8f64;
        // don't want really short beat periods, or really long ones
        (20..wv_len-20).for_each(|i| {
            (20..wv_len-20).for_each(|j| {
                let mu = i as f64;
                tmat[i][j] = (-((j as f64 -mu).powi(2)) / (2.0 * sigma.powi(2))).exp();
            })
        });

        // parameters for Viterbi decoding... this part is taken from
        // Murphy's matlab
        let mut delta = vec![vec![0.0; rcfmat[0].len()]; rcfmat.len()];
        let mut psi = vec![vec![0; rcfmat[0].len()]; rcfmat.len()];

        let T = delta.len();

        if T < 2 { return vec![0.0;0]; }; // can't do anything at all meaningful

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
                tmp_vec.iter_mut().enumerate().for_each(|(i, tv)| *tv = delta[t-1][i] * tmat[j][i]);
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
        let tmp_vec = delta[T-1].clone();

        // find starting point - best beat period for "last" frame
        bestpath[T-1] = math::max(&tmp_vec).0;

        // backtrace through index of maximum values in psi
        (0..T-1).rev().for_each(|t| {
            bestpath[t] = psi[t+1][bestpath[t+1]];
        });

        let mut lastind = 0;
        (0..T).for_each(|i| {
            let step = 128;
            (0..step).for_each(|j| {
                lastind = i * step + j;
                beat_period[lastind] = bestpath[i] as f64;
            });
            println!("bestpath[{}] = {} (used for beat_periods {} to {})",
        i, bestpath[i], i*step, i*step+step-1);
        });

        // fill in the last values...
        (lastind..beat_period.len()).for_each(|i| {
            beat_period[i] = beat_period[lastind];
        });

        beat_period.iter().map(|period| {
            60.0 * self.rate / self.increment as f64 / period
        }).collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_df_simple() {
        let mut df = vec![1.0; 8];
        TempoTrackV2::filter_df(&mut df);
        assert_ulps_eq!(
            df.as_slice(),
            [0.2064, 0.6942, 1.0379, 1.0734, 1.0417, 1.0451, 0.9791, 0.7028].as_slice(),
            epsilon = 1e-4
        );
    }
}
