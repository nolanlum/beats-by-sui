/*
This code is based on the QM DSP Library, badly ported to Rust.

The original code can be found at:
    - https://github.com/c4dm/qm-dsp/blob/master/dsp/rateconversion/Decimator.cpp
*/

/// Decimator carries out a fast downsample by a power-of-two
/// factor. Only a limited number of factors are supported, from two to
/// whatever getHighestSupportedFactor() returns. This is much faster
/// than Resampler but has a worse signal-noise ratio.
pub struct Decimator {
    dec_factor: u32,

    output_length: usize,

    a: [f64; 8],
    b: [f64; 8],

    // As far as I can tell, these are here for extreme memory locality.
    input: f64,
    output: f64,
    o: [f64; 7],
    dec_buffer: Vec<f64>,
}

impl Decimator {
    /// Construct a Decimator to operate on input blocks of length
    /// inLength, with decimation factor decFactor.  inLength should be
    /// a multiple of decFactor.  Output blocks will be of length
    /// inLength / decFactor.
    ///
    /// decFactor must be a power of two.  The highest supported factor
    /// is obtained through getHighestSupportedFactor(); for higher
    /// factors, you will need to chain more than one decimator.
    pub fn new(in_length: usize, dec_factor: u32) -> Self {
        let output_length = in_length / dec_factor as usize;
        let dec_buffer = vec![0.0; in_length];

        let (b, a) = match dec_factor {
            8 => (
                [
                    0.060111378492136,
                    -0.257323420830598,
                    0.420583503165928,
                    -0.222750785197418,
                    -0.222750785197418,
                    0.420583503165928,
                    -0.257323420830598,
                    0.060111378492136,
                ],
                [
                    1.0,
                    -5.667654878577432,
                    14.062452278088417,
                    -19.737303840697738,
                    16.88969887460864,
                    -8.796600612325928,
                    2.577553446979888,
                    -0.326903916815751,
                ],
            ),
            4 => (
                [
                    0.1013330690491862,
                    -0.2447523353702363,
                    0.33622528590120965,
                    -0.13936581560633518,
                    -0.13936581560633382,
                    0.3362252859012087,
                    -0.2447523353702358,
                    0.10133306904918594,
                ],
                [
                    1.0,
                    -3.9035590278139427,
                    7.529937998062113,
                    -8.689080379317751,
                    6.457866709609918,
                    -3.024297943122363,
                    0.8304338513674838,
                    -0.09442080083780933,
                ],
            ),
            2 => (
                [
                    0.20898944260075727,
                    0.40011234879814367,
                    0.819741973072733,
                    1.0087419911682323,
                    1.0087419911682325,
                    0.8197419730727316,
                    0.40011234879814295,
                    0.2089894426007566,
                ],
                [
                    1.0,
                    0.007733118420835822,
                    1.9853971155964376,
                    0.19296739275341004,
                    1.2330748872852182,
                    0.18705341389316466,
                    0.23659265908013868,
                    0.032352924250533946,
                ],
            ),

            // Unsupported decimation factors just get silently ignored
            _ => (
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        };

        Decimator {
            dec_factor,

            output_length,

            a,
            b,

            input: 0.0,
            output: 0.0,
            o: [0.0; 7],
            dec_buffer,
        }
    }

    /// Process inLength samples (as supplied to constructor) from src
    /// and write inLength / decFactor samples to dst.  Note that src
    /// and dst may be the same or overlap (an intermediate buffer is
    /// used).     
    pub fn process<'a>(&mut self, src: impl IntoIterator<Item = &'a f64>, dst: &mut [f64]) {
        if self.dec_factor == 1 {
            dst.iter_mut()
                .zip(src.into_iter().take(self.output_length))
                .for_each(|(dst_i, src_i)| *dst_i = *src_i);
            return;
        }

        self.do_anti_alias(src);

        dst.iter_mut()
            .enumerate()
            .take(self.output_length)
            .for_each(|(i, dst_i)| {
                *dst_i = self.dec_buffer[self.dec_factor as usize * i];
            });
    }

    fn do_anti_alias<'a>(&mut self, src: impl IntoIterator<Item = &'a f64>) {
        for (src_i, dst_i) in src.into_iter().zip(self.dec_buffer.iter_mut()) {
            self.input = *src_i;
            self.output = self.input * self.b[0] + self.o[0];

            self.o[0] = self.input * self.b[1] - self.output * self.a[1] + self.o[1];
            self.o[1] = self.input * self.b[2] - self.output * self.a[2] + self.o[2];
            self.o[2] = self.input * self.b[3] - self.output * self.a[3] + self.o[3];
            self.o[3] = self.input * self.b[4] - self.output * self.a[4] + self.o[4];
            self.o[4] = self.input * self.b[5] - self.output * self.a[5] + self.o[5];
            self.o[5] = self.input * self.b[6] - self.output * self.a[6] + self.o[6];
            self.o[6] = self.input * self.b[7] - self.output * self.a[7];

            *dst_i = self.output;
        }
    }
}
