/*
This code is based on the QM DSP Library, badly ported to Rust.

The original code can be found at:
    - https://github.com/c4dm/qm-dsp/blob/master/dsp/keydetection/GetKeyMode.cpp
*/

use super::{
    chromagram::Chromagram,
    math::{self, NormalizeType},
    rateconversion::Decimator,
};

const BINS_PER_OCTAVE: usize = 36;

// Chords profile
const MAJ_PROFILE: [f64; BINS_PER_OCTAVE] = [
    0.0384, 0.0629, 0.0258, 0.0121, 0.0146, 0.0106, 0.0364, 0.0610, 0.0267, 0.0126, 0.0121, 0.0086,
    0.0364, 0.0623, 0.0279, 0.0275, 0.0414, 0.0186, 0.0173, 0.0248, 0.0145, 0.0364, 0.0631, 0.0262,
    0.0129, 0.0150, 0.0098, 0.0312, 0.0521, 0.0235, 0.0129, 0.0142, 0.0095, 0.0289, 0.0478, 0.0239,
];

const MIN_PROFILE: [f64; BINS_PER_OCTAVE] = [
    0.0375, 0.0682, 0.0299, 0.0119, 0.0138, 0.0093, 0.0296, 0.0543, 0.0257, 0.0292, 0.0519, 0.0246,
    0.0159, 0.0234, 0.0135, 0.0291, 0.0544, 0.0248, 0.0137, 0.0176, 0.0104, 0.0352, 0.0670, 0.0302,
    0.0222, 0.0349, 0.0164, 0.0174, 0.0297, 0.0166, 0.0222, 0.0401, 0.0202, 0.0175, 0.0270, 0.0146,
];

fn get_frequency_for_pitch(midi_pitch: u32, cents_offset: f64, concert_a: f64) -> f64 {
    let p = midi_pitch as f64 + (cents_offset / 100.0);
    concert_a * 2f64.powf((p - 69.0) / 12.0)
}

struct GetKeyMode {
    chroma: Chromagram,
    decimator: Decimator,
    block_size: usize,
    hop_size: usize,
    decimation_factor: usize,

    chroma_buffer: Vec<f64>,
    chroma_buffer_size: usize,
    buffer_index: usize,
    chroma_buffer_filling: usize,

    median_filter_buffer: Vec<i32>,
    median_win_size: usize,
    median_buffer_filling: usize,
    sorted_buffer: Vec<i32>,

    decimated_buffer: Vec<f64>,
    mean_hpcp: Vec<f64>,

    maj_profile_norm: Vec<f64>,
    min_profile_norm: Vec<f64>,
    maj_corr: Vec<f64>,
    min_corr: Vec<f64>,
}

impl GetKeyMode {
    pub fn new(
        sample_rate: u32,
        tuning_frequency: f64,
        hpcp_average: f64,
        median_average: f64,
        frame_overlap_factor: u32,
        decimation_factor: u32,
    ) -> Self {
        // Chromagram configuration parameters
        let chroma_normalize = NormalizeType::UnitMax;
        let chroma_fs = 1f64.max(sample_rate as f64 / decimation_factor as f64);

        // Set C3 (= MIDI #48) as our base:
        // This implies that key = 1 => Cmaj, key = 12 => Bmaj, key = 13 => Cmin, etc.
        let chroma_min = get_frequency_for_pitch(48, 0.0, tuning_frequency);
        let chroma_max = get_frequency_for_pitch(96, 0.0, tuning_frequency);
        let chroma_bpo = BINS_PER_OCTAVE as u32;
        let chroma_cq_thresh = 0.0054;

        // Chromagram inst.
        let chroma = Chromagram::new(
            chroma_fs,
            chroma_min,
            chroma_max,
            chroma_bpo,
            chroma_cq_thresh,
            chroma_normalize,
        );

        // Get calculated parameters from chroma object
        let chroma_frame_size = chroma.get_frame_size() as usize;

        // override hopsize for this application
        let chroma_hop_size = chroma_frame_size / frame_overlap_factor as usize;

        // Chromagram average and estimated key median filter lengths
        let chroma_buffer_size =
            (hpcp_average * chroma_fs / chroma_frame_size as f64).ceil() as usize;
        let median_win_size =
            (median_average * chroma_fs / chroma_frame_size as f64).ceil() as usize;

        // Spawn objectc/arrays
        let decimated_buffer = vec![0.0; chroma_frame_size];
        let chroma_buffer = vec![0.0; BINS_PER_OCTAVE * chroma_buffer_size];

        let mean_hpcp = vec![0.0; BINS_PER_OCTAVE];

        let maj_corr = vec![0.0; BINS_PER_OCTAVE];
        let min_corr = vec![0.0; BINS_PER_OCTAVE];

        let mut maj_profile_norm = vec![0.0; BINS_PER_OCTAVE];
        let mut min_profile_norm = vec![0.0; BINS_PER_OCTAVE];

        let m_maj = MAJ_PROFILE.iter().sum::<f64>() / BINS_PER_OCTAVE as f64;
        let m_min = MIN_PROFILE.iter().sum::<f64>() / BINS_PER_OCTAVE as f64;

        for i in 0..BINS_PER_OCTAVE {
            maj_profile_norm[i] = MAJ_PROFILE[i] - m_maj;
            min_profile_norm[i] = MIN_PROFILE[i] - m_min;
        }

        let median_filter_buffer = vec![0; median_win_size];
        let sorted_buffer = vec![0; median_win_size];

        let decimator = Decimator::new(
            chroma_frame_size * decimation_factor as usize,
            decimation_factor,
        );
        // let key_strengths = vec![0; 24];

        GetKeyMode {
            chroma,
            decimator,
            block_size: chroma_frame_size,
            hop_size: chroma_hop_size,
            decimation_factor: decimation_factor as usize,

            chroma_buffer,
            chroma_buffer_size,
            buffer_index: 0,
            chroma_buffer_filling: 0,

            median_filter_buffer,
            median_win_size,
            median_buffer_filling: 0,
            sorted_buffer,

            decimated_buffer,
            mean_hpcp,

            maj_profile_norm,
            min_profile_norm,
            maj_corr,
            min_corr,
        }
    }

    pub fn new_with_defaults(sample_rate: u32, tuning_frequency: f64) -> Self {
        Self::new(sample_rate, tuning_frequency, 10.0, 10.0, 1, 8)
    }

    pub fn get_block_size(&self) -> usize {
        self.block_size * self.decimation_factor
    }

    pub fn get_hop_size(&self) -> usize {
        self.hop_size * self.decimation_factor
    }

    fn krum_corr(data_norm: &[f64], profile_norm: &[f64], shift_profile: i32, length: i32) -> f64 {
        let mut num = 0.0;
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;

        #[allow(clippy::needless_range_loop)]
        for i in 0..length as usize {
            let k = ((i as i32 - shift_profile + length) % length) as usize;

            num += data_norm[i] * profile_norm[k];

            sum1 += data_norm[i] * data_norm[i];
            sum2 += profile_norm[k] * profile_norm[k];
        }

        let den = (sum1 * sum2).sqrt();

        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    }

    /// Process a single time-domain input sample frame of length
    /// getBlockSize(). Successive calls should provide overlapped data
    /// with an advance of getHopSize() between frames.
    ///
    /// Return a key index in the range 0-24, where 0 indicates no key
    /// detected, 1 is C major, and 13 is C minor.
    pub fn process(&mut self, pcm_data: &[f64]) -> i32 {
        self.decimator.process(pcm_data, &mut self.decimated_buffer);

        let chroma_ptr = self.chroma.process(&self.decimated_buffer);

        // populate hpcp values
        for (j, item) in chroma_ptr.iter().enumerate().take(BINS_PER_OCTAVE) {
            self.chroma_buffer[self.buffer_index * BINS_PER_OCTAVE + j] = *item;
        }

        // keep track of input buffers
        self.buffer_index = (self.buffer_index + 1) % self.chroma_buffer_size;

        // track filling of chroma matrix
        self.chroma_buffer_filling = self.chroma_buffer_size.min(self.chroma_buffer_filling + 1);

        // calculate mean
        for k in 0..BINS_PER_OCTAVE {
            let sum = (0..self.chroma_buffer_filling)
                .map(|j| self.chroma_buffer[k + j * BINS_PER_OCTAVE])
                .sum::<f64>();
            self.mean_hpcp[k] = sum / self.chroma_buffer_filling as f64;
        }

        // Normalize for zero average
        let hpcp = self.mean_hpcp.iter().sum::<f64>() / self.mean_hpcp.len() as f64;
        self.mean_hpcp.iter_mut().for_each(|x| *x -= hpcp);

        for k in 0..BINS_PER_OCTAVE {
            // The Chromagram has the center of C at bin 0, while the major
            // and minor profiles have the center of C at 1. We want to have
            // the correlation for C result also at 1.
            // To achieve this we have to shift two times:
            self.maj_corr[k] = Self::krum_corr(
                &self.mean_hpcp,
                &self.maj_profile_norm,
                k as i32 - 2,
                BINS_PER_OCTAVE as i32,
            );
            self.min_corr[k] = Self::krum_corr(
                &self.mean_hpcp,
                &self.min_profile_norm,
                k as i32 - 2,
                BINS_PER_OCTAVE as i32,
            );
        }

        // m_MajCorr[1] is C center  1 / 3 + 1 = 1
        // m_MajCorr[4] is D center  4 / 3 + 1 = 2
        // '+ 1' because we number keys 1-24, not 0-23.
        let (max_maj_bin, max_maj) = math::max(&self.maj_corr);
        let (max_min_bin, max_min) = math::max(&self.min_corr);
        let max_bin = if max_maj > max_min {
            max_maj_bin
        } else {
            max_min_bin + BINS_PER_OCTAVE
        };
        let key = max_bin as i32 / 3 + 1;

        // Median filtering

        // track Median buffer initial filling
        self.median_buffer_filling = self.median_win_size.min(self.median_buffer_filling + 1);

        // shift median buffer
        for k in 1..self.median_win_size {
            self.median_filter_buffer[k - 1] = self.median_filter_buffer[k];
        }

        // write new key value into median buffer
        self.median_filter_buffer[self.median_win_size - 1] = key;

        // copy median into sorting buffer, reversed
        self.sorted_buffer
            .iter_mut()
            .zip(self.median_filter_buffer.iter().rev())
            .for_each(|(sorted, median)| *sorted = *median);
        self.sorted_buffer[..self.median_buffer_filling].sort();

        let sort_length = self.median_buffer_filling;
        let midpoint = ((sort_length as f64 / 2.0).ceil() as usize).min(1);

        self.sorted_buffer[midpoint - 1]
    }
}

#[cfg(test)]
mod tests {

    use std::f64::consts::PI;

    use super::*;

    fn generate_sinusoid(frequency: f64, sample_rate: f64, length: usize) -> Vec<f64> {
        (0..length)
            .map(|i| f64::sin(i as f64 * PI * 2.0 * frequency / sample_rate))
            .collect()
    }

    #[test]
    fn test_sinusoid_12tet() {
        let concert_a = 440.0;
        let sample_rate = 44100;

        for midi_pitch in 48..96 {
            let mut gkm = GetKeyMode::new_with_defaults(sample_rate, concert_a);
            let block_size = gkm.get_block_size();
            let hop_size = gkm.get_hop_size();

            let frequency = concert_a * 2f64.powf((midi_pitch - 69) as f64 / 12.0);

            let blocks = 4;
            let total_length = block_size * blocks;
            let signal = generate_sinusoid(frequency, sample_rate as f64, total_length);

            let mut key = 0;
            let mut offset = 0;
            while offset + block_size < total_length {
                let k = gkm.process(&signal[offset..offset + block_size]);

                if offset == 0 {
                    key = k;
                } else {
                    assert_eq!(key, k);
                }

                offset += hop_size
            }

            let minor = key > 12;
            let tonic = if minor { key - 12 } else { key };

            assert_eq!(1 + (midi_pitch % 12), tonic);
        }
    }
}
