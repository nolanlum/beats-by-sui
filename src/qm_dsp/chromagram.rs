/*
This code is based on the QM DSP Library, badly ported to Rust.

The original code can be found at:
    - https://github.com/c4dm/qm-dsp/blob/master/dsp/chromagram/Chromagram.cpp
    - https://github.com/c4dm/qm-dsp/blob/master/dsp/chromagram/ConstantQ.cpp
    - https://github.com/mixxxdj/mixxx/blob/main/lib/qm-dsp/dsp/chromagram/Chromagram.cpp
    - https://github.com/mixxxdj/mixxx/blob/main/lib/qm-dsp/dsp/chromagram/ConstantQ.cpp
*/

use std::{f64::consts::PI, sync::Arc};

use rustfft::{
    num_complex::{Complex64, ComplexFloat},
    num_traits::Zero,
    Fft, FftPlanner,
};

use super::math::{self, NormalizeType};

pub struct ConstantQ {
    q: f64,
    hop: u32,
    fft_length: u32,
    k: u32,

    sparse_kernel: SparseKernel,
}

struct SparseKernel {
    is: Vec<u32>,
    js: Vec<u32>,
    val: Vec<Complex64>,
}

impl ConstantQ {
    /// Creates a new ConstantQ function.
    ///
    /// `fs` is the sample rate
    /// `f_min` is the minimum frequency
    /// `f_max` is the maximum frequency
    /// `bpo` is the bins per octave
    /// `cq_thresh` is the threshold
    pub fn new(fs: f64, f_min: f64, f_max: f64, bpo: u32, cq_thresh: f64) -> Self {
        // Q value for filter bank
        let q = 1.0 / (f64::powf(2.0, 1.0 / bpo as f64) - 1.0);
        // No. of constant Q bins
        let k = (bpo as f64 * (f_max / f_min).ln() / f64::ln(2.0)).ceil() as u32;
        // Length of fft required for this Constant Q filter bank
        let fft_length = ((q * fs / f_min).ceil() as u32).next_power_of_two();
        // Hop from one frame to next
        let hop = fft_length / 8;

        let sparse_kernel =
            Self::create_sparse_kernel(fs, f_min, bpo, cq_thresh, q, fft_length as usize, k);

        ConstantQ {
            q,
            hop,
            fft_length,
            k,

            sparse_kernel,
        }
    }

    #[allow(unused)]
    pub fn get_q(&self) -> f64 {
        self.q
    }

    pub fn get_k(&self) -> u32 {
        self.k
    }

    pub fn get_fft_length(&self) -> u32 {
        self.fft_length
    }

    pub fn get_hop(&self) -> u32 {
        self.hop
    }

    fn create_sparse_kernel(
        fs: f64,
        f_min: f64,
        bpo: u32,
        cq_thresh: f64,
        q: f64,
        fft_length: usize,
        k: u32,
    ) -> SparseKernel {
        let mut is = Vec::new();
        let mut js = Vec::new();
        let mut val = Vec::new();

        let mut window = vec![Complex64::default(); fft_length];

        // for each bin value K, calculate temporal kernel, take its fft
        // to calculate the spectral kernel then threshold it to make it
        // sparse and add it to the sparse kernels matrix

        let square_threshold = cq_thresh.powi(2);

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_length);
        let mut fft_scratch = vec![Complex64::default(); fft.get_inplace_scratch_len()];

        for j in (0..k).rev() {
            window.iter_mut().for_each(|x| *x = Complex64::zero());

            // Compute a complex sinusoid windowed with a hamming window
            // of the right length
            let samples_per_cycle = fs / (f_min * f64::powf(2.0, j as f64 / bpo as f64));
            let window_length = (q * samples_per_cycle).ceil() as u32;

            let origin = fft_length as u32 / 2 - window_length / 2;

            window
                .iter_mut()
                .enumerate()
                .skip(origin as usize)
                .take(window_length as usize)
                .for_each(|(i, x)| *x = Complex64::cis((2.0 * PI * i as f64) / samples_per_cycle));

            // Shape with hamming window
            let hamming = apodize::hamming_iter(window_length as usize).collect::<Vec<_>>();
            window
                .iter_mut()
                .skip(origin as usize)
                .zip(hamming.iter())
                .for_each(|(x, y)| *x *= y);

            // Scale
            window
                .iter_mut()
                .skip(origin as usize)
                .take(window_length as usize)
                .for_each(|x| *x /= window_length as f64);

            // Input is expected to have been fftshifted, so do the
            // same to the input to the fft that contains the kernel
            for i in 0..fft_length / 2 {
                window.swap(i, i + fft_length / 2);
            }

            fft.process_with_scratch(&mut window, &mut fft_scratch);

            // convert to sparse form
            for (i, x) in window.iter().enumerate() {
                // perform thresholding
                let mag = x.norm_sqr();
                if mag <= square_threshold {
                    continue;
                }

                // Insert non-zero position indexes
                is.push(i as u32);
                js.push(j);

                // take conjugate, normalise and add to array for sparse kernel
                val.push(x.conj() / fft_length as f64);
            }
        }

        SparseKernel { is, js, val }
    }

    pub fn process(&mut self, fft: &[Complex64], cq: &mut [Complex64]) {
        cq.iter_mut().take(self.k as usize).for_each(|x| {
            *x = Complex64::zero();
        });

        let fftbin = &self.sparse_kernel.is;
        let cqbin = &self.sparse_kernel.js;
        let val = &self.sparse_kernel.val;
        let sparse_cells = self.sparse_kernel.val.len();

        for i in 0..sparse_cells {
            let row = cqbin[i];
            let col = fftbin[i];
            if col == 0 {
                continue;
            }

            let x = val[i];
            let y = fft[(self.fft_length - col) as usize];
            cq[row as usize] += x * y;
        }
    }
}

pub struct Chromagram {
    bpo: u32,
    k: u32,
    frame_size: u32,
    hop_size: u32,
    normalize: NormalizeType,

    constant_q: ConstantQ,
    chroma_data: Vec<f64>,

    fft: Arc<dyn Fft<f64>>,
    fft_data: Vec<Complex64>,
    fft_scratch: Vec<Complex64>,
    cq_data: Vec<Complex64>,

    window: Vec<f64>,
}

impl Chromagram {
    pub fn new(
        fs: f64,
        f_min: f64,
        f_max: f64,
        bpo: u32,
        cq_thresh: f64,
        normalize: NormalizeType,
    ) -> Self {
        // Extend range to a full octave
        let octaves = (f_max / f_min).ln() / f64::ln(2.0);
        let f_max = f_min * 2f64.powf(octaves.ceil());

        // Create array for chroma result
        let chroma_data = vec![0.0; bpo as usize];

        // Initialise ConstantQ operator
        let constant_q = ConstantQ::new(fs, f_min, f_max, bpo, cq_thresh);

        // No. of constant Q bins
        let k = constant_q.get_k();

        // Initialise working arrays
        let frame_size = constant_q.get_fft_length();
        let hop_size = constant_q.get_hop();

        // Initialise FFT object
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame_size as usize);
        let fft_scratch = vec![Complex64::default(); fft.get_inplace_scratch_len()];

        let fft_data = vec![Complex64::default(); frame_size as usize];
        let cq_data = vec![Complex64::default(); k as usize];

        let window = apodize::hamming_iter(frame_size as usize).collect::<Vec<_>>();

        Chromagram {
            bpo,
            k,
            frame_size,
            hop_size,
            normalize,

            constant_q,
            chroma_data,

            fft,
            fft_data,
            fft_scratch,
            cq_data,
            window,
        }
    }

    #[allow(unused)]
    pub fn get_k(&self) -> u32 {
        self.k
    }

    pub fn get_frame_size(&self) -> u32 {
        self.frame_size
    }

    #[allow(unused)]
    pub fn get_hop_size(&self) -> u32 {
        self.hop_size
    }

    pub fn process(&mut self, data: &[f64]) -> &[f64] {
        let windowed = data.iter().zip(self.window.iter()).map(|(x, y)| x * y);
        self.fft_data
            .iter_mut()
            .zip(windowed)
            .for_each(|(x, val)| *x = Complex64::new(val, 0.0));

        // The frequency-domain version expects pre-fftshifted input - so
        // we must do the same here
        let half_fft_size: usize = self.frame_size as usize / 2;
        for i in 0..half_fft_size {
            self.fft_data.swap(i, i + half_fft_size);
        }

        self.fft
            .process_with_scratch(&mut self.fft_data, &mut self.fft_scratch);

        self.process_complex_internal()
    }

    fn process_complex_internal(&mut self) -> &[f64] {
        // initialise chromadata to 0
        self.chroma_data.iter_mut().for_each(|x| *x = 0.0);

        // Calculate ConstantQ frame
        self.constant_q.process(&self.fft_data, &mut self.cq_data);

        // add each octave of cq data into Chromagram
        let octaves = self.k / self.bpo;
        for octave in 0..octaves {
            let first_bin = (octave * self.bpo) as usize;
            for i in 0..self.bpo as usize {
                self.chroma_data[i] += self.cq_data[first_bin + i].abs();
            }
        }

        math::normalize(&mut self.chroma_data, self.normalize);

        &self.chroma_data
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_sparse_kernel() {
        let cq = ConstantQ::new(4000.0, 20.0, 2000.0, 36, 0.0054);
        assert_ne!(0, cq.sparse_kernel.val.len());
    }

    #[allow(unused)]
    fn midi_pitch_name(midi_pitch: impl Into<usize>) -> String {
        const NAMES: [&str; 12] = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        NAMES[midi_pitch.into() % NAMES.len()].to_string()
    }

    fn generate_sinusoid(frequency: f64, sample_rate: f64, length: usize) -> Vec<f64> {
        (0..length)
            .map(|i| f64::sin(i as f64 * PI * 2.0 * frequency / sample_rate))
            .collect()
    }

    fn frequency_for_pitch(midi_pitch: u32, concert_a: f64) -> f64 {
        concert_a * 2f64.powf((midi_pitch as f64 - 69.0) / 12.0)
    }

    fn test_sinusoid_12tet(concert_a: f64, sample_rate: f64, bpo: u32) {
        let chroma_min_pitch = 36;
        let chroma_max_pitch = 108;

        let probe_min_pitch = 36;
        let probe_max_pitch = 108;

        let mut chroma = Chromagram::new(
            sample_rate,
            frequency_for_pitch(chroma_min_pitch, concert_a),
            frequency_for_pitch(chroma_max_pitch, concert_a),
            bpo,
            0.0054,
            NormalizeType::None,
        );

        let bins_per_semi = bpo / 12;

        for midi_pitch in probe_min_pitch..probe_max_pitch {
            let block_size = chroma.get_frame_size();

            let frequency = frequency_for_pitch(midi_pitch, concert_a);
            let expected_peak_bin = ((midi_pitch - chroma_min_pitch) * bins_per_semi) % bpo;

            let signal = generate_sinusoid(frequency, sample_rate, block_size as usize);

            let output = chroma.process(&signal);

            let (peak_bin, _peak_value) = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();

            assert_eq!(expected_peak_bin as usize, peak_bin);
        }
    }

    #[test]
    fn test_sinusoid_12tet_440_44100_36() {
        test_sinusoid_12tet(440.0, 44100.0, 36);
    }

    #[test]
    fn test_sinusoid_12tet_440_44100_60() {
        test_sinusoid_12tet(440.0, 44100.0, 60);
    }

    #[test]
    fn test_sinusoid_12tet_397_44100_60() {
        test_sinusoid_12tet(397.0, 44100.0, 60);
    }

    #[test]
    fn test_sinusoid_12tet_440_48000_60() {
        test_sinusoid_12tet(440.0, 48000.0, 60);
    }
}
