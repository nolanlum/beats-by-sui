extern crate apodize;
extern crate ringbuf;
extern crate rustfft;

#[cfg(test)]
#[macro_use]
extern crate approx;

pub mod qm_dsp;

use std::time::Duration;

use qm_dsp::DetectionFunction::DetectionFunction;
use qm_dsp::TempoTrackV2;
use ringbuf::{storage::Heap, traits::*, LocalRb};
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub struct BpmInfo {
    pub bpm: f64,

    pub range_lower: f64,
    pub range_upper: f64,
}

type FramePos = u64;

/// CoarseBeatDetector uses the qm-dsp library to detect the
/// coarse location of beats using some autocorrelation and vocoder
/// math magic. The end result is that we have a decent idea of where
/// beats are, within some ~12ms windows. However, that resolution
/// gives us only a limited number of BPM values, none of which are
/// whole numbers, and most music is written with whole-number BPMs.
///
/// Expressed in BPM it means we have for instance steps of these
/// BPM values around 120 BPM:
/// ```
/// 117.454 - 120.185 - 123.046 - 126.048
/// ```
/// A pure electronic 120.000 BPM track will detect as, many beats at
/// an interval of 120.185 BPM, and a few 117.454 BPM beats
/// to adjust the collected offset.
///
/// The results from CoarseBeatDetector are post-processed to detect
/// these "leap" beats, and create a more accurate bpm.
pub struct CoarseBeatDetector {
    sample_rate: u32,
    window_size: usize,
    step_size_frames: usize,

    window_buf: LocalRb<Heap<f64>>,
    processed_frame_count: u64,

    detection_function: DetectionFunction,
    detection_results: Vec<f64>,
}

impl CoarseBeatDetector {
    // This determines the resolution of the resulting BeatMap.
    // ~12 ms (86 Hz) is a fair compromise between accuracy and analysis speed,
    // also matching the preferred window/step sizes from BeatTrack VAMP.
    const STEP_SECS: f64 = 0.01161;
    // results in 43 Hz @ 44.1 kHz / 47 Hz @ 48 kHz / 47 Hz @ 96 kHz
    const MAX_BIN_SIZE_HZ: u32 = 50;

    pub fn new(sample_rate: u32) -> Self {
        let step_size_frames = (sample_rate as f64 * Self::STEP_SECS).floor() as usize;
        let window_size = (sample_rate / Self::MAX_BIN_SIZE_HZ).next_power_of_two() as usize;
        let detection_function =
            DetectionFunction::new_with_defaults(step_size_frames, window_size);

        // println!(
        //     "sample rate {}Hz, window size {} frames, step size {} frames",
        //     sample_rate, window_size, step_size_frames
        // );

        // make sure the first frame is centered into the fft window. This makes sure
        // that the result is significant starting from the first step.
        let mut window_buf = LocalRb::new(window_size);
        let frames_pushed = window_buf.push_iter((0..window_size / 2).map(|_| 0.0));
        assert!(frames_pushed == window_size / 2);

        CoarseBeatDetector {
            sample_rate,
            window_size,
            step_size_frames,
            window_buf,
            detection_function,

            processed_frame_count: 0,
            detection_results: vec![0.0; 0],
        }
    }

    pub fn process_samples(&mut self, samples: &[f32]) {
        let mut samples_read = 0;
        while samples_read < samples.len() {
            let samples_available = samples.len() - samples_read;
            let write_available = self.window_buf.vacant_len();
            assert!(write_available > 0);

            let read_size = samples_available.min(write_available);
            assert!(samples_read + read_size <= samples.len());

            let samples_pushed = self.window_buf.push_iter(
                samples[samples_read..samples_read + read_size]
                    .iter()
                    .map(|x| *x as f64),
            );
            assert!(samples_pushed == read_size);
            samples_read += read_size;

            if self.window_buf.is_full() {
                self.detection_results.push(
                    self.detection_function
                        .process_time_domain(self.window_buf.iter(), self.window_size),
                );

                let samples_dropped = self.window_buf.skip(self.step_size_frames);
                assert!(samples_dropped == self.step_size_frames);
            }
        }
        self.processed_frame_count += samples_read as u64
    }

    pub fn finalize(&mut self) -> Vec<FramePos> {
        // Process the last remaining samples by appending silence.
        let frames_to_fill_window = self.window_buf.vacant_len();
        if frames_to_fill_window < self.step_size_frames {
            let frames_pushed = self
                .window_buf
                .push_iter((0..frames_to_fill_window).map(|_| 0.0));
            assert!(frames_pushed == frames_to_fill_window);

            self.detection_results.push(
                self.detection_function
                    .process_time_domain(self.window_buf.iter(), self.window_size),
            )
        }

        let non_zero_count = self.detection_results.len()
            - self
                .detection_results
                .iter()
                .rev()
                .map_while(|x| if *x <= 0.0 { Some(1) } else { None })
                .sum::<usize>();

        // skip first 2 results as it might have detect noise as onset
        // that's how vamp does and seems works best this way
        let df = &self.detection_results[2.min(self.detection_results.len())..non_zero_count];
        let mut beat_period = vec![0; df.len()];

        let tt = TempoTrackV2::new(self.sample_rate, self.step_size_frames);
        tt.calculate_beat_period(df, &mut beat_period, 120.0, false);
        tt.calculate_beats(df, &beat_period, 0.9, 4.0)
            .into_iter()
            .map(|x| x as u64 * self.step_size_frames as u64)
            .collect()
    }

    pub fn processed_frames(&self) -> u64 {
        self.processed_frame_count
    }

    pub fn processed_frames_duration(&self) -> Duration {
        Duration::from_secs_f64(self.processed_frame_count as f64 / self.sample_rate as f64)
    }
}

/// Beatmania is adapted from the BeatUtils class of mixxx.
/// https://github.com/mixxxdj/mixxx/blob/master/src/track/beatutils.cpp
pub struct Beatmania {
    sample_rate: u32,
}

impl Beatmania {
    /// When ironing the grid for long sequences of const tempo we use
    /// a 25 ms tolerance because this small of a difference is inaudible
    /// This is > 2 * 12 ms, the step width of the QM beat detector
    const MAX_BEAT_DRIFT_ERROR_SECS: f64 = 0.025;

    /// This is set to avoid to use a constant region during an offset shift.
    /// That happens for instance when the beat instrument changes.
    const MAX_BEAT_DRIFT_ERROR_SUM_SECS: f64 = 0.1;

    const MAX_OUTLIERS_COUNT: u32 = 1;
    const MIN_REGION_BEAT_COUNT: u32 = 16;

    pub fn new(sample_rate: u32) -> Self {
        Beatmania { sample_rate }
    }

    pub fn calculateBpm(&self, rough_beats: &[FramePos]) -> f64 {
        if rough_beats.len() < 2 {
            return 0.0;
        }

        // If we don't have enough beats, just take the simple average.
        if rough_beats.len() < Self::MIN_REGION_BEAT_COUNT as usize {
            let num_beats = rough_beats.len() as f64;
            let sample_rate = self.sample_rate as f64;
            let num_frames = (rough_beats[rough_beats.len() - 1] - rough_beats[0]) as f64;
            return 60.0 * num_beats * sample_rate / num_frames;
        }

        0.0
    }
}

#[wasm_bindgen]
pub fn detect_bpm(samples: &[f32], sample_rate: u32) -> BpmInfo {
    let mut bpm_machine = CoarseBeatDetector::new(sample_rate);
    bpm_machine.process_samples(samples);
    bpm_machine.finalize();
    BpmInfo {
        bpm: 0.0,
        range_lower: 0.0,
        range_upper: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stellar_stellar() {
        let rough_beats = [
            1114u64, 20052, 40104, 60156, 84107, 108058, 132009, 154289, 176012, 197735, 217787,
            237839, 257891, 277943, 291868, 306350, 325288, 344226, 364278, 384330, 405496, 424434,
            444486, 465652, 486818, 506870, 527479, 548088, 568697, 589863, 609915, 630524, 651133,
            671185, 690680, 710175, 733569, 758077, 782585, 807093, 831044, 854438, 878389, 902340,
            926291, 949128, 973079, 997030, 1020981, 1046046, 1070554, 1095619, 1120127, 1145192,
            1170257, 1194765, 1219273, 1243781, 1268289, 1292240, 1317305, 1341813, 1366321,
            1391386, 1416451, 1440959, 1465467, 1489975, 1514483, 1539548, 1564056, 1588564,
            1613629, 1638137, 1663202, 1687710, 1712218, 1736726, 1761791, 1786299, 1810807,
            1835315, 1860380, 1884888, 1909953, 1934461, 1958969, 1983477, 2008542, 2033050,
            2058115, 2082623, 2107131, 2131639, 2156704, 2181212, 2205720, 2230785, 2255293,
            2280358, 2304866, 2328817, 2352211, 2376162, 2400113, 2424621, 2448572, 2472523,
            2495917, 2519311, 2543819, 2567770, 2592835, 2617343, 2642408, 2666916, 2691424,
            2716489, 2740997, 2765505, 2790570, 2815078, 2839586, 2864651, 2889159, 2913667,
            2938732, 2963240, 2987748, 3012813, 3037321, 3061829, 3086337, 3111402, 3135910,
            3160418, 3185483, 3209991, 3234499, 3259564, 3284072, 3308580, 3333645, 3358153,
            3382661, 3407726, 3432234, 3456742, 3481250, 3505758, 3530823, 3555888, 3588194,
            3621057, 3653920, 3686783, 3720203, 3753066, 3785372, 3818235, 3851098, 3883961,
            3916824, 3950244, 3983107, 4016527, 4049947, 4082810, 4115673, 4148536, 4181399,
            4214262, 4247125, 4279988, 4312851, 4345714, 4378577, 4411440, 4444303, 4477166,
            4510029, 4542892, 4575755, 4609175, 4642038, 4674344, 4707207, 4740627, 4773490,
            4806353, 4839216, 4872079, 4904942, 4937805, 4970668, 5003531, 5036951, 5069814,
            5102677, 5135540, 5168403, 5201266, 5234129, 5266435, 5299855, 5332718, 5366138,
            5398444, 5431864, 5464170, 5497590, 5530453, 5563316, 5596179, 5629042, 5661905,
            5694768, 5727631, 5760494, 5793357, 5826777, 5859640, 5892503, 5925366, 5958229,
            5991649, 6023955, 6056818, 6090238, 6123101, 6155964, 6189384, 6222247, 6255110,
            6287973, 6320836, 6353699, 6386005, 6418868, 6451731, 6484594, 6517457, 6550320,
            6583740, 6616603, 6649466, 6682329, 6715192, 6748055, 6780918, 6813781, 6846644,
            6879507, 6912370, 6945790, 6978653, 7011516, 7044379, 7077242, 7110105, 7142968,
            7175831, 7208694, 7241557, 7274977, 7307840, 7340703, 7373566, 7406429, 7439292,
            7472155, 7504461, 7537881, 7571301, 7604164, 7636470,
        ];

        let iidx = Beatmania::new(48000);
        assert_ulps_eq!(175.0, iidx.calculateBpm(&rough_beats), epsilon = 1e-6);
    }
}
