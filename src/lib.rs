extern crate apodize;
extern crate ringbuf;
extern crate rustfft;

#[cfg(test)]
#[macro_use]
extern crate approx;

pub mod qm_dsp;

use std::{fmt::Debug, time::Duration};

use qm_dsp::DetectionFunction::DetectionFunction;
use qm_dsp::TempoTrackV2;
use ringbuf::{storage::Heap, traits::*, LocalRb};
use wasm_bindgen::prelude::wasm_bindgen;

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
            // add step_size_frames/2 here, because the beat is detected between the two steps.
            .map(|x| x as u64 * self.step_size_frames as u64 + self.step_size_frames as u64 / 2)
            .collect()
    }

    pub fn processed_frames(&self) -> u64 {
        self.processed_frame_count
    }

    pub fn processed_frames_duration(&self) -> Duration {
        Duration::from_secs_f64(self.processed_frame_count as f64 / self.sample_rate as f64)
    }
}

/// When ironing the grid for long sequences of const tempo we use
/// a 25 ms tolerance because this small of a difference is inaudible
/// This is > 2 * 12 ms, the step width of the QM beat detector
const MAX_BEAT_DRIFT_ERROR_SECS: f64 = 0.025;

/// This is set to avoid to use a constant region during an offset shift.
/// That happens for instance when the beat instrument changes.
const MAX_TOTAL_BEAT_DRIFT_ERROR_SECS: f64 = 0.1;

#[derive(Clone, Copy)]
struct ExclusiveInterval<T>
where
    T: PartialOrd,
{
    pub min: T,
    pub max: T,
}

impl<T: PartialOrd> ExclusiveInterval<T> {
    fn new(min: T, max: T) -> Self {
        ExclusiveInterval { min, max }
    }

    fn contains(&self, other: T) -> bool {
        other > self.min && other < self.max
    }
}

impl<T: PartialOrd + Debug> Debug for ExclusiveInterval<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("").field(&self.min).field(&self.max).finish()
    }
}

#[derive(Clone, Debug)]
struct BpmRegion {
    /// Sample rate stored as a double, for convenience.
    sample_rate: f64,

    start_frame: FramePos,
    end_frame: FramePos,

    beat_count: u32,
    average_beat_length: f64,
}

impl BpmRegion {
    fn uncertainty(&self) -> f64 {
        (MAX_BEAT_DRIFT_ERROR_SECS * self.sample_rate) / self.beat_count as f64
    }

    fn beat_length_range(&self) -> ExclusiveInterval<f64> {
        let length = self.average_beat_length;
        let alpha = self.uncertainty();
        ExclusiveInterval::new(length - alpha, length + alpha)
    }

    fn round_bpm(&self) -> f64 {
        let min_bpm = 60.0 * self.sample_rate / self.beat_length_range().max;
        let max_bpm = 60.0 * self.sample_rate / self.beat_length_range().min;
        let average_bpm = 60.0 * self.sample_rate / self.average_beat_length;
        let bpm_in_range = |x| x > min_bpm && x < max_bpm;

        // First try simply rounding to an integer BPM.
        if bpm_in_range(average_bpm.round()) {
            return average_bpm.round();
        }

        // Probe the reasonable multipliers for 0.5
        let bpm_range_width = max_bpm - min_bpm;
        if bpm_range_width > 0.5 {
            // 0.5 BPM are only reasonable if the double value is not insane
            // or the 2/3 value is not too small.
            if average_bpm < 85.0 {
                // This can be actually up to 170 BPM, allow halved BPM values.
                return (average_bpm * 2.0).round() / 2.0;
            } else if average_bpm > 127.0 {
                // optimize for 2/3 going down to 85
                return (average_bpm / 3.0 * 2.0).round() * 3.0 / 2.0;
            }
        } else {
            // Even if the range is small, prefer the 1/2 bpm if it's in range.
            // (this departs from the original algorithm)
            let bpm_to_nearest_half = (average_bpm * 2.0).round() / 2.0;
            if bpm_in_range(bpm_to_nearest_half) {
                return bpm_to_nearest_half;
            }
        }

        let bpm_to_nearest_twelfth = (average_bpm * 12.0).round() / 12.0;
        if bpm_range_width > (1.0 / 12.0) {
            // This covers all sorts of 1/2 2/3 and 3/4 multiplier
            return bpm_to_nearest_twelfth;
        } else {
            // We are here if we have more than ~75 beats or ~30 s, try to snap to a 1/12 bpm.
            if bpm_in_range(bpm_to_nearest_twelfth) {
                return bpm_to_nearest_twelfth;
            }
        }

        // Give up and use the original BPM value.
        average_bpm
    }
}

/// Beatmania is adapted from the BeatUtils class of mixxx.
/// https://github.com/mixxxdj/mixxx/blob/master/src/track/beatutils.cpp
pub struct Beatmania {
    /// Sample rate stored as a double, for convenience.
    sample_rate: f64,
}

impl Beatmania {
    const MAX_OUTLIERS_COUNT: u32 = 1;
    const MIN_REGION_BEAT_COUNT: u32 = 16;

    pub fn new(sample_rate: u32) -> Self {
        Beatmania {
            sample_rate: sample_rate as f64,
        }
    }

    pub fn calculate_bpm(&self, rough_beats: &[FramePos]) -> f64 {
        if rough_beats.len() < 2 {
            return 0.0;
        }

        // If we don't have enough beats, just take the simple average.
        if rough_beats.len() < Self::MIN_REGION_BEAT_COUNT as usize {
            let num_beats = rough_beats.len() as f64;
            let num_frames = (rough_beats[rough_beats.len() - 1] - rough_beats[0]) as f64;
            return 60.0 * num_beats * self.sample_rate / num_frames;
        }

        let bpm_regions = self.find_consistent_bpm_regions(rough_beats);
        self.detect_consistent_bpm(&bpm_regions)
    }

    /// We attempt to find regions of the track that have roughly "consistent"
    /// internal tempo, defined as the simple average beat length (num beats/num frames).
    /// We construct these regions such that they have have `MAX_OUTLIERS_COUNT`
    /// number of beats that don't align with this "consistent" tempo.
    /// The hope is that averaging out the BPM over the track in this way acts as a
    /// smoothing function over the beat locations returned by the qm-dsp library, and
    /// gets us closer to the true bpm of the track.
    fn find_consistent_bpm_regions(&self, rough_beats: &[FramePos]) -> Vec<BpmRegion> {
        let (mut left, mut right) = (0, rough_beats.len() - 1);

        let max_beat_drift_error = self.sample_rate * MAX_BEAT_DRIFT_ERROR_SECS;
        let max_total_beat_drift_error = self.sample_rate * MAX_TOTAL_BEAT_DRIFT_ERROR_SECS;

        let mut bpm_regions = Vec::new();
        while left < rough_beats.len() - 1 {
            debug_assert!(right > left);

            let region_average_beat_length =
                (rough_beats[right] - rough_beats[left]) as f64 / (right - left) as f64;
            let mut num_outliers = 0;
            let mut total_beat_drift_error = 0.0;

            // println!(
            //     "Evaluating candidate region [{}, {}]. Average beat length {} frames",
            //     left, right, region_average_beat_length
            // );

            let region_end_idx = (left + 1..=right)
                .map_while(|i| {
                    let expected_beat_pos =
                        rough_beats[left] as f64 + (i - left) as f64 * region_average_beat_length;
                    let beat_drift_error = expected_beat_pos - rough_beats[i] as f64;
                    total_beat_drift_error += beat_drift_error;

                    // println!(
                    //     "  Beat {} (@{}) drifted {} from expected {}. [total drift {} frames]",
                    //     i,
                    //     rough_beats[i],
                    //     beat_drift_error,
                    //     expected_beat_pos,
                    //     total_beat_drift_error
                    // );

                    if beat_drift_error.abs() > max_beat_drift_error {
                        // First beat cannot be an outliar, or else the region is not consistent.
                        num_outliers += 1;
                        if num_outliers > Self::MAX_OUTLIERS_COUNT || i == left + 1 {
                            return None;
                        }
                    }
                    if total_beat_drift_error.abs() > max_total_beat_drift_error {
                        // The region is drifting away consistently from the average bpm,
                        // it cannot be consistent.
                        return None;
                    }

                    Some(i)
                })
                .last()
                .unwrap_or(left);

            if region_end_idx == right {
                // Verify that the first and the last beat are not correction beats
                // in the same direction. This would bend the region average beat length
                // unfavorably away from the optimum.
                let region_border_error = if right > left + 2 {
                    let first_beat_length = (rough_beats[left + 1] - rough_beats[left]) as f64;
                    let last_beat_length = (rough_beats[right] - rough_beats[right - 1]) as f64;
                    let expected_beat_length = 2.0 * region_average_beat_length;
                    (first_beat_length + last_beat_length - expected_beat_length).abs()
                } else {
                    0.0
                };
                if region_border_error < max_beat_drift_error / 2.0 {
                    // We've found a constant enough region, store for later.
                    let new_region = BpmRegion {
                        sample_rate: self.sample_rate,
                        start_frame: rough_beats[left],
                        end_frame: rough_beats[right],

                        beat_count: (right - left) as u32,
                        average_beat_length: region_average_beat_length,
                    };
                    bpm_regions.push(new_region);

                    left = right;
                    right = rough_beats.len() - 1;
                    continue;
                }
            }

            // Try again with a one-smaller beat region.
            right -= 1;
        }
        bpm_regions
    }

    /// Now that we have detected regions with consistent beat lengths (+/- MAX_BEAT_DRIFT_ERROR_SECS),
    /// we will attempt to enlarge the longest detected such region as far towards the start and end
    /// of the track as possible, assuming that said region is actually of constant tempo!
    ///
    /// To do this, we look for regions with consistent beat lengths that are of similar tempo
    /// and verify that their phase matches the original longest region, i.e. counting backwards
    /// from the original beat region would yield similar beat locations throughout the merge candidate region.
    fn detect_consistent_bpm(&self, bpm_regions: &[BpmRegion]) -> f64 {
        // We assume here the track was recorded with an unhear-able static metronome.
        // This metronome is likely at a full BPM.
        // The track may has intros, outros and bridges without detectable beats.
        // In these regions the detected beat might is floating around and is just wrong.
        // The track may also has regions with different rhythm giving instruments. They
        // have a different shape of onsets and introduce a static beat offset.
        // The track may also have break beats or other issues that makes the detector
        // hook onto a beat that is by an integer fraction off the original metronome.

        // This code aims to find the static metronome and a phase offset.
        let (lr_idx, lr) = bpm_regions
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| {
                (x.end_frame - x.start_frame).cmp(&(y.end_frame - y.start_frame))
            })
            .unwrap();

        // bpm_regions.iter().for_each(|x| println!("{:?}", x));
        // println!("Found longest region: {:?}", lr);
        // println!("  Beat length range: {:?}", lr.beat_length_range());

        // Find a region at the beginning of the track with a similar tempo and phase
        let mut back_extended_region_candidate = None;
        for candidate_region in bpm_regions.iter().take(lr_idx) {
            // Skip candidate regions that are too short, can be inconsistent.
            if candidate_region.beat_count < Self::MIN_REGION_BEAT_COUNT {
                continue;
            }

            // Skip candidate regions where our tempo doesn't match.
            if !candidate_region
                .beat_length_range()
                .contains(lr.average_beat_length)
            {
                continue;
            }

            // Now check if both regions are at the same phase.
            let merged_region_length = lr.end_frame - candidate_region.start_frame;
            let merged_beat_length_min = lr
                .beat_length_range()
                .min
                .max(candidate_region.beat_length_range().min);
            let merged_beat_length_max = lr
                .beat_length_range()
                .max
                .min(candidate_region.beat_length_range().max);

            // If the number of beats would change between the new candidate tempos,
            // these regions are unsuitable for merging.
            let min_num_beats =
                (merged_region_length as f64 / merged_beat_length_min).round() as u64;
            let max_num_beats =
                (merged_region_length as f64 / merged_beat_length_max).round() as u64;
            if min_num_beats != max_num_beats {
                continue;
            }

            let merged_beat_length = merged_region_length as f64 / min_num_beats as f64;
            // println!(
            //     "Evaluating candiate region ({} to {}), average beat length {}",
            //     candidate_region.start_frame, lr.end_frame, merged_beat_length
            // );
            if lr.beat_length_range().contains(merged_beat_length) {
                back_extended_region_candidate = Some(BpmRegion {
                    start_frame: candidate_region.start_frame,
                    end_frame: lr.end_frame,
                    beat_count: min_num_beats as u32,
                    average_beat_length: merged_beat_length,
                    ..*lr
                });
                break;
            }
        }

        let back_extended_region = back_extended_region_candidate.unwrap_or_else(|| lr.clone());
        // println!("Left expanded longest region: {:?}", back_extended_region);

        // Find a region at the end of the track with a similar tempo and phase
        let mut forwards_extended_region_candidate = None;
        for candidate_region in bpm_regions
            .iter()
            .rev()
            .take(bpm_regions.len() - lr_idx - 1)
        {
            if candidate_region.beat_count < Self::MIN_REGION_BEAT_COUNT {
                continue;
            }
            if !candidate_region
                .beat_length_range()
                .contains(back_extended_region.average_beat_length)
            {
                continue;
            }

            // Now check if both regions are at the same phase.
            let merged_region_length =
                candidate_region.end_frame - back_extended_region.start_frame;
            let merged_beat_length_min = back_extended_region
                .beat_length_range()
                .min
                .max(candidate_region.beat_length_range().min);
            let merged_beat_length_max = back_extended_region
                .beat_length_range()
                .max
                .min(candidate_region.beat_length_range().max);

            // If the number of beats would change between the new candidate tempos,
            // these regions are unsuitable for merging.
            let min_num_beats =
                (merged_region_length as f64 / merged_beat_length_min).round() as u64;
            let max_num_beats =
                (merged_region_length as f64 / merged_beat_length_max).round() as u64;
            if min_num_beats != max_num_beats {
                continue;
            }

            let merged_beat_length = merged_region_length as f64 / min_num_beats as f64;
            // println!(
            //     "Evaluating candiate region ({} to {}), average beat length {}",
            //     back_extended_region.start_frame, candidate_region.end_frame, merged_beat_length
            // );
            if back_extended_region
                .beat_length_range()
                .contains(merged_beat_length)
            {
                forwards_extended_region_candidate = Some(BpmRegion {
                    start_frame: back_extended_region.start_frame,
                    end_frame: candidate_region.end_frame,
                    beat_count: min_num_beats as u32,
                    average_beat_length: merged_beat_length,
                    ..*lr
                });
                break;
            }
        }

        let extended_region = forwards_extended_region_candidate.unwrap_or(back_extended_region);
        // println!("Expanded longest region: {:?}", extended_region);
        // println!(
        //     "  Beat length range: {:?}",
        //     extended_region.beat_length_range()
        // );
        // println!("  Rounded BPM value: {:.2}", extended_region.round_bpm());
        extended_region.round_bpm()
    }
}

#[wasm_bindgen]
pub fn detect_bpm(samples: &[f32], sample_rate: u32) -> f64 {
    let mut bpm_machine = CoarseBeatDetector::new(sample_rate);
    bpm_machine.process_samples(samples);
    let beats = bpm_machine.finalize();
    let iidx = Beatmania::new(sample_rate);
    iidx.calculate_bpm(&beats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stellar_stellar() {
        let rough_beats = [
            1392, 20330, 40382, 60434, 84385, 108336, 132287, 154567, 176290, 198013, 218065,
            238117, 258169, 278221, 292146, 306628, 325566, 344504, 364556, 384608, 405774, 424712,
            444764, 465930, 487096, 507148, 527757, 548366, 568975, 590141, 610193, 630802, 651411,
            671463, 690958, 710453, 733847, 758355, 782863, 807371, 831322, 854716, 878667, 902618,
            926569, 949406, 973357, 997308, 1021259, 1046324, 1070832, 1095897, 1120405, 1145470,
            1170535, 1195043, 1219551, 1244059, 1268567, 1292518, 1317583, 1342091, 1366599,
            1391664, 1416729, 1441237, 1465745, 1490253, 1514761, 1539826, 1564334, 1588842,
            1613907, 1638415, 1663480, 1687988, 1712496, 1737004, 1762069, 1786577, 1811085,
            1835593, 1860658, 1885166, 1910231, 1934739, 1959247, 1983755, 2008820, 2033328,
            2058393, 2082901, 2107409, 2131917, 2156982, 2181490, 2205998, 2231063, 2255571,
            2280636, 2305144, 2329095, 2353603, 2378111, 2403176, 2427684, 2452192, 2477257,
            2502322, 2526830, 2551895, 2576403, 2601468, 2625976, 2650484, 2675549, 2700057,
            2725122, 2749630, 2774138, 2798646, 2823711, 2848219, 2872727, 2897792, 2922300,
            2946808, 2971873, 2996381, 3020889, 3045954, 3070462, 3094970, 3120035, 3144543,
            3169051, 3194116, 3218624, 3243132, 3267640, 3292705, 3317213, 3342278, 3366786,
            3391294, 3415802, 3440867, 3465375, 3489883, 3514391, 3538899, 3563407, 3584573,
            3605739, 3626348, 3645843, 3665338, 3684833, 3704328, 3720481, 3736634, 3753344,
            3769497, 3785650, 3801803, 3818513, 3834666, 3851376, 3868086, 3884239, 3900392,
            3917102, 3933812, 3950522, 3967232, 3983385, 4000095, 4016805, 4032958, 4049668,
            4066378, 4083088, 4099241, 4115951, 4132661, 4148814, 4164967, 4181677, 4198387,
            4214540, 4231250, 4247403, 4263556, 4280266, 4296419, 4313129, 4329282, 4345992,
            4362145, 4378855, 4395008, 4411718, 4428428, 4444581, 4461291, 4477444, 4493597,
            4510307, 4526460, 4543170, 4559880, 4576033, 4592743, 4609453, 4625606, 4642316,
            4658469, 4674622, 4691332, 4707485, 4724195, 4740905, 4757058, 4773768, 4789921,
            4806631, 4822784, 4839494, 4855647, 4872357, 4888510, 4905220, 4921930, 4938083,
            4954236, 4970946, 4987656, 5003809, 5020519, 5037229, 5053382, 5069535, 5086245,
            5102955, 5119108, 5135261, 5151971, 5168681, 5184834, 5201544, 5217697, 5234407,
            5250560, 5266713, 5283423, 5300133, 5316286, 5332996, 5349706, 5366416, 5382569,
            5398722, 5415432, 5432142, 5448295, 5464448, 5481158, 5497868, 5514021, 5530731,
            5546884, 5563594, 5580304, 5596457, 5613167, 5629320, 5646030, 5662183, 5678893,
            5695046, 5711756, 5727909, 5744619, 5760772, 5777482, 5793635, 5810345, 5827055,
            5843765, 5859918, 5876628, 5892781, 5909491, 5925644, 5941797, 5958507, 5975217,
            5991927, 6008080, 6024233, 6040386, 6057096, 6073806, 6090516, 6106669, 6123379,
            6140089, 6156242, 6172395, 6189105, 6205258, 6221968, 6238678, 6255388, 6271541,
            6287694, 6304404, 6321114, 6337824, 6353977, 6370130, 6386283, 6402993, 6419146,
            6435856, 6452566, 6468719, 6484872, 6501582, 6517735, 6534445, 6550598, 6567308,
            6584018, 6600171, 6616881, 6633034, 6649744, 6665897, 6682607, 6699317, 6715470,
            6732180, 6748333, 6765043, 6781196, 6797906, 6814059, 6830769, 6846922, 6863632,
            6880342, 6897052, 6913205, 6929358, 6946068, 6962778, 6978931, 6995084, 7011794,
            7028504, 7044657, 7060810, 7077520, 7094230, 7110383, 7127093, 7143246, 7159956,
            7176109, 7192819, 7208972, 7225682, 7241835, 7258545, 7275255, 7291965, 7308118,
            7324828, 7341538, 7357691, 7373844, 7390554, 7406707, 7423417, 7439570, 7456280,
            7472990, 7489700, 7506410, 7522563, 7538716, 7555426, 7572136, 7588846, 7605556,
            7622266, 7638419,
        ];

        let iidx = Beatmania::new(48000);
        assert_ulps_eq!(175.0, iidx.calculate_bpm(&rough_beats), epsilon = 1e-6);
    }

    #[test]
    fn test_comet() {
        let rough_beats = [
            11418, 23672, 35926, 48180, 60434, 71574, 82714, 93854, 104994, 116134, 127274, 138414,
            153453, 178518, 198570, 221407, 245358, 269309, 288247, 308299, 326123, 347289, 366784,
            384608, 404103, 423598, 442536, 461474, 480412, 499907, 519402, 539454, 559506, 580115,
            601281, 621890, 645284, 669235, 693186, 716580, 739974, 763368, 785648, 807928, 830765,
            853602, 876439, 899276, 922670, 943279, 962774, 982269, 1001207, 1020702, 1039640,
            1058578, 1078073, 1097011, 1115949, 1135444, 1153825, 1173320, 1192258, 1211753,
            1231248, 1250743, 1269681, 1288062, 1307557, 1326495, 1345990, 1364928, 1383309,
            1402804, 1422299, 1441237, 1460175, 1479670, 1499165, 1518103, 1537041, 1556536,
            1576031, 1594412, 1613907, 1633402, 1653454, 1671835, 1690773, 1710268, 1729763,
            1748144, 1767639, 1786577, 1806072, 1825010, 1843948, 1863443, 1882938, 1901876,
            1920814, 1940309, 1959804, 1978742, 1997680, 2017175, 2036670, 2055608, 2074546,
            2094041, 2113536, 2132474, 2151412, 2170907, 2190402, 2209897, 2226050, 2242203,
            2258356, 2274509, 2290662, 2306815, 2322968, 2339121, 2353603, 2368085, 2382567,
            2397049, 2410974, 2425456, 2439938, 2454420, 2468902, 2482827, 2496752, 2511234,
            2525159, 2539084, 2553009, 2572504, 2592556, 2612608, 2632660, 2652712, 2671650,
            2690588, 2709526, 2728464, 2747402, 2765783, 2785278, 2804773, 2823711, 2842649,
            2862144, 2881639, 2900020, 2918958, 2938453, 2957948, 2976886, 2996381, 3015876,
            3034814, 3053752, 3073247, 3092185, 3111680, 3130618, 3149556, 3169051, 3188546,
            3207484, 3226422, 3245360, 3264855, 3284350, 3303288, 3322226, 3341721, 3360659,
            3380154, 3399092, 3418587, 3438082, 3457020, 3475958, 3495453, 3514391, 3533886,
            3552824, 3572319, 3591257, 3610752, 3629690, 3649185, 3668123, 3687618, 3706556,
            3726051, 3744989, 3763927, 3783422, 3802360, 3821855, 3840793, 3860288, 3879783,
            3898721, 3917659, 3936597, 3956092, 3975030, 3994525, 4013463, 4032401, 4051339,
            4070834, 4089772, 4109267, 4128762, 4148257, 4167752, 4186690, 4205628, 4225123,
            4244061, 4263556, 4282494, 4301432, 4320927, 4340422, 4359360, 4378855, 4397793,
            4417288, 4436226, 4455164, 4474659, 4493597, 4512535, 4532030, 4550968, 4570463,
            4589958, 4608896, 4628391, 4647329, 4666267, 4685762, 4704700, 4724195, 4743133,
            4762071, 4781566, 4801061, 4819999, 4839494, 4858432, 4877927, 4896865, 4916360,
            4935855, 4954793, 4973731, 4993226, 5012164, 5031659, 5050597, 5069535, 5089030,
            5108525, 5127463, 5146401, 5165339, 5184277, 5202101, 5215469, 5229394, 5243319,
            5257801, 5272283, 5284537, 5296791, 5309045, 5321299, 5333553, 5345807, 5358618,
            5370872, 5383126, 5395380, 5407634, 5419888, 5432142, 5444396, 5456650, 5468904,
            5481158, 5493412, 5506223, 5519034, 5531288, 5544099, 5556910, 5569721, 5582532,
            5594786, 5607040, 5619294, 5632662, 5646030, 5659955, 5673880, 5687805, 5701730,
            5715655, 5729580, 5743505, 5756873, 5775811, 5794749, 5813687, 5832625, 5851563,
            5869944, 5888882, 5907820, 5926758, 5945696, 5964077, 5983015, 6001396, 6020891,
            6039829, 6059324, 6078819, 6097757, 6116695, 6135633, 6154571, 6174066, 6193561,
            6212499, 6231437, 6250932, 6269870, 6288808, 6307746, 6327241, 6346736, 6365674,
            6385169, 6404107, 6423602, 6443097, 6462592, 6482087, 6501025, 6519963, 6538901,
            6557839, 6577334, 6596272, 6615210, 6634705, 6654200, 6673138, 6692076, 6711571,
            6730509, 6750004, 6768942, 6788437, 6807375, 6826870, 6845808, 6865303, 6884241,
            6903736, 6922674, 6941612, 6960550, 6979488, 6998426, 7017364, 7035745, 7060253,
            7084761, 7109269, 7133777, 7158285, 7183350, 7208972, 7234037, 7259659, 7289180,
            7318144, 7351564, 7385541, 7418961, 7452938, 7489143, 7523677, 7557654, 7591074,
            7624494, 7651787, 7679637, 7704145, 7729767, 7754832, 7780454, 7806076, 7832255,
            7857320, 7882942, 7908007, 7932515, 7957023, 7980974, 8004368, 8028319, 8052827,
            8071765, 8090703, 8109641, 8128579, 8146960, 8165898, 8184279, 8202660, 8221598,
            8240536, 8259474, 8276741, 8295679, 8315174, 8334112, 8353607, 8372545, 8391483,
            8409864, 8429359, 8448854, 8468349, 8487287, 8506782, 8525720, 8545215, 8564153,
            8583648, 8602586, 8622081, 8641019, 8659957, 8679452, 8698390, 8717885, 8736823,
            8756318, 8775813, 8794751, 8813689, 8833184, 8852122, 8871617, 8890555, 8909493,
            8928988, 8948483, 8967421, 8986359, 9005854, 9024792, 9044287, 9063225, 9082720,
            9101658, 9121153, 9140648, 9159586, 9178524, 9198019, 9216957, 9235895, 9255390,
            9274328, 9293823, 9312761, 9332256, 9351751, 9370689, 9389627, 9409122, 9428060,
            9446441, 9464822, 9483203, 9501584, 9519965, 9538346, 9556727, 9575108, 9593489,
            9611870, 9630808, 9649746, 9668684, 9687622, 9706560, 9725498, 9744993, 9763931,
            9783983, 9802921, 9821859, 9840797, 9860292, 9879787, 9898725, 9917663, 9937158,
            9956653, 9975591, 9994529, 10013467, 10032962, 10052457, 10071395, 10090890, 10109828,
            10128766, 10147704, 10167199, 10186137, 10205075, 10224570, 10244065, 10263560,
            10283055, 10302550, 10321488, 10340426, 10359364, 10378302, 10397797, 10421748,
            10445699, 10468536, 10495829, 10524236, 10552086, 10579936, 10608343, 10637864,
            10666828, 10695235, 10724199, 10752606, 10781570, 10810534, 10838384, 10866234,
            10894084, 10921377, 10948670, 10975406, 10999357, 11023865, 11048373, 11072881,
            11096832, 11121340, 11140835, 11160330, 11179268, 11198763, 11217701, 11236639,
            11256134, 11273401, 11290668, 11307935, 11325202, 11340241, 11355280, 11370319,
            11385358, 11400397, 11415436, 11431032, 11446071, 11461667, 11476706, 11491745,
            11506784, 11521823, 11536862, 11551901, 11566940, 11581979, 11597018, 11612057,
        ];

        let iidx = Beatmania::new(48000);
        assert_ulps_eq!(150.0, iidx.calculate_bpm(&rough_beats), epsilon = 1e-6);
    }

    #[test]
    fn test_next_color_planet() {
        let rough_beats = [
            4177, 27014, 48737, 70460, 94411, 117805, 140642, 164036, 189101, 212495, 236446,
            260954, 283791, 307185, 330579, 354530, 377924, 401318, 425269, 448663, 472614, 496565,
            520516, 543910, 566747, 590141, 614092, 638043, 661437, 684831, 708225, 731619, 755570,
            779521, 802915, 826309, 850260, 873654, 897605, 920999, 944393, 968344, 991738,
            1015132, 1038526, 1062477, 1085871, 1109265, 1132659, 1156610, 1180004, 1203955,
            1227349, 1251300, 1274694, 1298088, 1322039, 1345433, 1368827, 1392778, 1416172,
            1440123, 1463517, 1486911, 1510862, 1534256, 1558207, 1582158, 1605552, 1628946,
            1652340, 1675734, 1699685, 1723079, 1747030, 1770424, 1794375, 1817769, 1841163,
            1864557, 1888508, 1911902, 1935296, 1959247, 1982641, 2006035, 2029429, 2053380,
            2077331, 2100725, 2124119, 2148070, 2171464, 2194858, 2218809, 2242203, 2265597,
            2289548, 2313499, 2336893, 2360287, 2383681, 2407632, 2431026, 2454977, 2478371,
            2501765, 2525716, 2549110, 2573061, 2596455, 2619849, 2643800, 2667194, 2691145,
            2714539, 2737933, 2761884, 2785835, 2810343, 2834294, 2857688, 2881082, 2904476,
            2927313, 2950707, 2974101, 2998052, 3021446, 3045397, 3068791, 3092185, 3116136,
            3139530, 3163481, 3186875, 3210826, 3234220, 3257614, 3281008, 3304959, 3328353,
            3352304, 3375698, 3399092, 3423043, 3446437, 3470388, 3493782, 3517176, 3540570,
            3564521, 3587915, 3611309, 3634703, 3658654, 3682605, 3705999, 3729950, 3753344,
            3776738, 3800689, 3824083, 3847477, 3871428, 3894822, 3918773, 3942167, 3966118,
            3989512, 4012906, 4036300, 4060251, 4083645, 4107596, 4130990, 4154384, 4178335,
            4201729, 4225680, 4249074, 4272468, 4296419, 4319813, 4343764, 4367158, 4390552,
            4413946, 4437897, 4461848, 4485242, 4508636, 4532587, 4555981, 4579375, 4602769,
            4626163, 4650114, 4674065, 4697459, 4721410, 4744804, 4768198, 4791592, 4815543,
            4838937, 4862888, 4886282, 4910233, 4933627, 4957021, 4980972, 5004366, 5027760,
            5051711, 5075105, 5099056, 5123007, 5146401, 5169795, 5193189, 5217140, 5241091,
            5264485, 5287879, 5311273, 5335224, 5358618, 5382569, 5405963, 5429357, 5453308,
            5476702, 5500653, 5524047, 5547998, 5571392, 5594786, 5618180, 5641574, 5665525,
            5688919, 5712870, 5736264, 5759658, 5783052, 5807003, 5830397, 5854348, 5877742,
            5901693, 5925087, 5948481, 5972432, 5995826, 6019220, 6043171, 6066565, 6090516,
            6113910, 6137304, 6161255, 6184649, 6208043, 6231437, 6254831, 6278782, 6302733,
            6326684, 6351192, 6375143, 6398537, 6421931, 6445325, 6468719, 6492113, 6515507,
            6539458, 6563409, 6586803, 6610197, 6633591, 6656985, 6680379, 6704330, 6727724,
            6751675, 6775069, 6799020, 6822414, 6845808, 6869759, 6893153, 6916547, 6940498,
            6963892, 6987286, 7011237, 7034631, 7058582, 7081976, 7105370, 7129321, 7152715,
            7176109, 7200060, 7223454, 7247405, 7270799, 7294193, 7317587, 7341538, 7364932,
            7388883, 7411720, 7435671, 7459622, 7483573, 7506967, 7530361, 7553755, 7577706,
            7601100, 7625051, 7649002, 7671839, 7695790, 7719184, 7743135, 7766529, 7790480,
            7813874, 7837825, 7861219, 7885170, 7908564, 7931958, 7955909, 7979303, 8002697,
            8026091, 8049485, 8073436, 8096830, 8120781, 8144175, 8167569, 8191520, 8214914,
            8238308, 8262259, 8285653, 8309604, 8332998, 8356392, 8380343, 8403737, 8427131,
            8451082, 8474476, 8497870, 8521821, 8545772, 8569166, 8592560, 8615954, 8639905,
            8663299, 8686693, 8710087, 8734038, 8757989, 8781383, 8805334, 8828728, 8852122,
            8876073, 8899467, 8923418, 8946812, 8970763, 8994157, 9017551, 9040945, 9064896,
            9088290, 9112241, 9135635, 9159029, 9182423, 9206374, 9229768, 9253719, 9277113,
            9301064, 9324458, 9348409, 9371803, 9395197, 9418591, 9442542, 9465936, 9489887,
            9513281, 9537232, 9560626, 9584020, 9607971, 9631365, 9654759, 9678710, 9702104,
            9726055, 9749449, 9773400, 9796794, 9820188, 9844139, 9867533, 9890927, 9914878,
            9938272, 9961666, 9985617, 10009011, 10032405, 10056356, 10079750, 10103144, 10127095,
            10150489, 10174440, 10197834, 10221228, 10245179, 10268573, 10292524, 10315918,
            10339869, 10363820, 10387214, 10410608, 10434559, 10458510, 10481904, 10505298,
            10528135, 10552086, 10575480, 10599431, 10622825, 10646219, 10670170, 10693564,
            10717515, 10740909, 10764303, 10788254, 10811648, 10835599, 10858993, 10882387,
            10906338, 10929732, 10953126, 10977077, 11000471, 11023865, 11047259, 11071210,
            11095161, 11118555, 11142506, 11165900, 11189294, 11212688, 11236639, 11260033,
            11283984, 11307378, 11331329, 11354723, 11378117, 11401511, 11425462, 11448856,
            11472807, 11496201, 11520152, 11543546, 11566940, 11590891, 11614285, 11637679,
            11661630, 11685024, 11708975, 11732369, 11756320, 11779714, 11803108, 11827059,
            11850453, 11873847, 11897798, 11921192, 11944586, 11968537, 11991931, 12015882,
            12039276, 12062670, 12086621, 12110015, 12133966, 12157917, 12181311, 12204705,
            12228099, 12251493, 12275444, 12298838, 12322789, 12346183, 12369577, 12393528,
            12416922, 12440316, 12464267, 12487661, 12511612, 12535006, 12558400, 12582351,
            12605745, 12629139, 12653090, 12676484, 12700435, 12723829, 12747780, 12771174,
            12794568, 12818519, 12841913, 12865307, 12889258, 12913209, 12936603, 12959997,
            12983391, 13006785, 13030736, 13054130, 13078081, 13101475, 13125426, 13148820,
            13172214, 13196165, 13219559, 13242953, 13266347, 13289741, 13313692, 13337086,
        ];

        let iidx = Beatmania::new(48000);
        assert_ulps_eq!(122.0, iidx.calculate_bpm(&rough_beats), epsilon = 1e-6);
    }
}
