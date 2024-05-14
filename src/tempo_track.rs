/*
This code is based on the QM DSP Library, ported to WebAssembly-suitable Rust.

The original code can be found at https://github.com/c4dm/qm-dsp/blob/master/dsp/tempotracking/TempoTrackV2.cpp.
*/

/// A tempo tracker that will operate on beat detection function data calculated from
/// audio at the given sample rate with the given frame increment.
///
/// Currently the sample rate and increment are used only for the conversion from
/// beat frame location to bpm in the tempo array.
pub struct TempoTrackV2 {
    rate: f64,
    increment: i32,
}

/// !!! Question: how far is this actually sample rate dependent?  I
/// think it does produce plausible results for e.g. 48000 as well as
/// 44100, but surely the fixed window sizes and comb filtering will
/// make it prefer double or half time when run at e.g. 96000?
impl TempoTrackV2 {}
