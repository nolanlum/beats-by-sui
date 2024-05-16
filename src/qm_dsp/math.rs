/*
This code is based on the QM DSP Library, badly ported to Rust.

The original code can be found at:
    - https://github.com/c4dm/qm-dsp/tree/master/maths
*/

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
