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

#[allow(unused)]
#[derive(Clone, Copy, Debug)]
pub enum NormalizeType {
    None,
    UnitSum,
    UnitMax,
}

pub fn normalize(data: &mut [f64], normalize_type: NormalizeType) {
    match normalize_type {
        NormalizeType::None => {}

        NormalizeType::UnitSum => {
            let sum = data.iter().sum::<f64>();
            if sum != 0.0 {
                data.iter_mut().for_each(|x| *x /= sum);
            }
        }

        NormalizeType::UnitMax => {
            let max = data
                .iter()
                .map(|x| x.abs())
                .max_by(|a, b| a.total_cmp(b))
                .unwrap_or(0.0);
            if max != 0.0 {
                data.iter_mut().for_each(|x| *x /= max);
            }
        }
    };
}
