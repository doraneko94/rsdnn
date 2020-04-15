use num_traits::float::Float;
use num_traits::{Zero, One};

pub fn step<F: Float>(x: F) -> F {
    let zero: F = Zero::zero();
    let one: F = One::one();
    if x > zero { one }
    else { zero }
}