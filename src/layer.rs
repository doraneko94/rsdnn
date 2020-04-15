use ndarray::*;
#[allow(unused_imports)]
use ndarray_linalg::*;
use num_traits::float::Float;
use num_traits::{Zero, One};
use rand_distr::{Normal, Distribution};
use std::ops::{Add, Sub, Mul};

#[derive(PartialEq)]
pub enum LMode {
    Input,
    Linear,
    ReLU,
    Softmax,
    Output,
}

pub struct Layer<F: 'static + Float + Copy> {
    pub output: Array2<F>,
    pub grad: Array2<F>,
    pub w: Array2<F>,
    pub mode: LMode,
}

impl<F: 'static + Float + Copy> Layer<F> {
    pub fn new(mode: LMode, n_input: usize, n_output: usize, batch_size: usize) -> Self {
        let output = Array::<F, _>::zeros((n_output, batch_size));
        let grad = Array::<F, _>::zeros((n_input, batch_size));
        let nd = Normal::new(0.0, (2.0 / n_input as f64).sqrt()).unwrap();
        let mut w = Array::<F, _>::zeros((n_output, n_input));
        w.map_mut(|j| *j = F::from(nd.sample(&mut rand::thread_rng())).unwrap());

        Self { output, grad, w, mode }
    }

    pub fn forward(&mut self, input: &Array2<F>) {
        let zero = Zero::zero();
        match self.mode {
            LMode::Softmax => {
                self.output = input.to_owned();
                let batch_size = input.shape()[1];
                for i in 0..batch_size {
                    let m: F = input.column(i).iter().fold(zero, |m, &j| { if m < j { j } else { m } });
                    self.output.column_mut(i).map_mut(|j| *j = *j - m);
                }
                self.output = self.output.mapv(F::exp);
                for i in 0..batch_size {
                    let s = self.output.column(i).iter().fold(zero, |m, &j| m + j);
                    self.output.column_mut(i).map_mut(|j| *j = *j / s);
                }
            },
            LMode::Linear => {
                self.output = self.w.to_owned().dot(input);
            },
            LMode::ReLU => {
                self.output = input.to_owned();
                self.output.map_mut(|i| {
                    if *i < zero { *i = zero }
                });
            },
            LMode::Input => {
                self.output = input.to_owned();
            }
            _ => {},
        };
    }

    pub fn backward(&mut self, learning_rate: F, input: &Array2<F>, grad: &Array2<F>) {
        let zero: F = Zero::zero();
        let one: F = One::one();
        match self.mode {
            LMode::Softmax => {
                self.grad = self.output.to_owned().sub(grad);
            },
            LMode::Linear => {
                self.grad = self.w.to_owned().t().dot(grad);
                let mut dw = grad.to_owned().dot(&input.to_owned().t());
                dw.map_mut(|i| *i = *i * learning_rate);
                //println!("dw = {}", dw[[0, 0]].to_f64().unwrap());
                self.w = self.w.to_owned().sub(&dw);
            },
            LMode::ReLU => {
                let dx = input.map(|&i| {
                    if i > zero { one }
                    else { zero }
                });
                self.grad = dx.mul(grad);
            },
            LMode::Output => {
                self.grad = grad.to_owned();
            }
            _ => {},
        };
    }
}