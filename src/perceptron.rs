use ndarray::*;
#[allow(unused_imports)]
use ndarray_linalg::*;
use num_traits::float::Float;
use num_traits::{Zero, One};
use rand::distributions::{Uniform};
use rand::distributions::uniform::SampleUniform;
use ndarray_rand::RandomExt;

use crate::activation::step;

pub struct Perceptron<F: 'static + Float + SampleUniform + Copy> {
    pub w: Array1<F>,
    pub n_input: usize,
    pub learning_rate: F,
}

impl<F: 'static + Float + SampleUniform + Copy> Perceptron<F> {
    pub fn new(n_input: usize, learning_rate: F) -> Self {
        let zero: F = Zero::zero();
        let one: F = One::one();
        let ud = Uniform::new(zero, one);
        let w = Array::<F, _>::random(n_input, ud);

        Self { w, n_input, learning_rate }
    }

    pub fn forward(&self, x: &Array1<F>) -> F {
        let u = self.w.dot(x);
        step(u)
    }
    
    pub fn train_once(&mut self, x: &Array1<F>, t: F) {
        let z = self.forward(x);
        let dx = x.map(|&xi| xi * (t - z) * self.learning_rate);
        self.w = self.w.to_owned() + dx;
    }
    
    pub fn train_vec(&mut self, xv: &Vec<Array1<F>>, tv: &Vec<F>) {
        for (x, &t) in xv.iter().zip(tv.iter()) {
            self.train_once(x, t);
        }
    }

    pub fn train_epoch(&mut self, epoch: usize, xv: &Vec<Array1<F>>, tv: &Vec<F>) {
        for _ in 0..epoch {
            self.train_vec(xv, tv);
        }
    }
}