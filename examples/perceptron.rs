use rsdnn::perceptron::*;
use ndarray::*;

fn main() {
    let mut per = Perceptron::<f64>::new(3, 0.1);
    let x = vec![array![1.0, 0.0, 0.0], array![1.0, 0.0, 1.0], array![1.0, 0.0, 1.0], array![1.0, 1.0, 1.0]];
    let t = vec![0.0, 0.0, 0.0, 1.0];
    per.train_epoch(10, &x, &t);
    println!("*** weights ***");
    for (i, &w) in per.w.iter().enumerate() {
        println!("w{} = {}", i, w);
    }
    println!("*** y_pred ***");
    let y_pred: Vec<f64> = x.iter().map(|xi| per.forward(xi)).collect();
    println!("{:?}", y_pred);
}