use rsdnn::layer::*;
use ndarray::*;

fn main() {
    let mut net = Vec::new();
    net.push(Layer::<f64>::new(LMode::Input, 3, 3, 4));
    net.push(Layer::<f64>::new(LMode::Linear, 3, 100, 4));
    net.push(Layer::<f64>::new(LMode::ReLU, 100, 100, 4));
    net.push(Layer::<f64>::new(LMode::Linear, 100, 2, 4));
    net.push(Layer::<f64>::new(LMode::Softmax, 2, 2, 4));
    net.push(Layer::<f64>::new(LMode::Output, 2, 2, 4));
    let x = arr2(&[[1.0, 0.0, 0.0],
                   [1.0, 0.0, 1.0],
                   [1.0, 1.0, 0.0],
                   [1.0, 1.0, 1.0]]).t().to_owned();
    let t = arr2(&[[1.0, 0.0],
                   [0.0, 1.0],
                   [0.0, 1.0],
                   [1.0, 0.0]]).t().to_owned();
    net[0].forward(&x);
    net[5].backward(0.1, &x, &t);
    for j in 0..100 {
        for i in 1..5 {
            let input = &net[i-1].output.to_owned();
            net[i].forward(&input);
        }
        for i in (1..5).rev() {
            let input = &net[i-1].output.to_owned();
            let grad = &net[i+1].grad.to_owned();
            net[i].backward(0.1, &input, &grad);
        }
        //println!("{}, w = {:?}", j, net[3].w[[0, 0]]);
        //println!("{:?}", net[2].output);
        //println!("{:?}", net[4].output);
        //println!("{:?}", net[4].grad);
    }
    println!("{:?}", net[4].output);
    //println!("{:?}", net[4].grad);
    //println!("{:?}", net[3].w);
    //println!("{:?}", net[3].grad)
}