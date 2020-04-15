use ndarray::*;
use ndarray_linalg::*;

fn main() {
    let a = arr2(&[[1.0, 2.0, 3.0],
                   [4.0, 8.0, 6.0],
                   [7.0, 8.0, 9.0]]);
    let mut b = a.to_owned();
    for i in 0..3 {
        let m = &a.column(i).iter().fold(0.0, |m, &i| { if m < i { i } else { m } });
        b.column_mut(i).map_mut(|i| *i -= m);
    }
    b = b.mapv(f64::exp);
    for i in 0..3 {
        let s = b.column(i).iter().fold(0.0, |m, &i| m + i);
        b.column_mut(i).map_mut(|i| *i /= s);
    }
    let t = arr2(&[[0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [1.0, 0.0, 1.0]]);
    b = b - t;
    println!("{:?}", b);
}