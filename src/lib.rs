// This file is just a placeholder, which show how to expose Rust
// functions to Python, and also the deployment is set up around
// this.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;


#[pyfunction]
fn blank(a: usize) -> usize {
    a + 1
}

#[pyfunction]
fn dot(a: Vec<i32>, b: Vec<i32>) -> i32 {
    a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).sum()
}

#[pyfunction]
fn dot2(py: Python, a: Vec<i32>, b: Vec<i32>) -> i32 {
    py.allow_threads(|| dot(a, b))
}

#[pymodule]
fn numerics(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(blank, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(dot2, m)?)?;
    Ok(())
}
