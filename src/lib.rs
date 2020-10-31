use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn blank(a: usize) -> usize {
    a + 1
}


#[pymodule]
fn pysprint(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(blank, m)?)?;
    Ok(())
}
