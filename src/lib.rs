// This file basically sets up the panic hook for PySprint-CLI.
// Currently the other functions are just for testing purposes.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// A placeholder function to check if the Rust <-> Python connection works.
#[pyfunction]
fn blank(a: usize) -> usize {
    a + 1
}

/// This function sets up the CTRL+C exit hook, which may be used
/// with PySprint-CLI. Note that this terminates immediately,
/// so no destructors will be run at all.
#[pyfunction]
fn set_panic_hook() {
    ctrlc::set_handler(|| std::process::exit(130)).unwrap();
}

#[pymodule]
fn internals(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(blank, m)?)?;
    m.add_function(wrap_pyfunction!(set_panic_hook, m)?)?;
    Ok(())
}
