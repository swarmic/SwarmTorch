//! Standalone HTML report generator (artifact reader).
//!
//! This is intentionally offline-first:
//! - reads a run artifact bundle directory
//! - validates `manifest.json`
//! - generates a self-contained `report.html` without requiring a server/DB/UI framework

mod load;
mod model;
mod render;

pub use load::{load_report, load_report_with_warnings, LoadWarning};
pub use model::{is_node_unsafe, Report};
pub use render::{generate_report, generate_report_html};

#[cfg(test)]
mod tests;
