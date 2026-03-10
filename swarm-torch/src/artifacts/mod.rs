//! Run artifact bundle writer/validator (std-only).
//!
//! This implements the on-disk "artifact spine" described in ADR-0016:
//! `runs/<run_id>/...` with a path-addressed SHA-256 `manifest.json` and
//! NDJSON baselines for spans/events/metrics/materializations.

mod bundle;
mod io;
mod session;
mod sink;

pub use bundle::RunArtifactBundle;
pub use session::{DataOpsSession, OutputSpec, PredictError};
pub use sink::{ArtifactWriteProfile, ManifestRefreshPolicy, RunArtifactSink, SnapshotProfile};

#[cfg(test)]
pub(crate) use io::hex_lower;
#[cfg(test)]
pub(crate) use io::read_json;

pub(crate) fn record_validation_error_to_io<E: core::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string())
}

#[cfg(test)]
mod tests;
