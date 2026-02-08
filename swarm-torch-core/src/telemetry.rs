//! Telemetry integration (optional).
//!
//! The canonical offline-first signal schema lives in [`crate::observe`]. This
//! module exists to provide optional integration points (e.g., `tracing`) while
//! keeping SwarmTorch's on-disk artifact contract OTel-compatible but not
//! OTel-dependent.

pub use crate::observe::{ParseIdError, RunId, SpanId, TraceId};

#[cfg(feature = "alloc")]
pub use crate::observe::{AttrMap, AttrValue, EventRecord, MetricRecord, SpanRecord};

// Future work (ADR-0016 / ADR-0012):
// - Provide a `RunEventEmitter` abstraction for `no_std` + `alloc` targets.
// - Provide a `tracing` layer/exporter that maps spans/events/metrics into the
//   SwarmTorch record types for artifact emission.
