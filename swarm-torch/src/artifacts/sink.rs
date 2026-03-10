use std::io;
use std::sync::Mutex;

use swarm_torch_core::dataops::{
    DatasetEntryV1, DatasetLineageV1, DatasetRegistryV1, LineageEdgeV1, MaterializationRecordV1,
    MaterializationRecordV2,
};
use swarm_torch_core::observe::{
    validate_event_record, validate_metric_record, validate_span_record, EventRecord, MetricRecord,
    SpanRecord,
};
use swarm_torch_core::run_graph::GraphV1;

use super::record_validation_error_to_io;
use super::RunArtifactBundle;

/// Snapshot persistence policy for DataOps state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotProfile {
    /// Rewrite snapshot JSON files on every DataOps mutation.
    Strict,
    /// Append deltas and compact snapshots every N DataOps mutations.
    Streaming { snapshot_every_n_writes: u64 },
}

impl SnapshotProfile {
    pub fn strict() -> Self {
        Self::Strict
    }

    pub fn streaming(snapshot_every_n_writes: u64) -> Self {
        Self::Streaming {
            snapshot_every_n_writes,
        }
    }
}

/// Manifest refresh policy for artifact writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestRefreshPolicy {
    /// Refresh only when explicitly finalized.
    FinalOnly,
    /// Refresh every N writes.
    IntervalN(u64),
    /// Refresh after every write.
    Always,
}

/// Combined write profile used by artifact sinks/sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactWriteProfile {
    pub snapshot_profile: SnapshotProfile,
    pub manifest_policy: ManifestRefreshPolicy,
}

impl ArtifactWriteProfile {
    pub fn strict_final_only() -> Self {
        Self {
            snapshot_profile: SnapshotProfile::Strict,
            manifest_policy: ManifestRefreshPolicy::FinalOnly,
        }
    }
}

impl Default for ArtifactWriteProfile {
    fn default() -> Self {
        Self::strict_final_only()
    }
}

#[derive(Debug, Default)]
struct SinkState {
    write_count: u64,
}

/// Thread-safe artifact sink (single-writer enforced by an in-process mutex).
///
/// This is the simplest v0.1 strategy for multi-producer telemetry without risking
/// interleaved NDJSON lines.
#[derive(Debug)]
pub struct RunArtifactSink {
    bundle: RunArtifactBundle,
    profile: ArtifactWriteProfile,
    lock: Mutex<SinkState>,
}

impl swarm_torch_core::observe::RunEventEmitter for RunArtifactSink {
    type Error = io::Error;

    fn emit_span(&self, span: &SpanRecord) -> Result<(), Self::Error> {
        validate_span_record(span).map_err(record_validation_error_to_io)?;
        self.append_span(span)
    }

    fn emit_event(&self, event: &EventRecord) -> Result<(), Self::Error> {
        validate_event_record(event).map_err(record_validation_error_to_io)?;
        self.append_event(event)
    }

    fn emit_metric(&self, metric: &MetricRecord) -> Result<(), Self::Error> {
        validate_metric_record(metric).map_err(record_validation_error_to_io)?;
        self.append_metric(metric)
    }
}

impl RunArtifactSink {
    pub fn new(bundle: RunArtifactBundle) -> Self {
        Self::with_profile(bundle, ArtifactWriteProfile::default())
    }

    pub fn with_profile(bundle: RunArtifactBundle, profile: ArtifactWriteProfile) -> Self {
        Self {
            bundle,
            profile,
            lock: Mutex::new(SinkState::default()),
        }
    }

    pub fn bundle(&self) -> &RunArtifactBundle {
        &self.bundle
    }

    pub fn profile(&self) -> ArtifactWriteProfile {
        self.profile
    }

    fn guard(&self) -> io::Result<std::sync::MutexGuard<'_, SinkState>> {
        self.lock
            .lock()
            .map_err(|_| io::Error::other("artifact sink mutex poisoned"))
    }

    fn post_write_maybe_refresh_manifest(
        &self,
        state: &mut std::sync::MutexGuard<'_, SinkState>,
    ) -> io::Result<()> {
        state.write_count = state.write_count.saturating_add(1);

        match self.profile.manifest_policy {
            ManifestRefreshPolicy::FinalOnly => Ok(()),
            ManifestRefreshPolicy::Always => self.bundle.finalize_manifest(),
            ManifestRefreshPolicy::IntervalN(n) => {
                let period = n.max(1);
                if state.write_count % period == 0 {
                    self.bundle.finalize_manifest()?;
                }
                Ok(())
            }
        }
    }

    pub fn write_graph(&self, graph: &GraphV1) -> io::Result<()> {
        let mut state = self.guard()?;
        self.bundle.write_graph(graph)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_span(&self, span: &SpanRecord) -> io::Result<()> {
        let mut state = self.guard()?;
        self.bundle.append_span(span)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_event(&self, event: &EventRecord) -> io::Result<()> {
        let mut state = self.guard()?;
        self.bundle.append_event(event)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_metric(&self, metric: &MetricRecord) -> io::Result<()> {
        let mut state = self.guard()?;
        self.bundle.append_metric(metric)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_materialization(&self, m: &MaterializationRecordV1) -> io::Result<()> {
        let mut state = self.guard()?;
        self.bundle.append_materialization(m)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_materialization_v2(&self, m: &MaterializationRecordV2) -> io::Result<()> {
        let mut state = self.guard()?;
        self.bundle.append_materialization_v2(m)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_registry_update(&self, dataset: &DatasetEntryV1) -> io::Result<()> {
        let mut state = self.guard()?;
        self.bundle.append_registry_update(dataset)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_lineage_edge_update(&self, edge: &LineageEdgeV1) -> io::Result<()> {
        let mut state = self.guard()?;
        self.bundle.append_lineage_edge_update(edge)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn write_dataset_registry(&self, r: &DatasetRegistryV1) -> io::Result<()> {
        let mut state = self.guard()?;
        self.bundle.write_dataset_registry(r)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn write_dataset_lineage(&self, l: &DatasetLineageV1) -> io::Result<()> {
        let mut state = self.guard()?;
        self.bundle.write_dataset_lineage(l)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn finalize_manifest(&self) -> io::Result<()> {
        let _g = self.guard()?;
        self.bundle.finalize_manifest()
    }

    pub fn validate_manifest(&self) -> io::Result<()> {
        let _g = self.guard()?;
        self.bundle.validate_manifest()
    }
}
