use std::io;
use std::sync::Mutex;

use swarm_torch_core::crypto::KeyPair;
use swarm_torch_core::dataops::{
    DatasetEntryV1, DatasetLineageV1, DatasetRegistryV1, LineageEdgeV1, MaterializationRecordV1,
    MaterializationRecordV2,
};
use swarm_torch_core::observe::{
    validate_event_record, validate_metric_record, validate_span_record, EventRecord, MetricRecord,
    SpanRecord,
};
use swarm_torch_core::run_graph::GraphV1;

use super::bundle::ManifestV1;
use super::cas::{
    read_committed_head_v1, write_archived_generation_manifest_bytes, write_committed_head_v1,
    COMMITTED_HEAD_REL_PATH,
};
use super::lock::{BundleLockGuard, LockAcquisitionPolicy};
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
    sealed: bool,
    generation: u64,
    lock_guard: Option<BundleLockGuard>,
}

/// Thread-safe artifact sink (single-writer enforced by an in-process mutex and optional
/// cross-process [`BundleLockGuard`], with seal/generation fencing per `ADR-0023` §2.1).
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

    pub(crate) fn with_profile_and_lock(
        bundle: RunArtifactBundle,
        profile: ArtifactWriteProfile,
        lock_guard: BundleLockGuard,
    ) -> Self {
        Self {
            bundle,
            profile,
            lock: Mutex::new(SinkState {
                write_count: 0,
                sealed: false,
                generation: 1,
                lock_guard: Some(lock_guard),
            }),
        }
    }

    pub fn bundle(&self) -> &RunArtifactBundle {
        &self.bundle
    }

    pub fn profile(&self) -> ArtifactWriteProfile {
        self.profile
    }

    /// Return whether this sink has been sealed (`ADR-0023` §2.1).
    pub fn is_sealed(&self) -> io::Result<bool> {
        let state = self.guard_any()?;
        Ok(state.sealed)
    }

    /// Return the current monotonic fencing generation of this sink (`ADR-0023` §2.1).
    pub fn generation(&self) -> io::Result<u64> {
        let state = self.guard_any()?;
        Ok(state.generation)
    }

    /// Seal the sink without re-writing the manifest, fencing off any further mutations
    /// from cloned `Arc<RunArtifactSink>` handles and releasing any held `.bundle.lock`.
    pub fn seal(&self) -> io::Result<u64> {
        let mut state = self.guard_unsealed()?;
        state.sealed = true;
        state.generation = state.generation.saturating_add(1);
        let _ = state.lock_guard.take();
        Ok(state.generation)
    }

    /// Finalize (and optionally sign) `manifest.json` during an active round commit and archive the
    /// exact serialized manifest bytes for generation `generation` (`ST-04`).
    ///
    /// Returns `(manifest, manifest_sha256_hex)`.
    pub(crate) fn finalize_checkpoint_manifest(
        &self,
        generation: u64,
        signer: Option<&KeyPair>,
    ) -> io::Result<(ManifestV1, String)> {
        let state = self.guard_unsealed()?;
        let _ephemeral_lock = if state.lock_guard.is_none() {
            Some(self.bundle.acquire_lock(LockAcquisitionPolicy::default())?)
        } else {
            None
        };
        let (manifest, manifest_bytes, manifest_sha256) =
            self.bundle.finalize_manifest_with_sha256_unlocked(signer)?;
        write_archived_generation_manifest_bytes(
            self.bundle.run_dir(),
            generation,
            &manifest_bytes,
        )?;
        Ok((manifest, manifest_sha256))
    }

    /// Finalize (and optionally sign) `manifest.json`, seal the sink against any late writes
    /// from cloned `Arc<RunArtifactSink>` handles, and release any held `.bundle.lock`
    /// (`ADR-0023` §2.1, `W10-01`).
    pub fn seal_and_finalize(&self, signer: Option<&KeyPair>) -> io::Result<(u64, ManifestV1)> {
        let mut state = self.guard_unsealed()?;
        let _ephemeral_lock = if state.lock_guard.is_none() {
            Some(self.bundle.acquire_lock(LockAcquisitionPolicy::default())?)
        } else {
            None
        };
        let (manifest, manifest_bytes, manifest_sha256) =
            self.bundle.finalize_manifest_with_sha256_unlocked(signer)?;
        // If `checkpoints/committed_head.json` exists, re-pin its `manifest_sha256` and archived
        // generation manifest to the newly finalized (possibly signed) bundle manifest so subsequent
        // strict checkpoint resume stays consistent with `manifest.json` (`ST-04`).
        let head_path = self.bundle.run_dir().join(COMMITTED_HEAD_REL_PATH);
        if head_path.exists() {
            let mut head = read_committed_head_v1(self.bundle.run_dir())?;
            write_archived_generation_manifest_bytes(
                self.bundle.run_dir(),
                head.generation,
                &manifest_bytes,
            )?;
            head.manifest_sha256 = manifest_sha256;
            write_committed_head_v1(self.bundle.run_dir(), &head)?;
        }
        state.sealed = true;
        state.generation = state.generation.saturating_add(1);
        let gen = state.generation;
        let _ = state.lock_guard.take();
        Ok((gen, manifest))
    }

    fn guard_any(&self) -> io::Result<std::sync::MutexGuard<'_, SinkState>> {
        self.lock
            .lock()
            .map_err(|_| io::Error::other("artifact sink mutex poisoned"))
    }

    fn guard_unsealed(&self) -> io::Result<std::sync::MutexGuard<'_, SinkState>> {
        let guard = self.guard_any()?;
        if guard.sealed {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "artifact sink for run {} is sealed (generation {})",
                    self.bundle.run_id(),
                    guard.generation
                ),
            ));
        }
        Ok(guard)
    }

    fn refresh_manifest_with_state(
        &self,
        state: &std::sync::MutexGuard<'_, SinkState>,
    ) -> io::Result<()> {
        if state.lock_guard.is_some() {
            self.bundle.finalize_manifest_unlocked(None).map(|_| ())
        } else {
            self.bundle.finalize_manifest()
        }
    }

    fn post_write_maybe_refresh_manifest(
        &self,
        state: &mut std::sync::MutexGuard<'_, SinkState>,
    ) -> io::Result<()> {
        state.write_count = state.write_count.saturating_add(1);

        match self.profile.manifest_policy {
            ManifestRefreshPolicy::FinalOnly => Ok(()),
            ManifestRefreshPolicy::Always => self.refresh_manifest_with_state(state),
            ManifestRefreshPolicy::IntervalN(n) => {
                let period = n.max(1);
                if state.write_count % period == 0 {
                    self.refresh_manifest_with_state(state)?;
                }
                Ok(())
            }
        }
    }

    pub fn write_graph(&self, graph: &GraphV1) -> io::Result<()> {
        let mut state = self.guard_unsealed()?;
        if state.lock_guard.is_some() {
            self.bundle.write_graph_unlocked(graph)?;
        } else {
            self.bundle.write_graph(graph)?;
        }
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_span(&self, span: &SpanRecord) -> io::Result<()> {
        let mut state = self.guard_unsealed()?;
        self.bundle.append_span(span)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_event(&self, event: &EventRecord) -> io::Result<()> {
        let mut state = self.guard_unsealed()?;
        self.bundle.append_event(event)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_metric(&self, metric: &MetricRecord) -> io::Result<()> {
        let mut state = self.guard_unsealed()?;
        self.bundle.append_metric(metric)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_materialization(&self, m: &MaterializationRecordV1) -> io::Result<()> {
        let mut state = self.guard_unsealed()?;
        self.bundle.append_materialization(m)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_materialization_v2(&self, m: &MaterializationRecordV2) -> io::Result<()> {
        let mut state = self.guard_unsealed()?;
        self.bundle.append_materialization_v2(m)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_registry_update(&self, dataset: &DatasetEntryV1) -> io::Result<()> {
        let mut state = self.guard_unsealed()?;
        self.bundle.append_registry_update(dataset)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn append_lineage_edge_update(&self, edge: &LineageEdgeV1) -> io::Result<()> {
        let mut state = self.guard_unsealed()?;
        self.bundle.append_lineage_edge_update(edge)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn write_dataset_registry(&self, r: &DatasetRegistryV1) -> io::Result<()> {
        let mut state = self.guard_unsealed()?;
        self.bundle.write_dataset_registry(r)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn write_dataset_lineage(&self, l: &DatasetLineageV1) -> io::Result<()> {
        let mut state = self.guard_unsealed()?;
        self.bundle.write_dataset_lineage(l)?;
        self.post_write_maybe_refresh_manifest(&mut state)
    }

    pub fn finalize_manifest(&self) -> io::Result<()> {
        let state = self.guard_unsealed()?;
        self.refresh_manifest_with_state(&state)
    }

    pub fn validate_manifest(&self) -> io::Result<()> {
        let _g = self.guard_any()?;
        self.bundle.validate_manifest()
    }
}
