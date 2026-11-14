//! Run artifact bundle writer/validator (std-only).
//!
//! This implements the on-disk "artifact spine" described in ADR-0016:
//! `runs/<run_id>/...` with a path-addressed SHA-256 `manifest.json` and
//! NDJSON baselines for spans/events/metrics/materializations.

mod bundle;
pub mod cas;
mod io;
pub mod lock;
pub mod schema_check;
mod session;
mod sink;

pub mod dp_ledger;

pub use bundle::{
    compute_manifest_entries_hash_v1, sign_manifest_v1, verify_manifest_signature_v1,
    ManifestEntryV1, ManifestSignatureV1, ManifestV1, RunArtifactBundle, RunFileV1, TrustPolicy,
    VerificationStatus, VerifiedBundleV1, SCHEMA_VERSION_V1,
};
pub use cas::{
    get_verified_cclip_center_v1, get_verified_checkpoint_model,
    get_verified_coordinator_resume_state_v2, get_verified_historical_anchors_v1,
    get_verified_optimizer_state_snapshot_v1, get_verified_safetensors_checkpoint_v1,
    parse_cas_uri_digest, put_cclip_center_v1, put_checkpoint_model,
    put_coordinator_resume_state_v2, put_historical_anchors_v1, put_optimizer_state_snapshot_v1,
    put_safetensors_checkpoint_v1, read_committed_head_v1, read_verified_round_resume_state,
    verify_cas_uri_exists, write_committed_head_v1, write_round_resume_state,
    CenteredClippingCenterBlobV1, CheckpointContentDigestV1, CheckpointRecipeKeyV1,
    CommittedHeadV1, CoordinatorResumeStateV2, HistoricalAnchorsBlobV1, RestoreModeV1,
    RoundResumeStateV1, CCLIP_CENTER_BLOB_SCHEMA_V1, CHECKPOINT_MANIFESTS_DIR_REL_PREFIX,
    CHECKPOINT_RECIPE_DOMAIN_TAG_V1, COMMITTED_HEAD_REL_PATH, COMMITTED_HEAD_SCHEMA_V1,
    COORDINATOR_RESUME_STATE_SCHEMA_V2, HISTORICAL_ANCHORS_BLOB_SCHEMA_V1,
    ROUND_RESUME_STATE_REL_PATH, ROUND_RESUME_STATE_SCHEMA_V1,
};
pub use dp_ledger::{
    append_dp_release_record_durable, derive_dp_receipt_id, pre_release_debit_dp_accountant,
    read_dp_ledger_entries, reconstruct_dp_accountant_from_ledger, CryptoErrorOrIo,
    DpLedgerEntryV1, DP_LEDGER_REL_PATH, DP_LEDGER_SCHEMA_V1,
};
pub use io::{
    DurabilityPolicyV1, MAX_MANIFEST_BYTES, MAX_NDJSON_LINES, MAX_NDJSON_LINE_BYTES,
    MAX_SINGLE_ARTIFACT_BYTES, MAX_TOTAL_BUNDLE_BYTES,
};
pub use lock::{BundleLockError, BundleLockGuard, LockAcquisitionPolicy, BUNDLE_LOCK_FILE_NAME};
pub use session::{DataOpsSession, OutputSpec, PredictError, SessionCommitReceiptV1};
pub use sink::{ArtifactWriteProfile, ManifestRefreshPolicy, RunArtifactSink, SnapshotProfile};

#[cfg(test)]
pub(crate) use io::read_json;
pub(crate) use io::{
    hex_lower, parse_json_slice, parse_ndjson_slice_bounded, read_bytes_bounded,
    write_once_cas_blob,
};

pub(crate) fn record_validation_error_to_io<E: core::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string())
}

#[cfg(test)]
mod tests;
