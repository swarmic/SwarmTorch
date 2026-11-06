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

pub use bundle::{
    compute_manifest_entries_hash_v1, sign_manifest_v1, verify_manifest_signature_v1,
    ManifestEntryV1, ManifestSignatureV1, ManifestV1, RunArtifactBundle, RunFileV1, TrustPolicy,
    VerificationStatus, VerifiedBundleV1, SCHEMA_VERSION_V1,
};
pub use cas::{
    get_verified_checkpoint_model, parse_cas_uri_digest, put_checkpoint_model,
    read_verified_round_resume_state, verify_cas_uri_exists, write_round_resume_state,
    CheckpointContentDigestV1, CheckpointRecipeKeyV1, RoundResumeStateV1,
    CHECKPOINT_RECIPE_DOMAIN_TAG_V1, ROUND_RESUME_STATE_REL_PATH, ROUND_RESUME_STATE_SCHEMA_V1,
};
pub use io::{
    MAX_MANIFEST_BYTES, MAX_NDJSON_LINES, MAX_NDJSON_LINE_BYTES, MAX_SINGLE_ARTIFACT_BYTES,
    MAX_TOTAL_BUNDLE_BYTES,
};
pub use lock::{BundleLockError, BundleLockGuard, LockAcquisitionPolicy, BUNDLE_LOCK_FILE_NAME};
pub use session::{DataOpsSession, OutputSpec, PredictError, SessionCommitReceiptV1};
pub use sink::{ArtifactWriteProfile, ManifestRefreshPolicy, RunArtifactSink, SnapshotProfile};

#[cfg(test)]
pub(crate) use io::read_json;
pub(crate) use io::{hex_lower, parse_json_slice, parse_ndjson_slice_bounded};

pub(crate) fn record_validation_error_to_io<E: core::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string())
}

#[cfg(test)]
mod tests;
