//! Durable pre-release Rényi Differential Privacy (`RDP`) accounting ledger (`ADR-0023` §4.2, `W12-03`).
//!
//! Enforces strict fail-closed **pre-release debit**:
//! - Every round/step privacy debit is appended and `fsync`ed (`sync_all()`) to `dp/ledger.ndjson`
//!   **before** any aggregated checkpoint or update is published to CAS or emitted externally.
//! - A crash between durable ledger debit and checkpoint publication leaves the spent privacy budget
//!   debited and **non-refundable**.
//! - Idempotent retries keyed on `receipt_id` observe the persisted debit without double-charging.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use swarm_torch_core::crypto::{sha256_slices, CryptoError, PrivacyAccountantHook};
use swarm_torch_core::dp::{
    DpAdjacencyV1, DpConversionFormulaV1, DpMechanismProfileV1, DpUnitV1, RdpAccountantV1,
    RDP_GRID_LEN,
};
use swarm_torch_core::observe::RunId;

use super::io::{
    hex_lower, parse_ndjson_slice_bounded, read_bytes_bounded, sync_parent_dir, MAX_NDJSON_LINES,
    MAX_NDJSON_LINE_BYTES, MAX_SINGLE_ARTIFACT_BYTES,
};

/// Canonical relative path of the durable DP release ledger inside a run bundle.
pub const DP_LEDGER_REL_PATH: &str = "dp/ledger.ndjson";

/// Schema identifier embedded in durable DP ledger entries.
pub const DP_LEDGER_SCHEMA_V1: &str = "swarmtorch.dp_ledger.v1";

/// One durable pre-release DP debit record appended to `dp/ledger.ndjson` (`ADR-0023` §4.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DpLedgerEntryV1 {
    /// Schema marker (`"swarmtorch.dp_ledger.v1"`).
    pub schema_version: String,
    /// Deterministic idempotency receipt key preventing duplicate debits across retries.
    pub receipt_id: String,
    /// Round identifier (for synchronous rounds) or sequential commit round (for async steps).
    pub round_id: u64,
    /// Parent checkpoint SHA-256 hex digest active when the debit occurred.
    pub parent_digest_hex: String,
    /// Cumulative 16-order Rényi divergence array $\sum_t \rho_t(\alpha)$ AFTER this debit.
    pub cumulative_rdp_orders: Vec<f64>,
    /// Tightest converted $\varepsilon(\delta)$ after this debit.
    pub converted_epsilon: f64,
    /// Optimal Rényi order $\alpha^*$ minimizing $\varepsilon(\delta)$.
    pub optimal_alpha: f64,
    /// Configured failure probability $\delta$.
    pub target_delta: f64,
    /// Round subsampling rate $q \in (0, 1]$.
    pub subsampling_rate_q: f64,
    /// Applied $\ell_2$ clipping threshold $\tau$.
    pub clipping_radius_tau: f32,
    /// Gaussian noise multiplier $\sigma$.
    pub noise_multiplier: f32,
    /// Privacy unit (`ClientLevel` or `ExampleLevel`).
    pub unit: DpUnitV1,
    /// Adjacency relation (`AddRemoveOne` or `ReplaceOne`).
    pub adjacency: DpAdjacencyV1,
    /// RDP-to-DP conversion formula (`Mironov2017` or `Balle2020Tight`).
    pub conversion_formula: DpConversionFormulaV1,
    /// Monotonic count of debited rounds (`>= 1`).
    pub committed_rounds: u64,
    /// Monotonic Unix nanosecond timestamp when the pre-release debit was sealed.
    pub ts_unix_ns: u64,
}

/// Derive a deterministic 64-char hex idempotency `receipt_id` for a DP debit step.
pub fn derive_dp_receipt_id(
    run_id: RunId,
    round_id: u64,
    parent_digest_hex: &str,
    committed_rounds_before: u64,
) -> String {
    let digest = sha256_slices(&[
        b"swarmtorch.dp.receipt.v1",
        run_id.as_bytes(),
        &round_id.to_le_bytes(),
        parent_digest_hex.as_bytes(),
        &committed_rounds_before.to_le_bytes(),
    ]);
    hex_lower(&digest)
}

/// Read all persisted [`DpLedgerEntryV1`] records from `run_dir/dp/ledger.ndjson`.
///
/// Returns `Ok(Vec::new())` when no ledger has been created yet.
pub fn read_dp_ledger_entries(run_dir: &Path) -> io::Result<Vec<DpLedgerEntryV1>> {
    let path = run_dir.join(DP_LEDGER_REL_PATH);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = read_bytes_bounded(&path, MAX_SINGLE_ARTIFACT_BYTES)?;
    let entries: Vec<DpLedgerEntryV1> =
        parse_ndjson_slice_bounded(&bytes, MAX_NDJSON_LINE_BYTES, MAX_NDJSON_LINES)?;
    for entry in &entries {
        if entry.schema_version != DP_LEDGER_SCHEMA_V1
            || entry.cumulative_rdp_orders.len() != RDP_GRID_LEN
            || entry.receipt_id.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "malformed dp/ledger.ndjson entry",
            ));
        }
    }
    Ok(entries)
}

/// Durably append a pre-release DP debit record to `run_dir/dp/ledger.ndjson` (`sync_all()` +
/// parent dir sync).
///
/// - If `entry.receipt_id` already exists in the ledger with identical cumulative RDP orders and
///   round ID, returns `Ok(false)` idempotently without double-writing.
/// - If `entry.receipt_id` exists with conflicting payload, fails closed with `InvalidData`.
/// - Otherwise appends the NDJSON record, flushes, calls `sync_all()`, and returns `Ok(true)`.
pub fn append_dp_release_record_durable(
    run_dir: &Path,
    entry: &DpLedgerEntryV1,
) -> io::Result<bool> {
    if entry.schema_version != DP_LEDGER_SCHEMA_V1
        || entry.cumulative_rdp_orders.len() != RDP_GRID_LEN
        || entry.receipt_id.is_empty()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid DpLedgerEntryV1 payload",
        ));
    }
    if entry
        .cumulative_rdp_orders
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
        || !entry.converted_epsilon.is_finite()
        || entry.converted_epsilon < 0.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "non-finite cumulative RDP state in DpLedgerEntryV1",
        ));
    }

    let ledger_path = run_dir.join(DP_LEDGER_REL_PATH);
    if let Some(parent) = ledger_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let existing = read_dp_ledger_entries(run_dir)?;
    if let Some(prev) = existing.iter().find(|e| e.receipt_id == entry.receipt_id) {
        let orders_match = prev.cumulative_rdp_orders.len() == entry.cumulative_rdp_orders.len()
            && prev
                .cumulative_rdp_orders
                .iter()
                .zip(entry.cumulative_rdp_orders.iter())
                .all(|(a, b)| (a - b).abs() <= 1e-13 * a.abs().max(b.abs()).max(1.0));
        if prev.round_id != entry.round_id
            || prev.committed_rounds != entry.committed_rounds
            || !orders_match
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "conflicting duplicate receipt_id in dp/ledger.ndjson",
            ));
        }
        return Ok(false);
    }

    let line = serde_json::to_string(entry)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("serialize error: {e}")))?;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&ledger_path)?;
    file.write_all(line.as_bytes())?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.sync_all()?;
    sync_parent_dir(&ledger_path)?;
    Ok(true)
}

/// Preview, durably debit in `dp/ledger.ndjson`, and advance `accountant` BEFORE releasing a model
/// update or checkpoint (`ADR-0023` §4.2).
///
/// Crucially:
/// 1. `accountant.preview_clipped_round` checks $\varepsilon_{\text{next}} \le \varepsilon_{\max}$ without mutation.
/// 2. [`append_dp_release_record_durable`] syncs the debit to disk FIRST.
/// 3. Only after durable ledger commit succeeds is `accountant` mutated in memory, ensuring crash
///    or post-debit publication error NEVER refunds spent privacy budget.
#[allow(clippy::too_many_arguments)]
pub fn pre_release_debit_dp_accountant(
    run_dir: &Path,
    run_id: RunId,
    round_id: u64,
    parent_digest_hex: &str,
    accountant: &mut RdpAccountantV1,
    tau: f32,
    subsampling_rate_q: f64,
    ts_unix_ns: u64,
) -> Result<(bool, DpLedgerEntryV1), CryptoErrorOrIo> {
    let (desc, next_orders, next_eps) = accountant
        .preview_clipped_round(tau, subsampling_rate_q)
        .map_err(CryptoErrorOrIo::Crypto)?;

    let committed_before = accountant.committed_rounds();
    let receipt_id = derive_dp_receipt_id(run_id, round_id, parent_digest_hex, committed_before);
    let profile = *accountant.profile();
    let entry = DpLedgerEntryV1 {
        schema_version: DP_LEDGER_SCHEMA_V1.to_string(),
        receipt_id,
        round_id,
        parent_digest_hex: parent_digest_hex.to_string(),
        cumulative_rdp_orders: next_orders.to_vec(),
        converted_epsilon: next_eps,
        optimal_alpha: desc.alpha,
        target_delta: profile.target_delta,
        subsampling_rate_q,
        clipping_radius_tau: tau,
        noise_multiplier: profile.noise_multiplier,
        unit: profile.unit,
        adjacency: profile.adjacency,
        conversion_formula: profile.conversion_formula,
        committed_rounds: committed_before.saturating_add(1),
        ts_unix_ns,
    };

    let newly_appended =
        append_dp_release_record_durable(run_dir, &entry).map_err(CryptoErrorOrIo::Io)?;

    // Advance in-memory accountant only after durable disk sync succeeds.
    accountant
        .record_clipped_round(tau, subsampling_rate_q)
        .map_err(CryptoErrorOrIo::Crypto)?;

    Ok((newly_appended, entry))
}

/// Reconstruct an [`RdpAccountantV1`] from the latest entry in `run_dir/dp/ledger.ndjson` (or fresh if absent).
pub fn reconstruct_dp_accountant_from_ledger(
    run_dir: &Path,
    profile: DpMechanismProfileV1,
) -> Result<RdpAccountantV1, CryptoErrorOrIo> {
    let entries = read_dp_ledger_entries(run_dir).map_err(CryptoErrorOrIo::Io)?;
    match entries.last() {
        None => RdpAccountantV1::new(profile).map_err(CryptoErrorOrIo::Crypto),
        Some(last) => {
            if last.unit != profile.unit
                || last.adjacency != profile.adjacency
                || last.conversion_formula != profile.conversion_formula
                || (last.target_delta - profile.target_delta).abs() > 1e-15
                || (last.noise_multiplier - profile.noise_multiplier).abs() > 1e-6
            {
                return Err(CryptoErrorOrIo::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "dp/ledger.ndjson profile mismatch against configured DpMechanismProfileV1",
                )));
            }
            let mut orders = [0.0f64; RDP_GRID_LEN];
            orders.copy_from_slice(&last.cumulative_rdp_orders);
            RdpAccountantV1::restore_from_orders(profile, orders, last.committed_rounds)
                .map_err(CryptoErrorOrIo::Crypto)
        }
    }
}

/// Error returned by [`pre_release_debit_dp_accountant`] and [`reconstruct_dp_accountant_from_ledger`].
#[derive(Debug)]
pub enum CryptoErrorOrIo {
    /// Cryptographic / privacy budget error.
    Crypto(CryptoError),
    /// Durable storage I/O or ledger consistency error.
    Io(io::Error),
}
