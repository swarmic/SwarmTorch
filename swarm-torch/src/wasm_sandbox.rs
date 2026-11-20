//! Fail-closed experimental WebAssembly (`MVP`) test-fixture VM (`ADR-0018`, `ADR-0025`, `W12-04`, `ST-09`).
//!
//! > **Security Notice (`ST-09` Fail-Closed Containment):** `WasmSandboxRunnerV1::execute_byte_transform`
//! > and `WasmSandboxRunnerV1::execute_scalar_i32` fail closed unconditionally with
//! > `WasmSandboxError::UnsupportedUntrustedWasmSandbox`. Production `Scheduler`
//! > (`RunnerIsolationProfile::WasmSandboxedV1`) and `NativeOpRunner` also fail closed with
//! > `WasmSandboxError::UnsupportedUntrustedWasmSandbox` unless an external [`crate::wasm_sandbox::WasmExtensionExecutor`]
//! > (such as `adapters/swarm-torch-wasm`, supporting `WasmGuestAbiV1::RawMem0V0` and `WasmGuestAbiV1::SwarmTorchAllocV1`)
//! > is explicitly attached via `NativeOpRunner::with_wasm_executor`. The internal subset interpreter below is kept exclusively
//! > behind `#[doc(hidden)]` `WasmSandboxRunnerV1::run_experimental_trusted_test_fixture_only` for unit tests.

use std::collections::BTreeSet;
use std::fmt;

use sha2::{Digest, Sha256};

/// Canonical 8-byte WebAssembly binary magic + version header (`\0asm\x01\0\0\0`).
pub const WASM_MAGIC_AND_VERSION_V1: [u8; 8] = *b"\0asm\x01\0\0\0";

/// WebAssembly fixed linear page size (`65,536` bytes = 64 KiB).
pub const WASM_PAGE_BYTES: usize = 65_536;

/// Default maximum call stack depth enforced against unbounded guest recursion.
pub const DEFAULT_WASM_MAX_CALL_DEPTH: usize = 64;

/// Default maximum instruction fuel budget allowed during synchronous Wasm `start` execution.
pub const DEFAULT_WASM_MAX_START_FUEL: u64 = 10_000;

/// Hard cap on total declared locals per function across all local declaration groups (`ST-09`).
pub const DEFAULT_WASM_MAX_LOCALS_PER_FUNCTION: usize = 4_096;

/// Canonical guest allocation export required under [`WasmGuestAbiV1::SwarmTorchAllocV1`] (`ADR-0025`).
pub const SWARMTORCH_WASM_ALLOC_EXPORT_V1: &str = "swarmtorch_alloc";

/// Canonical guest deallocation export required under [`WasmGuestAbiV1::SwarmTorchAllocV1`] (`ADR-0025`).
pub const SWARMTORCH_WASM_DEALLOC_EXPORT_V1: &str = "swarmtorch_dealloc";

/// Default maximum output byte length allowed for a guest byte-transform return slice (`16 MiB`, `ADR-0025`).
pub const DEFAULT_WASM_MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;

/// Guest memory ABI contract governing how host and Wasm guest exchange byte-transform buffers (`ADR-0025`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WasmGuestAbiV1 {
    /// Legacy `raw_mem0_v0` ABI (`export(ptr: i32, len: i32) -> i32` writing input at linear memory offset `0`).
    #[default]
    RawMem0V0,
    /// Structured allocator ABI (`swarmtorch_alloc(len: i32) -> i32`,
    /// `export(in_ptr: i32, in_len: i32) -> i64` packing `(out_ptr << 32) | (out_len & 0xFFFF_FFFF)`,
    /// `swarmtorch_dealloc(ptr: i32, len: i32)`).
    SwarmTorchAllocV1,
}

/// Errors raised during pre-parse verification, static validation, or sandboxed Wasm execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmSandboxError {
    /// Fail-closed error returned when untrusted Wasm execution has no attached [`WasmExtensionExecutor`] (`ST-09`).
    UnsupportedUntrustedWasmSandbox,
    /// Module digest was not present in [`SandboxedExtensionPolicyV1::allowed_module_hashes`].
    UntrustedModuleHash { actual: [u8; 32] },
    /// Module byte length exceeded [`WasmStoreLimitsV1::max_module_bytes`].
    ModuleTooLarge { len: usize, max: usize },
    /// Module lacked valid `\0asm\x01\0\0\0` header.
    InvalidWasmHeader,
    /// Malformed LEB128 integer, truncated section, or invalid section ordering.
    MalformedSectionOrLeb128,
    /// Host/WASI or unauthorized import rejected before instantiation.
    ForbiddenImport { module: String, field: String },
    /// Multi-memory modules are rejected under single-linear-memory isolation.
    MultipleMemoriesForbidden,
    /// Multi-table modules are rejected under single-table isolation.
    MultipleTablesForbidden,
    /// Declared initial/maximum memory pages exceeded policy limit.
    MemoryLimitExceeded {
        requested_pages: u32,
        max_pages: u32,
    },
    /// Declared table element count exceeded policy limit.
    TableLimitExceeded { requested: u32, max: u32 },
    /// Exported symbol was missing or rejected by `allowed_exports`.
    ExportNotAllowed(String),
    /// Requested export signature did not match caller ABI expectation.
    ExportSignatureMismatch(String),
    /// Deterministic fuel budget exhausted during `start` or export execution.
    OutOfFuel { consumed: u64, limit: u64 },
    /// `memory.grow` exceeded [`WasmStoreLimitsV1::max_memory_pages`] and `trap_on_grow_failure` trapped.
    MemoryGrowTrap {
        current_pages: u32,
        delta_pages: u32,
        max_pages: u32,
    },
    /// Checked guest linear memory read/write or `(ptr, len)` slice exceeded bounds.
    OutOfBoundsMemoryAccess {
        addr: u64,
        len: usize,
        mem_len: usize,
    },
    /// Wall-clock execution time exceeded [`WasmStoreLimitsV1::timeout_ms`].
    ExecutionTimeout { elapsed_ms: u64, limit_ms: u64 },
    /// Division by zero, integer overflow trap, `unreachable`, or call stack overflow.
    ExecutionTrap(String),
    /// Guest transform export returned a negative status/error code (`ADR-0025`).
    GuestTransformFailed { export: String, code: i32 },
    /// Guest output length exceeded [`WasmStoreLimitsV1::max_output_bytes`] (`ADR-0025`).
    OutputTooLarge { len: usize, max: usize },
    /// Guest returned an invalid or partially overlapping `(out_ptr, out_len)` buffer alias (`ADR-0025`).
    OutputAliasingViolation {
        in_ptr: i32,
        in_len: i32,
        out_ptr: i32,
        out_len: i32,
    },
}

impl fmt::Display for WasmSandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedUntrustedWasmSandbox => write!(
                f,
                "UnsupportedUntrustedWasmSandbox: untrusted WebAssembly execution requires an attached external WasmExtensionExecutor"
            ),
            Self::UntrustedModuleHash { actual } => {
                write!(
                    f,
                    "wasm module SHA-256 {:02x?} is not in allowlist",
                    &actual[..4]
                )
            }
            Self::ModuleTooLarge { len, max } => {
                write!(f, "wasm module size {len} exceeds policy cap {max}")
            }
            Self::InvalidWasmHeader => write!(f, "invalid wasm magic/version header"),
            Self::MalformedSectionOrLeb128 => write!(f, "malformed wasm binary section or LEB128"),
            Self::ForbiddenImport { module, field } => {
                write!(f, "forbidden wasm host/WASI import: {module}::{field}")
            }
            Self::MultipleMemoriesForbidden => write!(f, "multiple wasm linear memories forbidden"),
            Self::MultipleTablesForbidden => write!(f, "multiple wasm tables forbidden"),
            Self::MemoryLimitExceeded {
                requested_pages,
                max_pages,
            } => write!(
                f,
                "wasm initial/max memory pages {requested_pages} exceeds cap {max_pages}"
            ),
            Self::TableLimitExceeded { requested, max } => {
                write!(f, "wasm table elements {requested} exceeds cap {max}")
            }
            Self::ExportNotAllowed(name) => write!(f, "wasm export '{name}' missing or forbidden"),
            Self::ExportSignatureMismatch(name) => {
                write!(
                    f,
                    "wasm export '{name}' signature incompatible with host ABI"
                )
            }
            Self::OutOfFuel { consumed, limit } => {
                write!(f, "wasm execution ran out of fuel ({consumed}/{limit})")
            }
            Self::MemoryGrowTrap {
                current_pages,
                delta_pages,
                max_pages,
            } => write!(
                f,
                "wasm memory.grow trapped ({current_pages} + {delta_pages} > {max_pages} pages)"
            ),
            Self::OutOfBoundsMemoryAccess { addr, len, mem_len } => write!(
                f,
                "wasm out-of-bounds memory access: [{addr}..+{len}) in {mem_len} bytes"
            ),
            Self::ExecutionTimeout {
                elapsed_ms,
                limit_ms,
            } => write!(
                f,
                "wasm execution timed out after {elapsed_ms} ms (limit {limit_ms} ms)"
            ),
            Self::ExecutionTrap(reason) => write!(f, "wasm execution trap: {reason}"),
            Self::GuestTransformFailed { export, code } => {
                write!(f, "wasm guest transform '{export}' failed with code {code}")
            }
            Self::OutputTooLarge { len, max } => {
                write!(
                    f,
                    "wasm guest output length {len} exceeds max_output_bytes {max}"
                )
            }
            Self::OutputAliasingViolation {
                in_ptr,
                in_len,
                out_ptr,
                out_len,
            } => write!(
                f,
                "wasm guest output [{out_ptr}..+{out_len}) invalidly aliases input [{in_ptr}..+{in_len})"
            ),
        }
    }
}

impl std::error::Error for WasmSandboxError {}

fn default_max_start_fuel() -> u64 {
    DEFAULT_WASM_MAX_START_FUEL
}

fn default_max_output_bytes() -> usize {
    DEFAULT_WASM_MAX_OUTPUT_BYTES
}

/// Hard resource and isolation limits enforced by [`WasmSandboxRunnerV1`] (`ADR-0018`, `W12-04`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WasmStoreLimitsV1 {
    /// Maximum accepted `.wasm` binary size in bytes.
    pub max_module_bytes: usize,
    /// Maximum linear memory size in 64 KiB Wasm pages (`16` = 1 MiB default).
    pub max_memory_pages: u32,
    /// Maximum indirect call table entries.
    pub max_table_elements: u32,
    /// Maximum number of linear memories allowed (`1`).
    pub max_memories: u32,
    /// Maximum number of tables allowed (`1`).
    pub max_tables: u32,
    /// Deterministic instruction fuel budget per invocation.
    pub max_fuel: u64,
    /// Maximum deterministic fuel allowed during synchronous `start` execution (`Decision A2`).
    #[serde(default = "default_max_start_fuel")]
    pub max_start_fuel: u64,
    /// Maximum accepted guest output payload size in bytes (`ADR-0025`).
    #[serde(default = "default_max_output_bytes")]
    pub max_output_bytes: usize,
    /// Trap immediately instead of returning `-1` when `memory.grow` would exceed `max_memory_pages`.
    pub trap_on_grow_failure: bool,
    /// Advisory wall-clock execution limit matching `RunnerIsolationProfile`.
    pub timeout_ms: u64,
}

impl Default for WasmStoreLimitsV1 {
    fn default() -> Self {
        Self {
            max_module_bytes: 1_048_576,
            max_memory_pages: 16,
            max_table_elements: 1_024,
            max_memories: 1,
            max_tables: 1,
            max_fuel: 100_000,
            max_start_fuel: DEFAULT_WASM_MAX_START_FUEL,
            max_output_bytes: DEFAULT_WASM_MAX_OUTPUT_BYTES,
            trap_on_grow_failure: true,
            timeout_ms: 5_000,
        }
    }
}

/// Fail-closed policy governing `ExecutionTrust::SandboxedExtension` execution (`ADR-0018`, `W12-04`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub struct SandboxedExtensionPolicyV1 {
    /// SHA-256 digests of cryptographically pinned Wasm modules permitted to run.
    pub allowed_module_hashes: BTreeSet<[u8; 32]>,
    /// Optional export symbol allowlist (if empty, caller-requested export is validated against module).
    pub allowed_exports: BTreeSet<String>,
    /// Store memory, table, fuel, and grow-trap caps.
    pub limits: WasmStoreLimitsV1,
    /// Guest memory ABI contract (`RawMem0V0` default or structured `SwarmTorchAllocV1`, `ADR-0025`).
    #[serde(default)]
    pub guest_abi: WasmGuestAbiV1,
}

impl SandboxedExtensionPolicyV1 {
    /// Construct a policy pinning one known module SHA-256 digest with default strict limits.
    pub fn with_allowed_hash(module_sha256: [u8; 32]) -> Self {
        let mut allowed_module_hashes = BTreeSet::new();
        allowed_module_hashes.insert(module_sha256);
        Self {
            allowed_module_hashes,
            allowed_exports: BTreeSet::new(),
            limits: WasmStoreLimitsV1::default(),
            guest_abi: WasmGuestAbiV1::default(),
        }
    }

    /// Builder method setting the guest memory ABI contract (`ADR-0025`).
    pub fn with_guest_abi(mut self, guest_abi: WasmGuestAbiV1) -> Self {
        self.guest_abi = guest_abi;
        self
    }

    /// Compute canonical SHA-256 digest of a raw `.wasm` module binary.
    pub fn sha256_module(wasm_bytes: &[u8]) -> [u8; 32] {
        let digest = Sha256::digest(wasm_bytes);
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest[..]);
        out
    }

    /// Compute a canonical domain-separated SHA-256 hex digest binding all allowlists, store
    /// limits, and `guest_abi` so cached `SandboxedExtension` outputs cannot replay across
    /// tightened policies or ABI changes (`ADR-0025`).
    pub fn policy_fingerprint_hex(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"swarmtorch.wasm.policy.v1");
        hasher.update((self.allowed_module_hashes.len() as u64).to_be_bytes());
        for hash in &self.allowed_module_hashes {
            hasher.update(hash);
        }
        hasher.update((self.allowed_exports.len() as u64).to_be_bytes());
        for export in &self.allowed_exports {
            hasher.update((export.len() as u64).to_be_bytes());
            hasher.update(export.as_bytes());
        }
        hasher.update((self.limits.max_module_bytes as u64).to_be_bytes());
        hasher.update(self.limits.max_memory_pages.to_be_bytes());
        hasher.update(self.limits.max_table_elements.to_be_bytes());
        hasher.update(self.limits.max_memories.to_be_bytes());
        hasher.update(self.limits.max_tables.to_be_bytes());
        hasher.update(self.limits.max_fuel.to_be_bytes());
        hasher.update(self.limits.max_start_fuel.to_be_bytes());
        hasher.update((self.limits.max_output_bytes as u64).to_be_bytes());
        hasher.update([u8::from(self.limits.trap_on_grow_failure)]);
        hasher.update(self.limits.timeout_ms.to_be_bytes());
        let abi_tag: u8 = match self.guest_abi {
            WasmGuestAbiV1::RawMem0V0 => 0,
            WasmGuestAbiV1::SwarmTorchAllocV1 => 1,
        };
        hasher.update([abi_tag]);
        crate::artifacts::hex_lower(&hasher.finalize())
    }
}

/// Deterministic fuel meter charged on every decoded Wasm opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WasmFuelMeterV1 {
    limit: u64,
    consumed: u64,
}

impl WasmFuelMeterV1 {
    /// Construct a fuel meter with total budget `limit`.
    pub const fn new(limit: u64) -> Self {
        Self { limit, consumed: 0 }
    }

    /// Consume 1 unit of fuel or fail closed with [`WasmSandboxError::OutOfFuel`].
    #[inline]
    pub fn consume(&mut self, units: u64) -> Result<(), WasmSandboxError> {
        let next = self.consumed.saturating_add(units);
        if next > self.limit {
            return Err(WasmSandboxError::OutOfFuel {
                consumed: self.consumed,
                limit: self.limit,
            });
        }
        self.consumed = next;
        Ok(())
    }

    /// Total fuel consumed so far.
    pub const fn consumed(&self) -> u64 {
        self.consumed
    }
}

/// Verified receipt emitted after sandboxed Wasm execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmExecutionReceiptV1 {
    /// Cryptographic SHA-256 digest of the executed Wasm binary.
    pub module_sha256: [u8; 32],
    /// Deterministic fuel units consumed (including `start` section if present).
    pub fuel_consumed: u64,
    /// Peak linear memory size in bytes reached during execution.
    pub peak_memory_bytes: usize,
    /// Integer result returned by the exported function.
    pub return_code: i32,
    /// Checked output slice read from guest memory (for byte-transform invocations).
    pub output_bytes: Vec<u8>,
}

/// Checked guest linear memory slice helper (`ADR-0018`, `W12-04`).
///
/// Rejects `ptr < 0`, `len < 0`, and `u32::checked_add` wrap-around attacks.
pub fn read_guest_slice(memory: &[u8], ptr: i32, len: i32) -> Result<Vec<u8>, WasmSandboxError> {
    if ptr < 0 || len < 0 {
        return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
            addr: ptr as u64,
            len: len.max(0) as usize,
            mem_len: memory.len(),
        });
    }
    let start = ptr as u32;
    let byte_len = len as u32;
    let end = start
        .checked_add(byte_len)
        .ok_or(WasmSandboxError::OutOfBoundsMemoryAccess {
            addr: start as u64,
            len: byte_len as usize,
            mem_len: memory.len(),
        })?;
    let start_usize = start as usize;
    let end_usize = end as usize;
    if end_usize > memory.len() {
        return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
            addr: start as u64,
            len: byte_len as usize,
            mem_len: memory.len(),
        });
    }
    Ok(memory[start_usize..end_usize].to_vec())
}

/// Unpack a 64-bit guest return value `packed = ((out_ptr as u32 as u64) << 32) | (out_len as u32 as u64)`
/// under [`WasmGuestAbiV1::SwarmTorchAllocV1`] (`ADR-0025`).
///
/// - If the high 32-bit signed word is negative (`high < 0`), returns
///   [`WasmSandboxError::GuestTransformFailed`].
/// - If `high >= 0` and the low 32-bit signed word is negative (`low < 0`), returns
///   [`WasmSandboxError::OutOfBoundsMemoryAccess`].
/// - Otherwise returns `Ok((out_ptr, out_len))` with `out_ptr >= 0` and `out_len >= 0`.
pub fn unpack_guest_ptr_len_i64(packed: i64) -> Result<(i32, i32), WasmSandboxError> {
    let raw = packed as u64;
    let high = (raw >> 32) as i32;
    let low = (raw & 0xFFFF_FFFF) as i32;
    if high < 0 {
        let code = if high == -1 && low < 0 { low } else { high };
        return Err(WasmSandboxError::GuestTransformFailed {
            export: String::new(),
            code,
        });
    }
    if low < 0 {
        return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
            addr: high as u64,
            len: (low as u32) as usize,
            mem_len: 0,
        });
    }
    Ok((high, low))
}

/// Narrow execution boundary for an isolated external WebAssembly sandbox runtime (`ST-09`).
///
/// Implemented outside `swarm-torch` by `adapters/swarm-torch-wasm` using `wasmi`, while keeping
/// `swarm-torch` dependency-free and fail-closed whenever no executor is configured.
pub trait WasmExtensionExecutor: Send + Sync {
    /// Execute a sandboxed byte-transform Wasm extension under `policy`, dispatching according to
    /// [`SandboxedExtensionPolicyV1::guest_abi`] ([`WasmGuestAbiV1::RawMem0V0`]
    /// `export(ptr: i32, len: i32) -> i32` or [`WasmGuestAbiV1::SwarmTorchAllocV1`]
    /// `export(in_ptr: i32, in_len: i32) -> i64` with `swarmtorch_alloc`/`swarmtorch_dealloc`),
    /// enforcing fuel, wall-clock timeout, memory/table/output limits, and SHA-256 pinning.
    fn execute_byte_transform(
        &self,
        policy: &SandboxedExtensionPolicyV1,
        wasm_bytes: &[u8],
        export_name: &str,
        input_payload: &[u8],
    ) -> Result<WasmExecutionReceiptV1, WasmSandboxError>;

    /// Execute a sandboxed scalar Wasm extension (`export(arg: i32) -> i32`)
    /// under `policy`, enforcing fuel, wall-clock timeout, memory/table limits, and SHA-256 pinning.
    fn execute_scalar_i32(
        &self,
        policy: &SandboxedExtensionPolicyV1,
        wasm_bytes: &[u8],
        export_name: &str,
        arg: i32,
    ) -> Result<(i32, WasmExecutionReceiptV1), WasmSandboxError>;
}

/// Decode a strict 64-character lowercase hexadecimal SHA-256 digest string.
pub fn parse_sha256_hex_v1(hex_str: &str) -> Option<[u8; 32]> {
    let bytes = hex_str.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let hex_nibble = |b: u8| -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            _ => None,
        }
    };
    let mut out = [0u8; 32];
    for (idx, chunk) in bytes.chunks_exact(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out[idx] = (hi << 4) | lo;
    }
    Some(out)
}

/// Shared fail-closed Wasm extension preflight executed prior to any runtime instantiation (`ST-09`).
///
/// Enforces module size cap, `\0asm\x01\0\0\0` magic header, non-zero fuel budget,
/// export allowlist policy, and SHA-256 pinning against
/// [`SandboxedExtensionPolicyV1::allowed_module_hashes`].
pub fn preflight_wasm_extension_v1(
    policy: &SandboxedExtensionPolicyV1,
    wasm_bytes: &[u8],
    export_name: &str,
) -> Result<[u8; 32], WasmSandboxError> {
    if wasm_bytes.len() > policy.limits.max_module_bytes {
        return Err(WasmSandboxError::ModuleTooLarge {
            len: wasm_bytes.len(),
            max: policy.limits.max_module_bytes,
        });
    }
    if wasm_bytes.len() < 8 || wasm_bytes[..8] != WASM_MAGIC_AND_VERSION_V1 {
        return Err(WasmSandboxError::InvalidWasmHeader);
    }
    let actual = SandboxedExtensionPolicyV1::sha256_module(wasm_bytes);
    if policy.allowed_module_hashes.is_empty() || !policy.allowed_module_hashes.contains(&actual) {
        return Err(WasmSandboxError::UntrustedModuleHash { actual });
    }
    if export_name.is_empty()
        || (!policy.allowed_exports.is_empty() && !policy.allowed_exports.contains(export_name))
        || (policy.guest_abi == WasmGuestAbiV1::SwarmTorchAllocV1
            && (export_name == SWARMTORCH_WASM_ALLOC_EXPORT_V1
                || export_name == SWARMTORCH_WASM_DEALLOC_EXPORT_V1))
    {
        return Err(WasmSandboxError::ExportNotAllowed(export_name.to_string()));
    }
    if policy.limits.max_fuel == 0 {
        return Err(WasmSandboxError::OutOfFuel {
            consumed: 0,
            limit: 0,
        });
    }
    Ok(actual)
}

pub(crate) fn wasm_error_to_io(err: WasmSandboxError) -> std::io::Error {
    use std::io;
    let kind = match &err {
        WasmSandboxError::UnsupportedUntrustedWasmSandbox => io::ErrorKind::Unsupported,
        WasmSandboxError::UntrustedModuleHash { .. }
        | WasmSandboxError::ForbiddenImport { .. }
        | WasmSandboxError::ExportNotAllowed(_) => io::ErrorKind::PermissionDenied,
        WasmSandboxError::ExecutionTimeout { .. } => io::ErrorKind::TimedOut,
        WasmSandboxError::ModuleTooLarge { .. }
        | WasmSandboxError::InvalidWasmHeader
        | WasmSandboxError::MalformedSectionOrLeb128
        | WasmSandboxError::MultipleMemoriesForbidden
        | WasmSandboxError::MultipleTablesForbidden
        | WasmSandboxError::MemoryLimitExceeded { .. }
        | WasmSandboxError::TableLimitExceeded { .. }
        | WasmSandboxError::ExportSignatureMismatch(_)
        | WasmSandboxError::OutOfBoundsMemoryAccess { .. }
        | WasmSandboxError::OutputTooLarge { .. }
        | WasmSandboxError::OutputAliasingViolation { .. } => io::ErrorKind::InvalidInput,
        WasmSandboxError::OutOfFuel { .. }
        | WasmSandboxError::MemoryGrowTrap { .. }
        | WasmSandboxError::ExecutionTrap(_)
        | WasmSandboxError::GuestTransformFailed { .. } => io::ErrorKind::Other,
    };
    io::Error::new(kind, err.to_string())
}

/// Shared fail-closed resolver & preflight check for `wasm_transform` nodes across scheduler admission
/// and physical execution (`ADR-0018`, `ST-09`).
pub(crate) fn resolve_and_preflight_wasm_node_v1<'a, 'b>(
    policy: &SandboxedExtensionPolicyV1,
    params: &'a swarm_torch_core::run_graph::CanonParams,
    registry: &'b std::collections::BTreeMap<[u8; 32], Vec<u8>>,
) -> std::io::Result<([u8; 32], &'a str, &'b [u8])> {
    use std::io;
    use swarm_torch_core::run_graph::CanonValue;

    if registry.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "wasm_transform requires a non-empty wasm_registry",
        ));
    }
    let module_hex = match params.get("module_sha256") {
        Some(CanonValue::Str(s)) => s.as_str(),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "wasm_transform requires string parameter 'module_sha256'",
            ));
        }
    };
    let declared_hash = parse_sha256_hex_v1(module_hex).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("wasm_transform invalid 64-hex lowercase module_sha256 '{module_hex}'"),
        )
    })?;
    let export_name = match params.get("export") {
        Some(CanonValue::Str(s)) if !s.is_empty() => s.as_str(),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "wasm_transform requires non-empty string parameter 'export'",
            ));
        }
    };
    let wasm_bytes = registry.get(&declared_hash).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("wasm_transform module '{module_hex}' not present in wasm_registry"),
        )
    })?;
    let verified_hash =
        preflight_wasm_extension_v1(policy, wasm_bytes, export_name).map_err(wasm_error_to_io)?;
    if verified_hash != declared_hash {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "wasm_registry entry bytes do not match declared module_sha256 key",
        ));
    }
    Ok((declared_hash, export_name, wasm_bytes.as_slice()))
}

#[cfg(test)]
mod legacy_vm {
    use super::*;

    // ---------------------------------------------------------------------------
    // Pure-Rust WebAssembly MVP Binary Decoder & Deterministic Stack Interpreter (test-only)
    // ---------------------------------------------------------------------------

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum WasmValType {
        I32,
        I64,
        F32,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct WasmFuncType {
        params: Vec<WasmValType>,
        results: Vec<WasmValType>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct WasmValue {
        raw_i32: i32,
        raw_f32: u32,
        raw_i64: i64,
        ty: WasmValType,
    }

    impl WasmValue {
        fn i32(v: i32) -> Self {
            Self {
                raw_i32: v,
                raw_f32: v as u32,
                raw_i64: v as i64,
                ty: WasmValType::I32,
            }
        }
        fn i64(v: i64) -> Self {
            Self {
                raw_i32: v as i32,
                raw_f32: 0,
                raw_i64: v,
                ty: WasmValType::I64,
            }
        }
        fn f32(v: f32) -> Self {
            Self {
                raw_i32: 0,
                raw_f32: v.to_bits(),
                raw_i64: 0,
                ty: WasmValType::F32,
            }
        }
        fn as_i32(self) -> Result<i32, WasmSandboxError> {
            if self.ty != WasmValType::I32 {
                return Err(WasmSandboxError::ExecutionTrap(
                    "type mismatch: expected i32".to_string(),
                ));
            }
            Ok(self.raw_i32)
        }
        fn as_f32(self) -> Result<f32, WasmSandboxError> {
            if self.ty != WasmValType::F32 {
                return Err(WasmSandboxError::ExecutionTrap(
                    "type mismatch: expected f32".to_string(),
                ));
            }
            Ok(f32::from_bits(self.raw_f32))
        }
    }

    #[derive(Debug, Clone)]
    struct WasmGlobalDecl {
        val: WasmValue,
        mutable: bool,
    }

    #[derive(Debug, Clone)]
    struct WasmDataSegment {
        offset: usize,
        bytes: Vec<u8>,
    }

    #[derive(Debug, Clone)]
    struct WasmCodeBody {
        locals: Vec<WasmValType>,
        expr: Vec<u8>,
    }

    #[derive(Debug, Clone)]
    struct DecodedWasmModule {
        types: Vec<WasmFuncType>,
        func_type_indices: Vec<u32>,
        initial_memory_pages: u32,
        exports: std::collections::BTreeMap<String, u32>,
        start_func: Option<u32>,
        globals: Vec<WasmGlobalDecl>,
        data_segments: Vec<WasmDataSegment>,
        codes: Vec<WasmCodeBody>,
    }

    struct Cursor<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> Cursor<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            Self { bytes, pos: 0 }
        }
        fn remaining(&self) -> usize {
            self.bytes.len().saturating_sub(self.pos)
        }
        fn read_u8(&mut self) -> Result<u8, WasmSandboxError> {
            let b = *self
                .bytes
                .get(self.pos)
                .ok_or(WasmSandboxError::MalformedSectionOrLeb128)?;
            self.pos += 1;
            Ok(b)
        }
        fn read_exact(&mut self, len: usize) -> Result<&'a [u8], WasmSandboxError> {
            let end = self
                .pos
                .checked_add(len)
                .ok_or(WasmSandboxError::MalformedSectionOrLeb128)?;
            let slice = self
                .bytes
                .get(self.pos..end)
                .ok_or(WasmSandboxError::MalformedSectionOrLeb128)?;
            self.pos = end;
            Ok(slice)
        }
        fn read_u32_leb(&mut self) -> Result<u32, WasmSandboxError> {
            let mut result: u32 = 0;
            let mut shift = 0u32;
            for _ in 0..5 {
                let b = self.read_u8()?;
                let low = (b & 0x7f) as u32;
                if shift == 28 && (b & 0x70) != 0 {
                    return Err(WasmSandboxError::MalformedSectionOrLeb128);
                }
                result |= low << shift;
                if (b & 0x80) == 0 {
                    return Ok(result);
                }
                shift += 7;
            }
            Err(WasmSandboxError::MalformedSectionOrLeb128)
        }
        fn read_i32_leb(&mut self) -> Result<i32, WasmSandboxError> {
            let mut result: i32 = 0;
            let mut shift = 0u32;
            let mut last_byte;
            for _ in 0..5 {
                let b = self.read_u8()?;
                last_byte = b;
                result |= ((b & 0x7f) as i32) << shift;
                shift += 7;
                if (b & 0x80) == 0 {
                    if shift < 32 && (last_byte & 0x40) != 0 {
                        result |= !0i32 << shift;
                    }
                    return Ok(result);
                }
            }
            Err(WasmSandboxError::MalformedSectionOrLeb128)
        }
        fn read_i64_leb(&mut self) -> Result<i64, WasmSandboxError> {
            let mut result: i64 = 0;
            let mut shift = 0u32;
            for _ in 0..10 {
                let b = self.read_u8()?;
                result |= ((b & 0x7f) as i64) << shift;
                shift += 7;
                if (b & 0x80) == 0 {
                    if shift < 64 && (b & 0x40) != 0 {
                        result |= !0i64 << shift;
                    }
                    return Ok(result);
                }
            }
            Err(WasmSandboxError::MalformedSectionOrLeb128)
        }
        fn read_name(&mut self) -> Result<String, WasmSandboxError> {
            let len = self.read_u32_leb()? as usize;
            let slice = self.read_exact(len)?;
            let s = std::str::from_utf8(slice)
                .map_err(|_| WasmSandboxError::MalformedSectionOrLeb128)?;
            Ok(s.to_string())
        }
        fn read_val_type(&mut self) -> Result<WasmValType, WasmSandboxError> {
            match self.read_u8()? {
                0x7f => Ok(WasmValType::I32),
                0x7e => Ok(WasmValType::I64),
                0x7d => Ok(WasmValType::F32),
                0x7c => Err(WasmSandboxError::ExecutionTrap(
                    "f64 value type is unsupported in experimental test fixture".to_string(),
                )),
                _ => Err(WasmSandboxError::MalformedSectionOrLeb128),
            }
        }
    }

    fn read_limits(cur: &mut Cursor<'_>) -> Result<(u32, Option<u32>), WasmSandboxError> {
        let flags = cur.read_u8()?;
        let initial = cur.read_u32_leb()?;
        let max = match flags {
            0x00 => None,
            0x01 => {
                let m = cur.read_u32_leb()?;
                if m < initial {
                    return Err(WasmSandboxError::MalformedSectionOrLeb128);
                }
                Some(m)
            }
            _ => return Err(WasmSandboxError::MalformedSectionOrLeb128),
        };
        Ok((initial, max))
    }

    fn eval_const_init_expr(cur: &mut Cursor<'_>) -> Result<WasmValue, WasmSandboxError> {
        let val = match cur.read_u8()? {
            0x41 => WasmValue::i32(cur.read_i32_leb()?),
            0x42 => WasmValue::i64(cur.read_i64_leb()?),
            0x43 => {
                let b = cur.read_exact(4)?;
                WasmValue::f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            _ => return Err(WasmSandboxError::MalformedSectionOrLeb128),
        };
        if cur.read_u8()? != 0x0b {
            return Err(WasmSandboxError::MalformedSectionOrLeb128);
        }
        Ok(val)
    }

    fn decode_and_validate_module(
        wasm_bytes: &[u8],
        limits: &WasmStoreLimitsV1,
        fuel: &mut WasmFuelMeterV1,
    ) -> Result<DecodedWasmModule, WasmSandboxError> {
        if wasm_bytes.len() > limits.max_module_bytes {
            return Err(WasmSandboxError::ModuleTooLarge {
                len: wasm_bytes.len(),
                max: limits.max_module_bytes,
            });
        }
        if wasm_bytes.len() < 8 || wasm_bytes[..8] != WASM_MAGIC_AND_VERSION_V1 {
            return Err(WasmSandboxError::InvalidWasmHeader);
        }

        let mut cur = Cursor::new(&wasm_bytes[8..]);
        let mut types = Vec::new();
        let mut func_type_indices = Vec::new();
        let mut initial_memory_pages = 0u32;
        let mut memory_count = 0u32;
        let mut table_count = 0u32;
        let mut exports = std::collections::BTreeMap::new();
        let mut start_func = None;
        let mut globals = Vec::new();
        let mut data_segments = Vec::new();
        let mut codes = Vec::new();
        let mut last_non_custom_section = 0u8;

        while cur.remaining() > 0 {
            fuel.consume(1)?;
            let section_id = cur.read_u8()?;
            let section_len = cur.read_u32_leb()? as usize;
            let section_bytes = cur.read_exact(section_len)?;

            if section_id != 0 {
                if section_id > 12 || section_id <= last_non_custom_section {
                    return Err(WasmSandboxError::MalformedSectionOrLeb128);
                }
                last_non_custom_section = section_id;
            }

            let mut sc = Cursor::new(section_bytes);
            match section_id {
                0 => {
                    // Custom section skipped cleanly (bounded inside section_bytes).
                }
                1 => {
                    let count = sc.read_u32_leb()?;
                    for _ in 0..count {
                        fuel.consume(1)?;
                        if sc.read_u8()? != 0x60 {
                            return Err(WasmSandboxError::MalformedSectionOrLeb128);
                        }
                        let p_len = sc.read_u32_leb()?;
                        let mut params = Vec::with_capacity((p_len as usize).min(256));
                        for _ in 0..p_len {
                            fuel.consume(1)?;
                            params.push(sc.read_val_type()?);
                        }
                        let r_len = sc.read_u32_leb()?;
                        let mut results = Vec::with_capacity((r_len as usize).min(256));
                        for _ in 0..r_len {
                            fuel.consume(1)?;
                            results.push(sc.read_val_type()?);
                        }
                        types.push(WasmFuncType { params, results });
                    }
                }
                2 => {
                    // Zero-capability isolation: reject ALL WASI / external host imports!
                    let count = sc.read_u32_leb()?;
                    if count > 0 {
                        let module = sc.read_name()?;
                        let field = sc.read_name()?;
                        return Err(WasmSandboxError::ForbiddenImport { module, field });
                    }
                }
                3 => {
                    let count = sc.read_u32_leb()?;
                    for _ in 0..count {
                        let tidx = sc.read_u32_leb()?;
                        if tidx as usize >= types.len() {
                            return Err(WasmSandboxError::MalformedSectionOrLeb128);
                        }
                        func_type_indices.push(tidx);
                    }
                }
                4 => {
                    let count = sc.read_u32_leb()?;
                    table_count = table_count.saturating_add(count);
                    if table_count > limits.max_tables {
                        return Err(WasmSandboxError::MultipleTablesForbidden);
                    }
                    for _ in 0..count {
                        let _elem_type = sc.read_u8()?;
                        let (init, max) = read_limits(&mut sc)?;
                        if init > limits.max_table_elements
                            || max.unwrap_or(init) > limits.max_table_elements
                        {
                            return Err(WasmSandboxError::TableLimitExceeded {
                                requested: max.unwrap_or(init),
                                max: limits.max_table_elements,
                            });
                        }
                    }
                }
                5 => {
                    let count = sc.read_u32_leb()?;
                    memory_count = memory_count.saturating_add(count);
                    if memory_count > limits.max_memories {
                        return Err(WasmSandboxError::MultipleMemoriesForbidden);
                    }
                    for _ in 0..count {
                        let (init, max) = read_limits(&mut sc)?;
                        if init > limits.max_memory_pages {
                            return Err(WasmSandboxError::MemoryLimitExceeded {
                                requested_pages: init,
                                max_pages: limits.max_memory_pages,
                            });
                        }
                        if let Some(m) = max {
                            if m > limits.max_memory_pages {
                                return Err(WasmSandboxError::MemoryLimitExceeded {
                                    requested_pages: m,
                                    max_pages: limits.max_memory_pages,
                                });
                            }
                        }
                        initial_memory_pages = init;
                    }
                }
                6 => {
                    let count = sc.read_u32_leb()?;
                    for _ in 0..count {
                        let _ty = sc.read_val_type()?;
                        let mutable = sc.read_u8()? == 1;
                        let val = eval_const_init_expr(&mut sc)?;
                        globals.push(WasmGlobalDecl { val, mutable });
                    }
                }
                7 => {
                    let count = sc.read_u32_leb()?;
                    for _ in 0..count {
                        let name = sc.read_name()?;
                        let kind = sc.read_u8()?;
                        let idx = sc.read_u32_leb()?;
                        if kind == 0x00 {
                            if idx as usize >= func_type_indices.len() {
                                return Err(WasmSandboxError::MalformedSectionOrLeb128);
                            }
                            exports.insert(name, idx);
                        }
                    }
                }
                8 => {
                    let fidx = sc.read_u32_leb()?;
                    if fidx as usize >= func_type_indices.len() {
                        return Err(WasmSandboxError::MalformedSectionOrLeb128);
                    }
                    start_func = Some(fidx);
                }
                9 => {
                    // Element segments validated cleanly
                    let count = sc.read_u32_leb()?;
                    for _ in 0..count {
                        let table_idx = sc.read_u32_leb()?;
                        if table_idx >= table_count {
                            return Err(WasmSandboxError::MalformedSectionOrLeb128);
                        }
                        let _ = eval_const_init_expr(&mut sc)?;
                        let num_fn = sc.read_u32_leb()?;
                        for _ in 0..num_fn {
                            let fidx = sc.read_u32_leb()?;
                            if fidx as usize >= func_type_indices.len() {
                                return Err(WasmSandboxError::MalformedSectionOrLeb128);
                            }
                        }
                    }
                }
                10 => {
                    let count = sc.read_u32_leb()?;
                    if count as usize != func_type_indices.len() {
                        return Err(WasmSandboxError::MalformedSectionOrLeb128);
                    }
                    for _ in 0..count {
                        fuel.consume(1)?;
                        let body_len = sc.read_u32_leb()? as usize;
                        let body_slice = sc.read_exact(body_len)?;
                        let mut bc = Cursor::new(body_slice);
                        let local_decls = bc.read_u32_leb()?;
                        let mut locals = Vec::new();
                        let mut total_locals = 0usize;
                        for _ in 0..local_decls {
                            fuel.consume(1)?;
                            let rep = bc.read_u32_leb()?;
                            let ty = bc.read_val_type()?;
                            total_locals = total_locals
                                .checked_add(rep as usize)
                                .ok_or(WasmSandboxError::MalformedSectionOrLeb128)?;
                            if total_locals > DEFAULT_WASM_MAX_LOCALS_PER_FUNCTION {
                                return Err(WasmSandboxError::ExecutionTrap(format!(
                                "wasm function local count {total_locals} exceeds cap {DEFAULT_WASM_MAX_LOCALS_PER_FUNCTION}"
                            )));
                            }
                            for _ in 0..rep {
                                fuel.consume(1)?;
                                locals.push(ty);
                            }
                        }
                        let expr = bc.bytes[bc.pos..].to_vec();
                        let scan_cost = (expr.len() as u64).max(1) / 8 + 1;
                        fuel.consume(scan_cost)?;
                        if expr.last().copied() != Some(0x0b) {
                            return Err(WasmSandboxError::MalformedSectionOrLeb128);
                        }
                        codes.push(WasmCodeBody { locals, expr });
                    }
                }
                11 => {
                    let count = sc.read_u32_leb()?;
                    if count > 0 && memory_count == 0 {
                        return Err(WasmSandboxError::MalformedSectionOrLeb128);
                    }
                    for _ in 0..count {
                        let mem_idx = sc.read_u32_leb()?;
                        if mem_idx != 0 {
                            return Err(WasmSandboxError::MultipleMemoriesForbidden);
                        }
                        let offset_val = eval_const_init_expr(&mut sc)?.as_i32()?;
                        if offset_val < 0 {
                            return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                                addr: offset_val as u64,
                                len: 0,
                                mem_len: (initial_memory_pages as usize) * WASM_PAGE_BYTES,
                            });
                        }
                        let dlen = sc.read_u32_leb()? as usize;
                        let dbytes = sc.read_exact(dlen)?.to_vec();
                        data_segments.push(WasmDataSegment {
                            offset: offset_val as usize,
                            bytes: dbytes,
                        });
                    }
                }
                _ => return Err(WasmSandboxError::MalformedSectionOrLeb128),
            }
            if section_id != 0 && sc.remaining() != 0 {
                return Err(WasmSandboxError::MalformedSectionOrLeb128);
            }
        }

        if codes.len() != func_type_indices.len() {
            return Err(WasmSandboxError::MalformedSectionOrLeb128);
        }

        Ok(DecodedWasmModule {
            types,
            func_type_indices,
            initial_memory_pages,
            exports,
            start_func,
            globals,
            data_segments,
            codes,
        })
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum LabelKind {
        Function,
        Block,
        Loop,
        If,
    }

    #[derive(Debug, Clone, Copy)]
    struct ControlFrame {
        kind: LabelKind,
        start_pc: usize,
        end_pc: usize,
        stack_height: usize,
    }

    struct WasmVm<'a> {
        module: &'a DecodedWasmModule,
        limits: &'a WasmStoreLimitsV1,
        memory: Vec<u8>,
        peak_memory_bytes: usize,
        globals: Vec<WasmValue>,
        fuel: WasmFuelMeterV1,
        call_depth: usize,
    }

    impl<'a> WasmVm<'a> {
        fn instantiate(
            module: &'a DecodedWasmModule,
            limits: &'a WasmStoreLimitsV1,
            fuel: WasmFuelMeterV1,
        ) -> Result<Self, WasmSandboxError> {
            let initial_bytes = (module.initial_memory_pages as usize)
                .checked_mul(WASM_PAGE_BYTES)
                .ok_or(WasmSandboxError::MemoryLimitExceeded {
                    requested_pages: module.initial_memory_pages,
                    max_pages: limits.max_memory_pages,
                })?;
            let mut memory = vec![0u8; initial_bytes];
            for seg in &module.data_segments {
                let end = seg.offset.checked_add(seg.bytes.len()).ok_or(
                    WasmSandboxError::OutOfBoundsMemoryAccess {
                        addr: seg.offset as u64,
                        len: seg.bytes.len(),
                        mem_len: memory.len(),
                    },
                )?;
                if end > memory.len() {
                    return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                        addr: seg.offset as u64,
                        len: seg.bytes.len(),
                        mem_len: memory.len(),
                    });
                }
                memory[seg.offset..end].copy_from_slice(&seg.bytes);
            }
            let globals = module.globals.iter().map(|g| g.val).collect();
            Ok(Self {
                module,
                limits,
                memory,
                peak_memory_bytes: initial_bytes,
                globals,
                fuel,
                call_depth: 0,
            })
        }

        fn effective_addr(
            &self,
            base: i32,
            offset: u32,
            width: usize,
        ) -> Result<usize, WasmSandboxError> {
            if base < 0 {
                return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                    addr: base as u64,
                    len: width,
                    mem_len: self.memory.len(),
                });
            }
            let addr = (base as usize).checked_add(offset as usize).ok_or(
                WasmSandboxError::OutOfBoundsMemoryAccess {
                    addr: base as u64,
                    len: width,
                    mem_len: self.memory.len(),
                },
            )?;
            let end = addr
                .checked_add(width)
                .ok_or(WasmSandboxError::OutOfBoundsMemoryAccess {
                    addr: addr as u64,
                    len: width,
                    mem_len: self.memory.len(),
                })?;
            if end > self.memory.len() {
                return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                    addr: addr as u64,
                    len: width,
                    mem_len: self.memory.len(),
                });
            }
            Ok(addr)
        }

        fn call(
            &mut self,
            func_idx: u32,
            args: &[WasmValue],
        ) -> Result<Vec<WasmValue>, WasmSandboxError> {
            if self.call_depth >= DEFAULT_WASM_MAX_CALL_DEPTH {
                return Err(WasmSandboxError::ExecutionTrap(
                    "call stack depth limit exceeded".to_string(),
                ));
            }
            let tidx = *self
                .module
                .func_type_indices
                .get(func_idx as usize)
                .ok_or(WasmSandboxError::MalformedSectionOrLeb128)?;
            let ftype = &self.module.types[tidx as usize];
            if args.len() != ftype.params.len() {
                return Err(WasmSandboxError::ExportSignatureMismatch(format!(
                    "func {func_idx}"
                )));
            }
            let code = &self.module.codes[func_idx as usize];
            let mut locals = Vec::with_capacity(args.len() + code.locals.len());
            locals.extend_from_slice(args);
            for &lty in &code.locals {
                let z = match lty {
                    WasmValType::I32 => WasmValue::i32(0),
                    WasmValType::I64 => WasmValue::i64(0),
                    WasmValType::F32 => WasmValue::f32(0.0),
                };
                locals.push(z);
            }

            self.call_depth += 1;
            let res = self.exec_body(&code.expr, ftype.results.len(), &mut locals);
            self.call_depth -= 1;
            res
        }

        fn exec_body(
            &mut self,
            expr: &[u8],
            result_arity: usize,
            locals: &mut [WasmValue],
        ) -> Result<Vec<WasmValue>, WasmSandboxError> {
            // Pre-scan structured control flow endpoints (block/loop/if/else/end) cleanly.
            let mut matches_end = vec![0usize; expr.len()];
            let mut matches_else: Vec<Option<usize>> = vec![None; expr.len()];
            let mut scan_stack: Vec<(usize, Option<usize>)> = Vec::new();
            {
                let mut sc = Cursor::new(expr);
                while sc.remaining() > 0 {
                    self.fuel.consume(1)?;
                    let op_pos = sc.pos;
                    let op = sc.read_u8()?;
                    match op {
                        0x02..=0x04 => {
                            let _bt = sc.read_u8()?;
                            scan_stack.push((op_pos, None));
                        }
                        0x05 => {
                            if let Some(top) = scan_stack.last_mut() {
                                top.1 = Some(op_pos);
                            } else {
                                return Err(WasmSandboxError::MalformedSectionOrLeb128);
                            }
                        }
                        0x0b => {
                            if let Some((start, else_p)) = scan_stack.pop() {
                                matches_end[start] = op_pos;
                                matches_else[start] = else_p;
                            } else if sc.remaining() == 0 {
                                matches_end[0] = op_pos;
                            } else {
                                return Err(WasmSandboxError::MalformedSectionOrLeb128);
                            }
                        }
                        0x0c | 0x0d | 0x10 | 0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x3f | 0x40 => {
                            let _ = sc.read_u32_leb()?;
                        }
                        0x41 => {
                            let _ = sc.read_i32_leb()?;
                        }
                        0x42 => {
                            let _ = sc.read_i64_leb()?;
                        }
                        0x43 => {
                            let _ = sc.read_exact(4)?;
                        }
                        0x44 => {
                            let _ = sc.read_exact(8)?;
                        }
                        0x28 | 0x2a | 0x2c | 0x2d | 0x36 | 0x38 | 0x3a => {
                            let _align = sc.read_u32_leb()?;
                            let _off = sc.read_u32_leb()?;
                        }
                        _ => {}
                    }
                }
                if !scan_stack.is_empty() {
                    return Err(WasmSandboxError::MalformedSectionOrLeb128);
                }
            }

            let mut stack: Vec<WasmValue> = Vec::new();
            let mut frames: Vec<ControlFrame> = vec![ControlFrame {
                kind: LabelKind::Function,
                start_pc: 0,
                end_pc: expr.len() - 1,
                stack_height: 0,
            }];
            let mut cur = Cursor::new(expr);

            loop {
                if cur.remaining() == 0 {
                    break;
                }
                self.fuel.consume(1)?;
                let op_pos = cur.pos;
                let op = cur.read_u8()?;
                match op {
                    0x00 => {
                        return Err(WasmSandboxError::ExecutionTrap(
                            "wasm unreachable instruction executed".to_string(),
                        ));
                    }
                    0x01 => {} // nop
                    0x02 => {
                        let _bt = cur.read_u8()?;
                        frames.push(ControlFrame {
                            kind: LabelKind::Block,
                            start_pc: op_pos,
                            end_pc: matches_end[op_pos],
                            stack_height: stack.len(),
                        });
                    }
                    0x03 => {
                        let _bt = cur.read_u8()?;
                        frames.push(ControlFrame {
                            kind: LabelKind::Loop,
                            start_pc: cur.pos,
                            end_pc: matches_end[op_pos],
                            stack_height: stack.len(),
                        });
                    }
                    0x04 => {
                        let _bt = cur.read_u8()?;
                        let cond = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("stack underflow".into())
                            })?
                            .as_i32()?;
                        let end_pc = matches_end[op_pos];
                        let else_pc = matches_else[op_pos];
                        if cond != 0 {
                            frames.push(ControlFrame {
                                kind: LabelKind::If,
                                start_pc: op_pos,
                                end_pc,
                                stack_height: stack.len(),
                            });
                        } else if let Some(ep) = else_pc {
                            frames.push(ControlFrame {
                                kind: LabelKind::Block,
                                start_pc: ep,
                                end_pc,
                                stack_height: stack.len(),
                            });
                            cur.pos = ep + 1;
                        } else {
                            cur.pos = end_pc + 1;
                        }
                    }
                    0x05 => {
                        // Reached end of true branch of `if`: jump to matching `end`
                        let top = frames.pop().ok_or_else(|| {
                            WasmSandboxError::ExecutionTrap("frame underflow".into())
                        })?;
                        cur.pos = top.end_pc + 1;
                    }
                    0x0b => {
                        let popped = frames.pop().ok_or_else(|| {
                            WasmSandboxError::ExecutionTrap("frame underflow".into())
                        })?;
                        if popped.kind == LabelKind::Function {
                            break;
                        }
                    }
                    0x0c | 0x0d => {
                        let depth = cur.read_u32_leb()? as usize;
                        let should_branch = if op == 0x0d {
                            stack
                                .pop()
                                .ok_or_else(|| {
                                    WasmSandboxError::ExecutionTrap("stack underflow".into())
                                })?
                                .as_i32()?
                                != 0
                        } else {
                            true
                        };
                        if should_branch {
                            if depth >= frames.len() {
                                return Err(WasmSandboxError::MalformedSectionOrLeb128);
                            }
                            let target_idx = frames.len() - 1 - depth;
                            let target = frames[target_idx];
                            if target.kind == LabelKind::Function {
                                break;
                            } else if target.kind == LabelKind::Loop {
                                frames.truncate(target_idx + 1);
                                stack.truncate(target.stack_height);
                                cur.pos = target.start_pc;
                            } else {
                                frames.truncate(target_idx);
                                cur.pos = target.end_pc + 1;
                            }
                        }
                    }
                    0x0f => break, // return
                    0x10 => {
                        let target_fn = cur.read_u32_leb()?;
                        let tidx = *self
                            .module
                            .func_type_indices
                            .get(target_fn as usize)
                            .ok_or(WasmSandboxError::MalformedSectionOrLeb128)?;
                        let arity = self.module.types[tidx as usize].params.len();
                        if stack.len() < arity {
                            return Err(WasmSandboxError::ExecutionTrap(
                                "call stack underflow".into(),
                            ));
                        }
                        let call_args = stack.split_off(stack.len() - arity);
                        let call_res = self.call(target_fn, &call_args)?;
                        stack.extend(call_res);
                    }
                    0x1a => {
                        stack.pop().ok_or_else(|| {
                            WasmSandboxError::ExecutionTrap("drop underflow".into())
                        })?;
                    }
                    0x1b => {
                        let c = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("select underflow".into())
                            })?
                            .as_i32()?;
                        let v2 = stack.pop().ok_or_else(|| {
                            WasmSandboxError::ExecutionTrap("select underflow".into())
                        })?;
                        let v1 = stack.pop().ok_or_else(|| {
                            WasmSandboxError::ExecutionTrap("select underflow".into())
                        })?;
                        stack.push(if c != 0 { v1 } else { v2 });
                    }
                    0x20 => {
                        let idx = cur.read_u32_leb()? as usize;
                        let v = *locals
                            .get(idx)
                            .ok_or(WasmSandboxError::MalformedSectionOrLeb128)?;
                        stack.push(v);
                    }
                    0x21 => {
                        let idx = cur.read_u32_leb()? as usize;
                        let v = stack.pop().ok_or_else(|| {
                            WasmSandboxError::ExecutionTrap("local.set underflow".into())
                        })?;
                        *locals
                            .get_mut(idx)
                            .ok_or(WasmSandboxError::MalformedSectionOrLeb128)? = v;
                    }
                    0x22 => {
                        let idx = cur.read_u32_leb()? as usize;
                        let v = *stack.last().ok_or_else(|| {
                            WasmSandboxError::ExecutionTrap("local.tee underflow".into())
                        })?;
                        *locals
                            .get_mut(idx)
                            .ok_or(WasmSandboxError::MalformedSectionOrLeb128)? = v;
                    }
                    0x23 => {
                        let idx = cur.read_u32_leb()? as usize;
                        let g = self
                            .globals
                            .get(idx)
                            .ok_or(WasmSandboxError::MalformedSectionOrLeb128)?;
                        stack.push(*g);
                    }
                    0x24 => {
                        let idx = cur.read_u32_leb()? as usize;
                        let mutable = self
                            .module
                            .globals
                            .get(idx)
                            .ok_or(WasmSandboxError::MalformedSectionOrLeb128)?
                            .mutable;
                        if !mutable {
                            return Err(WasmSandboxError::MalformedSectionOrLeb128);
                        }
                        let v = stack.pop().ok_or_else(|| {
                            WasmSandboxError::ExecutionTrap("global.set underflow".into())
                        })?;
                        self.globals[idx] = v;
                    }
                    0x28 => {
                        let _align = cur.read_u32_leb()?;
                        let off = cur.read_u32_leb()?;
                        let base = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("load underflow".into())
                            })?
                            .as_i32()?;
                        let addr = self.effective_addr(base, off, 4)?;
                        let val = i32::from_le_bytes([
                            self.memory[addr],
                            self.memory[addr + 1],
                            self.memory[addr + 2],
                            self.memory[addr + 3],
                        ]);
                        stack.push(WasmValue::i32(val));
                    }
                    0x2a => {
                        let _align = cur.read_u32_leb()?;
                        let off = cur.read_u32_leb()?;
                        let base = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("load underflow".into())
                            })?
                            .as_i32()?;
                        let addr = self.effective_addr(base, off, 4)?;
                        let bits = u32::from_le_bytes([
                            self.memory[addr],
                            self.memory[addr + 1],
                            self.memory[addr + 2],
                            self.memory[addr + 3],
                        ]);
                        stack.push(WasmValue::f32(f32::from_bits(bits)));
                    }
                    0x2c | 0x2d => {
                        let _align = cur.read_u32_leb()?;
                        let off = cur.read_u32_leb()?;
                        let base = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("load8 underflow".into())
                            })?
                            .as_i32()?;
                        let addr = self.effective_addr(base, off, 1)?;
                        let b = self.memory[addr];
                        let val = if op == 0x2c {
                            (b as i8) as i32
                        } else {
                            b as i32
                        };
                        stack.push(WasmValue::i32(val));
                    }
                    0x36 => {
                        let _align = cur.read_u32_leb()?;
                        let off = cur.read_u32_leb()?;
                        let val = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("store underflow".into())
                            })?
                            .as_i32()?;
                        let base = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("store underflow".into())
                            })?
                            .as_i32()?;
                        let addr = self.effective_addr(base, off, 4)?;
                        self.memory[addr..addr + 4].copy_from_slice(&val.to_le_bytes());
                    }
                    0x38 => {
                        let _align = cur.read_u32_leb()?;
                        let off = cur.read_u32_leb()?;
                        let val = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("fstore underflow".into())
                            })?
                            .as_f32()?;
                        let base = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("fstore underflow".into())
                            })?
                            .as_i32()?;
                        let addr = self.effective_addr(base, off, 4)?;
                        self.memory[addr..addr + 4].copy_from_slice(&val.to_bits().to_le_bytes());
                    }
                    0x3a => {
                        let _align = cur.read_u32_leb()?;
                        let off = cur.read_u32_leb()?;
                        let val = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("store8 underflow".into())
                            })?
                            .as_i32()?;
                        let base = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("store8 underflow".into())
                            })?
                            .as_i32()?;
                        let addr = self.effective_addr(base, off, 1)?;
                        self.memory[addr] = (val & 0xff) as u8;
                    }
                    0x3f => {
                        let _mem_idx = cur.read_u32_leb()?;
                        let pages = (self.memory.len() / WASM_PAGE_BYTES) as i32;
                        stack.push(WasmValue::i32(pages));
                    }
                    0x40 => {
                        let _mem_idx = cur.read_u32_leb()?;
                        let delta = stack
                            .pop()
                            .ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("grow underflow".into())
                            })?
                            .as_i32()?;
                        let current_pages = (self.memory.len() / WASM_PAGE_BYTES) as u32;
                        let delta_u32 = delta as u32;
                        let new_pages = current_pages.checked_add(delta_u32);
                        if delta < 0
                            || new_pages.map_or(true, |np| np > self.limits.max_memory_pages)
                        {
                            if self.limits.trap_on_grow_failure {
                                return Err(WasmSandboxError::MemoryGrowTrap {
                                    current_pages,
                                    delta_pages: delta_u32,
                                    max_pages: self.limits.max_memory_pages,
                                });
                            }
                            stack.push(WasmValue::i32(-1));
                        } else {
                            let np = new_pages.unwrap();
                            self.memory.resize((np as usize) * WASM_PAGE_BYTES, 0);
                            if self.memory.len() > self.peak_memory_bytes {
                                self.peak_memory_bytes = self.memory.len();
                            }
                            stack.push(WasmValue::i32(current_pages as i32));
                        }
                    }
                    0x41 => stack.push(WasmValue::i32(cur.read_i32_leb()?)),
                    0x42 => stack.push(WasmValue::i64(cur.read_i64_leb()?)),
                    0x43 => {
                        let b = cur.read_exact(4)?;
                        stack.push(WasmValue::f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
                    }
                    0x45 => {
                        let a = stack
                            .pop()
                            .ok_or_else(|| WasmSandboxError::ExecutionTrap("eqz".into()))?
                            .as_i32()?;
                        stack.push(WasmValue::i32(i32::from(a == 0)));
                    }
                    0x46..=0x4f => {
                        let b = stack
                            .pop()
                            .ok_or_else(|| WasmSandboxError::ExecutionTrap("cmp".into()))?
                            .as_i32()?;
                        let a = stack
                            .pop()
                            .ok_or_else(|| WasmSandboxError::ExecutionTrap("cmp".into()))?
                            .as_i32()?;
                        let res = match op {
                            0x46 => a == b,
                            0x47 => a != b,
                            0x48 => a < b,
                            0x49 => (a as u32) < (b as u32),
                            0x4a => a > b,
                            0x4b => (a as u32) > (b as u32),
                            0x4c => a <= b,
                            0x4d => (a as u32) <= (b as u32),
                            0x4e => a >= b,
                            0x4f => (a as u32) >= (b as u32),
                            _ => unreachable!(),
                        };
                        stack.push(WasmValue::i32(i32::from(res)));
                    }
                    0x6a..=0x78 => {
                        let b = stack
                            .pop()
                            .ok_or_else(|| WasmSandboxError::ExecutionTrap("alu".into()))?
                            .as_i32()?;
                        let a = stack
                            .pop()
                            .ok_or_else(|| WasmSandboxError::ExecutionTrap("alu".into()))?
                            .as_i32()?;
                        let res = match op {
                            0x6a => a.wrapping_add(b),
                            0x6b => a.wrapping_sub(b),
                            0x6c => a.wrapping_mul(b),
                            0x6d => a.checked_div(b).ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("div_s trap".into())
                            })?,
                            0x6e => (a as u32).checked_div(b as u32).ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("div_u trap".into())
                            })? as i32,
                            0x6f => a.checked_rem(b).ok_or_else(|| {
                                WasmSandboxError::ExecutionTrap("rem_s trap".into())
                            })?,
                            0x71 => a & b,
                            0x72 => a | b,
                            0x73 => a ^ b,
                            0x74 => a.wrapping_shl((b as u32) & 31),
                            0x75 => a.wrapping_shr((b as u32) & 31),
                            0x76 => ((a as u32).wrapping_shr((b as u32) & 31)) as i32,
                            _ => {
                                return Err(WasmSandboxError::ExecutionTrap(format!(
                                    "unsupported i32 opcode 0x{op:02x}"
                                )))
                            }
                        };
                        stack.push(WasmValue::i32(res));
                    }
                    _ => {
                        return Err(WasmSandboxError::ExecutionTrap(format!(
                            "unsupported wasm opcode 0x{op:02x}"
                        )));
                    }
                }
            }

            if stack.len() < result_arity {
                return Err(WasmSandboxError::ExecutionTrap(
                    "missing return value on wasm stack".to_string(),
                ));
            }
            Ok(stack.split_off(stack.len() - result_arity))
        }
    }

    pub(super) fn exec_trusted_fixture(
        policy: &SandboxedExtensionPolicyV1,
        wasm_bytes: &[u8],
        export_name: &str,
        invocation: ExperimentalTrustedFixtureCallV1<'_>,
    ) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
        let module_sha256 = SandboxedExtensionPolicyV1::sha256_module(wasm_bytes);
        if policy.allowed_module_hashes.is_empty()
            || !policy.allowed_module_hashes.contains(&module_sha256)
        {
            return Err(WasmSandboxError::UntrustedModuleHash {
                actual: module_sha256,
            });
        }
        if !policy.allowed_exports.is_empty() && !policy.allowed_exports.contains(export_name) {
            return Err(WasmSandboxError::ExportNotAllowed(export_name.to_string()));
        }

        let mut fuel = WasmFuelMeterV1::new(policy.limits.max_fuel);
        let module = decode_and_validate_module(wasm_bytes, &policy.limits, &mut fuel)?;
        let export_fn_idx = *module
            .exports
            .get(export_name)
            .ok_or_else(|| WasmSandboxError::ExportNotAllowed(export_name.to_string()))?;

        let mut vm = WasmVm::instantiate(&module, &policy.limits, fuel)?;
        if let Some(start_idx) = module.start_func {
            let _ = vm.call(start_idx, &[])?;
        }

        match invocation {
            ExperimentalTrustedFixtureCallV1::ByteTransform(input_payload) => {
                if input_payload.len() > vm.memory.len() {
                    return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                        addr: 0,
                        len: input_payload.len(),
                        mem_len: vm.memory.len(),
                    });
                }
                vm.memory[..input_payload.len()].copy_from_slice(input_payload);
                let input_len_i32 = i32::try_from(input_payload.len()).map_err(|_| {
                    WasmSandboxError::OutOfBoundsMemoryAccess {
                        addr: 0,
                        len: input_payload.len(),
                        mem_len: vm.memory.len(),
                    }
                })?;
                let results = vm.call(
                    export_fn_idx,
                    &[WasmValue::i32(0), WasmValue::i32(input_len_i32)],
                )?;
                let return_code = results
                    .first()
                    .copied()
                    .ok_or_else(|| {
                        WasmSandboxError::ExportSignatureMismatch(export_name.to_string())
                    })?
                    .as_i32()?;
                let output_bytes = if return_code < 0 {
                    return Err(WasmSandboxError::ExecutionTrap(format!(
                        "export '{export_name}' returned error code {return_code}"
                    )));
                } else {
                    read_guest_slice(&vm.memory, 0, return_code)?
                };
                Ok(WasmExecutionReceiptV1 {
                    module_sha256,
                    fuel_consumed: vm.fuel.consumed(),
                    peak_memory_bytes: vm.peak_memory_bytes,
                    return_code,
                    output_bytes,
                })
            }
            ExperimentalTrustedFixtureCallV1::ScalarI32(arg) => {
                let results = vm.call(export_fn_idx, &[WasmValue::i32(arg)])?;
                let ret = results
                    .first()
                    .copied()
                    .ok_or_else(|| {
                        WasmSandboxError::ExportSignatureMismatch(export_name.to_string())
                    })?
                    .as_i32()?;
                Ok(WasmExecutionReceiptV1 {
                    module_sha256,
                    fuel_consumed: vm.fuel.consumed(),
                    peak_memory_bytes: vm.peak_memory_bytes,
                    return_code: ret,
                    output_bytes: Vec::new(),
                })
            }
        }
    }
}

/// Invocation mode for internal test-only experimental fallback fixtures (`ST-09`).
#[cfg(test)]
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub enum ExperimentalTrustedFixtureCallV1<'a> {
    /// Invoke `(ptr: i32, len: i32) -> i32` over guest memory initialized with the given payload.
    ByteTransform(&'a [u8]),
    /// Invoke scalar `(arg: i32) -> i32`.
    ScalarI32(i32),
}

/// Fail-closed WebAssembly extension runner (`ADR-0018`, `W12-04`, `ST-09`).
///
/// Production entrypoints ([`Self::execute_byte_transform`] and [`Self::execute_scalar_i32`])
/// unconditionally fail closed with [`WasmSandboxError::UnsupportedUntrustedWasmSandbox`];
/// configure [`WasmExtensionExecutor`] (for example `WasmiExtensionExecutorV1` in
/// `adapters/swarm-torch-wasm`) to execute sandboxed extensions.
pub struct WasmSandboxRunnerV1;

impl WasmSandboxRunnerV1 {
    /// Legacy in-crate untrusted Wasm byte-transform execution is **disabled (fail-closed)**;
    /// configure an external [`WasmExtensionExecutor`] instead (`ST-09`).
    pub fn execute_byte_transform(
        _policy: &SandboxedExtensionPolicyV1,
        _wasm_bytes: &[u8],
        _export_name: &str,
        _input_payload: &[u8],
    ) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
        Err(WasmSandboxError::UnsupportedUntrustedWasmSandbox)
    }

    /// Legacy in-crate untrusted Wasm scalar execution is **disabled (fail-closed)**;
    /// configure an external [`WasmExtensionExecutor`] instead (`ST-09`).
    pub fn execute_scalar_i32(
        _policy: &SandboxedExtensionPolicyV1,
        _wasm_bytes: &[u8],
        _export_name: &str,
        _arg: i32,
    ) -> Result<(i32, WasmExecutionReceiptV1), WasmSandboxError> {
        Err(WasmSandboxError::UnsupportedUntrustedWasmSandbox)
    }

    /// **Experimental trusted local test fixture runner ONLY** (`#[cfg(test)]`, `#[doc(hidden)]`, `ST-09`).
    ///
    /// Must NEVER be called on untrusted manifests or in production scheduler/runner paths.
    #[cfg(test)]
    #[doc(hidden)]
    pub fn run_experimental_trusted_test_fixture_only(
        policy: &SandboxedExtensionPolicyV1,
        wasm_bytes: &[u8],
        export_name: &str,
        invocation: ExperimentalTrustedFixtureCallV1<'_>,
    ) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
        legacy_vm::exec_trusted_fixture(policy, wasm_bytes, export_name, invocation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_leb_u32(out: &mut Vec<u8>, mut val: u32) {
        loop {
            let mut b = (val & 0x7f) as u8;
            val >>= 7;
            if val != 0 {
                b |= 0x80;
            }
            out.push(b);
            if val == 0 {
                break;
            }
        }
    }

    fn push_section(module: &mut Vec<u8>, id: u8, payload: &[u8]) {
        module.push(id);
        push_leb_u32(module, payload.len() as u32);
        module.extend_from_slice(payload);
    }

    fn push_name(out: &mut Vec<u8>, name: &str) {
        push_leb_u32(out, name.len() as u32);
        out.extend_from_slice(name.as_bytes());
    }

    fn build_single_export_wasm(
        params: &[u8],
        results: &[u8],
        mem_pages: Option<(u32, Option<u32>)>,
        locals_i32: u32,
        export_name: &str,
        body_ops: &[u8],
    ) -> Vec<u8> {
        let mut m = WASM_MAGIC_AND_VERSION_V1.to_vec();

        // Section 1: Type
        let mut sec1 = vec![1]; // 1 type
        sec1.push(0x60);
        push_leb_u32(&mut sec1, params.len() as u32);
        sec1.extend_from_slice(params);
        push_leb_u32(&mut sec1, results.len() as u32);
        sec1.extend_from_slice(results);
        push_section(&mut m, 1, &sec1);

        // Section 3: Function (1 func -> type 0)
        push_section(&mut m, 3, &[1, 0]);

        // Section 5: Memory
        if let Some((init, max)) = mem_pages {
            let mut sec5 = vec![1];
            match max {
                None => {
                    sec5.push(0x00);
                    push_leb_u32(&mut sec5, init);
                }
                Some(mx) => {
                    sec5.push(0x01);
                    push_leb_u32(&mut sec5, init);
                    push_leb_u32(&mut sec5, mx);
                }
            }
            push_section(&mut m, 5, &sec5);
        }

        // Section 7: Export
        let mut sec7 = vec![1];
        push_name(&mut sec7, export_name);
        sec7.push(0x00); // func
        sec7.push(0x00); // idx 0
        push_section(&mut m, 7, &sec7);

        // Section 10: Code
        let mut body = Vec::new();
        if locals_i32 == 0 {
            body.push(0);
        } else {
            body.push(1);
            push_leb_u32(&mut body, locals_i32);
            body.push(0x7f);
        }
        body.extend_from_slice(body_ops);
        body.push(0x0b); // end

        let mut sec10 = vec![1];
        push_leb_u32(&mut sec10, body.len() as u32);
        sec10.extend_from_slice(&body);
        push_section(&mut m, 10, &sec10);
        m
    }

    #[test]
    fn wasm_sandbox_rejects_untrusted_hash_and_wasi_import() {
        let mut with_wasi = WASM_MAGIC_AND_VERSION_V1.to_vec();
        // Type () -> ()
        push_section(&mut with_wasi, 1, &[1, 0x60, 0, 0]);
        // Section 2: Import wasi_snapshot_preview1::fd_read
        let mut sec2 = vec![1];
        push_name(&mut sec2, "wasi_snapshot_preview1");
        push_name(&mut sec2, "fd_read");
        sec2.push(0x00);
        sec2.push(0x00);
        push_section(&mut with_wasi, 2, &sec2);

        // Static WasmSandboxRunnerV1 entrypoints MUST fail closed unconditionally even for pinned modules (`ST-09`).
        let pinned = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&with_wasi),
        );
        assert_eq!(
            WasmSandboxRunnerV1::execute_scalar_i32(&pinned, &with_wasi, "run", 0).unwrap_err(),
            WasmSandboxError::UnsupportedUntrustedWasmSandbox
        );
        assert_eq!(
            WasmSandboxRunnerV1::execute_byte_transform(&pinned, &with_wasi, "run", b"x")
                .unwrap_err(),
            WasmSandboxError::UnsupportedUntrustedWasmSandbox
        );

        // First: unallowlisted module fails BEFORE decoding WASI import on experimental fixture path
        let empty_policy = SandboxedExtensionPolicyV1::default();
        assert!(matches!(
            WasmSandboxRunnerV1::run_experimental_trusted_test_fixture_only(
                &empty_policy,
                &with_wasi,
                "run",
                ExperimentalTrustedFixtureCallV1::ScalarI32(0)
            ),
            Err(WasmSandboxError::UntrustedModuleHash { .. })
        ));

        // Even if explicitly pinned in hash allowlist, WASI import MUST fail closed!
        assert!(matches!(
            WasmSandboxRunnerV1::run_experimental_trusted_test_fixture_only(
                &pinned,
                &with_wasi,
                "run",
                ExperimentalTrustedFixtureCallV1::ScalarI32(0)
            ),
            Err(WasmSandboxError::ForbiddenImport { module, field })
                if module == "wasi_snapshot_preview1" && field == "fd_read"
        ));
    }

    #[test]
    fn wasm_sandbox_traps_infinite_loop_and_memory_grow_bomb() {
        // Infinite loop: loop { br 0 } end
        let loop_wasm = build_single_export_wasm(
            &[0x7f],
            &[0x7f],
            None,
            0,
            "spin",
            &[0x03, 0x40, 0x0c, 0x00, 0x0b, 0x41, 0x01],
        );
        let mut policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&loop_wasm),
        );
        policy.limits.max_fuel = 50;
        assert!(matches!(
            WasmSandboxRunnerV1::run_experimental_trusted_test_fixture_only(
                &policy,
                &loop_wasm,
                "spin",
                ExperimentalTrustedFixtureCallV1::ScalarI32(0)
            ),
            Err(WasmSandboxError::OutOfFuel { limit: 50, .. })
        ));

        // Memory grow bomb: push 32 pages, memory.grow(0)
        let grow_wasm = build_single_export_wasm(
            &[0x7f],
            &[0x7f],
            Some((1, None)),
            0,
            "grow_bomb",
            &[0x41, 0x20, 0x40, 0x00],
        );
        let mut grow_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&grow_wasm),
        );
        grow_policy.limits.max_memory_pages = 4;
        grow_policy.limits.trap_on_grow_failure = true;
        assert!(matches!(
            WasmSandboxRunnerV1::run_experimental_trusted_test_fixture_only(
                &grow_policy,
                &grow_wasm,
                "grow_bomb",
                ExperimentalTrustedFixtureCallV1::ScalarI32(0)
            ),
            Err(WasmSandboxError::MemoryGrowTrap {
                current_pages: 1,
                delta_pages: 32,
                max_pages: 4
            })
        ));
    }

    #[test]
    fn wasm_sandbox_executes_verified_byte_transform_and_checks_guest_slice_bounds() {
        // Function transform(ptr=local0, len=local1) -> len:
        // Loops i from 0..len and XORs mem[ptr + i] ^= 0x20 (ASCII case flip), returns len.
        let ops: Vec<u8> = vec![
            0x41, 0x00, // i32.const 0
            0x21, 0x02, // local.set 2 (i = 0)
            0x03, 0x40, // loop
            0x20, 0x02, // local.get 2 (i)
            0x20, 0x01, // local.get 1 (len)
            0x48, // i32.lt_s (i < len)
            0x04, 0x40, // if
            // addr = ptr + i
            0x20, 0x00, 0x20, 0x02, 0x6a, // val = mem.load8_u(addr) ^ 0x20
            0x20, 0x00, 0x20, 0x02, 0x6a, 0x2d, 0x00, 0x00, 0x41, 0x20, 0x73,
            // mem.store8(addr, val)
            0x3a, 0x00, 0x00, // i = i + 1
            0x20, 0x02, 0x41, 0x01, 0x6a, 0x21, 0x02, 0x0c, 0x01, // br 1 (continue loop!)
            0x0b, // end if
            0x0b, // end loop
            0x20, 0x01, // return len
        ];
        let wasm = build_single_export_wasm(
            &[0x7f, 0x7f],
            &[0x7f],
            Some((1, Some(1))),
            1,
            "transform",
            &ops,
        );
        let policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&wasm),
        );
        let receipt = WasmSandboxRunnerV1::run_experimental_trusted_test_fixture_only(
            &policy,
            &wasm,
            "transform",
            ExperimentalTrustedFixtureCallV1::ByteTransform(b"abcDEF"),
        )
        .unwrap();
        assert_eq!(receipt.output_bytes, b"ABCdef");
        assert!(receipt.fuel_consumed > 0);
        assert_eq!(receipt.peak_memory_bytes, WASM_PAGE_BYTES);

        // Host read_guest_slice overflow check
        assert!(matches!(
            read_guest_slice(&[0u8; 16], i32::MAX, 10),
            Err(WasmSandboxError::OutOfBoundsMemoryAccess { .. })
        ));
    }

    #[test]
    fn wasm_sandbox_traps_infinite_loop_in_start_section() {
        // Module with Start section pointing to func 0: () -> () { loop { br 0 } }
        // and exported func 1 "noop": (i32) -> i32 { i32.const 42 }.
        let mut m = WASM_MAGIC_AND_VERSION_V1.to_vec();
        // Sec 1: Type 0: () -> (), Type 1: (i32) -> i32
        push_section(&mut m, 1, &[2, 0x60, 0, 0, 0x60, 1, 0x7f, 1, 0x7f]);
        // Sec 3: Func 0 -> type 0, Func 1 -> type 1
        push_section(&mut m, 3, &[2, 0, 1]);
        // Sec 7: Export "noop" -> func 1
        let mut sec7 = vec![1];
        push_name(&mut sec7, "noop");
        sec7.push(0x00);
        sec7.push(0x01);
        push_section(&mut m, 7, &sec7);
        // Sec 8: Start func 0
        push_section(&mut m, 8, &[0]);
        // Sec 10: Code func 0 (loop br 0 end end) + func 1 (i32.const 42 end)
        let body0 = [0x00u8, 0x03, 0x40, 0x0c, 0x00, 0x0b, 0x0b];
        let body1 = [0x00u8, 0x41, 0x2a, 0x0b];
        let mut sec10 = vec![2];
        push_leb_u32(&mut sec10, body0.len() as u32);
        sec10.extend_from_slice(&body0);
        push_leb_u32(&mut sec10, body1.len() as u32);
        sec10.extend_from_slice(&body1);
        push_section(&mut m, 10, &sec10);

        let mut policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&m),
        );
        policy.limits.max_fuel = 128;
        let err = WasmSandboxRunnerV1::run_experimental_trusted_test_fixture_only(
            &policy,
            &m,
            "noop",
            ExperimentalTrustedFixtureCallV1::ScalarI32(0),
        )
        .unwrap_err();
        assert!(
            matches!(err, WasmSandboxError::OutOfFuel { limit: 128, .. }),
            "expected OutOfFuel from Start section infinite loop, got {err:?}"
        );
    }

    #[test]
    fn wasm_fixture_rejects_f64_types_cumulative_local_bomb_and_decode_fuel_exhaustion() {
        // 1. Module declaring f64 local MUST fail closed cleanly instead of coercing to i64(0).
        let mut f64_mod = WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut f64_mod, 1, &[1, 0x60, 0, 1, 0x7f]);
        push_section(&mut f64_mod, 3, &[1, 0]);
        let mut sec7 = vec![1];
        push_name(&mut sec7, "bad_f64");
        sec7.push(0x00);
        sec7.push(0x00);
        push_section(&mut f64_mod, 7, &sec7);
        // body: 1 decl (rep=1, type=0x7c f64), i32.const 0, end
        let body = [0x01u8, 0x01, 0x7c, 0x41, 0x00, 0x0b];
        let mut sec10 = vec![1];
        push_leb_u32(&mut sec10, body.len() as u32);
        sec10.extend_from_slice(&body);
        push_section(&mut f64_mod, 10, &sec10);
        let f64_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&f64_mod),
        );
        assert!(matches!(
            WasmSandboxRunnerV1::run_experimental_trusted_test_fixture_only(
                &f64_policy,
                &f64_mod,
                "bad_f64",
                ExperimentalTrustedFixtureCallV1::ScalarI32(0)
            ),
            Err(WasmSandboxError::ExecutionTrap(msg)) if msg.contains("f64")
        ));

        // 2. Cumulative local declarations > DEFAULT_WASM_MAX_LOCALS_PER_FUNCTION across 2 groups MUST fail.
        let mut locals_bomb = WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut locals_bomb, 1, &[1, 0x60, 0, 1, 0x7f]);
        push_section(&mut locals_bomb, 3, &[1, 0]);
        let mut sec7_b = vec![1];
        push_name(&mut sec7_b, "bomb");
        sec7_b.push(0x00);
        sec7_b.push(0x00);
        push_section(&mut locals_bomb, 7, &sec7_b);
        let mut bomb_body = vec![2u8]; // 2 decl groups: 3000 + 3000 > 4096
        push_leb_u32(&mut bomb_body, 3000);
        bomb_body.push(0x7f);
        push_leb_u32(&mut bomb_body, 3000);
        bomb_body.push(0x7f);
        bomb_body.extend_from_slice(&[0x41, 0x00, 0x0b]);
        let mut sec10_b = vec![1];
        push_leb_u32(&mut sec10_b, bomb_body.len() as u32);
        sec10_b.extend_from_slice(&bomb_body);
        push_section(&mut locals_bomb, 10, &sec10_b);
        let mut bomb_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&locals_bomb),
        );
        bomb_policy.limits.max_fuel = 100_000;
        assert!(matches!(
            WasmSandboxRunnerV1::run_experimental_trusted_test_fixture_only(
                &bomb_policy,
                &locals_bomb,
                "bomb",
                ExperimentalTrustedFixtureCallV1::ScalarI32(0)
            ),
            Err(WasmSandboxError::ExecutionTrap(msg)) if msg.contains("local count")
        ));

        // 3. Tiny max_fuel triggers OutOfFuel during decode/local-allocation before execution!
        let ok_mod = build_single_export_wasm(&[0x7f], &[0x7f], None, 32, "id", &[0x20, 0x00]);
        let mut low_fuel_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&ok_mod),
        );
        low_fuel_policy.limits.max_fuel = 5;
        assert!(matches!(
            WasmSandboxRunnerV1::run_experimental_trusted_test_fixture_only(
                &low_fuel_policy,
                &ok_mod,
                "id",
                ExperimentalTrustedFixtureCallV1::ScalarI32(7)
            ),
            Err(WasmSandboxError::OutOfFuel { limit: 5, .. })
        ));
    }

    #[test]
    fn wasm_store_limits_v1_json_backward_compat_defaults_max_start_fuel_and_binds_fingerprint() {
        let legacy_limits_json = r#"{
            "max_module_bytes": 1048576,
            "max_memory_pages": 16,
            "max_table_elements": 1024,
            "max_memories": 1,
            "max_tables": 1,
            "max_fuel": 100000,
            "trap_on_grow_failure": true,
            "timeout_ms": 5000
        }"#;
        let parsed_limits: WasmStoreLimitsV1 = serde_json::from_str(legacy_limits_json).unwrap();
        assert_eq!(parsed_limits.max_start_fuel, DEFAULT_WASM_MAX_START_FUEL);
        assert_eq!(parsed_limits, WasmStoreLimitsV1::default());

        let legacy_policy_json = r#"{
            "allowed_module_hashes": [],
            "allowed_exports": [],
            "limits": {
                "max_module_bytes": 1048576,
                "max_memory_pages": 16,
                "max_table_elements": 1024,
                "max_memories": 1,
                "max_tables": 1,
                "max_fuel": 100000,
                "trap_on_grow_failure": true,
                "timeout_ms": 5000
            }
        }"#;
        let parsed_policy: SandboxedExtensionPolicyV1 =
            serde_json::from_str(legacy_policy_json).unwrap();
        assert_eq!(
            parsed_policy.limits.max_start_fuel,
            DEFAULT_WASM_MAX_START_FUEL
        );
        assert_eq!(
            parsed_policy.policy_fingerprint_hex(),
            SandboxedExtensionPolicyV1::default().policy_fingerprint_hex()
        );

        let zero_start_json = legacy_limits_json.replace(
            "\"trap_on_grow_failure\": true",
            "\"max_start_fuel\": 0,\n            \"trap_on_grow_failure\": true",
        );
        let zero_start_limits: WasmStoreLimitsV1 = serde_json::from_str(&zero_start_json).unwrap();
        assert_eq!(zero_start_limits.max_start_fuel, 0);

        let ok_mod = build_single_export_wasm(&[0x7f], &[0x7f], None, 0, "id", &[0x20, 0x00]);
        let mut policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&ok_mod),
        );
        let default_fp = policy.policy_fingerprint_hex();
        policy.limits.max_start_fuel = 0;
        let zero_start_fp = policy.policy_fingerprint_hex();
        assert_ne!(default_fp, zero_start_fp);

        // `max_start_fuel == 0` with `max_fuel > 0` passes preflight so modules without `start` run,
        // whereas `max_fuel == 0` fails preflight unconditionally.
        assert!(preflight_wasm_extension_v1(&policy, &ok_mod, "id").is_ok());
        policy.limits.max_fuel = 0;
        assert_eq!(
            preflight_wasm_extension_v1(&policy, &ok_mod, "id").unwrap_err(),
            WasmSandboxError::OutOfFuel {
                consumed: 0,
                limit: 0,
            }
        );
    }

    #[test]
    fn wave16c_guest_abi_v1_policy_fingerprint_json_compat_and_unpack_ptr_len_i64() {
        // Default JSON deserialization preserves RawMem0V0 and DEFAULT_WASM_MAX_OUTPUT_BYTES.
        let default_policy = SandboxedExtensionPolicyV1::default();
        assert_eq!(default_policy.guest_abi, WasmGuestAbiV1::RawMem0V0);
        assert_eq!(
            default_policy.limits.max_output_bytes,
            DEFAULT_WASM_MAX_OUTPUT_BYTES
        );

        let alloc_policy = default_policy
            .clone()
            .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_ne!(
            default_policy.policy_fingerprint_hex(),
            alloc_policy.policy_fingerprint_hex()
        );
        let serialized = serde_json::to_string(&alloc_policy).unwrap();
        assert!(serialized.contains("\"swarm_torch_alloc_v1\""));
        let roundtrip: SandboxedExtensionPolicyV1 = serde_json::from_str(&serialized).unwrap();
        assert_eq!(roundtrip, alloc_policy);

        let mut smaller_out_policy = alloc_policy.clone();
        smaller_out_policy.limits.max_output_bytes = 1_024;
        assert_ne!(
            alloc_policy.policy_fingerprint_hex(),
            smaller_out_policy.policy_fingerprint_hex()
        );

        // unpack_guest_ptr_len_i64 valid cases
        let packed = ((4096u64) << 32) | 20u64;
        assert_eq!(unpack_guest_ptr_len_i64(packed as i64), Ok((4096, 20)));
        assert_eq!(unpack_guest_ptr_len_i64(0), Ok((0, 0)));
        assert_eq!(
            unpack_guest_ptr_len_i64(((i32::MAX as u64) << 32 | (i32::MAX as u64)) as i64),
            Ok((i32::MAX, i32::MAX))
        );

        // Negative sign-extended i64 error code (-7i64 -> high = -1, low = -7 -> code = -7)
        assert_eq!(
            unpack_guest_ptr_len_i64(-7i64),
            Err(WasmSandboxError::GuestTransformFailed {
                export: String::new(),
                code: -7,
            })
        );
        // Packed negative high word ((-9i32 as u32 as u64) << 32 -> code = -9)
        let neg_high = ((-9i32 as u32 as u64) << 32) as i64;
        assert_eq!(
            unpack_guest_ptr_len_i64(neg_high),
            Err(WasmSandboxError::GuestTransformFailed {
                export: String::new(),
                code: -9,
            })
        );
        // Negative low word with non-negative high word -> OutOfBoundsMemoryAccess
        let neg_low = ((4096u64 << 32) | 0x8000_0000u64) as i64;
        assert!(matches!(
            unpack_guest_ptr_len_i64(neg_low),
            Err(WasmSandboxError::OutOfBoundsMemoryAccess { addr: 4096, .. })
        ));
    }

    #[test]
    fn wave16c_adversarial_unpack_i64_wasm_error_to_io_and_reserved_export_preflight() {
        // 1. Boundary unpack_guest_ptr_len_i64 edge values
        assert_eq!(
            unpack_guest_ptr_len_i64(-1i64),
            Err(WasmSandboxError::GuestTransformFailed {
                export: String::new(),
                code: -1,
            })
        );
        assert_eq!(
            unpack_guest_ptr_len_i64(i64::from(i32::MIN)),
            Err(WasmSandboxError::GuestTransformFailed {
                export: String::new(),
                code: i32::MIN,
            })
        );
        assert_eq!(
            unpack_guest_ptr_len_i64(i64::MIN),
            Err(WasmSandboxError::GuestTransformFailed {
                export: String::new(),
                code: i32::MIN,
            })
        );
        assert_eq!(
            unpack_guest_ptr_len_i64(((-1i32 as u32 as u64) << 32 | 5u64) as i64),
            Err(WasmSandboxError::GuestTransformFailed {
                export: String::new(),
                code: -1,
            })
        );
        assert!(matches!(
            unpack_guest_ptr_len_i64(i64::MAX),
            Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                addr,
                len,
                mem_len: 0,
            }) if addr == i32::MAX as u64 && len == u32::MAX as usize
        ));
        assert_eq!(unpack_guest_ptr_len_i64((1u64 << 32) as i64), Ok((1, 0)));

        // 2. Explicit max_output_bytes == 0 deserializes as 0 (not overwritten by default 16 MiB)
        let zero_out_json = r#"{
            "allowed_module_hashes": [],
            "allowed_exports": [],
            "guest_abi": "swarm_torch_alloc_v1",
            "limits": {
                "max_module_bytes": 1048576,
                "max_memory_pages": 16,
                "max_table_elements": 1024,
                "max_memories": 1,
                "max_tables": 1,
                "max_fuel": 100000,
                "max_output_bytes": 0,
                "trap_on_grow_failure": true,
                "timeout_ms": 5000
            }
        }"#;
        let parsed: SandboxedExtensionPolicyV1 = serde_json::from_str(zero_out_json).unwrap();
        assert_eq!(parsed.guest_abi, WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(parsed.limits.max_output_bytes, 0);
        assert_eq!(parsed.limits.max_start_fuel, DEFAULT_WASM_MAX_START_FUEL);

        // Invalid guest_abi string fails closed during JSON deserialization
        let bad_abi_json = zero_out_json.replace("swarm_torch_alloc_v1", "unknown_abi_v9");
        assert!(serde_json::from_str::<SandboxedExtensionPolicyV1>(&bad_abi_json).is_err());

        // 3. wasm_error_to_io mapping & Display coverage for all three ADR-0025 error variants
        let io_fail = wasm_error_to_io(WasmSandboxError::GuestTransformFailed {
            export: "transform".to_string(),
            code: -42,
        });
        assert_eq!(io_fail.kind(), std::io::ErrorKind::Other);
        assert!(io_fail.to_string().contains("-42"));

        let io_big = wasm_error_to_io(WasmSandboxError::OutputTooLarge {
            len: 2048,
            max: 1024,
        });
        assert_eq!(io_big.kind(), std::io::ErrorKind::InvalidInput);
        assert!(io_big.to_string().contains("2048"));

        let io_alias = wasm_error_to_io(WasmSandboxError::OutputAliasingViolation {
            in_ptr: 4096,
            in_len: 100,
            out_ptr: 4106,
            out_len: 20,
        });
        assert_eq!(io_alias.kind(), std::io::ErrorKind::InvalidInput);
        assert!(io_alias.to_string().contains("4106"));

        // 4. Canonical preflight_wasm_extension_v1 must reject reserved ABI lifecycle exports
        //    under SwarmTorchAllocV1 (prevents Scheduler admission, scalar bypass, and pre-validation start execution).
        let dummy_mod = build_single_export_wasm(&[0x7f], &[0x7f], None, 0, "id", &[0x20, 0x00]);
        let dummy_hash = SandboxedExtensionPolicyV1::sha256_module(&dummy_mod);
        let alloc_policy = SandboxedExtensionPolicyV1::with_allowed_hash(dummy_hash)
            .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        for reserved in [
            SWARMTORCH_WASM_ALLOC_EXPORT_V1,
            SWARMTORCH_WASM_DEALLOC_EXPORT_V1,
        ] {
            let err = preflight_wasm_extension_v1(&alloc_policy, &dummy_mod, reserved).unwrap_err();
            assert!(
                matches!(
                    err,
                    WasmSandboxError::ExportNotAllowed(ref name)
                        | WasmSandboxError::ExportSignatureMismatch(ref name)
                        if name == reserved
                ),
                "preflight must reject reserved ABI export '{reserved}' under SwarmTorchAllocV1, got {err:?}"
            );
        }
    }

    #[test]
    fn wave16c_signoff_allowed_exports_reserved_override_rawmem0_compat_and_node_resolver() {
        use std::collections::BTreeMap;
        use swarm_torch_core::run_graph::{CanonParams, CanonValue};

        let dummy_mod =
            build_single_export_wasm(&[0x7f], &[0x7f], None, 0, "transform", &[0x20, 0x00]);
        let dummy_hash = SandboxedExtensionPolicyV1::sha256_module(&dummy_mod);
        let dummy_hex = crate::artifacts::hex_lower(&dummy_hash);

        // 1. Even if a caller explicitly lists reserved ABI exports inside `allowed_exports`,
        //    `preflight_wasm_extension_v1` MUST still reject them under SwarmTorchAllocV1.
        let mut policy_explicit_reserved =
            SandboxedExtensionPolicyV1::with_allowed_hash(dummy_hash)
                .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        policy_explicit_reserved
            .allowed_exports
            .insert(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string());
        policy_explicit_reserved
            .allowed_exports
            .insert(SWARMTORCH_WASM_DEALLOC_EXPORT_V1.to_string());
        for reserved in [
            SWARMTORCH_WASM_ALLOC_EXPORT_V1,
            SWARMTORCH_WASM_DEALLOC_EXPORT_V1,
        ] {
            assert_eq!(
                preflight_wasm_extension_v1(&policy_explicit_reserved, &dummy_mod, reserved)
                    .unwrap_err(),
                WasmSandboxError::ExportNotAllowed(reserved.to_string())
            );
        }

        // 2. Non-empty `allowed_exports = {"transform"}` (omitting lifecycle exports) passes preflight.
        let mut policy_transform_only = SandboxedExtensionPolicyV1::with_allowed_hash(dummy_hash)
            .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        policy_transform_only
            .allowed_exports
            .insert("transform".to_string());
        assert_eq!(
            preflight_wasm_extension_v1(&policy_transform_only, &dummy_mod, "transform").unwrap(),
            dummy_hash
        );

        // 3. Legacy RawMem0V0 does NOT reserve `swarmtorch_alloc`/`swarmtorch_dealloc` export names.
        let raw_policy = SandboxedExtensionPolicyV1::with_allowed_hash(dummy_hash);
        assert_eq!(
            preflight_wasm_extension_v1(&raw_policy, &dummy_mod, SWARMTORCH_WASM_ALLOC_EXPORT_V1)
                .unwrap(),
            dummy_hash
        );

        // 4. Canonical `resolve_and_preflight_wasm_node_v1` (used by Scheduler + NativeOpRunner)
        //    rejects reserved ABI exports under SwarmTorchAllocV1 with PermissionDenied.
        let mut registry = BTreeMap::new();
        registry.insert(dummy_hash, dummy_mod);
        let alloc_policy = SandboxedExtensionPolicyV1::with_allowed_hash(dummy_hash)
            .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        for reserved in [
            SWARMTORCH_WASM_ALLOC_EXPORT_V1,
            SWARMTORCH_WASM_DEALLOC_EXPORT_V1,
        ] {
            let mut params = CanonParams::new();
            params.insert(
                "module_sha256".to_string(),
                CanonValue::Str(dummy_hex.clone()),
            );
            params.insert("export".to_string(), CanonValue::Str(reserved.to_string()));
            let err =
                resolve_and_preflight_wasm_node_v1(&alloc_policy, &params, &registry).unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
            assert!(err.to_string().contains(reserved));
        }
    }
}
