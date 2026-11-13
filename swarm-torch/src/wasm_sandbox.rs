//! Deterministic, zero-unsafe WebAssembly (`MVP`) sandbox & fuel-metered bytecode VM (`ADR-0018`, `W12-04`).
//!
//! Enforces strict fail-closed pre-execution & runtime confinement without adding heavy JIT/Wasmtime
//! dependency graphs to root `Cargo.lock`:
//! 1. **Cryptographic allowlist BEFORE parsing:** Computes `SHA-256(wasm_bytes)` and verifies
//!    membership in `allowed_module_hashes` ([`crate::wasm_sandbox::SandboxedExtensionPolicyV1`])
//!    before touching untrusted module bytes.
//! 2. **Strict Wasm MVP decoding & import quarantine:** Rejects malformed LEB128/sections, ALL WASI
//!    or host OS imports (`wasi_snapshot_preview1`, `wasi_unstable`, filesystem, sockets, clocks,
//!    randomness), multiple memories/tables, and oversized initial memory/table declarations.
//! 3. **Deterministic execution traps:** Enforces [`crate::wasm_sandbox::WasmFuelMeterV1`] (every opcode consumes fuel),
//!    capped linear memory growth with `trap_on_grow_failure: true` (`MemoryGrowTrap`), call-depth
//!    caps (`64`), `Start` section execution before exports, and checked guest `(ptr, len)` slice
//!    bounds (`u32::checked_add`).

use std::collections::BTreeSet;
use std::fmt;

use sha2::{Digest, Sha256};

/// Canonical 8-byte WebAssembly binary magic + version header (`\0asm\x01\0\0\0`).
pub const WASM_MAGIC_AND_VERSION_V1: [u8; 8] = *b"\0asm\x01\0\0\0";

/// WebAssembly fixed linear page size (`65,536` bytes = 64 KiB).
pub const WASM_PAGE_BYTES: usize = 65_536;

/// Default maximum call stack depth enforced against unbounded guest recursion.
pub const DEFAULT_WASM_MAX_CALL_DEPTH: usize = 64;

/// Errors raised during pre-parse verification, static validation, or sandboxed Wasm execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmSandboxError {
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
    /// Division by zero, integer overflow trap, `unreachable`, or call stack overflow.
    ExecutionTrap(String),
}

impl fmt::Display for WasmSandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Self::ExecutionTrap(reason) => write!(f, "wasm execution trap: {reason}"),
        }
    }
}

impl std::error::Error for WasmSandboxError {}

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
        }
    }

    /// Compute canonical SHA-256 digest of a raw `.wasm` module binary.
    pub fn sha256_module(wasm_bytes: &[u8]) -> [u8; 32] {
        let digest = Sha256::digest(wasm_bytes);
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest[..]);
        out
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

// ---------------------------------------------------------------------------
// Pure-Rust WebAssembly MVP Binary Decoder & Deterministic Stack Interpreter
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WasmValType {
    I32,
    I64,
    F32,
    F64,
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
        let s =
            std::str::from_utf8(slice).map_err(|_| WasmSandboxError::MalformedSectionOrLeb128)?;
        Ok(s.to_string())
    }
    fn read_val_type(&mut self) -> Result<WasmValType, WasmSandboxError> {
        match self.read_u8()? {
            0x7f => Ok(WasmValType::I32),
            0x7e => Ok(WasmValType::I64),
            0x7d => Ok(WasmValType::F32),
            0x7c => Ok(WasmValType::F64),
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
                    if sc.read_u8()? != 0x60 {
                        return Err(WasmSandboxError::MalformedSectionOrLeb128);
                    }
                    let p_len = sc.read_u32_leb()?;
                    let mut params = Vec::with_capacity(p_len as usize);
                    for _ in 0..p_len {
                        params.push(sc.read_val_type()?);
                    }
                    let r_len = sc.read_u32_leb()?;
                    let mut results = Vec::with_capacity(r_len as usize);
                    for _ in 0..r_len {
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
                    let body_len = sc.read_u32_leb()? as usize;
                    let body_slice = sc.read_exact(body_len)?;
                    let mut bc = Cursor::new(body_slice);
                    let local_decls = bc.read_u32_leb()?;
                    let mut locals = Vec::new();
                    for _ in 0..local_decls {
                        let rep = bc.read_u32_leb()?;
                        let ty = bc.read_val_type()?;
                        for _ in 0..rep.min(4096) {
                            locals.push(ty);
                        }
                        if rep > 4096 {
                            return Err(WasmSandboxError::MalformedSectionOrLeb128);
                        }
                    }
                    let expr = bc.bytes[bc.pos..].to_vec();
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
            fuel: WasmFuelMeterV1::new(limits.max_fuel),
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
                WasmValType::F64 => WasmValue::i64(0),
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
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("stack underflow".into()))?
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
                    let top = frames
                        .pop()
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("frame underflow".into()))?;
                    cur.pos = top.end_pc + 1;
                }
                0x0b => {
                    let popped = frames
                        .pop()
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("frame underflow".into()))?;
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
                    stack
                        .pop()
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("drop underflow".into()))?;
                }
                0x1b => {
                    let c = stack
                        .pop()
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("select underflow".into()))?
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
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("load underflow".into()))?
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
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("load underflow".into()))?
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
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("load8 underflow".into()))?
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
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("store underflow".into()))?
                        .as_i32()?;
                    let base = stack
                        .pop()
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("store underflow".into()))?
                        .as_i32()?;
                    let addr = self.effective_addr(base, off, 4)?;
                    self.memory[addr..addr + 4].copy_from_slice(&val.to_le_bytes());
                }
                0x38 => {
                    let _align = cur.read_u32_leb()?;
                    let off = cur.read_u32_leb()?;
                    let val = stack
                        .pop()
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("fstore underflow".into()))?
                        .as_f32()?;
                    let base = stack
                        .pop()
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("fstore underflow".into()))?
                        .as_i32()?;
                    let addr = self.effective_addr(base, off, 4)?;
                    self.memory[addr..addr + 4].copy_from_slice(&val.to_bits().to_le_bytes());
                }
                0x3a => {
                    let _align = cur.read_u32_leb()?;
                    let off = cur.read_u32_leb()?;
                    let val = stack
                        .pop()
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("store8 underflow".into()))?
                        .as_i32()?;
                    let base = stack
                        .pop()
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("store8 underflow".into()))?
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
                        .ok_or_else(|| WasmSandboxError::ExecutionTrap("grow underflow".into()))?
                        .as_i32()?;
                    let current_pages = (self.memory.len() / WASM_PAGE_BYTES) as u32;
                    let delta_u32 = delta as u32;
                    let new_pages = current_pages.checked_add(delta_u32);
                    if delta < 0 || new_pages.map_or(true, |np| np > self.limits.max_memory_pages) {
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
                        0x6d => a
                            .checked_div(b)
                            .ok_or_else(|| WasmSandboxError::ExecutionTrap("div_s trap".into()))?,
                        0x6e => (a as u32)
                            .checked_div(b as u32)
                            .ok_or_else(|| WasmSandboxError::ExecutionTrap("div_u trap".into()))?
                            as i32,
                        0x6f => a
                            .checked_rem(b)
                            .ok_or_else(|| WasmSandboxError::ExecutionTrap("rem_s trap".into()))?,
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

/// Fail-closed WebAssembly extension runner (`ADR-0018`, `W12-04`).
pub struct WasmSandboxRunnerV1;

impl WasmSandboxRunnerV1 {
    /// Validate `wasm_bytes` against `policy` (hash allowlist BEFORE decode + import/memory/table
    /// checks) and invoke `(ptr: i32, len: i32) -> i32` byte-transform export cleanly.
    pub fn execute_byte_transform(
        policy: &SandboxedExtensionPolicyV1,
        wasm_bytes: &[u8],
        export_name: &str,
        input_payload: &[u8],
    ) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
        let module_sha256 = SandboxedExtensionPolicyV1::sha256_module(wasm_bytes);
        if !policy.allowed_module_hashes.contains(&module_sha256) {
            return Err(WasmSandboxError::UntrustedModuleHash {
                actual: module_sha256,
            });
        }
        if !policy.allowed_exports.is_empty() && !policy.allowed_exports.contains(export_name) {
            return Err(WasmSandboxError::ExportNotAllowed(export_name.to_string()));
        }

        let module = decode_and_validate_module(wasm_bytes, &policy.limits)?;
        let export_fn_idx = *module
            .exports
            .get(export_name)
            .ok_or_else(|| WasmSandboxError::ExportNotAllowed(export_name.to_string()))?;

        let mut vm = WasmVm::instantiate(&module, &policy.limits)?;

        // Always execute optional Section 8 start function before exports under fuel/memory limits.
        if let Some(start_idx) = module.start_func {
            let _ = vm.call(start_idx, &[])?;
        }

        // Write input payload safely at offset 0 of linear memory.
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
            .ok_or_else(|| WasmSandboxError::ExportSignatureMismatch(export_name.to_string()))?
            .as_i32()?;

        // Checked slice read `[0..return_code)` when positive (negative return code signals trap/status)
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

    /// Validate `wasm_bytes` against `policy` and invoke scalar export `(arg: i32) -> i32`.
    pub fn execute_scalar_i32(
        policy: &SandboxedExtensionPolicyV1,
        wasm_bytes: &[u8],
        export_name: &str,
        arg: i32,
    ) -> Result<(i32, WasmExecutionReceiptV1), WasmSandboxError> {
        let module_sha256 = SandboxedExtensionPolicyV1::sha256_module(wasm_bytes);
        if !policy.allowed_module_hashes.contains(&module_sha256) {
            return Err(WasmSandboxError::UntrustedModuleHash {
                actual: module_sha256,
            });
        }
        if !policy.allowed_exports.is_empty() && !policy.allowed_exports.contains(export_name) {
            return Err(WasmSandboxError::ExportNotAllowed(export_name.to_string()));
        }

        let module = decode_and_validate_module(wasm_bytes, &policy.limits)?;
        let export_fn_idx = *module
            .exports
            .get(export_name)
            .ok_or_else(|| WasmSandboxError::ExportNotAllowed(export_name.to_string()))?;

        let mut vm = WasmVm::instantiate(&module, &policy.limits)?;
        if let Some(start_idx) = module.start_func {
            let _ = vm.call(start_idx, &[])?;
        }

        let results = vm.call(export_fn_idx, &[WasmValue::i32(arg)])?;
        let ret = results
            .first()
            .copied()
            .ok_or_else(|| WasmSandboxError::ExportSignatureMismatch(export_name.to_string()))?
            .as_i32()?;
        let receipt = WasmExecutionReceiptV1 {
            module_sha256,
            fuel_consumed: vm.fuel.consumed(),
            peak_memory_bytes: vm.peak_memory_bytes,
            return_code: ret,
            output_bytes: Vec::new(),
        };
        Ok((ret, receipt))
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

        // First: unallowlisted module fails BEFORE decoding WASI import
        let empty_policy = SandboxedExtensionPolicyV1::default();
        assert!(matches!(
            WasmSandboxRunnerV1::execute_scalar_i32(&empty_policy, &with_wasi, "run", 0),
            Err(WasmSandboxError::UntrustedModuleHash { .. })
        ));

        // Even if explicitly pinned in hash allowlist, WASI import MUST fail closed!
        let pinned = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&with_wasi),
        );
        assert!(matches!(
            WasmSandboxRunnerV1::execute_scalar_i32(&pinned, &with_wasi, "run", 0),
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
            WasmSandboxRunnerV1::execute_scalar_i32(&policy, &loop_wasm, "spin", 0),
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
            WasmSandboxRunnerV1::execute_scalar_i32(&grow_policy, &grow_wasm, "grow_bomb", 0),
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
        let receipt =
            WasmSandboxRunnerV1::execute_byte_transform(&policy, &wasm, "transform", b"abcDEF")
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
        let err = WasmSandboxRunnerV1::execute_scalar_i32(&policy, &m, "noop", 0).unwrap_err();
        assert!(
            matches!(err, WasmSandboxError::OutOfFuel { limit: 128, .. }),
            "expected OutOfFuel from Start section infinite loop, got {err:?}"
        );
    }
}
