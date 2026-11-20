#![forbid(unsafe_code)]
//! Isolated `wasmi 2.0` WebAssembly extension sandbox adapter for SwarmTorch (`ST-09`, `Decision A2`, `ADR-0025`).
//!
//! **Maturity Tier:** `Adversarial conformance passed` (scoped to `RawMem0V0` and `SwarmTorchAllocV1`) —
//! integrates `wasmi 2.0.0` behind [`swarm_torch::wasm_sandbox::WasmExtensionExecutor`] with end-to-end
//! `Scheduler` + `NativeOpRunner` tests.
//!
//! **Guest ABI & `wasmi 2.0.0` Engine Scope (`Decision A2`, `ADR-0025`):**
//! - Byte-transform exports (`execute_byte_transform`) support two explicit ABI contracts via
//!   [`SandboxedExtensionPolicyV1::guest_abi`](swarm_torch::wasm_sandbox::SandboxedExtensionPolicyV1::guest_abi):
//!   1. [`WasmGuestAbiV1::RawMem0V0`]: legacy scratch-offset-`0` contract (`export(ptr: i32, len: i32) -> i32`).
//!   2. [`WasmGuestAbiV1::SwarmTorchAllocV1`]: structured guest allocator contract
//!      (`swarmtorch_alloc(len: i32) -> i32`, `export(in_ptr: i32, in_len: i32) -> i64`,
//!      `swarmtorch_dealloc(ptr: i32, len: i32)`), preserving `.rodata`/`.data` segments at offset `0`
//!      and enforcing strict ownership/aliasing and `max_output_bytes` bounds before host allocation.
//! - `wasmi 2.0.0` executes `Module::new` (eager compilation) and `Linker::instantiate_and_start`
//!   synchronously without streaming compile interruption or resumable `start` ticks; `timeout_ms`
//!   is checked pre-compile, post-compile, post-start, and between resumable `FUEL_SLICE_UNITS`
//!   export slices, while synchronous `start` execution is bounded explicitly by
//!   [`WasmStoreLimitsV1::max_start_fuel`](swarm_torch::wasm_sandbox::WasmStoreLimitsV1::max_start_fuel).
//! - Both exported and un-exported internal tables/memories are enforced via `StoreLimits` + stack
//!   depth (`DEFAULT_WASM_MAX_CALL_DEPTH`) and value-stack (`64 KiB`) engine hardening.

use std::sync::OnceLock;
use std::time::Instant;

use swarm_torch::wasm_sandbox::{
    preflight_wasm_extension_v1, read_guest_slice, unpack_guest_ptr_len_i64,
    SandboxedExtensionPolicyV1, WasmExecutionReceiptV1, WasmExtensionExecutor, WasmGuestAbiV1,
    WasmSandboxError, DEFAULT_WASM_MAX_CALL_DEPTH, SWARMTORCH_WASM_ALLOC_EXPORT_V1,
    SWARMTORCH_WASM_DEALLOC_EXPORT_V1, WASM_PAGE_BYTES,
};
use wasmi::{
    errors::{EnforcedLimitsError, ErrorKind, InstantiationError},
    CompilationMode, Config, EnforcedLimits, Engine, Extern, ExternType, Instance, Linker, Memory,
    Module, Store, StoreLimits, StoreLimitsBuilder, TrapCode, TypedResumableCall, ValType,
};

/// Deterministic fuel slice charged between wall-clock timeout checks (`ST-09`).
const FUEL_SLICE_UNITS: u64 = 10_000;

/// Minimum initial `wasmi` value stack height (1 KiB).
const DEFAULT_WASM_MIN_STACK_HEIGHT_BYTES: usize = 1_024;

/// Maximum `wasmi` value stack height (64 KiB, tightened from wasmi's 1 MiB default).
const DEFAULT_WASM_MAX_STACK_HEIGHT_BYTES: usize = 64 * 1024;

static HARDENED_ENGINE: OnceLock<Engine> = OnceLock::new();

fn shared_hardened_engine() -> &'static Engine {
    HARDENED_ENGINE.get_or_init(build_hardened_engine)
}

/// Isolated `wasmi 2.0` implementation of [`WasmExtensionExecutor`] (`ST-09`).
#[derive(Debug, Clone, Copy, Default)]
pub struct WasmiExtensionExecutorV1;

impl WasmiExtensionExecutorV1 {
    /// Construct a stateless `wasmi` extension executor.
    pub const fn new() -> Self {
        Self
    }
}

impl WasmExtensionExecutor for WasmiExtensionExecutorV1 {
    fn execute_byte_transform(
        &self,
        policy: &SandboxedExtensionPolicyV1,
        wasm_bytes: &[u8],
        export_name: &str,
        input_payload: &[u8],
    ) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
        let expected_abi = match policy.guest_abi {
            WasmGuestAbiV1::RawMem0V0 => ExpectedExportAbi::ByteTransformRawMem0V0,
            WasmGuestAbiV1::SwarmTorchAllocV1 => ExpectedExportAbi::ByteTransformAllocV1,
        };
        let mut prepared = prepare_instance(policy, wasm_bytes, export_name, expected_abi)?;
        let memory = find_single_memory(&prepared.instance, &prepared.store).ok_or_else(|| {
            WasmSandboxError::ExecutionTrap(format!(
                "byte-transform export '{export_name}' requires an exported linear memory"
            ))
        })?;

        match policy.guest_abi {
            WasmGuestAbiV1::RawMem0V0 => execute_byte_transform_raw_mem0_v0(
                &mut prepared,
                memory,
                policy,
                export_name,
                input_payload,
            ),
            WasmGuestAbiV1::SwarmTorchAllocV1 => execute_byte_transform_alloc_v1(
                &mut prepared,
                memory,
                policy,
                export_name,
                input_payload,
            ),
        }
    }

    fn execute_scalar_i32(
        &self,
        policy: &SandboxedExtensionPolicyV1,
        wasm_bytes: &[u8],
        export_name: &str,
        arg: i32,
    ) -> Result<(i32, WasmExecutionReceiptV1), WasmSandboxError> {
        let mut prepared = prepare_instance(
            policy,
            wasm_bytes,
            export_name,
            ExpectedExportAbi::ScalarI32,
        )?;
        let memory = find_single_memory(&prepared.instance, &prepared.store);
        let func = prepared
            .instance
            .get_func(&prepared.store, export_name)
            .ok_or_else(|| WasmSandboxError::ExportNotAllowed(export_name.to_string()))?;
        let typed = func
            .typed::<i32, i32>(&prepared.store)
            .map_err(|_| WasmSandboxError::ExportSignatureMismatch(export_name.to_string()))?;

        let (ret, fuel_consumed) = call_typed_with_limits(
            &mut prepared.store,
            memory.as_ref(),
            policy,
            prepared.started_at,
            |store| typed.call_resumable(store, arg),
        )?;

        let peak_memory_bytes = memory
            .as_ref()
            .map_or(0, |mem| mem.data_size(&prepared.store));
        let receipt = WasmExecutionReceiptV1 {
            module_sha256: prepared.module_sha256,
            fuel_consumed,
            peak_memory_bytes,
            return_code: ret,
            output_bytes: Vec::new(),
        };
        Ok((ret, receipt))
    }
}

struct PreparedModule {
    module_sha256: [u8; 32],
    started_at: Instant,
    store: Store<StoreLimits>,
    instance: Instance,
}

fn execute_byte_transform_raw_mem0_v0(
    prepared: &mut PreparedModule,
    memory: Memory,
    policy: &SandboxedExtensionPolicyV1,
    export_name: &str,
    input_payload: &[u8],
) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
    let mem_len = memory.data_size(&prepared.store);
    if input_payload.len() > mem_len {
        return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
            addr: 0,
            len: input_payload.len(),
            mem_len,
        });
    }
    let input_len_i32 = i32::try_from(input_payload.len()).map_err(|_| {
        WasmSandboxError::OutOfBoundsMemoryAccess {
            addr: 0,
            len: input_payload.len(),
            mem_len,
        }
    })?;
    memory
        .write(&mut prepared.store, 0, input_payload)
        .map_err(|_| WasmSandboxError::OutOfBoundsMemoryAccess {
            addr: 0,
            len: input_payload.len(),
            mem_len,
        })?;

    let func = prepared
        .instance
        .get_func(&prepared.store, export_name)
        .ok_or_else(|| WasmSandboxError::ExportNotAllowed(export_name.to_string()))?;
    let typed = func
        .typed::<(i32, i32), i32>(&prepared.store)
        .map_err(|_| WasmSandboxError::ExportSignatureMismatch(export_name.to_string()))?;

    let (return_code, fuel_consumed) = call_typed_with_limits(
        &mut prepared.store,
        Some(&memory),
        policy,
        prepared.started_at,
        |store| typed.call_resumable(store, (0, input_len_i32)),
    )?;

    if return_code < 0 {
        return Err(WasmSandboxError::ExecutionTrap(format!(
            "export '{export_name}' returned negative error code {return_code}"
        )));
    }
    let out_len_usize = return_code as usize;
    if out_len_usize > policy.limits.max_output_bytes {
        return Err(WasmSandboxError::OutputTooLarge {
            len: out_len_usize,
            max: policy.limits.max_output_bytes,
        });
    }
    let output_bytes = read_guest_slice(memory.data(&prepared.store), 0, return_code)?;
    let peak_memory_bytes = memory.data_size(&prepared.store);

    Ok(WasmExecutionReceiptV1 {
        module_sha256: prepared.module_sha256,
        fuel_consumed,
        peak_memory_bytes,
        return_code,
        output_bytes,
    })
}

fn execute_byte_transform_alloc_v1(
    prepared: &mut PreparedModule,
    memory: Memory,
    policy: &SandboxedExtensionPolicyV1,
    export_name: &str,
    input_payload: &[u8],
) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
    if export_name == SWARMTORCH_WASM_ALLOC_EXPORT_V1
        || export_name == SWARMTORCH_WASM_DEALLOC_EXPORT_V1
    {
        return Err(WasmSandboxError::ExportNotAllowed(export_name.to_string()));
    }

    let alloc_func = prepared
        .instance
        .get_func(&prepared.store, SWARMTORCH_WASM_ALLOC_EXPORT_V1)
        .ok_or_else(|| {
            WasmSandboxError::ExportNotAllowed(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string())
        })?;
    let alloc_typed = alloc_func.typed::<i32, i32>(&prepared.store).map_err(|_| {
        WasmSandboxError::ExportSignatureMismatch(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string())
    })?;

    let dealloc_func = prepared
        .instance
        .get_func(&prepared.store, SWARMTORCH_WASM_DEALLOC_EXPORT_V1)
        .ok_or_else(|| {
            WasmSandboxError::ExportNotAllowed(SWARMTORCH_WASM_DEALLOC_EXPORT_V1.to_string())
        })?;
    let dealloc_typed = dealloc_func
        .typed::<(i32, i32), ()>(&prepared.store)
        .map_err(|_| {
            WasmSandboxError::ExportSignatureMismatch(SWARMTORCH_WASM_DEALLOC_EXPORT_V1.to_string())
        })?;

    let transform_func = prepared
        .instance
        .get_func(&prepared.store, export_name)
        .ok_or_else(|| WasmSandboxError::ExportNotAllowed(export_name.to_string()))?;
    let transform_typed = transform_func
        .typed::<(i32, i32), i64>(&prepared.store)
        .map_err(|_| WasmSandboxError::ExportSignatureMismatch(export_name.to_string()))?;

    let initial_mem_len = memory.data_size(&prepared.store);
    let input_len_i32 = i32::try_from(input_payload.len()).map_err(|_| {
        WasmSandboxError::OutOfBoundsMemoryAccess {
            addr: 0,
            len: input_payload.len(),
            mem_len: initial_mem_len,
        }
    })?;

    let in_ptr = if input_len_i32 > 0 {
        let (ptr, _) = call_typed_with_limits(
            &mut prepared.store,
            Some(&memory),
            policy,
            prepared.started_at,
            |store| alloc_typed.call_resumable(store, input_len_i32),
        )?;
        let mem_len = memory.data_size(&prepared.store);
        if ptr <= 0 {
            return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                addr: if ptr < 0 { ptr as u64 } else { 0 },
                len: input_payload.len(),
                mem_len,
            });
        }
        let end = (ptr as u32).checked_add(input_len_i32 as u32).ok_or(
            WasmSandboxError::OutOfBoundsMemoryAccess {
                addr: ptr as u64,
                len: input_payload.len(),
                mem_len,
            },
        )?;
        if (end as usize) > mem_len {
            return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                addr: ptr as u64,
                len: input_payload.len(),
                mem_len,
            });
        }
        memory
            .write(&mut prepared.store, ptr as usize, input_payload)
            .map_err(|_| WasmSandboxError::OutOfBoundsMemoryAccess {
                addr: ptr as u64,
                len: input_payload.len(),
                mem_len,
            })?;
        ptr
    } else {
        0
    };

    let mut peak_memory_bytes = memory.data_size(&prepared.store);
    let (packed_ret, mut fuel_consumed) = call_typed_with_limits(
        &mut prepared.store,
        Some(&memory),
        policy,
        prepared.started_at,
        |store| transform_typed.call_resumable(store, (in_ptr, input_len_i32)),
    )?;
    peak_memory_bytes = peak_memory_bytes.max(memory.data_size(&prepared.store));

    let (out_ptr, out_len) = unpack_guest_ptr_len_i64(packed_ret).map_err(|err| match err {
        WasmSandboxError::GuestTransformFailed { code, .. } => {
            WasmSandboxError::GuestTransformFailed {
                export: export_name.to_string(),
                code,
            }
        }
        WasmSandboxError::OutOfBoundsMemoryAccess { addr, len, .. } => {
            WasmSandboxError::OutOfBoundsMemoryAccess {
                addr,
                len,
                mem_len: memory.data_size(&prepared.store),
            }
        }
        other => other,
    })?;

    // Enforce max_output_bytes BEFORE reading/allocating host Vec<u8> (`ADR-0025`).
    let out_len_usize = out_len as usize;
    if out_len_usize > policy.limits.max_output_bytes {
        return Err(WasmSandboxError::OutputTooLarge {
            len: out_len_usize,
            max: policy.limits.max_output_bytes,
        });
    }

    let output_bytes = if out_len == 0 {
        // Case 1: empty output -> free input once (if allocated), never free (out_ptr, 0).
        if input_len_i32 > 0 {
            let (_, f) = call_typed_with_limits(
                &mut prepared.store,
                Some(&memory),
                policy,
                prepared.started_at,
                |store| dealloc_typed.call_resumable(store, (in_ptr, input_len_i32)),
            )?;
            fuel_consumed = f;
            peak_memory_bytes = peak_memory_bytes.max(memory.data_size(&prepared.store));
        }
        Vec::new()
    } else if input_len_i32 > 0 {
        let in_start = in_ptr as u64;
        let in_end = in_start + (input_len_i32 as u64);
        let out_start = out_ptr as u64;
        let out_end = out_start + (out_len as u64);

        if out_ptr == in_ptr && out_len <= input_len_i32 {
            // Case 2: exact or prefix-shortened in-place reuse of input buffer ->
            // read output slice, then free (in_ptr, in_len) ONCE with original input_len_i32!
            let bytes = read_guest_slice(memory.data(&prepared.store), out_ptr, out_len)?;
            let (_, f) = call_typed_with_limits(
                &mut prepared.store,
                Some(&memory),
                policy,
                prepared.started_at,
                |store| dealloc_typed.call_resumable(store, (in_ptr, input_len_i32)),
            )?;
            fuel_consumed = f;
            peak_memory_bytes = peak_memory_bytes.max(memory.data_size(&prepared.store));
            bytes
        } else if out_end <= in_start || in_end <= out_start {
            // Case 3: disjoint fresh output buffer -> reject null/offset-0 pointer, read slice,
            // then free (in_ptr, in_len) once and (out_ptr, out_len) once.
            if out_ptr <= 0 {
                return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                    addr: 0,
                    len: out_len_usize,
                    mem_len: memory.data_size(&prepared.store),
                });
            }
            let bytes = read_guest_slice(memory.data(&prepared.store), out_ptr, out_len)?;
            let (_, _) = call_typed_with_limits(
                &mut prepared.store,
                Some(&memory),
                policy,
                prepared.started_at,
                |store| dealloc_typed.call_resumable(store, (in_ptr, input_len_i32)),
            )?;
            let (_, f) = call_typed_with_limits(
                &mut prepared.store,
                Some(&memory),
                policy,
                prepared.started_at,
                |store| dealloc_typed.call_resumable(store, (out_ptr, out_len)),
            )?;
            fuel_consumed = f;
            peak_memory_bytes = peak_memory_bytes.max(memory.data_size(&prepared.store));
            bytes
        } else {
            // Case 4: interior pointer, partial overlap, or overgrown in-place buffer.
            return Err(WasmSandboxError::OutputAliasingViolation {
                in_ptr,
                in_len: input_len_i32,
                out_ptr,
                out_len,
            });
        }
    } else {
        // Empty input (in_len == 0) with non-empty output allocation -> reject null/offset-0,
        // read output slice, and free (out_ptr, out_len) once.
        if out_ptr <= 0 {
            return Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                addr: 0,
                len: out_len_usize,
                mem_len: memory.data_size(&prepared.store),
            });
        }
        let bytes = read_guest_slice(memory.data(&prepared.store), out_ptr, out_len)?;
        let (_, f) = call_typed_with_limits(
            &mut prepared.store,
            Some(&memory),
            policy,
            prepared.started_at,
            |store| dealloc_typed.call_resumable(store, (out_ptr, out_len)),
        )?;
        fuel_consumed = f;
        peak_memory_bytes = peak_memory_bytes.max(memory.data_size(&prepared.store));
        bytes
    };

    Ok(WasmExecutionReceiptV1 {
        module_sha256: prepared.module_sha256,
        fuel_consumed,
        peak_memory_bytes,
        return_code: out_len,
        output_bytes,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedExportAbi {
    ScalarI32,
    ByteTransformRawMem0V0,
    ByteTransformAllocV1,
}

fn prepare_instance(
    policy: &SandboxedExtensionPolicyV1,
    wasm_bytes: &[u8],
    export_name: &str,
    expected_abi: ExpectedExportAbi,
) -> Result<PreparedModule, WasmSandboxError> {
    let module_sha256 = preflight_wasm_extension_v1(policy, wasm_bytes, export_name)?;
    let started_at = Instant::now();
    check_timeout(started_at, policy.limits.timeout_ms)?;

    let engine = shared_hardened_engine();
    let module = Module::new(engine, wasm_bytes).map_err(map_compile_error)?;
    check_timeout(started_at, policy.limits.timeout_ms)?;

    if let Some(import) = module.imports().next() {
        return Err(WasmSandboxError::ForbiddenImport {
            module: import.module().to_string(),
            field: import.name().to_string(),
        });
    }

    let require_alloc_abi_exports = matches!(expected_abi, ExpectedExportAbi::ByteTransformAllocV1);
    let require_exported_memory = matches!(
        expected_abi,
        ExpectedExportAbi::ByteTransformRawMem0V0 | ExpectedExportAbi::ByteTransformAllocV1
    );
    let mut memory_count = 0u32;
    let mut table_count = 0u32;
    let mut target_export_sig_ok: Option<bool> = None;
    let mut alloc_export_sig_ok: Option<bool> = None;
    let mut dealloc_export_sig_ok: Option<bool> = None;
    for export in module.exports() {
        let name = export.name();
        if name == export_name {
            let sig_ok = match export.ty() {
                ExternType::Func(func_ty) => match expected_abi {
                    ExpectedExportAbi::ScalarI32 => {
                        func_ty.params() == [ValType::I32] && func_ty.results() == [ValType::I32]
                    }
                    ExpectedExportAbi::ByteTransformRawMem0V0 => {
                        func_ty.params() == [ValType::I32, ValType::I32]
                            && func_ty.results() == [ValType::I32]
                    }
                    ExpectedExportAbi::ByteTransformAllocV1 => {
                        func_ty.params() == [ValType::I32, ValType::I32]
                            && func_ty.results() == [ValType::I64]
                    }
                },
                _ => false,
            };
            target_export_sig_ok = Some(sig_ok);
        }
        if require_alloc_abi_exports {
            if name == SWARMTORCH_WASM_ALLOC_EXPORT_V1 {
                let sig_ok = match export.ty() {
                    ExternType::Func(func_ty) => {
                        func_ty.params() == [ValType::I32] && func_ty.results() == [ValType::I32]
                    }
                    _ => false,
                };
                alloc_export_sig_ok = Some(sig_ok);
            } else if name == SWARMTORCH_WASM_DEALLOC_EXPORT_V1 {
                let sig_ok = match export.ty() {
                    ExternType::Func(func_ty) => {
                        func_ty.params() == [ValType::I32, ValType::I32]
                            && func_ty.results().is_empty()
                    }
                    _ => false,
                };
                dealloc_export_sig_ok = Some(sig_ok);
            }
        }
        match export.ty() {
            ExternType::Memory(mem_ty) => {
                memory_count = memory_count.saturating_add(1);
                if memory_count > policy.limits.max_memories {
                    return Err(WasmSandboxError::MultipleMemoriesForbidden);
                }
                let initial_pages = u32::try_from(mem_ty.minimum()).unwrap_or(u32::MAX);
                if initial_pages > policy.limits.max_memory_pages {
                    return Err(WasmSandboxError::MemoryLimitExceeded {
                        requested_pages: initial_pages,
                        max_pages: policy.limits.max_memory_pages,
                    });
                }
            }
            ExternType::Table(table_ty) => {
                table_count = table_count.saturating_add(1);
                if table_count > policy.limits.max_tables {
                    return Err(WasmSandboxError::MultipleTablesForbidden);
                }
                let initial_elems = u32::try_from(table_ty.minimum()).unwrap_or(u32::MAX);
                if initial_elems > policy.limits.max_table_elements {
                    return Err(WasmSandboxError::TableLimitExceeded {
                        requested: initial_elems,
                        max: policy.limits.max_table_elements,
                    });
                }
            }
            _ => {}
        }
    }
    if target_export_sig_ok.is_none() {
        return Err(WasmSandboxError::ExportNotAllowed(export_name.to_string()));
    }
    if require_alloc_abi_exports {
        match alloc_export_sig_ok {
            None => {
                return Err(WasmSandboxError::ExportNotAllowed(
                    SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string(),
                ));
            }
            Some(false) => {
                return Err(WasmSandboxError::ExportSignatureMismatch(
                    SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string(),
                ));
            }
            Some(true) => {}
        }
        match dealloc_export_sig_ok {
            None => {
                return Err(WasmSandboxError::ExportNotAllowed(
                    SWARMTORCH_WASM_DEALLOC_EXPORT_V1.to_string(),
                ));
            }
            Some(false) => {
                return Err(WasmSandboxError::ExportSignatureMismatch(
                    SWARMTORCH_WASM_DEALLOC_EXPORT_V1.to_string(),
                ));
            }
            Some(true) => {}
        }
    }
    if target_export_sig_ok == Some(false) {
        return Err(WasmSandboxError::ExportSignatureMismatch(
            export_name.to_string(),
        ));
    }
    if require_exported_memory && memory_count == 0 {
        return Err(WasmSandboxError::ExecutionTrap(format!(
            "byte-transform export '{export_name}' requires an exported linear memory"
        )));
    }

    let max_memory_bytes = usize::try_from(policy.limits.max_memory_pages)
        .unwrap_or(usize::MAX)
        .saturating_mul(WASM_PAGE_BYTES);
    let store_limits = StoreLimitsBuilder::new()
        .memory_size(max_memory_bytes)
        .table_elements(policy.limits.max_table_elements as usize)
        .memories(policy.limits.max_memories.min(1) as usize)
        .tables(policy.limits.max_tables.min(1) as usize)
        .instances(1)
        .trap_on_grow_failure(policy.limits.trap_on_grow_failure)
        .build();

    let mut store = Store::new(engine, store_limits);
    store.limiter(|limits| limits);
    let start_fuel_cap = policy.limits.max_fuel.min(policy.limits.max_start_fuel);
    store
        .set_fuel(start_fuel_cap)
        .map_err(|err| WasmSandboxError::ExecutionTrap(err.to_string()))?;

    let linker = Linker::<StoreLimits>::new(engine);
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .map_err(|err| map_instantiation_error(err, policy, start_fuel_cap))?;

    check_timeout(started_at, policy.limits.timeout_ms)?;
    let unspent_start_fuel = store.get_fuel().unwrap_or(0);
    let start_consumed = start_fuel_cap.saturating_sub(unspent_start_fuel);
    let remaining_after_start = policy.limits.max_fuel.saturating_sub(start_consumed);
    store
        .set_fuel(remaining_after_start)
        .map_err(|err| WasmSandboxError::ExecutionTrap(err.to_string()))?;

    Ok(PreparedModule {
        module_sha256,
        started_at,
        store,
        instance,
    })
}

fn find_single_memory(instance: &Instance, store: &Store<StoreLimits>) -> Option<Memory> {
    if let Some(Extern::Memory(mem)) = instance.get_export(store, "memory") {
        return Some(mem);
    }
    for export in instance.exports(store) {
        if let Some(mem) = export.into_memory() {
            return Some(mem);
        }
    }
    None
}

fn call_typed_with_limits<R, F>(
    store: &mut Store<StoreLimits>,
    memory: Option<&Memory>,
    policy: &SandboxedExtensionPolicyV1,
    started_at: Instant,
    invoke: F,
) -> Result<(R, u64), WasmSandboxError>
where
    R: wasmi::WasmResults,
    F: FnOnce(&mut Store<StoreLimits>) -> Result<TypedResumableCall<R>, wasmi::Error>,
{
    check_timeout(started_at, policy.limits.timeout_ms)?;
    let fuel_limit = policy.limits.max_fuel;
    let total_available = store.get_fuel().unwrap_or(0);
    let mut remaining_budget = total_available;
    let first_slice = remaining_budget.min(FUEL_SLICE_UNITS);
    remaining_budget -= first_slice;
    store
        .set_fuel(first_slice)
        .map_err(|e| WasmSandboxError::ExecutionTrap(e.to_string()))?;

    let mut step = invoke(store);
    loop {
        match step {
            Ok(TypedResumableCall::Finished(val)) => {
                let unspent_in_slice = store.get_fuel().unwrap_or(0);
                let total_remaining = remaining_budget.saturating_add(unspent_in_slice);
                store
                    .set_fuel(total_remaining)
                    .map_err(|e| WasmSandboxError::ExecutionTrap(e.to_string()))?;
                let fuel_consumed = fuel_limit.saturating_sub(total_remaining);
                return Ok((val, fuel_consumed));
            }
            Ok(TypedResumableCall::OutOfFuel(invocation)) => {
                check_timeout(started_at, policy.limits.timeout_ms)?;
                let unspent_in_slice = store.get_fuel().unwrap_or(0);
                if remaining_budget == 0 {
                    return Err(WasmSandboxError::OutOfFuel {
                        consumed: fuel_limit.saturating_sub(unspent_in_slice),
                        limit: fuel_limit,
                    });
                }
                let next_slice = remaining_budget.min(FUEL_SLICE_UNITS);
                remaining_budget -= next_slice;
                let refilled = unspent_in_slice.saturating_add(next_slice);
                store
                    .set_fuel(refilled)
                    .map_err(|e| WasmSandboxError::ExecutionTrap(e.to_string()))?;
                step = invocation.resume(&mut *store);
            }
            Ok(TypedResumableCall::HostTrap(trap)) => {
                return Err(WasmSandboxError::ExecutionTrap(format!(
                    "unexpected host trap in zero-import sandbox: {trap:?}"
                )));
            }
            Err(err) => {
                return Err(map_execution_error(err, store, memory, policy));
            }
        }
    }
}

fn check_timeout(started_at: Instant, limit_ms: u64) -> Result<(), WasmSandboxError> {
    let elapsed_ms = started_at.elapsed().as_millis();
    if elapsed_ms >= u128::from(limit_ms) {
        return Err(WasmSandboxError::ExecutionTimeout {
            elapsed_ms: u64::try_from(elapsed_ms).unwrap_or(u64::MAX),
            limit_ms,
        });
    }
    Ok(())
}

fn build_hardened_engine() -> Engine {
    let mut config = Config::default();
    config.consume_fuel(true);
    config.compilation_mode(CompilationMode::Eager);
    config.allow_start_fn(true);
    config.ignore_custom_sections(true);
    config.wasm_multi_memory(false);
    config.wasm_reference_types(false);
    config.wasm_tail_call(false);
    config.wasm_custom_page_sizes(false);
    config.set_min_stack_height(DEFAULT_WASM_MIN_STACK_HEIGHT_BYTES);
    config.set_max_stack_height(DEFAULT_WASM_MAX_STACK_HEIGHT_BYTES);
    config.set_max_recursion_depth(DEFAULT_WASM_MAX_CALL_DEPTH);
    config.enforced_limits(EnforcedLimits::strict());
    Engine::new(&config)
}

fn map_compile_error(err: wasmi::Error) -> WasmSandboxError {
    match err.kind() {
        ErrorKind::UserLimits(EnforcedLimitsError::TooManyMemories { .. }) => {
            WasmSandboxError::MultipleMemoriesForbidden
        }
        ErrorKind::UserLimits(EnforcedLimitsError::TooManyTables { .. }) => {
            WasmSandboxError::MultipleTablesForbidden
        }
        _ => {
            let msg = err.to_string();
            let lower = msg.to_ascii_lowercase();
            if lower.contains("multi-memory") || lower.contains("multiple memories") {
                WasmSandboxError::MultipleMemoriesForbidden
            } else if lower.contains("multiple tables") || lower.contains("tables count") {
                WasmSandboxError::MultipleTablesForbidden
            } else {
                WasmSandboxError::MalformedSectionOrLeb128
            }
        }
    }
}

fn map_instantiation_error(
    err: wasmi::Error,
    policy: &SandboxedExtensionPolicyV1,
    start_fuel_cap: u64,
) -> WasmSandboxError {
    match err.kind() {
        ErrorKind::TrapCode(TrapCode::OutOfFuel) | ErrorKind::Fuel(_) => {
            WasmSandboxError::OutOfFuel {
                consumed: start_fuel_cap,
                limit: start_fuel_cap,
            }
        }
        ErrorKind::Instantiation(InstantiationError::TooManyMemories) => {
            WasmSandboxError::MultipleMemoriesForbidden
        }
        ErrorKind::Instantiation(InstantiationError::TooManyTables) => {
            WasmSandboxError::MultipleTablesForbidden
        }
        ErrorKind::Instantiation(InstantiationError::FailedToInstantiateMemory(_))
        | ErrorKind::TrapCode(TrapCode::GrowthOperationLimited)
        | ErrorKind::Memory(_) => WasmSandboxError::MemoryLimitExceeded {
            requested_pages: policy.limits.max_memory_pages.saturating_add(1),
            max_pages: policy.limits.max_memory_pages,
        },
        ErrorKind::Instantiation(InstantiationError::FailedToInstantiateTable(_))
        | ErrorKind::Table(_) => WasmSandboxError::TableLimitExceeded {
            requested: policy.limits.max_table_elements.saturating_add(1),
            max: policy.limits.max_table_elements,
        },
        _ => WasmSandboxError::ExecutionTrap(err.to_string()),
    }
}

fn map_execution_error(
    err: wasmi::Error,
    store: &Store<StoreLimits>,
    memory: Option<&Memory>,
    policy: &SandboxedExtensionPolicyV1,
) -> WasmSandboxError {
    match err.kind() {
        ErrorKind::TrapCode(TrapCode::OutOfFuel) | ErrorKind::Fuel(_) => {
            WasmSandboxError::OutOfFuel {
                consumed: policy.limits.max_fuel,
                limit: policy.limits.max_fuel,
            }
        }
        ErrorKind::TrapCode(TrapCode::GrowthOperationLimited) => {
            let current_pages = memory
                .and_then(|m| u32::try_from(m.size(store)).ok())
                .unwrap_or(0);
            WasmSandboxError::MemoryGrowTrap {
                current_pages,
                delta_pages: 0,
                max_pages: policy.limits.max_memory_pages,
            }
        }
        ErrorKind::TrapCode(TrapCode::MemoryOutOfBounds) => {
            let mem_len = memory.map_or(0, |m| m.data_size(store));
            WasmSandboxError::OutOfBoundsMemoryAccess {
                addr: 0,
                len: 0,
                mem_len,
            }
        }
        ErrorKind::Memory(_) => {
            let mem_len = memory.map_or(0, |m| m.data_size(store));
            let msg = err.to_string();
            if msg.to_ascii_lowercase().contains("out of bounds") {
                WasmSandboxError::OutOfBoundsMemoryAccess {
                    addr: 0,
                    len: 0,
                    mem_len,
                }
            } else {
                WasmSandboxError::ExecutionTrap(msg)
            }
        }
        _ => WasmSandboxError::ExecutionTrap(err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use swarm_torch::artifacts::{DataOpsSession, RunArtifactBundle, RunArtifactSink};
    use swarm_torch::native_runner::{NativeOpRunner, CONTENT_TYPE_OCTET_STREAM};
    use swarm_torch::scheduler::{
        execute_graph_sequential, execute_graph_with_isolation, RunnerIsolationProfile,
    };
    use swarm_torch_core::dataops::{AuthModeMarker, SourceDescriptorV0, TrustClass};
    use swarm_torch_core::execution::PermissivePolicy;
    use swarm_torch_core::observe::RunId;
    use swarm_torch_core::run_graph::{
        AssetRefV1, CanonParams, CanonValue, ExecutionTrust, GraphV1, NodeV1, OpKind,
    };

    fn push_leb_u32(out: &mut Vec<u8>, mut val: u32) {
        loop {
            let byte = (val & 0x7f) as u8;
            val >>= 7;
            if val == 0 {
                out.push(byte);
                break;
            } else {
                out.push(byte | 0x80);
            }
        }
    }

    fn push_name(out: &mut Vec<u8>, s: &str) {
        push_leb_u32(out, s.len() as u32);
        out.extend_from_slice(s.as_bytes());
    }

    fn push_section(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
        out.push(id);
        push_leb_u32(out, payload.len() as u32);
        out.extend_from_slice(payload);
    }

    fn build_single_export_wasm(
        params: &[u8],
        results: &[u8],
        memory_pages: Option<(u32, Option<u32>)>,
        local_i32_count: u32,
        export_name: &str,
        body_ops: &[u8],
    ) -> Vec<u8> {
        let mut wasm = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();

        // Section 1: Type
        let mut sec1 = vec![1];
        sec1.push(0x60);
        push_leb_u32(&mut sec1, params.len() as u32);
        sec1.extend_from_slice(params);
        push_leb_u32(&mut sec1, results.len() as u32);
        sec1.extend_from_slice(results);
        push_section(&mut wasm, 1, &sec1);

        // Section 3: Function
        push_section(&mut wasm, 3, &[1, 0]);

        // Section 5: Memory
        if let Some((min_pages, max_pages)) = memory_pages {
            let mut sec5 = vec![1];
            if let Some(max) = max_pages {
                sec5.push(0x01);
                push_leb_u32(&mut sec5, min_pages);
                push_leb_u32(&mut sec5, max);
            } else {
                sec5.push(0x00);
                push_leb_u32(&mut sec5, min_pages);
            }
            push_section(&mut wasm, 5, &sec5);
        }

        // Section 7: Export
        let export_count = if memory_pages.is_some() { 2 } else { 1 };
        let mut sec7 = vec![export_count];
        push_name(&mut sec7, export_name);
        sec7.push(0x00);
        sec7.push(0x00);
        if memory_pages.is_some() {
            push_name(&mut sec7, "memory");
            sec7.push(0x02);
            sec7.push(0x00);
        }
        push_section(&mut wasm, 7, &sec7);

        // Section 10: Code
        let mut body = Vec::new();
        if local_i32_count > 0 {
            body.push(1);
            push_leb_u32(&mut body, local_i32_count);
            body.push(0x7f);
        } else {
            body.push(0);
        }
        body.extend_from_slice(body_ops);
        body.push(0x0b);

        let mut sec10 = vec![1];
        push_leb_u32(&mut sec10, body.len() as u32);
        sec10.extend_from_slice(&body);
        push_section(&mut wasm, 10, &sec10);

        wasm
    }

    #[test]
    fn wasmi_rejects_untrusted_hash_and_wasi_import() {
        let exec = WasmiExtensionExecutorV1::new();

        // Simple add10(x: i32) -> i32
        let valid = build_single_export_wasm(
            &[0x7f],
            &[0x7f],
            None,
            0,
            "add10",
            &[0x20, 0x00, 0x41, 0x0a, 0x6a],
        );
        let err = exec
            .execute_scalar_i32(&SandboxedExtensionPolicyV1::default(), &valid, "add10", 5)
            .unwrap_err();
        assert!(matches!(err, WasmSandboxError::UntrustedModuleHash { .. }));

        // Module importing wasi_snapshot_preview1.fd_read -> rejected even if hash pinned.
        let mut with_wasi = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut with_wasi, 1, &[1, 0x60, 1, 0x7f, 1, 0x7f]);
        let mut sec2 = vec![1];
        push_name(&mut sec2, "wasi_snapshot_preview1");
        push_name(&mut sec2, "fd_read");
        sec2.push(0x00);
        sec2.push(0x00);
        push_section(&mut with_wasi, 2, &sec2);
        push_section(&mut with_wasi, 3, &[1, 0]);
        let mut sec7 = vec![1];
        push_name(&mut sec7, "run");
        sec7.push(0x00);
        sec7.push(0x01);
        push_section(&mut with_wasi, 7, &sec7);
        push_section(&mut with_wasi, 10, &[1, 4, 0, 0x41, 0x00, 0x0b]);

        let pinned = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&with_wasi),
        );
        let err = exec
            .execute_scalar_i32(&pinned, &with_wasi, "run", 0)
            .unwrap_err();
        assert!(matches!(
            err,
            WasmSandboxError::ForbiddenImport { module, field }
                if module == "wasi_snapshot_preview1" && field == "fd_read"
        ));
    }

    #[test]
    fn wasmi_traps_infinite_loops_in_export_and_start_and_enforces_wall_clock_timeout() {
        let exec = WasmiExtensionExecutorV1::new();

        // 1. Export infinite loop traps on fuel exhaustion
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
        let err = exec
            .execute_scalar_i32(&policy, &loop_wasm, "spin", 0)
            .unwrap_err();
        assert!(
            matches!(err, WasmSandboxError::OutOfFuel { limit: 50, .. }),
            "expected OutOfFuel, got {err:?}"
        );

        // 2. Massive fuel budget + short timeout MUST stop cleanly via ExecutionTimeout!
        let mut timeout_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&loop_wasm),
        );
        timeout_policy.limits.max_fuel = 100_000_000;
        timeout_policy.limits.timeout_ms = 25;
        let err = exec
            .execute_scalar_i32(&timeout_policy, &loop_wasm, "spin", 0)
            .unwrap_err();
        assert!(
            matches!(err, WasmSandboxError::ExecutionTimeout { limit_ms: 25, .. }),
            "expected ExecutionTimeout, got {err:?}"
        );

        // 3. Infinite loop in Wasm start section MUST trap OutOfFuel before export runs
        let mut start_loop = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut start_loop, 1, &[2, 0x60, 0, 0, 0x60, 1, 0x7f, 1, 0x7f]);
        push_section(&mut start_loop, 3, &[2, 0, 1]);
        let mut sec7 = vec![1];
        push_name(&mut sec7, "noop");
        sec7.push(0x00);
        sec7.push(0x01);
        push_section(&mut start_loop, 7, &sec7);
        push_section(&mut start_loop, 8, &[0]);
        let body0 = [0x00u8, 0x03, 0x40, 0x0c, 0x00, 0x0b, 0x0b];
        let body1 = [0x00u8, 0x41, 0x2a, 0x0b];
        let mut sec10 = vec![2];
        push_leb_u32(&mut sec10, body0.len() as u32);
        sec10.extend_from_slice(&body0);
        push_leb_u32(&mut sec10, body1.len() as u32);
        sec10.extend_from_slice(&body1);
        push_section(&mut start_loop, 10, &sec10);

        let mut start_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&start_loop),
        );
        start_policy.limits.max_fuel = 128;
        let err = exec
            .execute_scalar_i32(&start_policy, &start_loop, "noop", 0)
            .unwrap_err();
        assert!(
            matches!(err, WasmSandboxError::OutOfFuel { limit: 128, .. }),
            "expected OutOfFuel from start fn, got {err:?}"
        );
    }

    #[test]
    fn wasmi_enforces_memory_grow_trap_and_initial_memory_limits() {
        let exec = WasmiExtensionExecutorV1::new();

        // 1. memory.grow(32) exceeds 4 pages -> trap_on_grow_failure=true traps MemoryGrowTrap
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
        let err = exec
            .execute_scalar_i32(&grow_policy, &grow_wasm, "grow_bomb", 0)
            .unwrap_err();
        assert!(
            matches!(
                err,
                WasmSandboxError::MemoryGrowTrap {
                    current_pages: 1,
                    max_pages: 4,
                    ..
                }
            ),
            "expected MemoryGrowTrap, got {err:?}"
        );

        // 2. When trap_on_grow_failure=false, memory.grow returns -1 without exceeding limit
        grow_policy.limits.trap_on_grow_failure = false;
        let (ret, _) = exec
            .execute_scalar_i32(&grow_policy, &grow_wasm, "grow_bomb", 0)
            .unwrap();
        assert_eq!(ret, -1);

        // 3. Declared initial memory 10 pages > max_memory_pages (4) -> MemoryLimitExceeded
        let big_init_wasm = build_single_export_wasm(
            &[0x7f],
            &[0x7f],
            Some((10, Some(10))),
            0,
            "ok",
            &[0x20, 0x00],
        );
        let mut big_init_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&big_init_wasm),
        );
        big_init_policy.limits.max_memory_pages = 4;
        let err = exec
            .execute_scalar_i32(&big_init_policy, &big_init_wasm, "ok", 7)
            .unwrap_err();
        assert!(matches!(
            err,
            WasmSandboxError::MemoryLimitExceeded {
                requested_pages: 10,
                max_pages: 4
            }
        ));

        // 4. Declared initial table elements 32 > max_table_elements (8) -> TableLimitExceeded
        let mut table_mod = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut table_mod, 1, &[1, 0x60, 1, 0x7f, 1, 0x7f]);
        push_section(&mut table_mod, 3, &[1, 0]);
        // Section 4: Table (1 funcref table, flags=0x00, min=32)
        let mut sec4 = vec![1, 0x70, 0x00];
        push_leb_u32(&mut sec4, 32);
        push_section(&mut table_mod, 4, &sec4);
        let mut sec7_t = vec![2];
        push_name(&mut sec7_t, "ok");
        sec7_t.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7_t, "tbl");
        sec7_t.extend_from_slice(&[0x01, 0x00]);
        push_section(&mut table_mod, 7, &sec7_t);
        push_section(&mut table_mod, 10, &[1, 4, 0, 0x20, 0x00, 0x0b]);

        let mut table_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&table_mod),
        );
        table_policy.limits.max_table_elements = 8;
        let err = exec
            .execute_scalar_i32(&table_policy, &table_mod, "ok", 7)
            .unwrap_err();
        assert!(matches!(
            err,
            WasmSandboxError::TableLimitExceeded {
                requested: 32,
                max: 8
            }
        ));

        // 5. Multi-memory module MUST be rejected fail-closed!
        let mut multi_mem = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut multi_mem, 1, &[1, 0x60, 1, 0x7f, 1, 0x7f]);
        push_section(&mut multi_mem, 3, &[1, 0]);
        push_section(&mut multi_mem, 5, &[2, 0x00, 1, 0x00, 1]);
        let mut sec7_m = vec![1];
        push_name(&mut sec7_m, "ok");
        sec7_m.extend_from_slice(&[0x00, 0x00]);
        push_section(&mut multi_mem, 7, &sec7_m);
        push_section(&mut multi_mem, 10, &[1, 4, 0, 0x20, 0x00, 0x0b]);
        let multi_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&multi_mem),
        );
        let err = exec
            .execute_scalar_i32(&multi_policy, &multi_mem, "ok", 1)
            .unwrap_err();
        assert!(matches!(
            err,
            WasmSandboxError::MultipleMemoriesForbidden
                | WasmSandboxError::MalformedSectionOrLeb128
        ));
    }

    fn build_ascii_xor_wasm() -> Vec<u8> {
        let ops: Vec<u8> = vec![
            0x41, 0x00, 0x21, 0x02, 0x03, 0x40, 0x20, 0x02, 0x20, 0x01, 0x48, 0x04, 0x40, 0x20,
            0x00, 0x20, 0x02, 0x6a, 0x20, 0x00, 0x20, 0x02, 0x6a, 0x2d, 0x00, 0x00, 0x41, 0x20,
            0x73, 0x3a, 0x00, 0x00, 0x20, 0x02, 0x41, 0x01, 0x6a, 0x21, 0x02, 0x0c, 0x01, 0x0b,
            0x0b, 0x20, 0x01,
        ];
        build_single_export_wasm(
            &[0x7f, 0x7f],
            &[0x7f],
            Some((1, Some(2))),
            1,
            "transform",
            &ops,
        )
    }

    #[test]
    fn wasmi_executes_verified_byte_transform_and_rejects_bad_abi_or_oob_return() {
        let exec = WasmiExtensionExecutorV1::new();

        let wasm = build_ascii_xor_wasm();
        let hash = SandboxedExtensionPolicyV1::sha256_module(&wasm);
        let policy = SandboxedExtensionPolicyV1::with_allowed_hash(hash);

        let receipt = exec
            .execute_byte_transform(&policy, &wasm, "transform", b"abcDEF")
            .unwrap();
        assert_eq!(receipt.module_sha256, hash);
        assert_eq!(receipt.return_code, 6);
        assert_eq!(receipt.output_bytes, b"ABCdef");
        assert!(receipt.fuel_consumed > 0);
        assert_eq!(receipt.peak_memory_bytes, WASM_PAGE_BYTES);

        // Wrong ABI signature (scalar call on 2-param byte_transform export) -> ExportSignatureMismatch
        let err = exec
            .execute_scalar_i32(&policy, &wasm, "transform", 1)
            .unwrap_err();
        assert!(matches!(
            err,
            WasmSandboxError::ExportSignatureMismatch(ref name) if name == "transform"
        ));

        // OOB return length (return 100_000 > 65_536 page) -> OutOfBoundsMemoryAccess
        let oob_ret = build_single_export_wasm(
            &[0x7f, 0x7f],
            &[0x7f],
            Some((1, Some(1))),
            0,
            "oob",
            &[0x41, 0xa0, 0x8d, 0x06],
        );
        let oob_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&oob_ret),
        );
        let err = exec
            .execute_byte_transform(&oob_policy, &oob_ret, "oob", b"hi")
            .unwrap_err();
        assert!(matches!(
            err,
            WasmSandboxError::OutOfBoundsMemoryAccess { .. }
        ));
    }

    fn test_clock() -> u64 {
        1_700_000_000_000_000_000
    }

    fn make_graph_node(
        key: &str,
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        trust: ExecutionTrust,
    ) -> NodeV1 {
        NodeV1 {
            node_key: key.to_string(),
            node_id: None,
            op_kind: OpKind::Data,
            op_type: op_type.to_string(),
            inputs: inputs
                .iter()
                .map(|asset_key| AssetRefV1 {
                    asset_key: asset_key.to_string(),
                    fingerprint: None,
                })
                .collect(),
            outputs: outputs
                .iter()
                .map(|asset_key| AssetRefV1 {
                    asset_key: asset_key.to_string(),
                    fingerprint: None,
                })
                .collect(),
            params: CanonParams::new(),
            code_ref: Some("test@0.1.0".to_string()),
            unsafe_surface: false,
            execution_trust: trust,
            node_def_hash: None,
            execution_hint: None,
            cache_policy: None,
            materialization_policy: None,
            resources: None,
            op_hash: None,
        }
    }

    #[test]
    fn end_to_end_scheduler_and_native_runner_execute_wasm_transform_over_physical_cas() {
        let wasm = build_ascii_xor_wasm();
        let hash = SandboxedExtensionPolicyV1::sha256_module(&wasm);
        let hex_hash: String = hash.iter().map(|b| format!("{b:02x}")).collect();

        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("swarmtorch_wasmi_e2e_{pid}"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([0x42; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(sink);
        let run_dir = session.sink().bundle().run_dir().to_path_buf();

        let input_bytes = b"abcDEF";
        let input_sha = SandboxedExtensionPolicyV1::sha256_module(input_bytes);
        let input_hex: String = input_sha.iter().map(|b| format!("{b:02x}")).collect();
        let cas_dir = run_dir.join("blobs").join("sha256");
        std::fs::create_dir_all(&cas_dir).unwrap();
        std::fs::write(cas_dir.join(&input_hex), input_bytes).unwrap();

        let asset_key = "dataset://ns/raw";
        let ingest = make_graph_node(
            "ingest/raw",
            "ingest",
            &[],
            &[asset_key],
            ExecutionTrust::Core,
        );
        session
            .register_source(
                asset_key,
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: format!("cas://sha256/{input_hex}"),
                    content_type: CONTENT_TYPE_OCTET_STREAM.to_string(),
                    auth_mode: AuthModeMarker::None,
                    etag_or_version: Some("v1".to_string()),
                },
                None,
                &ingest,
            )
            .unwrap();

        let mut wasm_node = make_graph_node(
            "node/wasm_op",
            "wasm_transform",
            &[asset_key],
            &["dataset://ns/wasm_out"],
            ExecutionTrust::SandboxedExtension,
        );
        wasm_node
            .params
            .insert("module_sha256".to_string(), CanonValue::Str(hex_hash));
        wasm_node.params.insert(
            "export".to_string(),
            CanonValue::Str("transform".to_string()),
        );

        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("wasmi-e2e-test".to_string()),
            nodes: vec![wasm_node.clone()],
            edges: vec![],
        };

        let wasm_policy = SandboxedExtensionPolicyV1::with_allowed_hash(hash);
        let mut registry = BTreeMap::new();
        registry.insert(hash, wasm);

        // 1. Default runner without WasmExtensionExecutor MUST fail closed.
        let r_unconfigured = execute_graph_with_isolation(
            &graph,
            &mut session,
            &NativeOpRunner::new(),
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(wasm_policy.clone()),
            Some(&registry),
            RunId::from_bytes([0x11; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r_unconfigured.failed_nodes, vec!["node/wasm_op"]);

        // 2. Configured WasmiExtensionExecutorV1 MUST fail under InProcessTrusted profile.
        let runner =
            NativeOpRunner::new().with_wasm_executor(Arc::new(WasmiExtensionExecutorV1::new()));
        let r_in_process = execute_graph_sequential(
            &graph,
            &mut session,
            &runner,
            &PermissivePolicy,
            RunId::from_bytes([0x22; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r_in_process.failed_nodes, vec!["node/wasm_op"]);

        // 3. Configured WasmiExtensionExecutorV1 + WasmSandboxedV1 executes end-to-end!
        let r_ok = execute_graph_with_isolation(
            &graph,
            &mut session,
            &runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(wasm_policy.clone()),
            Some(&registry),
            RunId::from_bytes([0x33; 16]),
            test_clock,
        )
        .unwrap();
        assert!(
            r_ok.failed_nodes.is_empty(),
            "unexpected failure: {:?}",
            r_ok.failed_nodes
        );
        assert_eq!(r_ok.executed_nodes, vec!["node/wasm_op"]);

        let out_inst = &r_ok.runner_outputs["node/wasm_op"][0];
        let digest = out_inst
            .uri
            .as_deref()
            .and_then(|u| u.strip_prefix("cas://sha256/"))
            .unwrap();
        let cas_bytes = std::fs::read(run_dir.join("blobs").join("sha256").join(digest)).unwrap();
        assert_eq!(cas_bytes, b"ABCdef");

        // 4. Native ops tagged SandboxedExtension MUST still fail closed.
        let mut native_in_sandbox = wasm_node;
        native_in_sandbox.node_key = "node/bad_passthrough".to_string();
        native_in_sandbox.op_type = "passthrough".to_string();
        let bad_graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("bad-sandbox-op".to_string()),
            nodes: vec![native_in_sandbox],
            edges: vec![],
        };
        let r_bad_op = execute_graph_with_isolation(
            &bad_graph,
            &mut session,
            &runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(wasm_policy),
            Some(&registry),
            RunId::from_bytes([0x44; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r_bad_op.failed_nodes, vec!["node/bad_passthrough"]);

        drop(session);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn wasm_start_section_loop_is_capped_by_explicit_max_start_fuel_policy() {
        let mut start_loop = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut start_loop, 1, &[2, 0x60, 0, 0, 0x60, 1, 0x7f, 1, 0x7f]);
        push_section(&mut start_loop, 3, &[2, 0, 1]);
        let mut sec7 = vec![1];
        push_name(&mut sec7, "run");
        sec7.push(0x00);
        sec7.push(0x01);
        push_section(&mut start_loop, 7, &sec7);
        push_section(&mut start_loop, 8, &[0]);
        let body0 = [0x00u8, 0x03, 0x40, 0x0c, 0x00, 0x0b, 0x0b];
        let body1 = [0x00u8, 0x20, 0x00, 0x0b];
        let mut sec10 = vec![2];
        push_leb_u32(&mut sec10, body0.len() as u32);
        sec10.extend_from_slice(&body0);
        push_leb_u32(&mut sec10, body1.len() as u32);
        sec10.extend_from_slice(&body1);
        push_section(&mut start_loop, 10, &sec10);

        let mut policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&start_loop),
        );
        policy.limits.max_fuel = 10_000_000;
        policy.limits.max_start_fuel = 2_048;
        policy.limits.timeout_ms = 50;

        let started = Instant::now();
        let err = WasmiExtensionExecutorV1::new()
            .execute_scalar_i32(&policy, &start_loop, "run", 7)
            .unwrap_err();
        assert_eq!(
            err,
            WasmSandboxError::OutOfFuel {
                consumed: 2_048,
                limit: 2_048,
            }
        );
        assert!(started.elapsed().as_millis() < 500);

        // Policy fingerprint must reflect max_start_fuel tightening.
        let mut alt = policy.clone();
        alt.limits.max_start_fuel = 1_024;
        assert_ne!(
            policy.policy_fingerprint_hex(),
            alt.policy_fingerprint_hex()
        );
    }

    #[test]
    fn wasmi_enforces_private_internal_table_memory_and_stack_recursion_limits() {
        let exec = WasmiExtensionExecutorV1::new();

        // 1. Non-exported private table (`min=32`) exceeding `max_table_elements=8` must map cleanly.
        let mut priv_table_mod = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut priv_table_mod, 1, &[1, 0x60, 1, 0x7f, 1, 0x7f]);
        push_section(&mut priv_table_mod, 3, &[1, 0]);
        let mut sec4 = vec![1, 0x70, 0x00];
        push_leb_u32(&mut sec4, 32);
        push_section(&mut priv_table_mod, 4, &sec4);
        let mut sec7 = vec![1];
        push_name(&mut sec7, "ok");
        sec7.extend_from_slice(&[0x00, 0x00]);
        push_section(&mut priv_table_mod, 7, &sec7);
        push_section(&mut priv_table_mod, 10, &[1, 4, 0, 0x20, 0x00, 0x0b]);

        let mut p_table = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&priv_table_mod),
        );
        p_table.limits.max_table_elements = 8;
        assert_eq!(
            exec.execute_scalar_i32(&p_table, &priv_table_mod, "ok", 7)
                .unwrap_err(),
            WasmSandboxError::TableLimitExceeded {
                requested: 9,
                max: 8,
            }
        );

        // With `max_tables = 0`, any private internal table must fail `MultipleTablesForbidden`.
        p_table.limits.max_table_elements = 64;
        p_table.limits.max_tables = 0;
        assert_eq!(
            exec.execute_scalar_i32(&p_table, &priv_table_mod, "ok", 7)
                .unwrap_err(),
            WasmSandboxError::MultipleTablesForbidden
        );

        // 2. Non-exported private memory (`min=10`) exceeding `max_memory_pages=4` must map cleanly.
        let mut priv_mem_mod = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut priv_mem_mod, 1, &[1, 0x60, 1, 0x7f, 1, 0x7f]);
        push_section(&mut priv_mem_mod, 3, &[1, 0]);
        let mut sec5 = vec![1, 0x00];
        push_leb_u32(&mut sec5, 10);
        push_section(&mut priv_mem_mod, 5, &sec5);
        let mut sec7_m = vec![1];
        push_name(&mut sec7_m, "ok");
        sec7_m.extend_from_slice(&[0x00, 0x00]);
        push_section(&mut priv_mem_mod, 7, &sec7_m);
        push_section(&mut priv_mem_mod, 10, &[1, 4, 0, 0x20, 0x00, 0x0b]);

        let mut p_mem = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&priv_mem_mod),
        );
        p_mem.limits.max_memory_pages = 4;
        assert_eq!(
            exec.execute_scalar_i32(&p_mem, &priv_mem_mod, "ok", 7)
                .unwrap_err(),
            WasmSandboxError::MemoryLimitExceeded {
                requested_pages: 5,
                max_pages: 4,
            }
        );

        // 3. Unbounded recursion (`call 0` in fn 0) traps via narrowed recursion/stack limit
        //    long before exhausting a huge 10M fuel allowance!
        let rec_wasm = build_single_export_wasm(
            &[0x7f],
            &[0x7f],
            None,
            0,
            "recurse",
            &[0x20, 0x00, 0x10, 0x00],
        );
        let mut rec_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&rec_wasm),
        );
        rec_policy.limits.max_fuel = 10_000_000;
        let rec_err = exec
            .execute_scalar_i32(&rec_policy, &rec_wasm, "recurse", 1)
            .unwrap_err();
        assert!(
            matches!(rec_err, WasmSandboxError::ExecutionTrap(_)),
            "deep recursion must trap on call stack depth rather than OutOfFuel: got {rec_err:?}"
        );
    }

    #[test]
    fn wasmi_rejects_max_memories_zero_exported_multitable_table_grow_and_zero_fuel_edges() {
        let exec = WasmiExtensionExecutorV1::new();

        // 1. `max_memories = 0` with an un-exported private `(memory 1)` must fail `MultipleMemoriesForbidden`.
        let mut priv_mem_mod = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut priv_mem_mod, 1, &[1, 0x60, 1, 0x7f, 1, 0x7f]);
        push_section(&mut priv_mem_mod, 3, &[1, 0]);
        push_section(&mut priv_mem_mod, 5, &[1, 0x00, 1]);
        let mut sec7 = vec![1];
        push_name(&mut sec7, "ok");
        sec7.extend_from_slice(&[0x00, 0x00]);
        push_section(&mut priv_mem_mod, 7, &sec7);
        push_section(&mut priv_mem_mod, 10, &[1, 4, 0, 0x20, 0x00, 0x0b]);

        let mut zero_mem_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&priv_mem_mod),
        );
        zero_mem_policy.limits.max_memories = 0;
        assert_eq!(
            exec.execute_scalar_i32(&zero_mem_policy, &priv_mem_mod, "ok", 7)
                .unwrap_err(),
            WasmSandboxError::MultipleMemoriesForbidden
        );

        // `max_memories = 0` with an exported `(memory 1)` also fails `MultipleMemoriesForbidden`.
        let exp_mem_mod =
            build_single_export_wasm(&[0x7f], &[0x7f], Some((1, Some(1))), 0, "ok", &[0x20, 0x00]);
        let mut zero_exp_mem_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&exp_mem_mod),
        );
        zero_exp_mem_policy.limits.max_memories = 0;
        assert_eq!(
            exec.execute_scalar_i32(&zero_exp_mem_policy, &exp_mem_mod, "ok", 7)
                .unwrap_err(),
            WasmSandboxError::MultipleMemoriesForbidden
        );

        // 2. Exported multi-table module (2 exported funcref tables) must fail `MultipleTablesForbidden`.
        let mut multi_table_mod = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut multi_table_mod, 1, &[1, 0x60, 1, 0x7f, 1, 0x7f]);
        push_section(&mut multi_table_mod, 3, &[1, 0]);
        push_section(&mut multi_table_mod, 4, &[2, 0x70, 0x00, 1, 0x70, 0x00, 1]);
        let mut sec7_mt = vec![3];
        push_name(&mut sec7_mt, "ok");
        sec7_mt.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7_mt, "t0");
        sec7_mt.extend_from_slice(&[0x01, 0x00]);
        push_name(&mut sec7_mt, "t1");
        sec7_mt.extend_from_slice(&[0x01, 0x01]);
        push_section(&mut multi_table_mod, 7, &sec7_mt);
        push_section(&mut multi_table_mod, 10, &[1, 4, 0, 0x20, 0x00, 0x0b]);

        let multi_table_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&multi_table_mod),
        );
        assert_eq!(
            exec.execute_scalar_i32(&multi_table_policy, &multi_table_mod, "ok", 7)
                .unwrap_err(),
            WasmSandboxError::MultipleTablesForbidden
        );

        // 3. `table.grow` (`0xfc 0x0f 0x00`) is rejected at validation (`wasm_reference_types(false)`).
        let mut table_grow_mod = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut table_grow_mod, 1, &[1, 0x60, 1, 0x7f, 1, 0x7f]);
        push_section(&mut table_grow_mod, 3, &[1, 0]);
        push_section(&mut table_grow_mod, 4, &[1, 0x70, 0x00, 1]);
        let mut sec7_tg = vec![2];
        push_name(&mut sec7_tg, "grow_tbl");
        sec7_tg.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7_tg, "tbl");
        sec7_tg.extend_from_slice(&[0x01, 0x00]);
        push_section(&mut table_grow_mod, 7, &sec7_tg);
        // body: unreachable (polymorphic stack), table.grow 0, drop, i32.const 0, end
        push_section(
            &mut table_grow_mod,
            10,
            &[1, 9, 0, 0x00, 0xfc, 0x0f, 0x00, 0x1a, 0x41, 0x00, 0x0b],
        );

        let table_grow_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&table_grow_mod),
        );
        assert_eq!(
            exec.execute_scalar_i32(&table_grow_policy, &table_grow_mod, "grow_tbl", 1)
                .unwrap_err(),
            WasmSandboxError::MalformedSectionOrLeb128
        );

        // 4. `policy.limits.max_fuel == 0` fails closed in preflight (`OutOfFuel { 0, 0 }`).
        let ok_mod = build_single_export_wasm(&[0x7f], &[0x7f], None, 0, "id", &[0x20, 0x00]);
        let mut zero_fuel_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&ok_mod),
        );
        zero_fuel_policy.limits.max_fuel = 0;
        assert_eq!(
            exec.execute_scalar_i32(&zero_fuel_policy, &ok_mod, "id", 9)
                .unwrap_err(),
            WasmSandboxError::OutOfFuel {
                consumed: 0,
                limit: 0,
            }
        );

        let byte_mod = build_ascii_xor_wasm();
        let mut zero_byte_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&byte_mod),
        );
        zero_byte_policy.limits.max_fuel = 0;
        assert_eq!(
            exec.execute_byte_transform(&zero_byte_policy, &byte_mod, "transform", b"ab")
                .unwrap_err(),
            WasmSandboxError::OutOfFuel {
                consumed: 0,
                limit: 0,
            }
        );

        // 5. `max_start_fuel == 0` with `max_fuel > 0`:
        //    - Module WITHOUT `start` section executes cleanly.
        let mut zero_start_no_sec = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&ok_mod),
        );
        zero_start_no_sec.limits.max_start_fuel = 0;
        let (val, _) = exec
            .execute_scalar_i32(&zero_start_no_sec, &ok_mod, "id", 42)
            .unwrap();
        assert_eq!(val, 42);

        //    - Module WITH a `start` section fails instantiation cleanly with OutOfFuel { 0, 0 }.
        let mut with_start_mod = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(
            &mut with_start_mod,
            1,
            &[2, 0x60, 0, 0, 0x60, 1, 0x7f, 1, 0x7f],
        );
        push_section(&mut with_start_mod, 3, &[2, 0, 1]);
        let mut sec7_s = vec![1];
        push_name(&mut sec7_s, "run");
        sec7_s.extend_from_slice(&[0x00, 0x01]);
        push_section(&mut with_start_mod, 7, &sec7_s);
        push_section(&mut with_start_mod, 8, &[0]);
        // start fn 0: nop; end (non-infinite, still disallowed when max_start_fuel == 0)
        // export fn 1: local.get 0; end
        let body0 = [0x00u8, 0x01, 0x0b];
        let body1 = [0x00u8, 0x20, 0x00, 0x0b];
        let mut sec10_s = vec![2];
        push_leb_u32(&mut sec10_s, body0.len() as u32);
        sec10_s.extend_from_slice(&body0);
        push_leb_u32(&mut sec10_s, body1.len() as u32);
        sec10_s.extend_from_slice(&body1);
        push_section(&mut with_start_mod, 10, &sec10_s);

        let mut zero_start_with_sec = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&with_start_mod),
        );
        zero_start_with_sec.limits.max_fuel = 100_000;
        zero_start_with_sec.limits.max_start_fuel = 0;
        assert_eq!(
            exec.execute_scalar_i32(&zero_start_with_sec, &with_start_mod, "run", 7)
                .unwrap_err(),
            WasmSandboxError::OutOfFuel {
                consumed: 0,
                limit: 0,
            }
        );

        // When `max_start_fuel` allows `start` to finish, `start` fuel is charged against total `max_fuel`.
        zero_start_with_sec.limits.max_start_fuel = 100;
        let (run_ret, run_receipt) = exec
            .execute_scalar_i32(&zero_start_with_sec, &with_start_mod, "run", 7)
            .unwrap();
        assert_eq!(run_ret, 7);
        assert!(run_receipt.fuel_consumed >= 2);

        // 6. `timeout_ms == 0` fails closed pre-compile with `ExecutionTimeout { limit_ms: 0, .. }`.
        let mut zero_timeout_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&ok_mod),
        );
        zero_timeout_policy.limits.timeout_ms = 0;
        assert!(matches!(
            exec.execute_scalar_i32(&zero_timeout_policy, &ok_mod, "id", 7),
            Err(WasmSandboxError::ExecutionTimeout { limit_ms: 0, .. })
        ));
    }

    #[test]
    fn wasmi_enforces_64kib_value_stack_height_below_call_depth_and_reuses_shared_engine() {
        let exec = WasmiExtensionExecutorV1::new();

        // Depth-8 recursion where each frame allocates 1,500 i64 locals (~12 KiB/frame):
        // 8 * 12 KiB = 96 KiB > 64 KiB hardened max_stack_height (would succeed under wasmi's 1 MiB default!),
        // while depth 8 < DEFAULT_WASM_MAX_CALL_DEPTH (64), isolating value-stack enforcement.
        let mut stack_bomb = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(&mut stack_bomb, 1, &[1, 0x60, 1, 0x7f, 1, 0x7f]);
        push_section(&mut stack_bomb, 3, &[1, 0]);
        let mut sec7 = vec![1];
        push_name(&mut sec7, "deep_locals");
        sec7.extend_from_slice(&[0x00, 0x00]);
        push_section(&mut stack_bomb, 7, &sec7);

        let mut body = Vec::new();
        body.push(1); // 1 local decl group
        push_leb_u32(&mut body, 1_500);
        body.push(0x7e); // i64
        body.extend_from_slice(&[
            0x20, 0x00, // local.get 0
            0x41, 0x00, // i32.const 0
            0x4a, // i32.gt_s
            0x04, 0x7f, // if (result i32)
            0x20, 0x00, // local.get 0
            0x41, 0x01, // i32.const 1
            0x6b, // i32.sub
            0x10, 0x00, // call 0
            0x05, // else
            0x41, 0x00, // i32.const 0
            0x0b, // end if
            0x0b, // end func
        ]);
        let mut sec10 = vec![1];
        push_leb_u32(&mut sec10, body.len() as u32);
        sec10.extend_from_slice(&body);
        push_section(&mut stack_bomb, 10, &sec10);

        let mut policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&stack_bomb),
        );
        policy.limits.max_fuel = 10_000_000;
        let (ok_ret, _) = exec
            .execute_scalar_i32(&policy, &stack_bomb, "deep_locals", 1)
            .unwrap();
        assert_eq!(ok_ret, 0);

        let err = exec
            .execute_scalar_i32(&policy, &stack_bomb, "deep_locals", 8)
            .unwrap_err();
        assert!(
            matches!(err, WasmSandboxError::ExecutionTrap(ref msg) if msg.contains("call stack exhausted")),
            "expected value stack overflow ExecutionTrap before fuel/recursion exhaustion, got {err:?}"
        );

        // Concurrent reuse of the shared OnceLock<Engine> across threads must remain isolated and deterministic.
        let byte_mod = Arc::new(build_ascii_xor_wasm());
        let byte_policy = Arc::new(SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&byte_mod),
        ));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let m = Arc::clone(&byte_mod);
                let p = Arc::clone(&byte_policy);
                std::thread::spawn(move || {
                    let r = WasmiExtensionExecutorV1::new()
                        .execute_byte_transform(&p, &m, "transform", b"AbCdEf")
                        .unwrap();
                    assert_eq!(r.output_bytes, b"aBcDeF");
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    fn push_sleb_i32(out: &mut Vec<u8>, mut val: i32) {
        loop {
            let byte = (val & 0x7f) as u8;
            val >>= 7;
            let done = (val == 0 && (byte & 0x40) == 0) || (val == -1 && (byte & 0x40) != 0);
            if done {
                out.push(byte);
                break;
            } else {
                out.push(byte | 0x80);
            }
        }
    }

    fn push_sleb_i64(out: &mut Vec<u8>, mut val: i64) {
        loop {
            let byte = (val & 0x7f) as u8;
            val >>= 7;
            let done = (val == 0 && (byte & 0x40) == 0) || (val == -1 && (byte & 0x40) != 0);
            if done {
                out.push(byte);
                break;
            } else {
                out.push(byte | 0x80);
            }
        }
    }

    fn push_i32_const(out: &mut Vec<u8>, val: i32) {
        out.push(0x41);
        push_sleb_i32(out, val);
    }

    fn push_i64_const(out: &mut Vec<u8>, val: i64) {
        out.push(0x42);
        push_sleb_i64(out, val);
    }

    fn build_alloc_abi_test_wasm(
        rodata_at_0: &[u8],
        dealloc_ops: &[u8],
        transform_local_i32s: u32,
        transform_ops: &[u8],
    ) -> Vec<u8> {
        let mut wasm = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();

        // Section 1: Type
        // type 0: (i32) -> (i32)       [swarmtorch_alloc]
        // type 1: (i32, i32) -> ()     [swarmtorch_dealloc]
        // type 2: (i32, i32) -> (i64)  [transform]
        let sec1 = vec![
            3, 0x60, 1, 0x7f, 1, 0x7f, // type 0
            0x60, 2, 0x7f, 0x7f, 0, // type 1
            0x60, 2, 0x7f, 0x7f, 1, 0x7e, // type 2
        ];
        push_section(&mut wasm, 1, &sec1);

        // Section 3: Function [alloc=0, dealloc=1, transform=2]
        push_section(&mut wasm, 3, &[3, 0, 1, 2]);

        // Section 5: Memory (1 page min, 2 pages max)
        push_section(&mut wasm, 5, &[1, 0x01, 1, 2]);

        // Section 6: Globals
        // global 0: mut i32 heap_ptr = 4096
        // global 1: mut i32 dealloc_calls = 0
        let mut sec6 = vec![2];
        sec6.extend_from_slice(&[0x7f, 0x01]);
        push_i32_const(&mut sec6, 4096);
        sec6.push(0x0b);
        sec6.extend_from_slice(&[0x7f, 0x01]);
        push_i32_const(&mut sec6, 0);
        sec6.push(0x0b);
        push_section(&mut wasm, 6, &sec6);

        // Section 7: Export
        let mut sec7 = vec![4];
        push_name(&mut sec7, "memory");
        sec7.extend_from_slice(&[0x02, 0x00]);
        push_name(&mut sec7, SWARMTORCH_WASM_ALLOC_EXPORT_V1);
        sec7.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7, SWARMTORCH_WASM_DEALLOC_EXPORT_V1);
        sec7.extend_from_slice(&[0x00, 0x01]);
        push_name(&mut sec7, "transform");
        sec7.extend_from_slice(&[0x00, 0x02]);
        push_section(&mut wasm, 7, &sec7);

        // Section 10: Code
        // Func 0: bump allocator starting at global 0 (4096)
        let mut body0 = vec![1, 1, 0x7f]; // 1 local i32 (ptr)
        body0.extend_from_slice(&[
            0x23, 0x00, // global.get 0
            0x21, 0x01, // local.set 1 (ptr)
            0x20, 0x01, // local.get 1
            0x20, 0x00, // local.get 0 (len)
            0x6a, // i32.add
            0x24, 0x00, // global.set 0
            0x20, 0x01, // local.get 1
            0x0b, // end
        ]);

        // Func 1: dealloc
        let mut body1 = vec![0]; // 0 locals
        body1.extend_from_slice(dealloc_ops);
        body1.push(0x0b);

        // Func 2: transform
        let mut body2 = Vec::new();
        if transform_local_i32s > 0 {
            body2.push(1);
            push_leb_u32(&mut body2, transform_local_i32s);
            body2.push(0x7f);
        } else {
            body2.push(0);
        }
        body2.extend_from_slice(transform_ops);
        body2.push(0x0b);

        let mut sec10 = vec![3];
        for body in [&body0, &body1, &body2] {
            push_leb_u32(&mut sec10, body.len() as u32);
            sec10.extend_from_slice(body);
        }
        push_section(&mut wasm, 10, &sec10);

        // Section 11: Active Data segment at memory 0 offset 0
        if !rodata_at_0.is_empty() {
            let mut sec11 = vec![1, 0x00, 0x41, 0x00, 0x0b];
            push_leb_u32(&mut sec11, rodata_at_0.len() as u32);
            sec11.extend_from_slice(rodata_at_0);
            push_section(&mut wasm, 11, &sec11);
        }

        wasm
    }

    /// Generate `swarmtorch_dealloc` bytecode that asserts it is called strictly ONCE with `(exp_ptr, exp_len)`
    /// and traps (`unreachable`) on any mismatch or double-free.
    fn dealloc_assert_once(exp_ptr: i32, exp_len: i32) -> Vec<u8> {
        let mut ops = Vec::new();
        // if global.get 1 != 0 { unreachable }
        ops.extend_from_slice(&[0x23, 0x01, 0x04, 0x40, 0x00, 0x0b]);
        // if local.get 0 != exp_ptr { unreachable }
        ops.extend_from_slice(&[0x20, 0x00]);
        push_i32_const(&mut ops, exp_ptr);
        ops.extend_from_slice(&[0x47, 0x04, 0x40, 0x00, 0x0b]);
        // if local.get 1 != exp_len { unreachable }
        ops.extend_from_slice(&[0x20, 0x01]);
        push_i32_const(&mut ops, exp_len);
        ops.extend_from_slice(&[0x47, 0x04, 0x40, 0x00, 0x0b]);
        // global.set 1 = 1
        ops.extend_from_slice(&[0x41, 0x01, 0x24, 0x01]);
        ops
    }

    /// Generate `swarmtorch_dealloc` bytecode that asserts it is called strictly TWICE:
    /// first with `(ptr0, len0)` and second with `(ptr1, len1)`, trapping on any mismatch or 3rd call.
    fn dealloc_assert_twice(ptr0: i32, len0: i32, ptr1: i32, len1: i32) -> Vec<u8> {
        let mut ops = Vec::new();
        // if global.get 1 == 0 { assert (ptr0, len0); global 1 = 1; return }
        ops.extend_from_slice(&[0x23, 0x01, 0x45, 0x04, 0x40]);
        ops.extend_from_slice(&[0x20, 0x00]);
        push_i32_const(&mut ops, ptr0);
        ops.extend_from_slice(&[0x47, 0x04, 0x40, 0x00, 0x0b]);
        ops.extend_from_slice(&[0x20, 0x01]);
        push_i32_const(&mut ops, len0);
        ops.extend_from_slice(&[0x47, 0x04, 0x40, 0x00, 0x0b]);
        ops.extend_from_slice(&[0x41, 0x01, 0x24, 0x01, 0x0f, 0x0b]);

        // if global.get 1 == 1 { assert (ptr1, len1); global 1 = 2; return }
        ops.extend_from_slice(&[0x23, 0x01, 0x41, 0x01, 0x46, 0x04, 0x40]);
        ops.extend_from_slice(&[0x20, 0x00]);
        push_i32_const(&mut ops, ptr1);
        ops.extend_from_slice(&[0x47, 0x04, 0x40, 0x00, 0x0b]);
        ops.extend_from_slice(&[0x20, 0x01]);
        push_i32_const(&mut ops, len1);
        ops.extend_from_slice(&[0x47, 0x04, 0x40, 0x00, 0x0b]);
        ops.extend_from_slice(&[0x41, 0x02, 0x24, 0x01, 0x0f, 0x0b]);

        // Any 3rd call traps!
        ops.push(0x00);
        ops
    }

    /// Build a `SwarmTorchAllocV1` module with `.rodata` at offset 0 (`b"RODATA_MAGIC_V1!"`) that
    /// allocates a disjoint output buffer (`16 + in_len`), concatenates `.rodata[0..16] || input`,
    /// and validates both `swarmtorch_dealloc` calls strictly.
    fn build_rodata_prefix_disjoint_alloc_wasm(expected_in_len: i32) -> Vec<u8> {
        let rodata = b"RODATA_MAGIC_V1!";
        let expected_out_ptr = 4096 + expected_in_len;
        let expected_out_len = 16 + expected_in_len;
        let dealloc_ops =
            dealloc_assert_twice(4096, expected_in_len, expected_out_ptr, expected_out_len);

        // transform(in_ptr=0, in_len=1) -> i64; locals: out_len=2, out_ptr=3, i=4
        let mut t = Vec::new();
        // out_len = 16 + in_len
        t.extend_from_slice(&[0x41, 16, 0x20, 0x01, 0x6a, 0x21, 0x02]);
        // out_ptr = call swarmtorch_alloc(out_len)
        t.extend_from_slice(&[0x20, 0x02, 0x10, 0x00, 0x21, 0x03]);
        // Loop 1: copy 16 bytes from mem[0..16] to mem[out_ptr..out_ptr+16]
        t.extend_from_slice(&[
            0x41, 0x00, 0x21, 0x04, // i = 0
            0x03, 0x40, // loop
            0x20, 0x04, 0x41, 16, 0x48, // i < 16
            0x04, 0x40, // if
            0x20, 0x03, 0x20, 0x04, 0x6a, // out_ptr + i
            0x20, 0x04, 0x2d, 0x00, 0x00, // mem.load8_u(i)
            0x3a, 0x00, 0x00, // mem.store8
            0x20, 0x04, 0x41, 0x01, 0x6a, 0x21, 0x04, // i += 1
            0x0c, 0x01, // br 1 (continue loop)
            0x0b, 0x0b,
        ]);
        // Loop 2: copy in_len bytes from mem[in_ptr..in_ptr+in_len] to mem[out_ptr+16..]
        t.extend_from_slice(&[
            0x41, 0x00, 0x21, 0x04, // i = 0
            0x03, 0x40, // loop
            0x20, 0x04, 0x20, 0x01, 0x48, // i < in_len
            0x04, 0x40, // if
            0x20, 0x03, 0x41, 16, 0x6a, 0x20, 0x04, 0x6a, // out_ptr + 16 + i
            0x20, 0x00, 0x20, 0x04, 0x6a, 0x2d, 0x00, 0x00, // mem.load8_u(in_ptr + i)
            0x3a, 0x00, 0x00, // mem.store8
            0x20, 0x04, 0x41, 0x01, 0x6a, 0x21, 0x04, // i += 1
            0x0c, 0x01, // br 1
            0x0b, 0x0b,
        ]);
        // return ((out_ptr as u64) << 32) | (out_len as u64)
        t.extend_from_slice(&[
            0x20, 0x03, 0xad, // i64.extend_i32_u(out_ptr)
            0x42, 32, 0x86, // i64.shl 32
            0x20, 0x02, 0xad, // i64.extend_i32_u(out_len)
            0x84, // i64.or
        ]);

        build_alloc_abi_test_wasm(rodata, &dealloc_ops, 3, &t)
    }

    #[test]
    fn wave16c_swarmtorch_alloc_v1_preserves_rodata_and_enforces_strict_aliasing_and_dealloc() {
        let exec = WasmiExtensionExecutorV1::new();

        // 1. Disjoint fresh output allocation + `.rodata` at linear memory offset 0 preserved!
        //    Dealloc strictly asserts both (4096, 5) and (4101, 21) are freed exactly once!
        let disjoint_wasm = build_rodata_prefix_disjoint_alloc_wasm(5);
        let disjoint_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&disjoint_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let receipt = exec
            .execute_byte_transform(&disjoint_policy, &disjoint_wasm, "transform", b"hello")
            .unwrap();
        assert_eq!(receipt.return_code, 21);
        assert_eq!(receipt.output_bytes, b"RODATA_MAGIC_V1!hello");
        assert!(receipt.fuel_consumed > 0);

        // 2. Exact in-place alias ((4096, 6) -> (4096, 6)): dealloc must be called ONCE with (4096, 6).
        let mut inplace_ops = vec![
            0x41, 0x00, 0x21, 0x02, // i = 0
            0x03, 0x40, // loop
            0x20, 0x02, 0x20, 0x01, 0x48, // i < in_len
            0x04, 0x40, // if
            0x20, 0x00, 0x20, 0x02, 0x6a, // addr = in_ptr + i
            0x20, 0x00, 0x20, 0x02, 0x6a, 0x2d, 0x00, 0x00, 0x41, 0x20, 0x73, // ^ 0x20
            0x3a, 0x00, 0x00, // store8
            0x20, 0x02, 0x41, 0x01, 0x6a, 0x21, 0x02, // i += 1
            0x0c, 0x01, 0x0b, 0x0b, // return ((in_ptr as u64) << 32) | (in_len as u64)
            0x20, 0x00, 0xad, 0x42, 32, 0x86, 0x20, 0x01, 0xad, 0x84,
        ];
        let exact_alias_wasm =
            build_alloc_abi_test_wasm(b"RODATA", &dealloc_assert_once(4096, 6), 1, &inplace_ops);
        let exact_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&exact_alias_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let r_exact = exec
            .execute_byte_transform(&exact_policy, &exact_alias_wasm, "transform", b"abcDEF")
            .unwrap();
        assert_eq!(r_exact.return_code, 6);
        assert_eq!(r_exact.output_bytes, b"ABCdef");

        // 3. Shortened prefix alias ((4096, 100) -> (4096, 20)):
        //    Host MUST read 20 bytes AND call swarmtorch_dealloc(4096, 100) ONCE with original in_len=100!
        //    (dealloc_assert_once(4096, 100) traps if len == 20 or if called twice!)
        let mut prefix_ops = Vec::new();
        prefix_ops.extend_from_slice(&[0x20, 0x00, 0xad, 0x42, 32, 0x86]); // (in_ptr as u64) << 32
        push_i64_const(&mut prefix_ops, 20);
        prefix_ops.push(0x84); // i64.or
        let prefix_wasm =
            build_alloc_abi_test_wasm(b"RODATA", &dealloc_assert_once(4096, 100), 0, &prefix_ops);
        let prefix_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&prefix_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let input_100: Vec<u8> = (0u8..100u8).collect();
        let r_prefix = exec
            .execute_byte_transform(&prefix_policy, &prefix_wasm, "transform", &input_100)
            .unwrap();
        assert_eq!(r_prefix.return_code, 20);
        assert_eq!(r_prefix.output_bytes, &input_100[..20]);

        // 4. Interior pointer ((4096 + 10, 20)) MUST fail closed with OutputAliasingViolation!
        let mut interior_ops = Vec::new();
        interior_ops.extend_from_slice(&[0x20, 0x00, 0x41, 10, 0x6a, 0xad, 0x42, 32, 0x86]);
        push_i64_const(&mut interior_ops, 20);
        interior_ops.push(0x84);
        let interior_wasm = build_alloc_abi_test_wasm(b"", &[], 0, &interior_ops);
        let interior_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&interior_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&interior_policy, &interior_wasm, "transform", &input_100)
                .unwrap_err(),
            WasmSandboxError::OutputAliasingViolation {
                in_ptr: 4096,
                in_len: 100,
                out_ptr: 4106,
                out_len: 20,
            }
        );

        // 5. Overgrown in-place alias ((4096, 120) for in_len=100) MUST fail with OutputAliasingViolation!
        let mut overgrown_ops = Vec::new();
        overgrown_ops.extend_from_slice(&[0x20, 0x00, 0xad, 0x42, 32, 0x86]);
        push_i64_const(&mut overgrown_ops, 120);
        overgrown_ops.push(0x84);
        let overgrown_wasm = build_alloc_abi_test_wasm(b"", &[], 0, &overgrown_ops);
        let overgrown_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&overgrown_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(
                &overgrown_policy,
                &overgrown_wasm,
                "transform",
                &input_100,
            )
            .unwrap_err(),
            WasmSandboxError::OutputAliasingViolation {
                in_ptr: 4096,
                in_len: 100,
                out_ptr: 4096,
                out_len: 120,
            }
        );

        // 6. Partial left-overlap ((4080, 32) overlapping [4096, 4196)) MUST fail with OutputAliasingViolation!
        let mut left_overlap_ops = Vec::new();
        push_i64_const(&mut left_overlap_ops, ((4080u64 << 32) | 32u64) as i64);
        let left_overlap_wasm = build_alloc_abi_test_wasm(b"", &[], 0, &left_overlap_ops);
        let left_overlap_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&left_overlap_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(
                &left_overlap_policy,
                &left_overlap_wasm,
                "transform",
                &input_100,
            )
            .unwrap_err(),
            WasmSandboxError::OutputAliasingViolation {
                in_ptr: 4096,
                in_len: 100,
                out_ptr: 4080,
                out_len: 32,
            }
        );

        // 7. Empty output (out_len == 0): frees (4096, 5) once and never frees (0, 0).
        let mut empty_out_ops = Vec::new();
        push_i64_const(&mut empty_out_ops, 0);
        let empty_out_wasm =
            build_alloc_abi_test_wasm(b"RODATA", &dealloc_assert_once(4096, 5), 0, &empty_out_ops);
        let empty_out_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&empty_out_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let r_empty = exec
            .execute_byte_transform(&empty_out_policy, &empty_out_wasm, "transform", b"hello")
            .unwrap();
        assert_eq!(r_empty.return_code, 0);
        assert!(r_empty.output_bytes.is_empty());

        // 8. Negative error return (-7i64) -> GuestTransformFailed { export: "transform", code: -7 }.
        let mut err_ops = Vec::new();
        push_i64_const(&mut err_ops, -7);
        let err_wasm = build_alloc_abi_test_wasm(b"", &[], 0, &err_ops);
        let err_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&err_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&err_policy, &err_wasm, "transform", b"hi")
                .unwrap_err(),
            WasmSandboxError::GuestTransformFailed {
                export: "transform".to_string(),
                code: -7,
            }
        );

        // 9. Oversized output (> policy.limits.max_output_bytes) rejected BEFORE read_guest_slice!
        let mut big_out_ops = Vec::new();
        push_i64_const(&mut big_out_ops, ((8192u64 << 32) | 1024u64) as i64);
        let big_out_wasm = build_alloc_abi_test_wasm(b"", &[], 0, &big_out_ops);
        let mut big_out_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&big_out_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        big_out_policy.limits.max_output_bytes = 512;
        assert_eq!(
            exec.execute_byte_transform(&big_out_policy, &big_out_wasm, "transform", b"hi")
                .unwrap_err(),
            WasmSandboxError::OutputTooLarge {
                len: 1024,
                max: 512,
            }
        );

        // 10. Cleanup trap in swarmtorch_dealloc MUST fail closed (never silently ignored)!
        let trap_dealloc_wasm = build_alloc_abi_test_wasm(b"", &[0x00], 1, &inplace_ops);
        let trap_dealloc_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&trap_dealloc_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert!(matches!(
            exec.execute_byte_transform(
                &trap_dealloc_policy,
                &trap_dealloc_wasm,
                "transform",
                b"abc"
            ),
            Err(WasmSandboxError::ExecutionTrap(_))
        ));

        // 11. Cumulative fuel exhaustion inside swarmtorch_dealloc fails closed with OutOfFuel!
        let loop_dealloc_wasm =
            build_alloc_abi_test_wasm(b"", &[0x03, 0x40, 0x0c, 0x00, 0x0b], 1, &inplace_ops);
        let mut loop_dealloc_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&loop_dealloc_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        loop_dealloc_policy.limits.max_fuel = 500;
        assert!(matches!(
            exec.execute_byte_transform(
                &loop_dealloc_policy,
                &loop_dealloc_wasm,
                "transform",
                b"a"
            ),
            Err(WasmSandboxError::OutOfFuel { limit: 500, .. })
        ));

        // 12. Missing swarmtorch_alloc export (legacy RawMem0V0 module run under SwarmTorchAllocV1) fails closed!
        let legacy_wasm = build_ascii_xor_wasm();
        let legacy_as_alloc_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&legacy_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&legacy_as_alloc_policy, &legacy_wasm, "transform", b"abc")
                .unwrap_err(),
            WasmSandboxError::ExportNotAllowed(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string())
        );

        // Silence unused mut warning if refactored
        inplace_ops.clear();
    }

    #[test]
    fn wave16c_scheduler_end_to_end_executes_swarmtorch_alloc_v1_with_rodata() {
        let wasm = build_rodata_prefix_disjoint_alloc_wasm(6);
        let hash = SandboxedExtensionPolicyV1::sha256_module(&wasm);
        let hex_hash: String = hash.iter().map(|b| format!("{b:02x}")).collect();

        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("swarmtorch_wasmi_alloc_v1_e2e_{pid}"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([0x43; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(sink);
        let run_dir = session.sink().bundle().run_dir().to_path_buf();

        let input_bytes = b"abcDEF";
        let input_sha = SandboxedExtensionPolicyV1::sha256_module(input_bytes);
        let input_hex: String = input_sha.iter().map(|b| format!("{b:02x}")).collect();
        let cas_dir = run_dir.join("blobs").join("sha256");
        std::fs::create_dir_all(&cas_dir).unwrap();
        std::fs::write(cas_dir.join(&input_hex), input_bytes).unwrap();

        let asset_key = "dataset://ns/raw";
        let ingest = make_graph_node(
            "ingest/raw",
            "ingest",
            &[],
            &[asset_key],
            ExecutionTrust::Core,
        );
        session
            .register_source(
                asset_key,
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: format!("cas://sha256/{input_hex}"),
                    content_type: CONTENT_TYPE_OCTET_STREAM.to_string(),
                    auth_mode: AuthModeMarker::None,
                    etag_or_version: Some("v1".to_string()),
                },
                None,
                &ingest,
            )
            .unwrap();

        let mut wasm_node = make_graph_node(
            "node/wasm_alloc_op",
            "wasm_transform",
            &[asset_key],
            &["dataset://ns/wasm_alloc_out"],
            ExecutionTrust::SandboxedExtension,
        );
        wasm_node
            .params
            .insert("module_sha256".to_string(), CanonValue::Str(hex_hash));
        wasm_node.params.insert(
            "export".to_string(),
            CanonValue::Str("transform".to_string()),
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("wasmi-alloc-v1-e2e".to_string()),
            nodes: vec![wasm_node],
            edges: vec![],
        };

        let wasm_policy = SandboxedExtensionPolicyV1::with_allowed_hash(hash)
            .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let mut registry = BTreeMap::new();
        registry.insert(hash, wasm);
        let runner =
            NativeOpRunner::new().with_wasm_executor(Arc::new(WasmiExtensionExecutorV1::new()));

        let report = execute_graph_with_isolation(
            &graph,
            &mut session,
            &runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(wasm_policy),
            Some(&registry),
            RunId::from_bytes([0x55; 16]),
            test_clock,
        )
        .unwrap();
        assert!(
            report.failed_nodes.is_empty(),
            "unexpected failures: {:?}",
            report.failed_nodes
        );
        assert_eq!(report.executed_nodes, vec!["node/wasm_alloc_op"]);

        let out_inst = &report.runner_outputs["node/wasm_alloc_op"][0];
        let digest = out_inst
            .uri
            .as_deref()
            .and_then(|u| u.strip_prefix("cas://sha256/"))
            .unwrap();
        let cas_bytes = std::fs::read(run_dir.join("blobs").join("sha256").join(digest)).unwrap();
        assert_eq!(cas_bytes, b"RODATA_MAGIC_V1!abcDEF");

        drop(session);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[allow(clippy::too_many_arguments)]
    fn build_custom_alloc_abi_wasm(
        min_pages: u32,
        max_pages: u32,
        rodata_at_0: &[u8],
        start_ops: Option<&[u8]>,
        alloc_local_i32s: u32,
        alloc_ops: &[u8],
        dealloc_local_i32s: u32,
        dealloc_ops: &[u8],
        transform_local_i32s: u32,
        transform_ops: &[u8],
    ) -> Vec<u8> {
        let mut wasm = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();

        // Types:
        // 0: (i32) -> (i32)      [alloc]
        // 1: (i32, i32) -> ()    [dealloc]
        // 2: (i32, i32) -> (i64) [transform]
        // 3: () -> ()            [optional start]
        let mut sec1 = vec![if start_ops.is_some() { 4 } else { 3 }];
        sec1.extend_from_slice(&[
            0x60, 1, 0x7f, 1, 0x7f, // type 0
            0x60, 2, 0x7f, 0x7f, 0, // type 1
            0x60, 2, 0x7f, 0x7f, 1, 0x7e, // type 2
        ]);
        if start_ops.is_some() {
            sec1.extend_from_slice(&[0x60, 0, 0]); // type 3
        }
        push_section(&mut wasm, 1, &sec1);

        // Functions: [alloc=0, dealloc=1, transform=2, optional start=3]
        if start_ops.is_some() {
            push_section(&mut wasm, 3, &[4, 0, 1, 2, 3]);
        } else {
            push_section(&mut wasm, 3, &[3, 0, 1, 2]);
        }

        // Memory
        let mut sec5 = vec![1, 0x01];
        push_leb_u32(&mut sec5, min_pages);
        push_leb_u32(&mut sec5, max_pages);
        push_section(&mut wasm, 5, &sec5);

        // Globals: global 0 = heap_ptr (4096), global 1 = dealloc_calls (0)
        let mut sec6 = vec![2];
        sec6.extend_from_slice(&[0x7f, 0x01]);
        push_i32_const(&mut sec6, 4096);
        sec6.push(0x0b);
        sec6.extend_from_slice(&[0x7f, 0x01]);
        push_i32_const(&mut sec6, 0);
        sec6.push(0x0b);
        push_section(&mut wasm, 6, &sec6);

        // Exports
        let mut sec7 = vec![4];
        push_name(&mut sec7, "memory");
        sec7.extend_from_slice(&[0x02, 0x00]);
        push_name(&mut sec7, SWARMTORCH_WASM_ALLOC_EXPORT_V1);
        sec7.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7, SWARMTORCH_WASM_DEALLOC_EXPORT_V1);
        sec7.extend_from_slice(&[0x00, 0x01]);
        push_name(&mut sec7, "transform");
        sec7.extend_from_slice(&[0x00, 0x02]);
        push_section(&mut wasm, 7, &sec7);

        // Optional start section (fn 3)
        if start_ops.is_some() {
            push_section(&mut wasm, 8, &[3]);
        }

        let encode_body = |local_i32s: u32, ops: &[u8]| -> Vec<u8> {
            let mut body = Vec::new();
            if local_i32s > 0 {
                body.push(1);
                push_leb_u32(&mut body, local_i32s);
                body.push(0x7f);
            } else {
                body.push(0);
            }
            body.extend_from_slice(ops);
            body.push(0x0b);
            body
        };

        let body0 = encode_body(alloc_local_i32s, alloc_ops);
        let body1 = encode_body(dealloc_local_i32s, dealloc_ops);
        let body2 = encode_body(transform_local_i32s, transform_ops);
        let mut bodies = vec![body0, body1, body2];
        if let Some(s_ops) = start_ops {
            bodies.push(encode_body(1, s_ops));
        }

        let mut sec10 = vec![bodies.len() as u8];
        for body in &bodies {
            push_leb_u32(&mut sec10, body.len() as u32);
            sec10.extend_from_slice(body);
        }
        push_section(&mut wasm, 10, &sec10);

        if !rodata_at_0.is_empty() {
            let mut sec11 = vec![1, 0x00, 0x41, 0x00, 0x0b];
            push_leb_u32(&mut sec11, rodata_at_0.len() as u32);
            sec11.extend_from_slice(rodata_at_0);
            push_section(&mut wasm, 11, &sec11);
        }

        wasm
    }

    fn default_bump_alloc_ops() -> Vec<u8> {
        vec![
            0x23, 0x00, // global.get 0
            0x21, 0x01, // local.set 1
            0x20, 0x01, // local.get 1
            0x20, 0x00, // local.get 0
            0x6a, // i32.add
            0x24, 0x00, // global.set 0
            0x20, 0x01, // local.get 1
        ]
    }

    #[test]
    fn wave16c_adversarial_edge_cases_empty_input_bad_ptrs_aliasing_rawmem0_cap_and_metering() {
        let exec = WasmiExtensionExecutorV1::new();

        // 1. Empty input (in_len == 0) with non-empty output:
        //    - Host MUST NOT call swarmtorch_alloc (we make alloc trap `unreachable` to prove it!).
        //    - Guest writes bytes at 4096 and returns (4096, 4).
        //    - Host MUST read b"OKAY" and call swarmtorch_dealloc(4096, 4) strictly ONCE.
        let mut empty_in_transform = Vec::new();
        // Assert in_ptr == 0 and in_len == 0, else trap
        empty_in_transform.extend_from_slice(&[0x20, 0x00, 0x04, 0x40, 0x00, 0x0b]);
        empty_in_transform.extend_from_slice(&[0x20, 0x01, 0x04, 0x40, 0x00, 0x0b]);
        for (idx, &b) in b"OKAY".iter().enumerate() {
            push_i32_const(&mut empty_in_transform, 4096 + idx as i32);
            push_i32_const(&mut empty_in_transform, i32::from(b));
            empty_in_transform.extend_from_slice(&[0x3a, 0x00, 0x00]);
        }
        push_i64_const(&mut empty_in_transform, ((4096u64 << 32) | 4u64) as i64);

        let empty_in_wasm = build_custom_alloc_abi_wasm(
            1,
            2,
            b"RODATA_SAFE",
            None,
            0,
            &[0x00], // alloc traps if host calls it!
            0,
            &dealloc_assert_once(4096, 4),
            0,
            &empty_in_transform,
        );
        let empty_in_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&empty_in_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let r = exec
            .execute_byte_transform(&empty_in_policy, &empty_in_wasm, "transform", b"")
            .unwrap();
        assert_eq!(r.return_code, 4);
        assert_eq!(r.output_bytes, b"OKAY");

        // 2. Empty input (in_len == 0) with out_ptr == 0 and out_len > 0 -> OutOfBoundsMemoryAccess
        let mut zero_out_ptr_ops = Vec::new();
        push_i64_const(&mut zero_out_ptr_ops, 8); // (0 << 32) | 8
        let zero_out_ptr_wasm = build_alloc_abi_test_wasm(b"", &[0x00], 0, &zero_out_ptr_ops);
        let zero_out_ptr_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&zero_out_ptr_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert!(matches!(
            exec.execute_byte_transform(&zero_out_ptr_policy, &zero_out_ptr_wasm, "transform", b""),
            Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                addr: 0,
                len: 8,
                ..
            })
        ));
        // Also when in_len > 0 (disjoint out_ptr == 0, out_len == 8 vs in_ptr == 4096) -> OutOfBoundsMemoryAccess
        assert!(matches!(
            exec.execute_byte_transform(
                &zero_out_ptr_policy,
                &zero_out_ptr_wasm,
                "transform",
                b"hello",
            ),
            Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                addr: 0,
                len: 8,
                ..
            })
        ));

        // 3. Empty input + empty output with non-zero dangling pointer (1, 0):
        //    Neither alloc nor dealloc may be called (both trap if invoked!).
        let mut dangling_empty_ops = Vec::new();
        push_i64_const(&mut dangling_empty_ops, (1u64 << 32) as i64);
        let dangling_empty_wasm = build_custom_alloc_abi_wasm(
            1,
            1,
            b"",
            None,
            0,
            &[0x00], // alloc traps
            0,
            &[0x00], // dealloc traps
            0,
            &dangling_empty_ops,
        );
        let dangling_empty_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&dangling_empty_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let r_dangling = exec
            .execute_byte_transform(
                &dangling_empty_policy,
                &dangling_empty_wasm,
                "transform",
                b"",
            )
            .unwrap();
        assert_eq!(r_dangling.return_code, 0);
        assert!(r_dangling.output_bytes.is_empty());

        // 4. Non-empty input + empty output with non-zero dangling out_ptr (4096, 0):
        //    Frees (4096, 5) once and NEVER frees (4096, 0).
        let mut nonnull_empty_out = Vec::new();
        push_i64_const(&mut nonnull_empty_out, (4096u64 << 32) as i64);
        let nonnull_empty_wasm =
            build_alloc_abi_test_wasm(b"", &dealloc_assert_once(4096, 5), 0, &nonnull_empty_out);
        let nonnull_empty_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&nonnull_empty_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let r_nn = exec
            .execute_byte_transform(
                &nonnull_empty_policy,
                &nonnull_empty_wasm,
                "transform",
                b"hello",
            )
            .unwrap();
        assert_eq!(r_nn.return_code, 0);
        assert!(r_nn.output_bytes.is_empty());

        // 5. Bad in_ptr from swarmtorch_alloc: 0, -1, i32::MIN, i32::MAX (u32 wrap), OOB (65530 for len=16).
        for bad_ptr in [0i32, -1i32, i32::MIN, i32::MAX, 65_530i32] {
            let mut alloc_bad = Vec::new();
            push_i32_const(&mut alloc_bad, bad_ptr);
            let mut t_noop = Vec::new();
            push_i64_const(&mut t_noop, 0);
            let bad_alloc_wasm =
                build_custom_alloc_abi_wasm(1, 1, b"", None, 0, &alloc_bad, 0, &[], 0, &t_noop);
            let bad_alloc_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
                SandboxedExtensionPolicyV1::sha256_module(&bad_alloc_wasm),
            )
            .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
            let err = exec
                .execute_byte_transform(
                    &bad_alloc_policy,
                    &bad_alloc_wasm,
                    "transform",
                    b"0123456789abcdef",
                )
                .unwrap_err();
            assert!(
                matches!(err, WasmSandboxError::OutOfBoundsMemoryAccess { .. }),
                "expected OutOfBoundsMemoryAccess for alloc ptr={bad_ptr}, got {err:?}"
            );
        }

        // 6. swarmtorch_alloc growing memory from 0 initial pages via memory.grow(1) succeeds & updates peak_memory_bytes.
        let mut grow_alloc = Vec::new();
        grow_alloc.extend_from_slice(&[0x41, 0x01, 0x40, 0x00, 0x1a]); // drop(memory.grow(1))
        push_i32_const(&mut grow_alloc, 4096);
        let inplace_id_ops = vec![0x20, 0x00, 0xad, 0x42, 32, 0x86, 0x20, 0x01, 0xad, 0x84];
        let grow_alloc_wasm = build_custom_alloc_abi_wasm(
            0,
            2,
            b"",
            None,
            0,
            &grow_alloc,
            0,
            &dealloc_assert_once(4096, 4),
            0,
            &inplace_id_ops,
        );
        let grow_alloc_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&grow_alloc_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let r_grow = exec
            .execute_byte_transform(&grow_alloc_policy, &grow_alloc_wasm, "transform", b"grow")
            .unwrap();
        assert_eq!(r_grow.output_bytes, b"grow");
        assert_eq!(r_grow.peak_memory_bytes, WASM_PAGE_BYTES);

        // 7. Disjoint out_ptr wraparound (i32::MAX, 16) and OOB (65530, 16) fail with OutOfBoundsMemoryAccess
        //    and NEVER call dealloc (dealloc traps to verify).
        for bad_out_ptr in [i32::MAX, 65_530i32] {
            let mut oob_out = Vec::new();
            push_i64_const(
                &mut oob_out,
                (((bad_out_ptr as u32 as u64) << 32) | 16u64) as i64,
            );
            let oob_out_wasm = build_alloc_abi_test_wasm(b"", &[0x00], 0, &oob_out);
            let oob_out_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
                SandboxedExtensionPolicyV1::sha256_module(&oob_out_wasm),
            )
            .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
            assert!(matches!(
                exec.execute_byte_transform(&oob_out_policy, &oob_out_wasm, "transform", b"hi"),
                Err(WasmSandboxError::OutOfBoundsMemoryAccess { .. })
            ));
        }

        // 8. Negative out_len ((4096 << 32) | 0x8000_0000) fails with OutOfBoundsMemoryAccess with live mem_len.
        let mut neg_len_ops = Vec::new();
        push_i64_const(&mut neg_len_ops, ((4096u64 << 32) | 0x8000_0000u64) as i64);
        let neg_len_wasm = build_alloc_abi_test_wasm(b"", &[0x00], 0, &neg_len_ops);
        let neg_len_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&neg_len_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert!(matches!(
            exec.execute_byte_transform(&neg_len_policy, &neg_len_wasm, "transform", b"hi"),
            Err(WasmSandboxError::OutOfBoundsMemoryAccess {
                addr: 4096,
                mem_len,
                ..
            }) if mem_len == WASM_PAGE_BYTES
        ));

        // 9. Adjacent-left disjoint buffer ([4080..4096) vs input [4096..4196)) succeeds and frees both once!
        let mut adj_left_ops = Vec::new();
        for (idx, &b) in b"ADJACENT_LEFT_OK".iter().enumerate() {
            push_i32_const(&mut adj_left_ops, 4080 + idx as i32);
            push_i32_const(&mut adj_left_ops, i32::from(b));
            adj_left_ops.extend_from_slice(&[0x3a, 0x00, 0x00]);
        }
        push_i64_const(&mut adj_left_ops, ((4080u64 << 32) | 16u64) as i64);
        let adj_left_wasm = build_alloc_abi_test_wasm(
            b"",
            &dealloc_assert_twice(4096, 100, 4080, 16),
            0,
            &adj_left_ops,
        );
        let adj_left_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&adj_left_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let input_100: Vec<u8> = (0u8..100u8).collect();
        let r_adj = exec
            .execute_byte_transform(&adj_left_policy, &adj_left_wasm, "transform", &input_100)
            .unwrap();
        assert_eq!(r_adj.output_bytes, b"ADJACENT_LEFT_OK");

        // 10. Right-edge overlap (4190, 16), suffix alias (4150, 46), and superset (4080, 132) -> OutputAliasingViolation!
        for (out_ptr, out_len) in [(4190i32, 16i32), (4150i32, 46i32), (4080i32, 132i32)] {
            let mut alias_ops = Vec::new();
            push_i64_const(
                &mut alias_ops,
                (((out_ptr as u32 as u64) << 32) | (out_len as u32 as u64)) as i64,
            );
            let alias_wasm = build_alloc_abi_test_wasm(b"", &[0x00], 0, &alias_ops);
            let alias_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
                SandboxedExtensionPolicyV1::sha256_module(&alias_wasm),
            )
            .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
            assert_eq!(
                exec.execute_byte_transform(&alias_policy, &alias_wasm, "transform", &input_100)
                    .unwrap_err(),
                WasmSandboxError::OutputAliasingViolation {
                    in_ptr: 4096,
                    in_len: 100,
                    out_ptr,
                    out_len,
                }
            );
        }

        // 11. RawMem0V0 sibling parity for max_output_bytes (exact boundary succeeds, +1 and >mem_len fail OutputTooLarge!).
        let raw_wasm = build_ascii_xor_wasm();
        let mut raw_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&raw_wasm),
        );
        raw_policy.limits.max_output_bytes = 6;
        let r_exact_cap = exec
            .execute_byte_transform(&raw_policy, &raw_wasm, "transform", b"abcDEF")
            .unwrap();
        assert_eq!(r_exact_cap.output_bytes, b"ABCdef");
        raw_policy.limits.max_output_bytes = 5;
        assert_eq!(
            exec.execute_byte_transform(&raw_policy, &raw_wasm, "transform", b"abcDEF")
                .unwrap_err(),
            WasmSandboxError::OutputTooLarge { len: 6, max: 5 }
        );
        // Even when returned len > mem_len (100_000 > 65_536) AND > max_output_bytes (512), OutputTooLarge fires BEFORE read_guest_slice!
        let oob_ret = build_single_export_wasm(
            &[0x7f, 0x7f],
            &[0x7f],
            Some((1, Some(1))),
            0,
            "oob",
            &[0x41, 0xa0, 0x8d, 0x06],
        );
        let mut oob_cap_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&oob_ret),
        );
        oob_cap_policy.limits.max_output_bytes = 512;
        assert_eq!(
            exec.execute_byte_transform(&oob_cap_policy, &oob_ret, "oob", b"hi")
                .unwrap_err(),
            WasmSandboxError::OutputTooLarge {
                len: 100_000,
                max: 512,
            }
        );

        // 12. SwarmTorchAllocV1 module accidentally run under RawMem0V0 -> ExportSignatureMismatch("transform"),
        //     plus missing swarmtorch_dealloc / wrong-signature alloc / wrong-signature dealloc / alloc trap & OOF!
        let alloc_mod = build_rodata_prefix_disjoint_alloc_wasm(5);
        let raw_policy_on_alloc = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&alloc_mod),
        );
        assert_eq!(
            exec.execute_byte_transform(&raw_policy_on_alloc, &alloc_mod, "transform", b"hello")
                .unwrap_err(),
            WasmSandboxError::ExportSignatureMismatch("transform".to_string())
        );

        // Missing swarmtorch_dealloc -> ExportNotAllowed("swarmtorch_dealloc")
        let mut missing_dealloc = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(
            &mut missing_dealloc,
            1,
            &[2, 0x60, 1, 0x7f, 1, 0x7f, 0x60, 2, 0x7f, 0x7f, 1, 0x7e],
        );
        push_section(&mut missing_dealloc, 3, &[2, 0, 1]);
        push_section(&mut missing_dealloc, 5, &[1, 0x01, 1, 1]);
        let mut sec7_md = vec![3];
        push_name(&mut sec7_md, "memory");
        sec7_md.extend_from_slice(&[0x02, 0x00]);
        push_name(&mut sec7_md, SWARMTORCH_WASM_ALLOC_EXPORT_V1);
        sec7_md.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7_md, "transform");
        sec7_md.extend_from_slice(&[0x00, 0x01]);
        push_section(&mut missing_dealloc, 7, &sec7_md);
        push_section(
            &mut missing_dealloc,
            10,
            &[2, 5, 0, 0x41, 0x80, 0x20, 0x0b, 4, 0, 0x42, 0x00, 0x0b],
        );
        let md_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&missing_dealloc),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&md_policy, &missing_dealloc, "transform", b"a")
                .unwrap_err(),
            WasmSandboxError::ExportNotAllowed(SWARMTORCH_WASM_DEALLOC_EXPORT_V1.to_string())
        );

        // Wrong-signature swarmtorch_alloc ((i32, i32) -> i32) -> ExportSignatureMismatch("swarmtorch_alloc")
        let mut bad_alloc_sig = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(
            &mut bad_alloc_sig,
            1,
            &[
                3, 0x60, 2, 0x7f, 0x7f, 1, 0x7f, 0x60, 2, 0x7f, 0x7f, 0, 0x60, 2, 0x7f, 0x7f, 1,
                0x7e,
            ],
        );
        push_section(&mut bad_alloc_sig, 3, &[3, 0, 1, 2]);
        push_section(&mut bad_alloc_sig, 5, &[1, 0x01, 1, 1]);
        let mut sec7_ba = vec![4];
        push_name(&mut sec7_ba, "memory");
        sec7_ba.extend_from_slice(&[0x02, 0x00]);
        push_name(&mut sec7_ba, SWARMTORCH_WASM_ALLOC_EXPORT_V1);
        sec7_ba.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7_ba, SWARMTORCH_WASM_DEALLOC_EXPORT_V1);
        sec7_ba.extend_from_slice(&[0x00, 0x01]);
        push_name(&mut sec7_ba, "transform");
        sec7_ba.extend_from_slice(&[0x00, 0x02]);
        push_section(&mut bad_alloc_sig, 7, &sec7_ba);
        push_section(
            &mut bad_alloc_sig,
            10,
            &[
                3, 4, 0, 0x41, 0x01, 0x0b, 2, 0, 0x0b, 4, 0, 0x42, 0x00, 0x0b,
            ],
        );
        let ba_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&bad_alloc_sig),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&ba_policy, &bad_alloc_sig, "transform", b"a")
                .unwrap_err(),
            WasmSandboxError::ExportSignatureMismatch(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string())
        );

        // Wrong-signature swarmtorch_dealloc ((i32, i32) -> i32) -> ExportSignatureMismatch("swarmtorch_dealloc")
        let mut bad_dealloc_sig = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(
            &mut bad_dealloc_sig,
            1,
            &[
                3, 0x60, 1, 0x7f, 1, 0x7f, 0x60, 2, 0x7f, 0x7f, 1, 0x7f, 0x60, 2, 0x7f, 0x7f, 1,
                0x7e,
            ],
        );
        push_section(&mut bad_dealloc_sig, 3, &[3, 0, 1, 2]);
        push_section(&mut bad_dealloc_sig, 5, &[1, 0x01, 1, 1]);
        push_section(&mut bad_dealloc_sig, 7, &sec7_ba);
        push_section(
            &mut bad_dealloc_sig,
            10,
            &[
                3, 4, 0, 0x41, 0x01, 0x0b, 4, 0, 0x41, 0x00, 0x0b, 4, 0, 0x42, 0x00, 0x0b,
            ],
        );
        let bd_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&bad_dealloc_sig),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&bd_policy, &bad_dealloc_sig, "transform", b"a")
                .unwrap_err(),
            WasmSandboxError::ExportSignatureMismatch(
                SWARMTORCH_WASM_DEALLOC_EXPORT_V1.to_string()
            )
        );

        // Trap & OOF inside swarmtorch_alloc fail closed cleanly!
        let alloc_trap_wasm =
            build_custom_alloc_abi_wasm(1, 1, b"", None, 0, &[0x00], 0, &[], 0, &[0x42, 0x00]);
        let alloc_trap_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&alloc_trap_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert!(matches!(
            exec.execute_byte_transform(&alloc_trap_policy, &alloc_trap_wasm, "transform", b"a"),
            Err(WasmSandboxError::ExecutionTrap(_))
        ));

        let alloc_loop_wasm = build_custom_alloc_abi_wasm(
            1,
            1,
            b"",
            None,
            0,
            &[0x03, 0x40, 0x0c, 0x00, 0x0b, 0x41, 0x01],
            0,
            &[],
            0,
            &[0x42, 0x00],
        );
        let mut alloc_loop_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&alloc_loop_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        alloc_loop_policy.limits.max_fuel = 200;
        assert!(matches!(
            exec.execute_byte_transform(&alloc_loop_policy, &alloc_loop_wasm, "transform", b"a"),
            Err(WasmSandboxError::OutOfFuel { limit: 200, .. })
        ));

        // 13. Wall-clock timeout inside swarmtorch_dealloc fails closed with ExecutionTimeout!
        let timeout_dealloc_wasm =
            build_alloc_abi_test_wasm(b"", &[0x03, 0x40, 0x0c, 0x00, 0x0b], 0, &inplace_id_ops);
        let mut timeout_dealloc_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&timeout_dealloc_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        timeout_dealloc_policy.limits.max_fuel = 100_000_000;
        timeout_dealloc_policy.limits.timeout_ms = 25;
        assert!(matches!(
            exec.execute_byte_transform(
                &timeout_dealloc_policy,
                &timeout_dealloc_wasm,
                "transform",
                b"a",
            ),
            Err(WasmSandboxError::ExecutionTimeout { limit_ms: 25, .. })
        ));

        // 14. Trap in the SECOND swarmtorch_dealloc(out_ptr, out_len) call (disjoint Case 3) fails closed!
        let trap_on_second_dealloc = vec![
            0x23, 0x01, 0x45, 0x04, 0x40, // if global 1 == 0
            0x41, 0x01, 0x24, 0x01, 0x0f, //   global 1 = 1; return
            0x0b, 0x00, // else unreachable!
        ];
        let mut disjoint_t = Vec::new();
        push_i64_const(&mut disjoint_t, ((8192u64 << 32) | 4u64) as i64);
        let second_dealloc_trap_wasm =
            build_alloc_abi_test_wasm(b"", &trap_on_second_dealloc, 0, &disjoint_t);
        let second_dealloc_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&second_dealloc_trap_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert!(matches!(
            exec.execute_byte_transform(
                &second_dealloc_policy,
                &second_dealloc_trap_wasm,
                "transform",
                b"abcd",
            ),
            Err(WasmSandboxError::ExecutionTrap(_))
        ));

        // 15. Hardening guest dealloc that zeroes/poisons freed buffer does NOT corrupt output_bytes
        //     (proves host copies output_bytes BEFORE invoking swarmtorch_dealloc).
        let poison_dealloc = vec![
            // mem[ptr] = 0xFF
            0x20, 0x00, 0x41, 0x7f, 0x3a, 0x00, 0x00,
        ];
        let poison_wasm = build_alloc_abi_test_wasm(b"", &poison_dealloc, 0, &inplace_id_ops);
        let poison_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&poison_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let r_poison = exec
            .execute_byte_transform(&poison_policy, &poison_wasm, "transform", b"SAFE")
            .unwrap();
        assert_eq!(r_poison.output_bytes, b"SAFE");

        // 16. Cumulative fuel metering across start + alloc + transform + dealloc:
        //     Each of start, alloc, transform, dealloc executes a small deterministic loop (~40 iterations).
        let make_spin_n = |local_idx: u8, n: i32| -> Vec<u8> {
            let mut ops = Vec::new();
            push_i32_const(&mut ops, n);
            ops.extend_from_slice(&[
                0x21, local_idx, // local.set i = n
                0x03, 0x40, // loop
                0x20, local_idx, 0x41, 0x00, 0x4a, // i > 0
                0x04, 0x40, // if
                0x20, local_idx, 0x41, 0x01, 0x6b, 0x21, local_idx, // i -= 1
                0x0c, 0x01, // br 1
                0x0b, 0x0b,
            ]);
            ops
        };
        let start_spin = make_spin_n(0, 40);
        let mut alloc_spin = make_spin_n(2, 40);
        alloc_spin.extend_from_slice(&default_bump_alloc_ops());
        let dealloc_spin = make_spin_n(2, 40);
        let mut transform_spin = make_spin_n(2, 40);
        transform_spin.extend_from_slice(&inplace_id_ops);

        let cum_wasm = build_custom_alloc_abi_wasm(
            1,
            1,
            b"",
            Some(&start_spin),
            2,
            &alloc_spin,
            1,
            &dealloc_spin,
            1,
            &transform_spin,
        );
        let mut cum_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&cum_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        cum_policy.limits.max_fuel = 100_000;
        let full_receipt = exec
            .execute_byte_transform(&cum_policy, &cum_wasm, "transform", b"fuel")
            .unwrap();
        assert_eq!(full_receipt.output_bytes, b"fuel");
        assert!(full_receipt.fuel_consumed > 100);

        // Tightening max_fuel to (fuel_consumed - 1) MUST fail with OutOfFuel inside dealloc,
        // proving cumulative budget carries across start + alloc + transform + dealloc without reset!
        let mut tight_policy = cum_policy.clone();
        tight_policy.limits.max_fuel = full_receipt.fuel_consumed - 1;
        assert!(matches!(
            exec.execute_byte_transform(&tight_policy, &cum_wasm, "transform", b"fuel"),
            Err(WasmSandboxError::OutOfFuel { limit, .. })
                if limit == full_receipt.fuel_consumed - 1
        ));
    }

    #[test]
    fn wave16c_adversarial_scheduler_cache_invalidates_across_guest_abi_and_max_output_bytes() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingExecutor {
            calls: AtomicUsize,
        }
        impl WasmExtensionExecutor for CountingExecutor {
            fn execute_byte_transform(
                &self,
                policy: &SandboxedExtensionPolicyV1,
                wasm_bytes: &[u8],
                export_name: &str,
                input_payload: &[u8],
            ) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
                let _ = preflight_wasm_extension_v1(policy, wasm_bytes, export_name)?;
                self.calls.fetch_add(1, Ordering::SeqCst);
                let mut out = input_payload.to_vec();
                out.push(match policy.guest_abi {
                    WasmGuestAbiV1::RawMem0V0 => b'0',
                    WasmGuestAbiV1::SwarmTorchAllocV1 => b'1',
                });
                Ok(WasmExecutionReceiptV1 {
                    module_sha256: SandboxedExtensionPolicyV1::sha256_module(wasm_bytes),
                    fuel_consumed: 10,
                    peak_memory_bytes: WASM_PAGE_BYTES,
                    return_code: out.len() as i32,
                    output_bytes: out,
                })
            }

            fn execute_scalar_i32(
                &self,
                _policy: &SandboxedExtensionPolicyV1,
                _wasm_bytes: &[u8],
                _export_name: &str,
                _arg: i32,
            ) -> Result<(i32, WasmExecutionReceiptV1), WasmSandboxError> {
                Err(WasmSandboxError::UnsupportedUntrustedWasmSandbox)
            }
        }

        let wasm = build_rodata_prefix_disjoint_alloc_wasm(6);
        let hash = SandboxedExtensionPolicyV1::sha256_module(&wasm);
        let hex_hash: String = hash.iter().map(|b| format!("{b:02x}")).collect();

        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("swarmtorch_wasmi_abi_cache_inval_{pid}"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([0x66; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(sink);
        let run_dir = session.sink().bundle().run_dir().to_path_buf();

        let input_bytes = b"abcDEF";
        let input_sha = SandboxedExtensionPolicyV1::sha256_module(input_bytes);
        let input_hex: String = input_sha.iter().map(|b| format!("{b:02x}")).collect();
        let cas_dir = run_dir.join("blobs").join("sha256");
        std::fs::create_dir_all(&cas_dir).unwrap();
        std::fs::write(cas_dir.join(&input_hex), input_bytes).unwrap();

        let asset_key = "dataset://ns/raw";
        let ingest = make_graph_node(
            "ingest/raw",
            "ingest",
            &[],
            &[asset_key],
            ExecutionTrust::Core,
        );
        session
            .register_source(
                asset_key,
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: format!("cas://sha256/{input_hex}"),
                    content_type: CONTENT_TYPE_OCTET_STREAM.to_string(),
                    auth_mode: AuthModeMarker::None,
                    etag_or_version: Some("v1".to_string()),
                },
                None,
                &ingest,
            )
            .unwrap();

        let mut wasm_node = make_graph_node(
            "node/wasm_abi_cache",
            "wasm_transform",
            &[asset_key],
            &["dataset://ns/wasm_abi_out"],
            ExecutionTrust::SandboxedExtension,
        );
        wasm_node
            .params
            .insert("module_sha256".to_string(), CanonValue::Str(hex_hash));
        wasm_node.params.insert(
            "export".to_string(),
            CanonValue::Str("transform".to_string()),
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("wasmi-abi-cache-inval".to_string()),
            nodes: vec![wasm_node],
            edges: vec![],
        };

        let mut registry = BTreeMap::new();
        registry.insert(hash, wasm);

        // Part A: Counting executor proves guest_abi change AND max_output_bytes change bust cache!
        let counter = Arc::new(CountingExecutor {
            calls: AtomicUsize::new(0),
        });
        let counting_runner = NativeOpRunner::new().with_wasm_executor(counter.clone());

        let raw_policy = SandboxedExtensionPolicyV1::with_allowed_hash(hash);
        let r1 = execute_graph_with_isolation(
            &graph,
            &mut session,
            &counting_runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(raw_policy.clone()),
            Some(&registry),
            run_id,
            test_clock,
        )
        .unwrap();
        assert_eq!(r1.executed_nodes, vec!["node/wasm_abi_cache"]);
        assert_eq!(counter.calls.load(Ordering::SeqCst), 1);

        // Same policy -> cache hit
        let r1_hit = execute_graph_with_isolation(
            &graph,
            &mut session,
            &counting_runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(raw_policy.clone()),
            Some(&registry),
            run_id,
            test_clock,
        )
        .unwrap();
        assert_eq!(r1_hit.cached_nodes, vec!["node/wasm_abi_cache"]);
        assert_eq!(counter.calls.load(Ordering::SeqCst), 1);

        // Switching guest_abi from RawMem0V0 -> SwarmTorchAllocV1 MUST bust cache!
        let alloc_policy = raw_policy.with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let r2 = execute_graph_with_isolation(
            &graph,
            &mut session,
            &counting_runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(alloc_policy.clone()),
            Some(&registry),
            run_id,
            test_clock,
        )
        .unwrap();
        assert!(r2.cached_nodes.is_empty());
        assert_eq!(r2.executed_nodes, vec!["node/wasm_abi_cache"]);
        assert_eq!(counter.calls.load(Ordering::SeqCst), 2);

        // Part B: Real WasmiExtensionExecutorV1 — tightening max_output_bytes below output length
        //         MUST bust cache, fail closed with OutputTooLarge, and invalidate stale output!
        let real_runner =
            NativeOpRunner::new().with_wasm_executor(Arc::new(WasmiExtensionExecutorV1::new()));
        let mut real_ok_policy = alloc_policy.clone();
        real_ok_policy.limits.max_output_bytes = 64;
        let r_real_ok = execute_graph_with_isolation(
            &graph,
            &mut session,
            &real_runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(real_ok_policy.clone()),
            Some(&registry),
            run_id,
            test_clock,
        )
        .unwrap();
        assert_eq!(r_real_ok.executed_nodes, vec!["node/wasm_abi_cache"]);
        assert!(session.fingerprint("dataset://ns/wasm_abi_out").is_some());

        let mut real_tight_policy = real_ok_policy;
        real_tight_policy.limits.max_output_bytes = 4; // output is 22 bytes -> OutputTooLarge!
        let r_real_fail = execute_graph_with_isolation(
            &graph,
            &mut session,
            &real_runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(real_tight_policy),
            Some(&registry),
            run_id,
            test_clock,
        )
        .unwrap();
        assert!(r_real_fail.cached_nodes.is_empty());
        assert_eq!(r_real_fail.failed_nodes, vec!["node/wasm_abi_cache"]);
        assert!(session.fingerprint("dataset://ns/wasm_abi_out").is_none());

        drop(session);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn wave16c_adversarial_reserved_abi_export_names_rejected_before_start_and_in_scalar() {
        let exec = WasmiExtensionExecutorV1::new();

        // 1. Direct calls to reserved exports "swarmtorch_alloc" / "swarmtorch_dealloc" must fail closed.
        let normal_wasm = build_rodata_prefix_disjoint_alloc_wasm(5);
        let normal_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&normal_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        for reserved in [
            SWARMTORCH_WASM_ALLOC_EXPORT_V1,
            SWARMTORCH_WASM_DEALLOC_EXPORT_V1,
        ] {
            let err = exec
                .execute_byte_transform(&normal_policy, &normal_wasm, reserved, b"hello")
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    WasmSandboxError::ExportNotAllowed(ref n)
                        | WasmSandboxError::ExportSignatureMismatch(ref n)
                        if n == reserved
                ),
                "expected reserved export rejection for '{reserved}', got {err:?}"
            );
        }

        // 2. Module with a trapping `(start)` section MUST reject reserved ABI export name
        //    BEFORE executing `(start)` (never returning ExecutionTrap from start!).
        let trap_start_wasm = build_custom_alloc_abi_wasm(
            1,
            1,
            b"",
            Some(&[0x00]), // start executes `unreachable`!
            1,
            &default_bump_alloc_ops(),
            0,
            &[],
            0,
            &[0x42, 0x00],
        );
        let trap_start_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&trap_start_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let err = exec
            .execute_byte_transform(
                &trap_start_policy,
                &trap_start_wasm,
                SWARMTORCH_WASM_ALLOC_EXPORT_V1,
                b"hi",
            )
            .unwrap_err();
        assert!(
            matches!(
                err,
                WasmSandboxError::ExportNotAllowed(ref n)
                    | WasmSandboxError::ExportSignatureMismatch(ref n)
                    if n == SWARMTORCH_WASM_ALLOC_EXPORT_V1
            ),
            "reserved ABI export must fail in preflight BEFORE executing guest (start), got {err:?}"
        );
    }

    #[test]
    fn wave16c_adversarial_execute_scalar_rejects_reserved_swarmtorch_alloc_export() {
        let exec = WasmiExtensionExecutorV1::new();
        let normal_wasm = build_rodata_prefix_disjoint_alloc_wasm(5);
        let normal_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&normal_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);

        // execute_scalar_i32 MUST NOT bypass reserved ABI export restrictions under SwarmTorchAllocV1
        // (swarmtorch_alloc has signature (i32) -> i32!).
        let scalar_err = exec
            .execute_scalar_i32(
                &normal_policy,
                &normal_wasm,
                SWARMTORCH_WASM_ALLOC_EXPORT_V1,
                64,
            )
            .unwrap_err();
        assert!(
            matches!(
                scalar_err,
                WasmSandboxError::ExportNotAllowed(ref n)
                    | WasmSandboxError::ExportSignatureMismatch(ref n)
                    if n == SWARMTORCH_WASM_ALLOC_EXPORT_V1
            ),
            "execute_scalar_i32 must reject reserved ABI export 'swarmtorch_alloc' under SwarmTorchAllocV1, got {scalar_err:?}"
        );
    }

    #[test]
    fn wave16c_signoff_pre_start_alloc_dealloc_allowed_exports_multislice_fuel_and_grow() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let exec = WasmiExtensionExecutorV1::new();

        // 1. Trapping `(start)` section MUST NOT run when `swarmtorch_alloc` or `swarmtorch_dealloc`
        //    is missing, has wrong function signature, or is a non-function export (e.g., Global).
        // 1a. Missing `swarmtorch_alloc` + trapping `(start)` -> ExportNotAllowed("swarmtorch_alloc")
        let mut trap_no_alloc = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        // type 0: (i32,i32)->(), type 1: (i32,i32)->i64, type 2: ()->()
        push_section(
            &mut trap_no_alloc,
            1,
            &[
                3, 0x60, 2, 0x7f, 0x7f, 0, 0x60, 2, 0x7f, 0x7f, 1, 0x7e, 0x60, 0, 0,
            ],
        );
        push_section(&mut trap_no_alloc, 3, &[3, 0, 1, 2]);
        push_section(&mut trap_no_alloc, 5, &[1, 0x01, 1, 1]);
        let mut sec7 = vec![3];
        push_name(&mut sec7, "memory");
        sec7.extend_from_slice(&[0x02, 0x00]);
        push_name(&mut sec7, SWARMTORCH_WASM_DEALLOC_EXPORT_V1);
        sec7.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7, "transform");
        sec7.extend_from_slice(&[0x00, 0x01]);
        push_section(&mut trap_no_alloc, 7, &sec7);
        push_section(&mut trap_no_alloc, 8, &[2]); // start = fn 2 (unreachable)
        push_section(
            &mut trap_no_alloc,
            10,
            &[3, 2, 0, 0x0b, 4, 0, 0x42, 0x00, 0x0b, 3, 0, 0x00, 0x0b],
        );
        let p_no_alloc = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&trap_no_alloc),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&p_no_alloc, &trap_no_alloc, "transform", b"a")
                .unwrap_err(),
            WasmSandboxError::ExportNotAllowed(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string())
        );

        // 1b. Missing `swarmtorch_dealloc` + trapping `(start)` -> ExportNotAllowed("swarmtorch_dealloc")
        let mut trap_no_dealloc = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(
            &mut trap_no_dealloc,
            1,
            &[
                3, 0x60, 1, 0x7f, 1, 0x7f, 0x60, 2, 0x7f, 0x7f, 1, 0x7e, 0x60, 0, 0,
            ],
        );
        push_section(&mut trap_no_dealloc, 3, &[3, 0, 1, 2]);
        push_section(&mut trap_no_dealloc, 5, &[1, 0x01, 1, 1]);
        let mut sec7_nd = vec![3];
        push_name(&mut sec7_nd, "memory");
        sec7_nd.extend_from_slice(&[0x02, 0x00]);
        push_name(&mut sec7_nd, SWARMTORCH_WASM_ALLOC_EXPORT_V1);
        sec7_nd.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7_nd, "transform");
        sec7_nd.extend_from_slice(&[0x00, 0x01]);
        push_section(&mut trap_no_dealloc, 7, &sec7_nd);
        push_section(&mut trap_no_dealloc, 8, &[2]);
        push_section(
            &mut trap_no_dealloc,
            10,
            &[
                3, 4, 0, 0x41, 0x01, 0x0b, 4, 0, 0x42, 0x00, 0x0b, 3, 0, 0x00, 0x0b,
            ],
        );
        let p_no_dealloc = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&trap_no_dealloc),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&p_no_dealloc, &trap_no_dealloc, "transform", b"a")
                .unwrap_err(),
            WasmSandboxError::ExportNotAllowed(SWARMTORCH_WASM_DEALLOC_EXPORT_V1.to_string())
        );

        // 1c. Wrong-signature `swarmtorch_alloc` ((i32, i32) -> i32) + trapping `(start)`
        let mut trap_bad_alloc_sig = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(
            &mut trap_bad_alloc_sig,
            1,
            &[
                4, 0x60, 2, 0x7f, 0x7f, 1, 0x7f, // bad alloc
                0x60, 2, 0x7f, 0x7f, 0, // dealloc
                0x60, 2, 0x7f, 0x7f, 1, 0x7e, // transform
                0x60, 0, 0, // start
            ],
        );
        push_section(&mut trap_bad_alloc_sig, 3, &[4, 0, 1, 2, 3]);
        push_section(&mut trap_bad_alloc_sig, 5, &[1, 0x01, 1, 1]);
        let mut sec7_4 = vec![4];
        push_name(&mut sec7_4, "memory");
        sec7_4.extend_from_slice(&[0x02, 0x00]);
        push_name(&mut sec7_4, SWARMTORCH_WASM_ALLOC_EXPORT_V1);
        sec7_4.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7_4, SWARMTORCH_WASM_DEALLOC_EXPORT_V1);
        sec7_4.extend_from_slice(&[0x00, 0x01]);
        push_name(&mut sec7_4, "transform");
        sec7_4.extend_from_slice(&[0x00, 0x02]);
        push_section(&mut trap_bad_alloc_sig, 7, &sec7_4);
        push_section(&mut trap_bad_alloc_sig, 8, &[3]);
        push_section(
            &mut trap_bad_alloc_sig,
            10,
            &[
                4, 4, 0, 0x41, 0x01, 0x0b, 2, 0, 0x0b, 4, 0, 0x42, 0x00, 0x0b, 3, 0, 0x00, 0x0b,
            ],
        );
        let p_bad_alloc = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&trap_bad_alloc_sig),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&p_bad_alloc, &trap_bad_alloc_sig, "transform", b"a")
                .unwrap_err(),
            WasmSandboxError::ExportSignatureMismatch(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string())
        );

        // 1d. Wrong-signature `swarmtorch_dealloc` ((i32, i32) -> i32) + trapping `(start)`
        let mut trap_bad_dealloc_sig =
            swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(
            &mut trap_bad_dealloc_sig,
            1,
            &[
                4, 0x60, 1, 0x7f, 1, 0x7f, // alloc
                0x60, 2, 0x7f, 0x7f, 1, 0x7f, // bad dealloc
                0x60, 2, 0x7f, 0x7f, 1, 0x7e, // transform
                0x60, 0, 0, // start
            ],
        );
        push_section(&mut trap_bad_dealloc_sig, 3, &[4, 0, 1, 2, 3]);
        push_section(&mut trap_bad_dealloc_sig, 5, &[1, 0x01, 1, 1]);
        push_section(&mut trap_bad_dealloc_sig, 7, &sec7_4);
        push_section(&mut trap_bad_dealloc_sig, 8, &[3]);
        push_section(
            &mut trap_bad_dealloc_sig,
            10,
            &[
                4, 4, 0, 0x41, 0x01, 0x0b, 4, 0, 0x41, 0x00, 0x0b, 4, 0, 0x42, 0x00, 0x0b, 3, 0,
                0x00, 0x0b,
            ],
        );
        let p_bad_dealloc = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&trap_bad_dealloc_sig),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&p_bad_dealloc, &trap_bad_dealloc_sig, "transform", b"a")
                .unwrap_err(),
            WasmSandboxError::ExportSignatureMismatch(
                SWARMTORCH_WASM_DEALLOC_EXPORT_V1.to_string()
            )
        );

        // 1e. Non-function export named `swarmtorch_alloc` (Global i32) + trapping `(start)`
        let mut trap_global_alloc = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(
            &mut trap_global_alloc,
            1,
            &[
                3, 0x60, 2, 0x7f, 0x7f, 0, // dealloc
                0x60, 2, 0x7f, 0x7f, 1, 0x7e, // transform
                0x60, 0, 0, // start
            ],
        );
        push_section(&mut trap_global_alloc, 3, &[3, 0, 1, 2]);
        push_section(&mut trap_global_alloc, 5, &[1, 0x01, 1, 1]);
        // Section 6: Global 0 = const i32 = 4096
        let mut sec6_g = vec![1, 0x7f, 0x00];
        push_i32_const(&mut sec6_g, 4096);
        sec6_g.push(0x0b);
        push_section(&mut trap_global_alloc, 6, &sec6_g);
        let mut sec7_g = vec![4];
        push_name(&mut sec7_g, "memory");
        sec7_g.extend_from_slice(&[0x02, 0x00]);
        push_name(&mut sec7_g, SWARMTORCH_WASM_ALLOC_EXPORT_V1);
        sec7_g.extend_from_slice(&[0x03, 0x00]); // kind=global, idx=0
        push_name(&mut sec7_g, SWARMTORCH_WASM_DEALLOC_EXPORT_V1);
        sec7_g.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7_g, "transform");
        sec7_g.extend_from_slice(&[0x00, 0x01]);
        push_section(&mut trap_global_alloc, 7, &sec7_g);
        push_section(&mut trap_global_alloc, 8, &[2]);
        push_section(
            &mut trap_global_alloc,
            10,
            &[3, 2, 0, 0x0b, 4, 0, 0x42, 0x00, 0x0b, 3, 0, 0x00, 0x0b],
        );
        let p_global_alloc = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&trap_global_alloc),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&p_global_alloc, &trap_global_alloc, "transform", b"a")
                .unwrap_err(),
            WasmSandboxError::ExportSignatureMismatch(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string())
        );

        // 2. Non-empty `allowed_exports = {"transform"}` succeeds under SwarmTorchAllocV1 without
        //    requiring lifecycle exports in `allowed_exports`, while explicitly listing reserved
        //    exports in `allowed_exports` STILL rejects invoking them directly.
        let disjoint_wasm = build_rodata_prefix_disjoint_alloc_wasm(5);
        let disjoint_hash = SandboxedExtensionPolicyV1::sha256_module(&disjoint_wasm);
        let mut allow_transform_only = SandboxedExtensionPolicyV1::with_allowed_hash(disjoint_hash)
            .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        allow_transform_only
            .allowed_exports
            .insert("transform".to_string());
        let r_allow = exec
            .execute_byte_transform(&allow_transform_only, &disjoint_wasm, "transform", b"hello")
            .unwrap();
        assert_eq!(r_allow.output_bytes, b"RODATA_MAGIC_V1!hello");

        allow_transform_only
            .allowed_exports
            .insert(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string());
        assert_eq!(
            exec.execute_byte_transform(
                &allow_transform_only,
                &disjoint_wasm,
                SWARMTORCH_WASM_ALLOC_EXPORT_V1,
                b"hello",
            )
            .unwrap_err(),
            WasmSandboxError::ExportNotAllowed(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string())
        );
        assert_eq!(
            exec.execute_scalar_i32(
                &allow_transform_only,
                &disjoint_wasm,
                SWARMTORCH_WASM_ALLOC_EXPORT_V1,
                16,
            )
            .unwrap_err(),
            WasmSandboxError::ExportNotAllowed(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string())
        );

        // 3. Scheduler admission rejects a `wasm_transform` node targeting `swarmtorch_alloc`
        //    under SwarmTorchAllocV1 BEFORE calling the executor (0 executor calls!).
        struct ZeroCallGuard {
            calls: AtomicUsize,
        }
        impl WasmExtensionExecutor for ZeroCallGuard {
            fn execute_byte_transform(
                &self,
                _policy: &SandboxedExtensionPolicyV1,
                _wasm_bytes: &[u8],
                _export_name: &str,
                _input_payload: &[u8],
            ) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Err(WasmSandboxError::UnsupportedUntrustedWasmSandbox)
            }
            fn execute_scalar_i32(
                &self,
                _policy: &SandboxedExtensionPolicyV1,
                _wasm_bytes: &[u8],
                _export_name: &str,
                _arg: i32,
            ) -> Result<(i32, WasmExecutionReceiptV1), WasmSandboxError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Err(WasmSandboxError::UnsupportedUntrustedWasmSandbox)
            }
        }
        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("swarmtorch_wasmi_signoff_sched_{pid}"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([0x77; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let mut session = DataOpsSession::new(Arc::new(RunArtifactSink::new(bundle)));
        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let in_bytes = b"hello";
        let in_hex: String = SandboxedExtensionPolicyV1::sha256_module(in_bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let cas_dir = run_dir.join("blobs").join("sha256");
        std::fs::create_dir_all(&cas_dir).unwrap();
        std::fs::write(cas_dir.join(&in_hex), in_bytes).unwrap();
        let ingest = make_graph_node(
            "ingest/raw",
            "ingest",
            &[],
            &["dataset://ns/raw"],
            ExecutionTrust::Core,
        );
        session
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: format!("cas://sha256/{in_hex}"),
                    content_type: CONTENT_TYPE_OCTET_STREAM.to_string(),
                    auth_mode: AuthModeMarker::None,
                    etag_or_version: Some("v1".to_string()),
                },
                None,
                &ingest,
            )
            .unwrap();
        let hex_hash: String = disjoint_hash.iter().map(|b| format!("{b:02x}")).collect();
        let mut bad_node = make_graph_node(
            "node/reserved_export",
            "wasm_transform",
            &["dataset://ns/raw"],
            &["dataset://ns/out"],
            ExecutionTrust::SandboxedExtension,
        );
        bad_node
            .params
            .insert("module_sha256".to_string(), CanonValue::Str(hex_hash));
        bad_node.params.insert(
            "export".to_string(),
            CanonValue::Str(SWARMTORCH_WASM_ALLOC_EXPORT_V1.to_string()),
        );
        let bad_graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("reserved-export-admission".to_string()),
            nodes: vec![bad_node],
            edges: vec![],
        };
        let mut registry = BTreeMap::new();
        registry.insert(disjoint_hash, disjoint_wasm);
        let guard = Arc::new(ZeroCallGuard {
            calls: AtomicUsize::new(0),
        });
        let runner = NativeOpRunner::new().with_wasm_executor(guard.clone());
        let sched_report = execute_graph_with_isolation(
            &bad_graph,
            &mut session,
            &runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(allow_transform_only),
            Some(&registry),
            run_id,
            test_clock,
        )
        .unwrap();
        assert_eq!(sched_report.failed_nodes, vec!["node/reserved_export"]);
        assert_eq!(guard.calls.load(Ordering::SeqCst), 0);
        drop(session);
        let _ = std::fs::remove_dir_all(&base);

        // 4. Multi-slice (> FUEL_SLICE_UNITS = 10_000) fuel restoration across alloc -> transform -> dealloc!
        let make_spin_n = |local_idx: u8, n: i32| -> Vec<u8> {
            let mut ops = Vec::new();
            push_i32_const(&mut ops, n);
            ops.extend_from_slice(&[
                0x21, local_idx, 0x03, 0x40, 0x20, local_idx, 0x41, 0x00, 0x4a, 0x04, 0x40, 0x20,
                local_idx, 0x41, 0x01, 0x6b, 0x21, local_idx, 0x0c, 0x01, 0x0b, 0x0b,
            ]);
            ops
        };
        let inplace_id_ops = vec![0x20, 0x00, 0xad, 0x42, 32, 0x86, 0x20, 0x01, 0xad, 0x84];
        let mut alloc_multi = make_spin_n(2, 4_000);
        alloc_multi.extend_from_slice(&default_bump_alloc_ops());
        let dealloc_multi = make_spin_n(2, 4_000);
        let mut transform_multi = make_spin_n(2, 4_000);
        transform_multi.extend_from_slice(&inplace_id_ops);
        let multi_slice_wasm = build_custom_alloc_abi_wasm(
            1,
            1,
            b"",
            None,
            2,
            &alloc_multi,
            1,
            &dealloc_multi,
            1,
            &transform_multi,
        );
        let mut multi_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&multi_slice_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        multi_policy.limits.max_fuel = 200_000;
        let multi_receipt = exec
            .execute_byte_transform(&multi_policy, &multi_slice_wasm, "transform", b"slice")
            .unwrap();
        assert_eq!(multi_receipt.output_bytes, b"slice");
        assert!(
            multi_receipt.fuel_consumed > FUEL_SLICE_UNITS * 3,
            "expected multi-slice fuel > {}, got {}",
            FUEL_SLICE_UNITS * 3,
            multi_receipt.fuel_consumed
        );
        let mut tight_multi = multi_policy.clone();
        tight_multi.limits.max_fuel = multi_receipt.fuel_consumed - 1;
        assert!(matches!(
            exec.execute_byte_transform(&tight_multi, &multi_slice_wasm, "transform", b"slice"),
            Err(WasmSandboxError::OutOfFuel { limit, .. })
                if limit == multi_receipt.fuel_consumed - 1
        ));

        // 5. Live `max_output_bytes == 0` enforcement under BOTH SwarmTorchAllocV1 and RawMem0V0:
        //    empty output succeeds, 1-byte output fails OutputTooLarge { len: 1, max: 0 }.
        let mut empty_out_ops = Vec::new();
        push_i64_const(&mut empty_out_ops, 0);
        let empty_out_wasm =
            build_alloc_abi_test_wasm(b"", &dealloc_assert_once(4096, 3), 0, &empty_out_ops);
        let mut zero_cap_alloc = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&empty_out_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        zero_cap_alloc.limits.max_output_bytes = 0;
        let r_zero_ok = exec
            .execute_byte_transform(&zero_cap_alloc, &empty_out_wasm, "transform", b"abc")
            .unwrap();
        assert!(r_zero_ok.output_bytes.is_empty());

        let one_byte_wasm = build_alloc_abi_test_wasm(b"", &[0x00], 0, &inplace_id_ops);
        let mut zero_cap_one = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&one_byte_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        zero_cap_one.limits.max_output_bytes = 0;
        assert_eq!(
            exec.execute_byte_transform(&zero_cap_one, &one_byte_wasm, "transform", b"x")
                .unwrap_err(),
            WasmSandboxError::OutputTooLarge { len: 1, max: 0 }
        );

        let raw_wasm = build_ascii_xor_wasm();
        let mut zero_cap_raw = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&raw_wasm),
        );
        zero_cap_raw.limits.max_output_bytes = 0;
        assert!(exec
            .execute_byte_transform(&zero_cap_raw, &raw_wasm, "transform", b"")
            .unwrap()
            .output_bytes
            .is_empty());
        assert_eq!(
            exec.execute_byte_transform(&zero_cap_raw, &raw_wasm, "transform", b"a")
                .unwrap_err(),
            WasmSandboxError::OutputTooLarge { len: 1, max: 0 }
        );

        // 6. `transform` growing memory into page 2 (ptr=65536) updates `peak_memory_bytes` to 2 pages
        //    and deallocates both buffers once; `alloc` exceeding `max_memory_pages` hits `MemoryGrowTrap`.
        let mut grow_transform = vec![0x41, 0x01, 0x40, 0x00, 0x1a]; // drop(memory.grow(1))
        for (idx, &b) in b"PAGE".iter().enumerate() {
            push_i32_const(&mut grow_transform, 65_536 + idx as i32);
            push_i32_const(&mut grow_transform, i32::from(b));
            grow_transform.extend_from_slice(&[0x3a, 0x00, 0x00]);
        }
        push_i64_const(&mut grow_transform, ((65_536u64 << 32) | 4u64) as i64);
        let grow_t_wasm = build_alloc_abi_test_wasm(
            b"",
            &dealloc_assert_twice(4096, 4, 65_536, 4),
            0,
            &grow_transform,
        );
        let grow_t_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&grow_t_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let r_page2 = exec
            .execute_byte_transform(&grow_t_policy, &grow_t_wasm, "transform", b"init")
            .unwrap();
        assert_eq!(r_page2.output_bytes, b"PAGE");
        assert_eq!(r_page2.peak_memory_bytes, 2 * WASM_PAGE_BYTES);

        let mut bomb_alloc = vec![0x41, 0x08, 0x40, 0x00, 0x1a]; // drop(memory.grow(8))
        push_i32_const(&mut bomb_alloc, 4096);
        let bomb_alloc_wasm =
            build_custom_alloc_abi_wasm(1, 16, b"", None, 0, &bomb_alloc, 0, &[], 0, &[0x42, 0x00]);
        let mut bomb_alloc_policy = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&bomb_alloc_wasm),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        bomb_alloc_policy.limits.max_memory_pages = 2;
        bomb_alloc_policy.limits.trap_on_grow_failure = true;
        assert!(matches!(
            exec.execute_byte_transform(&bomb_alloc_policy, &bomb_alloc_wasm, "transform", b"x"),
            Err(WasmSandboxError::MemoryGrowTrap {
                current_pages: 1,
                max_pages: 2,
                ..
            })
        ));
    }

    #[test]
    fn wave16c_signoff_pre_start_target_export_signature_and_memory_validation() {
        let exec = WasmiExtensionExecutorV1::new();

        // 1. SwarmTorchAllocV1 module with valid alloc/dealloc, trapping `(start)`,
        //    and wrong-signature `"transform"` ((i32, i32) -> i32 instead of -> i64):
        //    MUST fail closed BEFORE executing `(start)` with ExportSignatureMismatch("transform")!
        let mut trap_bad_transform = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(
            &mut trap_bad_transform,
            1,
            &[
                4, 0x60, 1, 0x7f, 1, 0x7f, // alloc: (i32) -> i32
                0x60, 2, 0x7f, 0x7f, 0, // dealloc: (i32, i32) -> ()
                0x60, 2, 0x7f, 0x7f, 1, 0x7f, // WRONG transform: (i32, i32) -> i32
                0x60, 0, 0, // start: () -> ()
            ],
        );
        push_section(&mut trap_bad_transform, 3, &[4, 0, 1, 2, 3]);
        push_section(&mut trap_bad_transform, 5, &[1, 0x01, 1, 1]);
        let mut sec7 = vec![4];
        push_name(&mut sec7, "memory");
        sec7.extend_from_slice(&[0x02, 0x00]);
        push_name(&mut sec7, SWARMTORCH_WASM_ALLOC_EXPORT_V1);
        sec7.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7, SWARMTORCH_WASM_DEALLOC_EXPORT_V1);
        sec7.extend_from_slice(&[0x00, 0x01]);
        push_name(&mut sec7, "transform");
        sec7.extend_from_slice(&[0x00, 0x02]);
        push_section(&mut trap_bad_transform, 7, &sec7);
        push_section(&mut trap_bad_transform, 8, &[3]); // start = fn 3 (unreachable)
        push_section(
            &mut trap_bad_transform,
            10,
            &[
                4, 4, 0, 0x41, 0x01, 0x0b, 2, 0, 0x0b, 4, 0, 0x41, 0x00, 0x0b, 3, 0, 0x00, 0x0b,
            ],
        );
        let p_alloc = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&trap_bad_transform),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        assert_eq!(
            exec.execute_byte_transform(&p_alloc, &trap_bad_transform, "transform", b"a")
                .unwrap_err(),
            WasmSandboxError::ExportSignatureMismatch("transform".to_string()),
            "SwarmTorchAllocV1 must validate target export signature BEFORE executing (start)"
        );

        // 2. RawMem0V0 sibling: valid SwarmTorchAllocV1 module (`transform: (i32, i32) -> i64`)
        //    with trapping `(start)` invoked under RawMem0V0 MUST fail BEFORE `(start)`!
        let trap_alloc_mod = build_custom_alloc_abi_wasm(
            1,
            1,
            b"",
            Some(&[0x00]), // start traps
            1,
            &default_bump_alloc_ops(),
            0,
            &[],
            0,
            &[0x42, 0x00],
        );
        let p_raw = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&trap_alloc_mod),
        );
        assert_eq!(
            exec.execute_byte_transform(&p_raw, &trap_alloc_mod, "transform", b"a")
                .unwrap_err(),
            WasmSandboxError::ExportSignatureMismatch("transform".to_string()),
            "RawMem0V0 must validate target export signature BEFORE executing (start)"
        );

        // 3. Scalar sibling: `execute_scalar_i32` targeting 2-param `transform` with trapping `(start)`
        //    MUST fail BEFORE `(start)`!
        assert_eq!(
            exec.execute_scalar_i32(&p_raw, &trap_alloc_mod, "transform", 1)
                .unwrap_err(),
            WasmSandboxError::ExportSignatureMismatch("transform".to_string()),
            "execute_scalar_i32 must validate target export signature BEFORE executing (start)"
        );

        // 4. Non-function target export (`export_name = "memory"`) with trapping `(start)`
        //    MUST fail BEFORE `(start)` (never returning ExecutionTrap from start)!
        let mem_err = exec
            .execute_byte_transform(&p_alloc, &trap_bad_transform, "memory", b"a")
            .unwrap_err();
        assert!(
            matches!(
                mem_err,
                WasmSandboxError::ExportNotAllowed(ref n)
                    | WasmSandboxError::ExportSignatureMismatch(ref n)
                    if n == "memory"
            ),
            "non-function export 'memory' must be rejected BEFORE executing (start), got {mem_err:?}"
        );

        // 5. Byte-transform module with NO exported linear memory and trapping `(start)`
        //    MUST fail BEFORE executing `(start)` (missing memory error, NOT unreachable trap)!
        let mut trap_no_mem = swarm_torch::wasm_sandbox::WASM_MAGIC_AND_VERSION_V1.to_vec();
        push_section(
            &mut trap_no_mem,
            1,
            &[
                4, 0x60, 1, 0x7f, 1, 0x7f, // alloc
                0x60, 2, 0x7f, 0x7f, 0, // dealloc
                0x60, 2, 0x7f, 0x7f, 1, 0x7e, // transform
                0x60, 0, 0, // start
            ],
        );
        push_section(&mut trap_no_mem, 3, &[4, 0, 1, 2, 3]);
        let mut sec7_nm = vec![3];
        push_name(&mut sec7_nm, SWARMTORCH_WASM_ALLOC_EXPORT_V1);
        sec7_nm.extend_from_slice(&[0x00, 0x00]);
        push_name(&mut sec7_nm, SWARMTORCH_WASM_DEALLOC_EXPORT_V1);
        sec7_nm.extend_from_slice(&[0x00, 0x01]);
        push_name(&mut sec7_nm, "transform");
        sec7_nm.extend_from_slice(&[0x00, 0x02]);
        push_section(&mut trap_no_mem, 7, &sec7_nm);
        push_section(&mut trap_no_mem, 8, &[3]);
        push_section(
            &mut trap_no_mem,
            10,
            &[
                4, 4, 0, 0x41, 0x01, 0x0b, 2, 0, 0x0b, 4, 0, 0x42, 0x00, 0x0b, 3, 0, 0x00, 0x0b,
            ],
        );
        let p_no_mem = SandboxedExtensionPolicyV1::with_allowed_hash(
            SandboxedExtensionPolicyV1::sha256_module(&trap_no_mem),
        )
        .with_guest_abi(WasmGuestAbiV1::SwarmTorchAllocV1);
        let no_mem_err = exec
            .execute_byte_transform(&p_no_mem, &trap_no_mem, "transform", b"a")
            .unwrap_err();
        assert!(
            matches!(
                no_mem_err,
                WasmSandboxError::ExecutionTrap(ref msg)
                    if msg.contains("requires an exported linear memory")
            ),
            "missing exported linear memory must fail BEFORE executing (start), got {no_mem_err:?}"
        );
    }
}
