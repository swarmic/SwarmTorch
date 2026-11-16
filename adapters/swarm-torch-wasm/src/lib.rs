#![forbid(unsafe_code)]
//! Isolated `wasmi 2.0` WebAssembly extension sandbox adapter for SwarmTorch (`ST-09`).
//!
//! Implements [`swarm_torch::wasm_sandbox::WasmExtensionExecutor`] outside the core `swarm-torch`
//! workspace so untrusted WebAssembly execution is isolated behind an explicit adapter boundary
//! with:
//! - SHA-256 module digest pinning and size/header validation via [`preflight_wasm_extension_v1`],
//! - Zero host/WASI imports (`module.imports()` rejected fail-closed + empty [`wasmi::Linker`]),
//! - Deterministic `wasmi` fuel metering across `start` and sliced resumable export calls,
//! - Bounded wall-clock timeout enforcement (`WasmStoreLimitsV1::timeout_ms`) on every fuel slice,
//! - Strict linear memory and table bounds via [`wasmi::StoreLimits`], and
//! - Checked host-to-guest / guest-to-host linear memory transfers (`read_guest_slice`).

use std::time::Instant;

use swarm_torch::wasm_sandbox::{
    preflight_wasm_extension_v1, read_guest_slice, SandboxedExtensionPolicyV1,
    WasmExecutionReceiptV1, WasmExtensionExecutor, WasmSandboxError, WASM_PAGE_BYTES,
};
use wasmi::{
    errors::ErrorKind, CompilationMode, Config, EnforcedLimits, Engine, Extern, ExternType,
    Instance, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder, TrapCode,
    TypedResumableCall,
};

/// Deterministic fuel slice charged between wall-clock timeout checks (`ST-09`).
const FUEL_SLICE_UNITS: u64 = 10_000;

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
        let mut prepared = prepare_instance(policy, wasm_bytes, export_name)?;
        let memory = find_single_memory(&prepared.instance, &prepared.store).ok_or_else(|| {
            WasmSandboxError::ExecutionTrap(format!(
                "byte-transform export '{export_name}' requires an exported linear memory"
            ))
        })?;

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

        let output_bytes = if return_code < 0 {
            return Err(WasmSandboxError::ExecutionTrap(format!(
                "export '{export_name}' returned negative error code {return_code}"
            )));
        } else {
            read_guest_slice(memory.data(&prepared.store), 0, return_code)?
        };

        let peak_memory_bytes = memory.data_size(&prepared.store);

        Ok(WasmExecutionReceiptV1 {
            module_sha256: prepared.module_sha256,
            fuel_consumed,
            peak_memory_bytes,
            return_code,
            output_bytes,
        })
    }

    fn execute_scalar_i32(
        &self,
        policy: &SandboxedExtensionPolicyV1,
        wasm_bytes: &[u8],
        export_name: &str,
        arg: i32,
    ) -> Result<(i32, WasmExecutionReceiptV1), WasmSandboxError> {
        let mut prepared = prepare_instance(policy, wasm_bytes, export_name)?;
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

fn prepare_instance(
    policy: &SandboxedExtensionPolicyV1,
    wasm_bytes: &[u8],
    export_name: &str,
) -> Result<PreparedModule, WasmSandboxError> {
    let module_sha256 = preflight_wasm_extension_v1(policy, wasm_bytes, export_name)?;
    let started_at = Instant::now();
    check_timeout(started_at, policy.limits.timeout_ms)?;

    let engine = build_hardened_engine();
    let module = Module::new(&engine, wasm_bytes).map_err(map_compile_error)?;

    if let Some(import) = module.imports().next() {
        return Err(WasmSandboxError::ForbiddenImport {
            module: import.module().to_string(),
            field: import.name().to_string(),
        });
    }

    let mut memory_count = 0u32;
    let mut table_count = 0u32;
    let mut export_found = false;
    for export in module.exports() {
        if export.name() == export_name {
            export_found = true;
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
    if !export_found {
        return Err(WasmSandboxError::ExportNotAllowed(export_name.to_string()));
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

    let mut store = Store::new(&engine, store_limits);
    store.limiter(|limits| limits);
    store
        .set_fuel(policy.limits.max_fuel)
        .map_err(|err| WasmSandboxError::ExecutionTrap(err.to_string()))?;

    let linker = Linker::<StoreLimits>::new(&engine);
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .map_err(|err| map_instantiation_error(err, policy))?;

    check_timeout(started_at, policy.limits.timeout_ms)?;

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
    config.enforced_limits(EnforcedLimits::strict());
    Engine::new(&config)
}

fn map_compile_error(err: wasmi::Error) -> WasmSandboxError {
    let msg = err.to_string();
    let lower = msg.to_ascii_lowercase();
    if lower.contains("multi-memory") || lower.contains("multiple memories") {
        WasmSandboxError::MultipleMemoriesForbidden
    } else if lower.contains("multiple tables") {
        WasmSandboxError::MultipleTablesForbidden
    } else {
        WasmSandboxError::MalformedSectionOrLeb128
    }
}

fn map_instantiation_error(
    err: wasmi::Error,
    policy: &SandboxedExtensionPolicyV1,
) -> WasmSandboxError {
    match err.kind() {
        ErrorKind::TrapCode(TrapCode::OutOfFuel) | ErrorKind::Fuel(_) => {
            WasmSandboxError::OutOfFuel {
                consumed: policy.limits.max_fuel,
                limit: policy.limits.max_fuel,
            }
        }
        ErrorKind::TrapCode(TrapCode::GrowthOperationLimited) | ErrorKind::Memory(_) => {
            WasmSandboxError::MemoryLimitExceeded {
                requested_pages: policy.limits.max_memory_pages.saturating_add(1),
                max_pages: policy.limits.max_memory_pages,
            }
        }
        ErrorKind::Table(_) => WasmSandboxError::TableLimitExceeded {
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
}
