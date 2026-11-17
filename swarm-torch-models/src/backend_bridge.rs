//! Framework-agnostic external tensor backend bridge (`ADR-0024` §4, `W12-01`).
//!
//! Decouples `swarm-torch-models` (`no_std + alloc`, MSRV 1.75) from heavy deep-learning frameworks
//! (`Burn`, `Candle`) by defining [`BackendTensorBridge`] and [`BackendModelHandleV1`], which wrap
//! device-resident backend models behind [`SwarmModel`] with canonical [`ModelTensorSchemaV1`]
//! (`model_schema_hash`) verification and transactional `.is_finite()` gating before device upload.

use alloc::string::String;
use alloc::vec::Vec;

use swarm_torch_core::traits::SwarmModel;
use swarm_torch_core::Error as CoreError;

use crate::safetensors_v1::ModelTensorSchemaV1;

/// Physical compute device placement reported by a [`BackendTensorBridge`] (`ADR-0024` §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DevicePlacementV1 {
    /// Host CPU memory.
    Cpu,
    /// Portable WebGPU / Vulkan / Metal compute device (`Burn` `Wgpu` backend).
    Wgpu { device_index: u32 },
    /// NVIDIA CUDA device (`Candle` / `Burn` CUDA backend).
    Cuda { device_index: u32 },
}

/// Errors returned by external tensor backend bridges (`ADR-0024` §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendBridgeError {
    /// Backend tensor schema or `model_schema_hash` does not match expected schema.
    SchemaMismatch,
    /// Parameter buffer length does not match `schema.total_elements()`.
    ParameterCountMismatch { expected: usize, got: usize },
    /// Non-finite (`NaN` or `±Inf`) parameter or activation rejected at the bridge boundary.
    NonFiniteValue,
    /// Target compute device is unavailable on this node.
    DeviceUnavailable,
    /// Backend-specific runtime failure.
    BackendOperationFailed(String),
}

impl core::fmt::Display for BackendBridgeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SchemaMismatch => write!(
                f,
                "backend model_schema_hash does not match expected ModelTensorSchemaV1"
            ),
            Self::ParameterCountMismatch { expected, got } => write!(
                f,
                "backend parameter count mismatch: expected {expected}, got {got}"
            ),
            Self::NonFiniteValue => {
                write!(
                    f,
                    "non-finite (NaN or Inf) value rejected at backend bridge"
                )
            }
            Self::DeviceUnavailable => write!(f, "target backend compute device unavailable"),
            Self::BackendOperationFailed(msg) => write!(f, "backend operation failed: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BackendBridgeError {}

/// Portable tensor synchronization contract between external autodiff/GPU frameworks (`Burn`, `Candle`)
/// and SwarmTorch's contiguous-buffer [`SwarmModel`] / `SwarmRoundCoordinator` pipeline (`ADR-0024` §4).
pub trait BackendTensorBridge: Send + Sync {
    /// Human-readable backend adapter identifier (e.g., `"burn-ndarray"`, `"candle-cuda"`).
    fn backend_name(&self) -> &str;

    /// Active device placement for this model handle.
    fn placement(&self) -> DevicePlacementV1;

    /// Canonical [`ModelTensorSchemaV1`] defining tensor names, dtypes, shapes, and parameter order.
    fn schema(&self) -> &ModelTensorSchemaV1;

    /// Export all parameters in canonical [`ModelTensorSchemaV1`] order as a flat `f32` vector.
    fn pull_flat_parameters(&self) -> Result<Vec<f32>, BackendBridgeError>;

    /// Atomically overwrite backend device parameters from a flat `f32` vector in canonical schema order.
    fn push_flat_parameters(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError>;

    /// Execute a forward inference pass over flattened row input.
    fn forward_slice(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError>;
}

/// Verified wrapper binding a [`BackendTensorBridge`] to SwarmTorch's [`SwarmModel`] trait
/// with canonical `model_schema_hash` enforcement and transactional finiteness guards (`ADR-0024` §4).
#[derive(Debug, Clone)]
pub struct BackendModelHandleV1<B: BackendTensorBridge> {
    backend: B,
    expected_schema_hash: [u8; 32],
    expected_param_count: usize,
    shadow_parameters: Vec<f32>,
}

impl<B: BackendTensorBridge> BackendModelHandleV1<B> {
    /// Wrap a [`BackendTensorBridge`], verify its [`ModelTensorSchemaV1`] against `expected_schema_hash`
    /// (if provided), and pull initial finite parameters into host shadow storage.
    pub fn try_new(
        backend: B,
        expected_schema_hash: Option<[u8; 32]>,
    ) -> Result<Self, BackendBridgeError> {
        let schema = backend.schema();
        let actual_hash = schema.schema_hash();
        if let Some(expected) = expected_schema_hash {
            if actual_hash != expected {
                return Err(BackendBridgeError::SchemaMismatch);
            }
        }
        let expected_param_count = schema
            .total_elements()
            .map_err(|_| BackendBridgeError::SchemaMismatch)?;
        let initial_params = backend.pull_flat_parameters()?;
        validate_param_slice(&initial_params, expected_param_count)?;

        Ok(Self {
            backend,
            expected_schema_hash: actual_hash,
            expected_param_count,
            shadow_parameters: initial_params,
        })
    }

    /// Expected canonical 32-byte `model_schema_hash`.
    pub const fn schema_hash(&self) -> [u8; 32] {
        self.expected_schema_hash
    }

    /// Borrow the underlying backend adapter.
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Pull updated parameters from the backend device after local training, verifying schema
    /// immutability, length, and finiteness before updating host shadow state.
    pub fn sync_from_backend(&mut self) -> Result<&[f32], BackendBridgeError> {
        if self.backend.schema().schema_hash() != self.expected_schema_hash {
            return Err(BackendBridgeError::SchemaMismatch);
        }
        let fresh = self.backend.pull_flat_parameters()?;
        validate_param_slice(&fresh, self.expected_param_count)?;
        self.shadow_parameters = fresh;
        Ok(&self.shadow_parameters)
    }

    /// Push the current host `shadow_parameters` back onto the backend compute device
    /// transactionally after verifying finiteness and schema immutability.
    pub fn commit_to_backend(&mut self) -> Result<(), BackendBridgeError> {
        if self.backend.schema().schema_hash() != self.expected_schema_hash {
            return Err(BackendBridgeError::SchemaMismatch);
        }
        validate_param_slice(&self.shadow_parameters, self.expected_param_count)?;
        self.backend.push_flat_parameters(&self.shadow_parameters)
    }

    /// Compute the local update delta $\Delta = w_{\text{local}} - w_{\text{baseline}}$ in
    /// widened `f64` for submission via `UpdateKindV1::ModelDeltaV1`.
    pub fn delta_since(&self, baseline: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
        validate_param_slice(baseline, self.expected_param_count)?;
        validate_param_slice(&self.shadow_parameters, self.expected_param_count)?;
        let mut delta = Vec::with_capacity(self.expected_param_count);
        for (&local, &base) in self.shadow_parameters.iter().zip(baseline.iter()) {
            let d = ((local as f64) - (base as f64)) as f32;
            if !d.is_finite() {
                return Err(BackendBridgeError::NonFiniteValue);
            }
            delta.push(d);
        }
        Ok(delta)
    }
}

fn validate_param_slice(params: &[f32], expected_count: usize) -> Result<(), BackendBridgeError> {
    if params.len() != expected_count {
        return Err(BackendBridgeError::ParameterCountMismatch {
            expected: expected_count,
            got: params.len(),
        });
    }
    if params.iter().any(|p| !p.is_finite()) {
        return Err(BackendBridgeError::NonFiniteValue);
    }
    Ok(())
}

impl<B: BackendTensorBridge> SwarmModel for BackendModelHandleV1<B> {
    type Input = [f32];
    type Output = Vec<f32>;
    type Error = BackendBridgeError;

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
        if input.is_empty() || input.iter().any(|v| !v.is_finite()) {
            return Err(BackendBridgeError::NonFiniteValue);
        }
        let output = self.backend.forward_slice(input)?;
        if output.iter().any(|v| !v.is_finite()) {
            return Err(BackendBridgeError::NonFiniteValue);
        }
        Ok(output)
    }

    fn parameters(&self) -> &[f32] {
        &self.shadow_parameters
    }

    /// Borrow host `shadow_parameters` mutably. Callers mutating this slice directly MUST invoke
    /// [`Self::commit_to_backend`] afterward to push changes onto the backend device; [`Self::load_parameters`]
    /// and [`crate::optim::SgdOptimizer`] (`SwarmOptimizer::step`) synchronize the device backend automatically.
    fn parameters_mut(&mut self) -> &mut [f32] {
        &mut self.shadow_parameters
    }

    fn load_parameters(&mut self, params: &[f32]) -> swarm_torch_core::Result<()> {
        if self.backend.schema().schema_hash() != self.expected_schema_hash {
            return Err(CoreError::InvalidGradient);
        }
        if params.len() != self.expected_param_count {
            return Err(CoreError::InvalidGradient);
        }
        for &p in params {
            if p.is_nan() {
                return Err(CoreError::NanDetected);
            }
            if p.is_infinite() {
                return Err(CoreError::InfDetected);
            }
        }
        self.backend
            .push_flat_parameters(params)
            .map_err(|_| CoreError::Serialization)?;
        self.shadow_parameters.clear();
        self.shadow_parameters.extend_from_slice(params);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[derive(Debug)]
    struct ReferenceLinearBackend {
        schema: ModelTensorSchemaV1,
        params: Vec<f32>,
        placement: DevicePlacementV1,
        fail_push: bool,
    }

    impl ReferenceLinearBackend {
        fn new(bias: f32, weight: f32) -> Self {
            Self {
                schema: ModelTensorSchemaV1::for_linear(1, 1).unwrap(),
                // Canonical UTF-8 tensor order: "linear.bias" then "linear.weight"
                params: vec![bias, weight],
                placement: DevicePlacementV1::Cpu,
                fail_push: false,
            }
        }
    }

    impl BackendTensorBridge for ReferenceLinearBackend {
        fn backend_name(&self) -> &str {
            "reference-cpu-mock"
        }

        fn placement(&self) -> DevicePlacementV1 {
            self.placement
        }

        fn schema(&self) -> &ModelTensorSchemaV1 {
            &self.schema
        }

        fn pull_flat_parameters(&self) -> Result<Vec<f32>, BackendBridgeError> {
            Ok(self.params.clone())
        }

        fn push_flat_parameters(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
            if self.fail_push {
                return Err(BackendBridgeError::DeviceUnavailable);
            }
            if flat_params.len() != self.params.len() {
                return Err(BackendBridgeError::ParameterCountMismatch {
                    expected: self.params.len(),
                    got: flat_params.len(),
                });
            }
            self.params.clear();
            self.params.extend_from_slice(flat_params);
            Ok(())
        }

        fn forward_slice(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
            if input.len() != 1 {
                return Err(BackendBridgeError::ParameterCountMismatch {
                    expected: 1,
                    got: input.len(),
                });
            }
            let y = self.params[1] * input[0] + self.params[0];
            Ok(vec![y])
        }
    }

    #[test]
    fn backend_handle_enforces_schema_hash_and_transactional_load() {
        let backend = ReferenceLinearBackend::new(1.0, 2.0);
        let expected_hash = backend.schema().schema_hash();

        // Wrong schema hash is rejected before handle construction
        let mut wrong_hash = expected_hash;
        wrong_hash[0] ^= 0xff;
        assert_eq!(
            BackendModelHandleV1::try_new(ReferenceLinearBackend::new(1.0, 2.0), Some(wrong_hash))
                .unwrap_err(),
            BackendBridgeError::SchemaMismatch
        );

        let mut handle = BackendModelHandleV1::try_new(backend, Some(expected_hash)).unwrap();
        assert_eq!(handle.parameters(), &[1.0, 2.0]);
        assert_eq!(handle.forward(&[3.0]).unwrap(), vec![7.0]);

        // Non-finite load is rejected without mutating either backend or shadow state
        assert_eq!(
            handle.load_parameters(&[f32::NAN, 5.0]).unwrap_err(),
            CoreError::NanDetected
        );
        assert_eq!(handle.parameters(), &[1.0, 2.0]);

        // Valid load updates both backend and shadow parameters
        handle.load_parameters(&[0.5, 4.0]).unwrap();
        assert_eq!(handle.parameters(), &[0.5, 4.0]);
        assert_eq!(handle.forward(&[2.0]).unwrap(), vec![8.5]);

        // Delta extraction works relative to baseline
        let delta = handle.delta_since(&[0.0, 1.0]).unwrap();
        assert_eq!(delta, vec![0.5, 3.0]);
    }

    #[test]
    fn sgd_optimizer_step_syncs_backend_device_for_forward_and_rolls_back_on_push_failure() {
        use crate::optim::{OptimError, SgdOptimizer};
        use swarm_torch_core::traits::SwarmOptimizer;

        // Params order for `for_linear(1, 1)`: ["linear.bias", "linear.weight"] = [1.0, 2.0].
        // forward(&[3.0]) = 2.0 * 3.0 + 1.0 = 7.0.
        let backend = ReferenceLinearBackend::new(1.0, 2.0);
        let mut handle = BackendModelHandleV1::try_new(backend, None).unwrap();
        assert_eq!(handle.forward(&[3.0]).unwrap(), vec![7.0]);

        let mut opt =
            SgdOptimizer::<BackendModelHandleV1<ReferenceLinearBackend>, 8>::new(0.5).unwrap();
        // Step with grad=[2.0, 4.0], lr=0.5 => bias=0.0, weight=0.0.
        opt.step(&mut handle, &[2.0, 4.0]).unwrap();
        assert_eq!(handle.parameters(), &[0.0, 0.0]);
        assert_eq!(handle.backend().params, vec![0.0, 0.0]);
        assert_eq!(handle.forward(&[3.0]).unwrap(), vec![0.0]);

        // Another step with lr=0.5, grad=[-1.0, -2.0] => bias=0.5, weight=1.0 -> forward(&[3.0]) = 3.5.
        opt.step(&mut handle, &[-1.0, -2.0]).unwrap();
        assert_eq!(handle.forward(&[3.0]).unwrap(), vec![3.5]);

        // Simulated device push failure during optimizer step must leave BOTH device/shadow
        // parameters AND optimizer velocity completely intact.
        let mut failing_backend = ReferenceLinearBackend::new(1.0, 2.0);
        failing_backend.fail_push = true;
        let mut failing_handle = BackendModelHandleV1::try_new(failing_backend, None).unwrap();
        let mut failing_opt =
            SgdOptimizer::<BackendModelHandleV1<ReferenceLinearBackend>, 8>::with_hyperparams(
                0.5, 0.9, 0.0,
            )
            .unwrap();
        assert_eq!(
            failing_opt.step(&mut failing_handle, &[1.0, 1.0]),
            Err(OptimError::ModelUpdateFailed)
        );
        assert_eq!(failing_handle.parameters(), &[1.0, 2.0]);
        assert_eq!(failing_handle.forward(&[3.0]).unwrap(), vec![7.0]);
        assert!(failing_opt.velocity().is_empty());

        // Step-2 push failure after a successful Step 1 with momentum > 0 and weight_decay > 0
        // MUST preserve BOTH prior non-empty optimizer velocity and prior shadow/device weights.
        let mut step2_handle =
            BackendModelHandleV1::try_new(ReferenceLinearBackend::new(1.0, 2.0), None).unwrap();
        let mut step2_opt =
            SgdOptimizer::<BackendModelHandleV1<ReferenceLinearBackend>, 8>::with_hyperparams(
                0.5, 0.9, 0.1,
            )
            .unwrap();
        step2_opt.step(&mut step2_handle, &[1.0, 1.0]).unwrap();
        let step1_params = step2_handle.parameters().to_vec();
        let step1_vel = step2_opt.velocity().to_vec();
        let step1_fwd = step2_handle.forward(&[3.0]).unwrap();
        assert!(!step1_vel.is_empty());

        // Simulate subsequent backend device unavailability without modifying schema/params.
        // We wrap a failing backend initialized to the exact step-1 weights to verify rollback
        // of non-empty velocity during Step 2 push failure.
        let mut failing_step2_backend =
            ReferenceLinearBackend::new(step1_params[0], step1_params[1]);
        failing_step2_backend.fail_push = true;
        let mut failing_step2_handle =
            BackendModelHandleV1::try_new(failing_step2_backend, None).unwrap();
        let snap_after_step1 = step2_opt.snapshot_state().unwrap().unwrap();
        let mut resumed_step2_opt =
            SgdOptimizer::<BackendModelHandleV1<ReferenceLinearBackend>, 8>::new(0.1).unwrap();
        resumed_step2_opt.restore_state(&snap_after_step1).unwrap();
        assert_eq!(
            resumed_step2_opt.step(&mut failing_step2_handle, &[0.5, -0.5]),
            Err(OptimError::ModelUpdateFailed)
        );
        assert_eq!(failing_step2_handle.parameters(), step1_params.as_slice());
        assert_eq!(failing_step2_handle.forward(&[3.0]).unwrap(), step1_fwd);
        assert_eq!(resumed_step2_opt.velocity(), step1_vel.as_slice());
    }

    #[test]
    fn backend_handle_sync_commit_forward_and_delta_reject_non_finite_and_sync_manual_edits() {
        use core::sync::atomic::{AtomicBool, Ordering};

        #[derive(Debug)]
        struct MutableMockBackend {
            schema: ModelTensorSchemaV1,
            params: Vec<f32>,
            corrupt_next_pull: AtomicBool,
            next_output_override: Option<Vec<f32>>,
        }

        impl BackendTensorBridge for MutableMockBackend {
            fn backend_name(&self) -> &str {
                "mutable-mock"
            }

            fn placement(&self) -> DevicePlacementV1 {
                DevicePlacementV1::Cpu
            }

            fn schema(&self) -> &ModelTensorSchemaV1 {
                &self.schema
            }

            fn pull_flat_parameters(&self) -> Result<Vec<f32>, BackendBridgeError> {
                if self.corrupt_next_pull.load(Ordering::SeqCst) {
                    return Ok(vec![1.0, f32::NEG_INFINITY]);
                }
                Ok(self.params.clone())
            }

            fn push_flat_parameters(
                &mut self,
                flat_params: &[f32],
            ) -> Result<(), BackendBridgeError> {
                self.params = flat_params.to_vec();
                Ok(())
            }

            fn forward_slice(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
                if let Some(out) = &self.next_output_override {
                    return Ok(out.clone());
                }
                Ok(vec![self.params[1] * input[0] + self.params[0]])
            }
        }

        let backend = MutableMockBackend {
            schema: ModelTensorSchemaV1::for_linear(1, 1).unwrap(),
            params: vec![1.0, 2.0],
            corrupt_next_pull: AtomicBool::new(false),
            next_output_override: None,
        };
        let mut handle = BackendModelHandleV1::try_new(backend, None).unwrap();

        // Direct parameters_mut() followed by commit_to_backend() pushes to device for forward().
        handle.parameters_mut()[0] = 5.0;
        assert_eq!(handle.forward(&[3.0]).unwrap(), vec![7.0]); // still device value until commit
        handle.commit_to_backend().unwrap();
        assert_eq!(handle.forward(&[3.0]).unwrap(), vec![11.0]);

        // Non-finite mutation via parameters_mut() is blocked before device upload by commit_to_backend().
        handle.parameters_mut()[0] = f32::NAN;
        assert_eq!(
            handle.commit_to_backend().unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
        assert_eq!(handle.backend().params, vec![5.0, 2.0]);
        handle.parameters_mut()[0] = 5.0;

        // forward() rejects empty input and non-finite input.
        assert_eq!(
            handle.forward(&[]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
        assert_eq!(
            handle.forward(&[f32::INFINITY]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );

        // delta_since() rejects wrong length, non-finite baseline, and f32 overflow.
        assert_eq!(
            handle.delta_since(&[0.0]).unwrap_err(),
            BackendBridgeError::ParameterCountMismatch {
                expected: 2,
                got: 1,
            }
        );
        assert_eq!(
            handle.delta_since(&[0.0, f32::NAN]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
        handle.load_parameters(&[f32::MAX, 1.0]).unwrap();
        assert_eq!(
            handle.delta_since(&[-f32::MAX, 0.0]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
        handle.load_parameters(&[5.0, 2.0]).unwrap();

        // sync_from_backend() rejects non-finite device pulls transactionally without clobbering shadow.
        handle
            .backend()
            .corrupt_next_pull
            .store(true, Ordering::SeqCst);
        assert_eq!(
            handle.sync_from_backend().unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
        assert_eq!(handle.parameters(), &[5.0, 2.0]);
        handle
            .backend()
            .corrupt_next_pull
            .store(false, Ordering::SeqCst);
        assert_eq!(handle.sync_from_backend().unwrap(), &[5.0, 2.0]);

        // forward() rejects non-finite backend output without mutating shadow state.
        let h_bad_out = BackendModelHandleV1::try_new(
            MutableMockBackend {
                schema: ModelTensorSchemaV1::for_linear(1, 1).unwrap(),
                params: vec![1.0, 2.0],
                corrupt_next_pull: AtomicBool::new(false),
                next_output_override: Some(vec![f32::NAN]),
            },
            None,
        )
        .unwrap();
        assert_eq!(
            h_bad_out.forward(&[1.0]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
        assert_eq!(h_bad_out.parameters(), &[1.0, 2.0]);

        // load_parameters with wrong length and NaN/Inf leaves shadow untouched.
        assert_eq!(
            handle.load_parameters(&[1.0]).unwrap_err(),
            CoreError::InvalidGradient
        );
        assert_eq!(
            handle.load_parameters(&[1.0, f32::INFINITY]).unwrap_err(),
            CoreError::InfDetected
        );
        assert_eq!(handle.parameters(), &[5.0, 2.0]);
    }
}
