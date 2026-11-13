#![forbid(unsafe_code)]
//! Workspace-excluded tensor backend bridges for `Burn` and `Candle` (`ADR-0024` §4, `W12-01`).
//!
//! Keeps heavy DL SDK trees isolated outside root `Cargo.lock` while enforcing canonical
//! `ModelTensorSchemaV1` (`model_schema_hash`), schema-sorted parameter serialization, and
//! transactional `.is_finite()` verification across device synchronization boundaries.

use swarm_torch_models::backend_bridge::{
    BackendBridgeError, BackendModelHandleV1, BackendTensorBridge, DevicePlacementV1,
};
use swarm_torch_models::safetensors_v1::ModelTensorSchemaV1;

/// Minimal framework-facing trait implemented by downstream `Burn` module containers (`burn` feature).
pub trait BurnModuleOps: Send + Sync {
    /// Device placement (`Cpu` or `Wgpu { device_index }`).
    fn device_placement(&self) -> DevicePlacementV1;
    /// Canonical schema describing tensor names, shapes, and byte order.
    fn tensor_schema(&self) -> &ModelTensorSchemaV1;
    /// Export parameters in canonical [`ModelTensorSchemaV1`] tensor-name order.
    fn export_flat_f32(&self) -> Result<Vec<f32>, BackendBridgeError>;
    /// Atomically import parameters into backend tensors in canonical schema order.
    fn import_flat_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError>;
    /// Execute one forward pass over flattened input activations.
    fn forward_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError>;
}

/// [`BackendTensorBridge`] implementation wrapping any [`BurnModuleOps`] model (`ADR-0024` §4).
#[derive(Debug, Clone)]
pub struct BurnModuleBridgeV1<M: BurnModuleOps> {
    inner: M,
}

impl<M: BurnModuleOps> BurnModuleBridgeV1<M> {
    /// Construct a new Burn module bridge.
    pub const fn new(inner: M) -> Self {
        Self { inner }
    }

    /// Wrap this Burn bridge inside a verified [`BackendModelHandleV1`].
    pub fn into_handle(
        self,
        expected_schema_hash: Option<[u8; 32]>,
    ) -> Result<BackendModelHandleV1<Self>, BackendBridgeError> {
        BackendModelHandleV1::try_new(self, expected_schema_hash)
    }
}

impl<M: BurnModuleOps> BackendTensorBridge for BurnModuleBridgeV1<M> {
    fn backend_name(&self) -> &str {
        "burn-adapter-v1"
    }

    fn placement(&self) -> DevicePlacementV1 {
        self.inner.device_placement()
    }

    fn schema(&self) -> &ModelTensorSchemaV1 {
        self.inner.tensor_schema()
    }

    fn pull_flat_parameters(&self) -> Result<Vec<f32>, BackendBridgeError> {
        self.inner.export_flat_f32()
    }

    fn push_flat_parameters(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
        self.inner.import_flat_f32(flat_params)
    }

    fn forward_slice(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
        self.inner.forward_flat(input)
    }
}

/// Minimal framework-facing trait implemented by downstream `Candle` `VarMap` / module containers.
pub trait CandleModuleOps: Send + Sync {
    /// Target device placement (`Cpu` or `Cuda { device_index }`).
    fn device_placement(&self) -> DevicePlacementV1;
    /// Canonical schema describing tensor names, shapes, and byte order.
    fn tensor_schema(&self) -> &ModelTensorSchemaV1;
    /// Read flat parameters in canonical [`ModelTensorSchemaV1`] tensor order.
    fn read_varmap_f32(&self) -> Result<Vec<f32>, BackendBridgeError>;
    /// Write flat parameters into `VarMap` transactionally.
    fn write_varmap_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError>;
    /// Execute a forward inference pass over flat input activations.
    fn eval_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError>;
}

/// [`BackendTensorBridge`] implementation wrapping any [`CandleModuleOps`] model (`ADR-0024` §4).
#[derive(Debug, Clone)]
pub struct CandleModuleBridgeV1<M: CandleModuleOps> {
    inner: M,
}

impl<M: CandleModuleOps> CandleModuleBridgeV1<M> {
    /// Construct a new Candle module bridge.
    pub const fn new(inner: M) -> Self {
        Self { inner }
    }

    /// Wrap this Candle bridge inside a verified [`BackendModelHandleV1`].
    pub fn into_handle(
        self,
        expected_schema_hash: Option<[u8; 32]>,
    ) -> Result<BackendModelHandleV1<Self>, BackendBridgeError> {
        BackendModelHandleV1::try_new(self, expected_schema_hash)
    }
}

impl<M: CandleModuleOps> BackendTensorBridge for CandleModuleBridgeV1<M> {
    fn backend_name(&self) -> &str {
        "candle-adapter-v1"
    }

    fn placement(&self) -> DevicePlacementV1 {
        self.inner.device_placement()
    }

    fn schema(&self) -> &ModelTensorSchemaV1 {
        self.inner.tensor_schema()
    }

    fn pull_flat_parameters(&self) -> Result<Vec<f32>, BackendBridgeError> {
        self.inner.read_varmap_f32()
    }

    fn push_flat_parameters(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
        self.inner.write_varmap_f32(flat_params)
    }

    fn forward_slice(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
        self.inner.eval_flat(input)
    }
}

/// In-memory reference affine model (`y = W * x + b`) validating schema-ordered Burn/Candle parity.
#[derive(Debug, Clone)]
pub struct ReferenceLinearModule {
    schema: ModelTensorSchemaV1,
    placement: DevicePlacementV1,
    // Canonical sorted order from ModelTensorSchemaV1::for_linear:
    // ["linear.bias" (out_dim), "linear.weight" (out_dim * in_dim)]
    params: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl ReferenceLinearModule {
    /// Construct an affine module matching `ModelTensorSchemaV1::for_linear(in_dim, out_dim)`.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        placement: DevicePlacementV1,
    ) -> Result<Self, BackendBridgeError> {
        let schema = ModelTensorSchemaV1::for_linear(in_dim, out_dim)
            .map_err(|_| BackendBridgeError::SchemaMismatch)?;
        let total = schema
            .total_elements()
            .map_err(|_| BackendBridgeError::SchemaMismatch)?;
        Ok(Self {
            schema,
            placement,
            params: vec![0.0; total],
            in_dim,
            out_dim,
        })
    }

    fn eval_affine(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
        if input.len() != self.in_dim {
            return Err(BackendBridgeError::ParameterCountMismatch {
                expected: self.in_dim,
                got: input.len(),
            });
        }
        let bias = &self.params[..self.out_dim];
        let weight = &self.params[self.out_dim..];
        let mut out = Vec::with_capacity(self.out_dim);
        for row in 0..self.out_dim {
            let mut acc = bias[row] as f64;
            for col in 0..self.in_dim {
                acc += (weight[row * self.in_dim + col] as f64) * (input[col] as f64);
            }
            out.push(acc as f32);
        }
        Ok(out)
    }
}

impl BurnModuleOps for ReferenceLinearModule {
    fn device_placement(&self) -> DevicePlacementV1 {
        self.placement
    }

    fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
        &self.schema
    }

    fn export_flat_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
        Ok(self.params.clone())
    }

    fn import_flat_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
        if flat_params.len() != self.params.len() {
            return Err(BackendBridgeError::ParameterCountMismatch {
                expected: self.params.len(),
                got: flat_params.len(),
            });
        }
        self.params.copy_from_slice(flat_params);
        Ok(())
    }

    fn forward_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
        self.eval_affine(input)
    }
}

impl CandleModuleOps for ReferenceLinearModule {
    fn device_placement(&self) -> DevicePlacementV1 {
        self.placement
    }

    fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
        &self.schema
    }

    fn read_varmap_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
        Ok(self.params.clone())
    }

    fn write_varmap_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
        if flat_params.len() != self.params.len() {
            return Err(BackendBridgeError::ParameterCountMismatch {
                expected: self.params.len(),
                got: flat_params.len(),
            });
        }
        self.params.copy_from_slice(flat_params);
        Ok(())
    }

    fn eval_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
        self.eval_affine(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_torch_core::traits::SwarmModel;

    #[test]
    fn burn_and_candle_bridges_enforce_schema_hash_and_finiteness() {
        let schema = ModelTensorSchemaV1::for_linear(2, 1).unwrap();
        let expected_hash = schema.schema_hash();

        let burn_mod =
            ReferenceLinearModule::new(2, 1, DevicePlacementV1::Wgpu { device_index: 0 }).unwrap();
        let candle_mod =
            ReferenceLinearModule::new(2, 1, DevicePlacementV1::Cuda { device_index: 0 }).unwrap();

        let mut burn_handle = BurnModuleBridgeV1::new(burn_mod)
            .into_handle(Some(expected_hash))
            .unwrap();
        let mut candle_handle = CandleModuleBridgeV1::new(candle_mod)
            .into_handle(Some(expected_hash))
            .unwrap();

        // Wrong schema hash fails closed
        let wrong_hash_mod = ReferenceLinearModule::new(3, 1, DevicePlacementV1::Cpu).unwrap();
        assert_eq!(
            BurnModuleBridgeV1::new(wrong_hash_mod)
                .into_handle(Some(expected_hash))
                .unwrap_err(),
            BackendBridgeError::SchemaMismatch
        );

        // Load identical canonical params [bias, w0, w1] into both bridges
        let weights = [0.5_f32, 2.0, -1.0];
        burn_handle.load_parameters(&weights).unwrap();
        candle_handle.load_parameters(&weights).unwrap();

        assert_eq!(burn_handle.forward(&[3.0, 1.0]).unwrap(), vec![5.5]);
        assert_eq!(candle_handle.forward(&[3.0, 1.0]).unwrap(), vec![5.5]);

        // NaN parameters are rejected before device write, preserving prior parameters
        assert!(burn_handle.load_parameters(&[f32::NAN, 1.0, 1.0]).is_err());
        assert_eq!(burn_handle.parameters(), &weights);
    }
}
