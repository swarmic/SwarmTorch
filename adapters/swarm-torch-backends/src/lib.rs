#![forbid(unsafe_code)]
//! Workspace-excluded tensor backend bridges for `Burn` (CPU `NdArray` and GPU `Wgpu`) and `Candle`
//! (`ADR-0015`, `ADR-0024` §4, `W12-01`, `13D-4`, `14B`, `15D-4`, `Wave 16D`).
//!
//! **Maturity Tier:** `Integration verified` — live `burn 0.21` (`Autodiff<NdArray<f32>>` CPU and
//! `Autodiff<burn_wgpu::Wgpu<f32, i32, u32>>` GPU under `real-burn-wgpu`) and `candle 0.10`
//! (`VarMap` / `Linear` CPU) linear (`[W | b]`) and 2-layer ReLU MLP (`[W_1 | b_1 | W_2 | b_2]`)
//! bridges enforce canonical [`ModelTensorSchemaV1`] (`model_schema_hash`),
//! [`swarm_torch_models::safetensors_v1::ModelFlatTensorLayoutV1`] parameter/gradient alignment with `SwarmTorch::LinearModel` and `SwarmTorch::MlpModel`, and
//! transactional `.is_finite()` verification on every import, export, forward, and gradient path.
//! `real-burn-wgpu` enumerates per-type WGPU adapters using a serialized, backend-scoped
//! `wgpu::Instance` matching `AutoGraphicsApi::backend()` (avoiding both `cubecl-wgpu 0.10.0`'s
//! mixed-GPU global-index panic bug in `enumerate_all_devices` and its `Backends::all()` WGL
//! thread-crash hazard in `enumerate_devices`) and fails closed with
//! [`BackendBridgeError::DeviceUnavailable`] when no matching adapter is available; `Cuda` placement
//! remains reserved for future CUDA-runner verification.

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
    /// Export parameters in canonical declaration layout (`[W | b]` or `[W_1 | b_1 | W_2 | b_2]`).
    fn export_flat_f32(&self) -> Result<Vec<f32>, BackendBridgeError>;
    /// Atomically import parameters into backend tensors in canonical declaration layout.
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

    /// Borrow the underlying [`BurnModuleOps`] module.
    pub const fn inner(&self) -> &M {
        &self.inner
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
    /// Read flat parameters in canonical declaration layout (`[W | b]` or `[W_1 | b_1 | W_2 | b_2]`).
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

    /// Borrow the underlying [`CandleModuleOps`] module.
    pub const fn inner(&self) -> &M {
        &self.inner
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

/// In-memory reference affine model (`y = W * x + b`) in canonical `[W | b]` layout (`AR-14-07`).
#[derive(Debug, Clone)]
pub struct ReferenceLinearModule {
    schema: ModelTensorSchemaV1,
    placement: DevicePlacementV1,
    // Canonical declaration layout matching SwarmTorch::LinearModel (`ModelFlatTensorLayoutV1::for_linear`):
    // [weight (out_dim * in_dim) | bias (out_dim)]
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

    /// Batched affine evaluation over `[batch, in_dim]` row-major input, returning
    /// `[batch, out_dim]`, with the same finiteness and shape checks as the live bridges.
    fn eval_affine(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
        let batch = validate_batch_input(input, self.in_dim)?;
        let weight_len = self.out_dim * self.in_dim;
        let weight = &self.params[..weight_len];
        let bias = &self.params[weight_len..];
        let mut out = Vec::with_capacity(batch * self.out_dim);
        for x in input.chunks_exact(self.in_dim) {
            for row in 0..self.out_dim {
                // `f32` in the live bridges' order (dot product, then bias) so the reference
                // overflows exactly where Burn and Candle do.
                let mut dot = 0.0f32;
                for col in 0..self.in_dim {
                    dot += weight[row * self.in_dim + col] * x[col];
                }
                out.push(dot + bias[row]);
            }
        }
        validate_finite_f32_slice(&out, Some(batch * self.out_dim))?;
        Ok(out)
    }

    /// Validate length and finiteness completely before replacing any parameter.
    fn set_params(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
        validate_finite_f32_slice(flat_params, Some(self.params.len()))?;
        self.params.copy_from_slice(flat_params);
        Ok(())
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
        self.set_params(flat_params)
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
        self.set_params(flat_params)
    }

    fn eval_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
        self.eval_affine(input)
    }
}

fn validate_finite_f32_slice(
    values: &[f32],
    expected_len: Option<usize>,
) -> Result<(), BackendBridgeError> {
    if let Some(expected) = expected_len {
        if values.len() != expected {
            return Err(BackendBridgeError::ParameterCountMismatch {
                expected,
                got: values.len(),
            });
        }
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(BackendBridgeError::NonFiniteValue);
    }
    Ok(())
}

fn validate_batch_input(input: &[f32], in_dim: usize) -> Result<usize, BackendBridgeError> {
    if in_dim == 0 || input.is_empty() || !input.len().is_multiple_of(in_dim) {
        return Err(BackendBridgeError::ParameterCountMismatch {
            expected: in_dim,
            got: input.len(),
        });
    }
    validate_finite_f32_slice(input, None)?;
    Ok(input.len() / in_dim)
}

#[cfg(feature = "real-burn")]
mod burn_generic {
    use std::sync::Mutex;

    use burn::backend::Autodiff;
    use burn::module::Param;
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::activation::relu;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Tensor, TensorData};
    use swarm_torch_models::backend_bridge::{BackendBridgeError, DevicePlacementV1};
    use swarm_torch_models::safetensors_v1::ModelTensorSchemaV1;

    use crate::{validate_batch_input, validate_finite_f32_slice};

    // The bridge mutex is deliberately held across graph construction and `backward()`
    // (`15D-4` T-Agent review). Computing on a `Module::clone` snapshot instead lets concurrent
    // calls share `Param` autodiff leaves, which deadlocks burn-autodiff 0.21 (its global `STATE`
    // and per-graph locks are taken in opposite orders), and cloning a not-yet-initialized lazy
    // `Param` re-draws independent random weights per clone. Concurrent calls therefore serialize.

    #[derive(Debug)]
    pub(crate) struct BurnLinearCore<B: Backend> {
        schema: ModelTensorSchemaV1,
        placement: DevicePlacementV1,
        in_dim: usize,
        out_dim: usize,
        device: B::Device,
        linear: Mutex<Linear<Autodiff<B>>>,
    }

    impl<B: Backend> BurnLinearCore<B> {
        pub(crate) fn new(
            in_dim: usize,
            out_dim: usize,
            device: B::Device,
            placement: DevicePlacementV1,
        ) -> Result<Self, BackendBridgeError> {
            let schema = ModelTensorSchemaV1::for_linear(in_dim, out_dim)
                .map_err(|_| BackendBridgeError::SchemaMismatch)?;
            Ok(Self::new_with_schema(
                schema, in_dim, out_dim, device, placement,
            ))
        }

        pub(crate) fn new_with_schema(
            schema: ModelTensorSchemaV1,
            in_dim: usize,
            out_dim: usize,
            device: B::Device,
            placement: DevicePlacementV1,
        ) -> Self {
            let linear = LinearConfig::new(in_dim, out_dim)
                .with_bias(true)
                .init::<Autodiff<B>>(&device);
            Self {
                schema,
                placement,
                in_dim,
                out_dim,
                device,
                linear: Mutex::new(linear),
            }
        }

        pub(crate) fn device_placement(&self) -> DevicePlacementV1 {
            self.placement
        }

        pub(crate) fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
            &self.schema
        }

        pub(crate) fn backward_mse(
            &self,
            input: &[f32],
            target: &[f32],
        ) -> Result<(f32, Vec<f32>), BackendBridgeError> {
            let batch = validate_batch_input(input, self.in_dim)?;
            let expected_out = batch
                .checked_mul(self.out_dim)
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            validate_finite_f32_slice(target, Some(expected_out))?;

            let guard = self
                .linear
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;

            let x = Tensor::<Autodiff<B>, 2>::from_data(
                TensorData::new(input.to_vec(), [batch, self.in_dim]),
                &self.device,
            );
            let t = Tensor::<Autodiff<B>, 2>::from_data(
                TensorData::new(target.to_vec(), [batch, self.out_dim]),
                &self.device,
            );
            let pred = guard.forward(x);
            let diff = pred - t;
            let loss = diff.clone().mul(diff).mean();

            let loss_vec = loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;
            let loss_scalar = *loss_vec.first().ok_or(BackendBridgeError::NonFiniteValue)?;
            validate_finite_f32_slice(&[loss_scalar], Some(1))?;

            let grads = loss.backward();
            let bias_param = guard
                .bias
                .as_ref()
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            let b_grad = bias_param
                .grad(&grads)
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed("missing bias gradient".to_string())
                })?
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;

            // Burn stores `Linear.weight` in `[d_input, d_output]`; transpose gradient to
            // canonical `[d_output, d_input]` row-major layout (`ModelTensorSchemaV1::for_linear`).
            let w_grad = guard
                .weight
                .grad(&grads)
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed(
                        "missing weight gradient".to_string(),
                    )
                })?
                .transpose()
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;

            let expected = self.out_dim * self.in_dim + self.out_dim;
            let mut flat_grads = Vec::with_capacity(expected);
            flat_grads.extend_from_slice(&w_grad);
            flat_grads.extend_from_slice(&b_grad);
            validate_finite_f32_slice(&flat_grads, Some(expected))?;
            Ok((loss_scalar, flat_grads))
        }

        pub(crate) fn export_flat_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
            let guard = self
                .linear
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let bias_vec = guard
                .bias
                .as_ref()
                .ok_or(BackendBridgeError::SchemaMismatch)?
                .val()
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;
            // Transpose Burn `[in_dim, out_dim]` weight to canonical `[out_dim, in_dim]`.
            let weight_vec = guard
                .weight
                .val()
                .transpose()
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;

            let expected = self.out_dim * self.in_dim + self.out_dim;
            let mut out = Vec::with_capacity(expected);
            out.extend_from_slice(&weight_vec);
            out.extend_from_slice(&bias_vec);
            validate_finite_f32_slice(&out, Some(expected))?;
            Ok(out)
        }

        pub(crate) fn import_flat_f32(
            &mut self,
            flat_params: &[f32],
        ) -> Result<(), BackendBridgeError> {
            let weight_len = self.out_dim * self.in_dim;
            let expected = weight_len + self.out_dim;
            validate_finite_f32_slice(flat_params, Some(expected))?;

            // Prepare leaf tensors completely before acquiring lock or replacing parameters.
            let inner_weight_t = Tensor::<B, 2>::from_data(
                TensorData::new(
                    flat_params[..weight_len].to_vec(),
                    [self.out_dim, self.in_dim],
                ),
                &self.device,
            )
            .transpose();
            let inner_bias = Tensor::<B, 1>::from_data(
                TensorData::new(flat_params[weight_len..].to_vec(), [self.out_dim]),
                &self.device,
            );

            let new_weight = Param::from_tensor(
                Tensor::<Autodiff<B>, 2>::from_inner(inner_weight_t).require_grad(),
            );
            let new_bias =
                Param::from_tensor(Tensor::<Autodiff<B>, 1>::from_inner(inner_bias).require_grad());

            let mut guard = self
                .linear
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            guard.weight = new_weight;
            guard.bias = Some(new_bias);
            Ok(())
        }

        pub(crate) fn forward_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
            let batch = validate_batch_input(input, self.in_dim)?;
            let expected_out = batch
                .checked_mul(self.out_dim)
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            let guard = self
                .linear
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let x = Tensor::<Autodiff<B>, 2>::from_data(
                TensorData::new(input.to_vec(), [batch, self.in_dim]),
                &self.device,
            );
            let out = guard
                .forward(x)
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;
            validate_finite_f32_slice(&out, Some(expected_out))?;
            Ok(out)
        }
    }

    type MlpLayerPair<B> = (Linear<Autodiff<B>>, Linear<Autodiff<B>>);

    #[derive(Debug)]
    pub(crate) struct BurnMlpCore<B: Backend> {
        schema: ModelTensorSchemaV1,
        placement: DevicePlacementV1,
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        device: B::Device,
        layers: Mutex<MlpLayerPair<B>>,
    }

    impl<B: Backend> BurnMlpCore<B> {
        pub(crate) fn new(
            in_dim: usize,
            hidden_dim: usize,
            out_dim: usize,
            device: B::Device,
            placement: DevicePlacementV1,
        ) -> Result<Self, BackendBridgeError> {
            let schema = ModelTensorSchemaV1::for_mlp(in_dim, hidden_dim, out_dim)
                .map_err(|_| BackendBridgeError::SchemaMismatch)?;
            Ok(Self::new_with_schema(
                schema, in_dim, hidden_dim, out_dim, device, placement,
            ))
        }

        pub(crate) fn new_with_schema(
            schema: ModelTensorSchemaV1,
            in_dim: usize,
            hidden_dim: usize,
            out_dim: usize,
            device: B::Device,
            placement: DevicePlacementV1,
        ) -> Self {
            let fc1 = LinearConfig::new(in_dim, hidden_dim)
                .with_bias(true)
                .init::<Autodiff<B>>(&device);
            let fc2 = LinearConfig::new(hidden_dim, out_dim)
                .with_bias(true)
                .init::<Autodiff<B>>(&device);
            Self {
                schema,
                placement,
                in_dim,
                hidden_dim,
                out_dim,
                device,
                layers: Mutex::new((fc1, fc2)),
            }
        }

        fn total_elements(&self) -> usize {
            self.hidden_dim * self.in_dim
                + self.hidden_dim
                + self.out_dim * self.hidden_dim
                + self.out_dim
        }

        pub(crate) fn device_placement(&self) -> DevicePlacementV1 {
            self.placement
        }

        pub(crate) fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
            &self.schema
        }

        pub(crate) fn backward_mse(
            &self,
            input: &[f32],
            target: &[f32],
        ) -> Result<(f32, Vec<f32>), BackendBridgeError> {
            let batch = validate_batch_input(input, self.in_dim)?;
            let expected_out = batch
                .checked_mul(self.out_dim)
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            validate_finite_f32_slice(target, Some(expected_out))?;

            let guard = self
                .layers
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let (ref fc1, ref fc2) = *guard;

            let x = Tensor::<Autodiff<B>, 2>::from_data(
                TensorData::new(input.to_vec(), [batch, self.in_dim]),
                &self.device,
            );
            let t = Tensor::<Autodiff<B>, 2>::from_data(
                TensorData::new(target.to_vec(), [batch, self.out_dim]),
                &self.device,
            );
            let h = relu(fc1.forward(x));
            let pred = fc2.forward(h);
            let diff = pred - t;
            let loss = diff.clone().mul(diff).mean();

            let loss_vec = loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;
            let loss_scalar = *loss_vec.first().ok_or(BackendBridgeError::NonFiniteValue)?;
            validate_finite_f32_slice(&[loss_scalar], Some(1))?;

            let grads = loss.backward();

            let w1_grad = fc1
                .weight
                .grad(&grads)
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed(
                        "missing fc1.weight grad".to_string(),
                    )
                })?
                .transpose()
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;
            let b1_grad = fc1
                .bias
                .as_ref()
                .ok_or(BackendBridgeError::SchemaMismatch)?
                .grad(&grads)
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed("missing fc1.bias grad".to_string())
                })?
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;

            let w2_grad = fc2
                .weight
                .grad(&grads)
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed(
                        "missing fc2.weight grad".to_string(),
                    )
                })?
                .transpose()
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;
            let b2_grad = fc2
                .bias
                .as_ref()
                .ok_or(BackendBridgeError::SchemaMismatch)?
                .grad(&grads)
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed("missing fc2.bias grad".to_string())
                })?
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;

            let expected = self.total_elements();
            let mut flat_grads = Vec::with_capacity(expected);
            flat_grads.extend_from_slice(&w1_grad);
            flat_grads.extend_from_slice(&b1_grad);
            flat_grads.extend_from_slice(&w2_grad);
            flat_grads.extend_from_slice(&b2_grad);
            validate_finite_f32_slice(&flat_grads, Some(expected))?;
            Ok((loss_scalar, flat_grads))
        }

        pub(crate) fn export_flat_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
            let guard = self
                .layers
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let (ref fc1, ref fc2) = *guard;

            let w1_vec = fc1
                .weight
                .val()
                .transpose()
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;
            let b1_vec = fc1
                .bias
                .as_ref()
                .ok_or(BackendBridgeError::SchemaMismatch)?
                .val()
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;
            let w2_vec = fc2
                .weight
                .val()
                .transpose()
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;
            let b2_vec = fc2
                .bias
                .as_ref()
                .ok_or(BackendBridgeError::SchemaMismatch)?
                .val()
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;

            let expected = self.total_elements();
            let mut out = Vec::with_capacity(expected);
            out.extend_from_slice(&w1_vec);
            out.extend_from_slice(&b1_vec);
            out.extend_from_slice(&w2_vec);
            out.extend_from_slice(&b2_vec);
            validate_finite_f32_slice(&out, Some(expected))?;
            Ok(out)
        }

        pub(crate) fn import_flat_f32(
            &mut self,
            flat_params: &[f32],
        ) -> Result<(), BackendBridgeError> {
            let w1_len = self.hidden_dim * self.in_dim;
            let b1_end = w1_len + self.hidden_dim;
            let w2_end = b1_end + self.out_dim * self.hidden_dim;
            let expected = w2_end + self.out_dim;
            validate_finite_f32_slice(flat_params, Some(expected))?;

            let inner_w1_t = Tensor::<B, 2>::from_data(
                TensorData::new(
                    flat_params[..w1_len].to_vec(),
                    [self.hidden_dim, self.in_dim],
                ),
                &self.device,
            )
            .transpose();
            let inner_b1 = Tensor::<B, 1>::from_data(
                TensorData::new(flat_params[w1_len..b1_end].to_vec(), [self.hidden_dim]),
                &self.device,
            );
            let inner_w2_t = Tensor::<B, 2>::from_data(
                TensorData::new(
                    flat_params[b1_end..w2_end].to_vec(),
                    [self.out_dim, self.hidden_dim],
                ),
                &self.device,
            )
            .transpose();
            let inner_b2 = Tensor::<B, 1>::from_data(
                TensorData::new(flat_params[w2_end..].to_vec(), [self.out_dim]),
                &self.device,
            );

            let new_w1 =
                Param::from_tensor(Tensor::<Autodiff<B>, 2>::from_inner(inner_w1_t).require_grad());
            let new_b1 =
                Param::from_tensor(Tensor::<Autodiff<B>, 1>::from_inner(inner_b1).require_grad());
            let new_w2 =
                Param::from_tensor(Tensor::<Autodiff<B>, 2>::from_inner(inner_w2_t).require_grad());
            let new_b2 =
                Param::from_tensor(Tensor::<Autodiff<B>, 1>::from_inner(inner_b2).require_grad());

            let mut guard = self
                .layers
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            guard.0.weight = new_w1;
            guard.0.bias = Some(new_b1);
            guard.1.weight = new_w2;
            guard.1.bias = Some(new_b2);
            Ok(())
        }

        pub(crate) fn forward_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
            let batch = validate_batch_input(input, self.in_dim)?;
            let expected_out = batch
                .checked_mul(self.out_dim)
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            let guard = self
                .layers
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let x = Tensor::<Autodiff<B>, 2>::from_data(
                TensorData::new(input.to_vec(), [batch, self.in_dim]),
                &self.device,
            );
            let h = relu(guard.0.forward(x));
            let out = guard
                .1
                .forward(h)
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;
            validate_finite_f32_slice(&out, Some(expected_out))?;
            Ok(out)
        }
    }
}

/// Live `Burn 0.21` (`Autodiff<NdArray<f32>>`) CPU linear and 2-layer MLP backend bridges (`ST-08`, `AR-14-07`, `14B-1`).
#[cfg(feature = "real-burn")]
pub mod real_burn {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    use swarm_torch_models::backend_bridge::{BackendBridgeError, DevicePlacementV1};
    use swarm_torch_models::safetensors_v1::ModelTensorSchemaV1;

    use crate::burn_generic::{BurnLinearCore, BurnMlpCore};
    use crate::BurnModuleOps;

    /// CPU autodiff Burn backend used by [`BurnLinearBridgeV1`] and [`BurnMlpBridgeV1`].
    pub type BurnNdArrayBackend = Autodiff<NdArray<f32>>;

    /// Live `Burn` CPU affine module (`y = x @ W_t + b`) synchronized transactionally in canonical
    /// `[weight(out_dim * in_dim) | bias(out_dim)]` layout matching [`swarm_torch_models::simple::LinearModel`] (`AR-14-07`).
    #[derive(Debug)]
    pub struct BurnLinearBridgeV1 {
        core: BurnLinearCore<NdArray<f32>>,
    }

    impl BurnLinearBridgeV1 {
        /// Construct a new CPU autodiff `Burn` linear module matching `for_linear(in_dim, out_dim)`.
        pub fn new(in_dim: usize, out_dim: usize) -> Result<Self, BackendBridgeError> {
            Ok(Self {
                core: BurnLinearCore::new(
                    in_dim,
                    out_dim,
                    NdArrayDevice::Cpu,
                    DevicePlacementV1::Cpu,
                )?,
            })
        }

        /// Compute MSE loss `(1/N) * sum((y - target)^2)` and reverse-mode autodiff gradients
        /// in canonical `[weight_grad(out_dim * in_dim) | bias_grad(out_dim)]` order (`AR-14-07`).
        pub fn backward_mse(
            &self,
            input: &[f32],
            target: &[f32],
        ) -> Result<(f32, Vec<f32>), BackendBridgeError> {
            self.core.backward_mse(input, target)
        }
    }

    impl BurnModuleOps for BurnLinearBridgeV1 {
        fn device_placement(&self) -> DevicePlacementV1 {
            self.core.device_placement()
        }

        fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
            self.core.tensor_schema()
        }

        fn export_flat_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
            self.core.export_flat_f32()
        }

        fn import_flat_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
            self.core.import_flat_f32(flat_params)
        }

        fn forward_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
            self.core.forward_flat(input)
        }
    }

    /// Live `Burn` CPU 2-layer ReLU MLP (`y = fc2(relu(fc1(x)))`) synchronized transactionally in
    /// canonical `[W_1 | b_1 | W_2 | b_2]` layout matching [`swarm_torch_models::mlp::MlpModel`] (`14B-1`).
    #[derive(Debug)]
    pub struct BurnMlpBridgeV1 {
        core: BurnMlpCore<NdArray<f32>>,
    }

    impl BurnMlpBridgeV1 {
        /// Construct a new CPU autodiff `Burn` 2-layer ReLU MLP matching `for_mlp(in_dim, hidden_dim, out_dim)`.
        pub fn new(
            in_dim: usize,
            hidden_dim: usize,
            out_dim: usize,
        ) -> Result<Self, BackendBridgeError> {
            Ok(Self {
                core: BurnMlpCore::new(
                    in_dim,
                    hidden_dim,
                    out_dim,
                    NdArrayDevice::Cpu,
                    DevicePlacementV1::Cpu,
                )?,
            })
        }

        /// Compute MSE loss `(1/N) * sum((y - target)^2)` and reverse-mode autodiff gradients
        /// in canonical `[dW_1 | db_1 | dW_2 | db_2]` order matching [`swarm_torch_models::mlp::MlpModel`].
        pub fn backward_mse(
            &self,
            input: &[f32],
            target: &[f32],
        ) -> Result<(f32, Vec<f32>), BackendBridgeError> {
            self.core.backward_mse(input, target)
        }
    }

    impl BurnModuleOps for BurnMlpBridgeV1 {
        fn device_placement(&self) -> DevicePlacementV1 {
            self.core.device_placement()
        }

        fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
            self.core.tensor_schema()
        }

        fn export_flat_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
            self.core.export_flat_f32()
        }

        fn import_flat_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
            self.core.import_flat_f32(flat_params)
        }

        fn forward_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
            self.core.forward_flat(input)
        }
    }
}

/// Live `Burn 0.21` (`Autodiff<burn_wgpu::Wgpu<f32, i32, u32>>`) GPU backend bridges & fail-closed
/// per-type WGPU device selector (`ADR-0015`, `Wave 16D`).
#[cfg(feature = "real-burn-wgpu")]
pub mod real_burn_wgpu {
    use std::collections::BTreeSet;
    #[cfg(not(target_family = "wasm"))]
    use std::sync::Mutex;

    use burn::backend::Autodiff;
    use burn_wgpu::graphics::{AutoGraphicsApi, GraphicsApi};
    use burn_wgpu::Wgpu;
    pub use burn_wgpu::WgpuDevice;
    #[cfg(target_family = "wasm")]
    use burn_wgpu::WgpuRuntime;
    use cubecl::device::DeviceId;
    #[cfg(target_family = "wasm")]
    use cubecl::Runtime;
    use swarm_torch_models::backend_bridge::{BackendBridgeError, DevicePlacementV1};
    use swarm_torch_models::safetensors_v1::ModelTensorSchemaV1;

    use crate::burn_generic::{BurnLinearCore, BurnMlpCore};
    use crate::BurnModuleOps;

    /// Inner WGPU Burn backend (`f32` float, `i32` int, `u32` bool).
    pub type BurnWgpuInnerBackend = Wgpu<f32, i32, u32>;

    /// Autodiff WGPU Burn backend used by [`BurnWgpuLinearBridgeV1`] and [`BurnWgpuMlpBridgeV1`].
    pub type BurnWgpuBackend = Autodiff<BurnWgpuInnerBackend>;

    /// WGPU adapter hardware classification in deterministic preference order (`ADR-0015`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum WgpuDeviceKindV1 {
        /// Dedicated discrete GPU (`type_id = 0` in `cubecl-wgpu`).
        DiscreteGpu,
        /// Integrated GPU (`type_id = 1` in `cubecl-wgpu`).
        IntegratedGpu,
        /// Virtual GPU (`type_id = 2` in `cubecl-wgpu`).
        VirtualGpu,
        /// Software CPU WGPU adapter (`type_id = 3` in `cubecl-wgpu`).
        Cpu,
    }

    /// Validated per-type WGPU adapter descriptor produced by [`enumerate_wgpu_devices`] (`Wave 16D`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct WgpuDeviceDescriptorV1 {
        /// Deterministic logical device index mapped to [`DevicePlacementV1::Wgpu`].
        pub device_index: u32,
        /// Hardware class of the adapter.
        pub kind: WgpuDeviceKindV1,
        /// Per-type adapter ordinal valid for `cubecl-wgpu`'s `select_from_adapter_list`.
        pub type_ordinal: usize,
        /// Exact `burn_wgpu::WgpuDevice` ready to pass to Burn tensor initialization.
        pub wgpu_device: WgpuDevice,
    }

    /// Explicit selector for targeting a validated WGPU adapter (`Wave 16D`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum WgpuDeviceSelectorV1 {
        /// Select the `device_index`-th device in deterministic preference order
        /// (`DiscreteGpu` -> `IntegratedGpu` -> `VirtualGpu` -> `Cpu`).
        LogicalIndex(u32),
        /// Select an explicit discrete GPU per-type ordinal (`WgpuDevice::DiscreteGpu(ordinal)`).
        DiscreteGpu(usize),
        /// Select an explicit integrated GPU per-type ordinal (`WgpuDevice::IntegratedGpu(ordinal)`).
        IntegratedGpu(usize),
        /// Select an explicit virtual GPU per-type ordinal (`WgpuDevice::VirtualGpu(ordinal)`).
        VirtualGpu(usize),
        /// Select the WGPU CPU software adapter (`WgpuDevice::Cpu`).
        Cpu,
    }

    /// Build a deterministic, per-type-validated WGPU device inventory from the four per-type
    /// `WgpuRuntime::enumerate_devices(type_id, &backend)` buckets (`type_id` `0..=3`).
    ///
    /// **Why per-type buckets (`Wave 16D`):** `cubecl-wgpu 0.10.0`'s `enumerate_all_devices`
    /// places the global adapter index into per-type `DeviceId(type_id, global_index)`, causing
    /// `select_from_adapter_list` to panic on mixed-GPU hosts (e.g., `DiscreteGpu(0)` +
    /// `IntegratedGpu(0)`). Per-type enumeration via `enumerate_devices(type_id, &backend)`
    /// filters by `type_id` prior to `.enumerate()`, preserving exact per-type ordinals.
    pub fn build_wgpu_device_inventory_from_type_buckets(
        discrete_ids: &[DeviceId],
        integrated_ids: &[DeviceId],
        virtual_ids: &[DeviceId],
        cpu_ids: &[DeviceId],
    ) -> Vec<WgpuDeviceDescriptorV1> {
        let mut out = Vec::new();

        let mut push_bucket = |ids: &[DeviceId], expected_type_id: u16, kind: WgpuDeviceKindV1| {
            if expected_type_id == 3 {
                if ids.iter().any(|id| id.type_id == 3) {
                    let device_index = u32::try_from(out.len())
                        .expect("total enumerated WGPU adapters bounded by 3 * 65536 + 1");
                    out.push(WgpuDeviceDescriptorV1 {
                        device_index,
                        kind: WgpuDeviceKindV1::Cpu,
                        type_ordinal: 0,
                        wgpu_device: WgpuDevice::Cpu,
                    });
                }
                return;
            }
            let unique_ordinals: BTreeSet<u16> = ids
                .iter()
                .filter(|id| id.type_id == expected_type_id)
                .map(|id| id.index_id)
                .collect();
            for ord in unique_ordinals {
                let device_index = u32::try_from(out.len())
                    .expect("total enumerated WGPU adapters bounded by 3 * 65536 + 1");
                let type_ordinal = usize::from(ord);
                let wgpu_device = match kind {
                    WgpuDeviceKindV1::DiscreteGpu => WgpuDevice::DiscreteGpu(type_ordinal),
                    WgpuDeviceKindV1::IntegratedGpu => WgpuDevice::IntegratedGpu(type_ordinal),
                    WgpuDeviceKindV1::VirtualGpu => WgpuDevice::VirtualGpu(type_ordinal),
                    WgpuDeviceKindV1::Cpu => WgpuDevice::Cpu,
                };
                out.push(WgpuDeviceDescriptorV1 {
                    device_index,
                    kind,
                    type_ordinal,
                    wgpu_device,
                });
            }
        };

        push_bucket(discrete_ids, 0, WgpuDeviceKindV1::DiscreteGpu);
        push_bucket(integrated_ids, 1, WgpuDeviceKindV1::IntegratedGpu);
        push_bucket(virtual_ids, 2, WgpuDeviceKindV1::VirtualGpu);
        push_bucket(cpu_ids, 3, WgpuDeviceKindV1::Cpu);
        out
    }

    /// Resolve a [`WgpuDeviceSelectorV1`] against a previously built device inventory, failing
    /// closed with [`BackendBridgeError::DeviceUnavailable`] when absent or out of range.
    pub fn resolve_wgpu_device_from_inventory(
        inventory: &[WgpuDeviceDescriptorV1],
        selector: WgpuDeviceSelectorV1,
    ) -> Result<WgpuDeviceDescriptorV1, BackendBridgeError> {
        let found = match selector {
            WgpuDeviceSelectorV1::LogicalIndex(idx) => {
                inventory.iter().find(|d| d.device_index == idx)
            }
            WgpuDeviceSelectorV1::DiscreteGpu(ord) => inventory
                .iter()
                .find(|d| d.kind == WgpuDeviceKindV1::DiscreteGpu && d.type_ordinal == ord),
            WgpuDeviceSelectorV1::IntegratedGpu(ord) => inventory
                .iter()
                .find(|d| d.kind == WgpuDeviceKindV1::IntegratedGpu && d.type_ordinal == ord),
            WgpuDeviceSelectorV1::VirtualGpu(ord) => inventory
                .iter()
                .find(|d| d.kind == WgpuDeviceKindV1::VirtualGpu && d.type_ordinal == ord),
            WgpuDeviceSelectorV1::Cpu => inventory.iter().find(|d| d.kind == WgpuDeviceKindV1::Cpu),
        };
        found.cloned().ok_or(BackendBridgeError::DeviceUnavailable)
    }

    /// Enumerate available WGPU adapters for `AutoGraphicsApi::backend()` with exact per-type
    /// ordinals matching `cubecl-wgpu`'s `select_from_adapter_list`.
    ///
    /// On native targets, serializes enumeration behind a process-wide mutex and restricts
    /// `wgpu::Instance` strictly to `backends: AutoGraphicsApi::backend().into()` (matching
    /// `cubecl-wgpu`'s `request_adapter`), avoiding both `enumerate_all_devices`'s mixed-GPU
    /// global-index mismatch and `enumerate_devices`'s `Backends::all()` WGL thread-crash hazard
    /// on Windows (`Wave 16D`).
    pub fn enumerate_wgpu_devices() -> Vec<WgpuDeviceDescriptorV1> {
        #[cfg(not(target_family = "wasm"))]
        {
            static WGPU_ENUMERATE_LOCK: Mutex<()> = Mutex::new(());
            let _guard = WGPU_ENUMERATE_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let backend = AutoGraphicsApi::backend();
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: backend.into(),
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let adapters = cubecl::future::block_on(instance.enumerate_adapters(backend.into()));
            let mut discrete = Vec::new();
            let mut integrated = Vec::new();
            let mut virtual_gpu = Vec::new();
            let mut cpu = Vec::new();
            for adapter in adapters {
                match adapter.get_info().device_type {
                    wgpu::DeviceType::DiscreteGpu => {
                        if let Ok(ord) = u16::try_from(discrete.len()) {
                            discrete.push(DeviceId::new(0, ord));
                        }
                    }
                    wgpu::DeviceType::IntegratedGpu => {
                        if let Ok(ord) = u16::try_from(integrated.len()) {
                            integrated.push(DeviceId::new(1, ord));
                        }
                    }
                    wgpu::DeviceType::VirtualGpu => {
                        if let Ok(ord) = u16::try_from(virtual_gpu.len()) {
                            virtual_gpu.push(DeviceId::new(2, ord));
                        }
                    }
                    wgpu::DeviceType::Cpu => {
                        cpu.push(DeviceId::new(3, 0));
                    }
                    wgpu::DeviceType::Other => {}
                }
            }
            build_wgpu_device_inventory_from_type_buckets(
                &discrete,
                &integrated,
                &virtual_gpu,
                &cpu,
            )
        }

        #[cfg(target_family = "wasm")]
        {
            let backend = AutoGraphicsApi::backend();
            let discrete = <WgpuRuntime as Runtime>::enumerate_devices(0, &backend);
            let integrated = <WgpuRuntime as Runtime>::enumerate_devices(1, &backend);
            let virtual_gpu = <WgpuRuntime as Runtime>::enumerate_devices(2, &backend);
            let cpu = <WgpuRuntime as Runtime>::enumerate_devices(3, &backend);
            build_wgpu_device_inventory_from_type_buckets(
                &discrete,
                &integrated,
                &virtual_gpu,
                &cpu,
            )
        }
    }

    /// Probe live WGPU adapters and resolve `selector`, returning
    /// [`BackendBridgeError::DeviceUnavailable`] without panicking if no matching adapter exists.
    pub fn resolve_wgpu_device(
        selector: WgpuDeviceSelectorV1,
    ) -> Result<WgpuDeviceDescriptorV1, BackendBridgeError> {
        let inventory = enumerate_wgpu_devices();
        resolve_wgpu_device_from_inventory(&inventory, selector)
    }

    /// Probe live WGPU adapters for logical `device_index` (`DevicePlacementV1::Wgpu { device_index }`).
    pub fn probe_wgpu_device(
        device_index: u32,
    ) -> Result<WgpuDeviceDescriptorV1, BackendBridgeError> {
        resolve_wgpu_device(WgpuDeviceSelectorV1::LogicalIndex(device_index))
    }

    /// Live `Burn` WGPU GPU affine module (`y = x @ W_t + b`) synchronized transactionally in
    /// canonical `[weight(out_dim * in_dim) | bias(out_dim)]` layout (`ADR-0015`, `Wave 16D`).
    #[derive(Debug)]
    pub struct BurnWgpuLinearBridgeV1 {
        descriptor: WgpuDeviceDescriptorV1,
        core: BurnLinearCore<BurnWgpuInnerBackend>,
    }

    impl BurnWgpuLinearBridgeV1 {
        /// Construct a WGPU linear bridge targeting logical `device_index`, failing closed with
        /// [`BackendBridgeError::DeviceUnavailable`] if no matching WGPU adapter exists.
        pub fn new(
            in_dim: usize,
            out_dim: usize,
            device_index: u32,
        ) -> Result<Self, BackendBridgeError> {
            Self::new_with_selector(
                in_dim,
                out_dim,
                WgpuDeviceSelectorV1::LogicalIndex(device_index),
            )
        }

        /// Construct a WGPU linear bridge targeting an explicit [`WgpuDeviceSelectorV1`].
        pub fn new_with_selector(
            in_dim: usize,
            out_dim: usize,
            selector: WgpuDeviceSelectorV1,
        ) -> Result<Self, BackendBridgeError> {
            let schema = ModelTensorSchemaV1::for_linear(in_dim, out_dim)
                .map_err(|_| BackendBridgeError::SchemaMismatch)?;
            let descriptor = resolve_wgpu_device(selector)?;
            let placement = DevicePlacementV1::Wgpu {
                device_index: descriptor.device_index,
            };
            let core = BurnLinearCore::new_with_schema(
                schema,
                in_dim,
                out_dim,
                descriptor.wgpu_device.clone(),
                placement,
            );
            Ok(Self { descriptor, core })
        }

        /// Borrow the validated WGPU device descriptor bound to this bridge.
        pub const fn device_descriptor(&self) -> &WgpuDeviceDescriptorV1 {
            &self.descriptor
        }

        /// Compute MSE loss `(1/N) * sum((y - target)^2)` and reverse-mode autodiff gradients on
        /// the WGPU device in canonical `[weight_grad | bias_grad]` order.
        pub fn backward_mse(
            &self,
            input: &[f32],
            target: &[f32],
        ) -> Result<(f32, Vec<f32>), BackendBridgeError> {
            self.core.backward_mse(input, target)
        }
    }

    impl BurnModuleOps for BurnWgpuLinearBridgeV1 {
        fn device_placement(&self) -> DevicePlacementV1 {
            self.core.device_placement()
        }

        fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
            self.core.tensor_schema()
        }

        fn export_flat_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
            self.core.export_flat_f32()
        }

        fn import_flat_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
            self.core.import_flat_f32(flat_params)
        }

        fn forward_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
            self.core.forward_flat(input)
        }
    }

    /// Live `Burn` WGPU GPU 2-layer ReLU MLP (`y = fc2(relu(fc1(x)))`) synchronized transactionally
    /// in canonical `[W_1 | b_1 | W_2 | b_2]` layout (`ADR-0015`, `Wave 16D`).
    #[derive(Debug)]
    pub struct BurnWgpuMlpBridgeV1 {
        descriptor: WgpuDeviceDescriptorV1,
        core: BurnMlpCore<BurnWgpuInnerBackend>,
    }

    impl BurnWgpuMlpBridgeV1 {
        /// Construct a WGPU 2-layer ReLU MLP bridge targeting logical `device_index`, failing
        /// closed with [`BackendBridgeError::DeviceUnavailable`] if no matching WGPU adapter exists.
        pub fn new(
            in_dim: usize,
            hidden_dim: usize,
            out_dim: usize,
            device_index: u32,
        ) -> Result<Self, BackendBridgeError> {
            Self::new_with_selector(
                in_dim,
                hidden_dim,
                out_dim,
                WgpuDeviceSelectorV1::LogicalIndex(device_index),
            )
        }

        /// Construct a WGPU 2-layer ReLU MLP bridge targeting an explicit [`WgpuDeviceSelectorV1`].
        pub fn new_with_selector(
            in_dim: usize,
            hidden_dim: usize,
            out_dim: usize,
            selector: WgpuDeviceSelectorV1,
        ) -> Result<Self, BackendBridgeError> {
            let schema = ModelTensorSchemaV1::for_mlp(in_dim, hidden_dim, out_dim)
                .map_err(|_| BackendBridgeError::SchemaMismatch)?;
            let descriptor = resolve_wgpu_device(selector)?;
            let placement = DevicePlacementV1::Wgpu {
                device_index: descriptor.device_index,
            };
            let core = BurnMlpCore::new_with_schema(
                schema,
                in_dim,
                hidden_dim,
                out_dim,
                descriptor.wgpu_device.clone(),
                placement,
            );
            Ok(Self { descriptor, core })
        }

        /// Borrow the validated WGPU device descriptor bound to this bridge.
        pub const fn device_descriptor(&self) -> &WgpuDeviceDescriptorV1 {
            &self.descriptor
        }

        /// Compute MSE loss `(1/N) * sum((y - target)^2)` and reverse-mode autodiff gradients on
        /// the WGPU device in canonical `[dW_1 | db_1 | dW_2 | db_2]` order.
        pub fn backward_mse(
            &self,
            input: &[f32],
            target: &[f32],
        ) -> Result<(f32, Vec<f32>), BackendBridgeError> {
            self.core.backward_mse(input, target)
        }
    }

    impl BurnModuleOps for BurnWgpuMlpBridgeV1 {
        fn device_placement(&self) -> DevicePlacementV1 {
            self.core.device_placement()
        }

        fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
            self.core.tensor_schema()
        }

        fn export_flat_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
            self.core.export_flat_f32()
        }

        fn import_flat_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
            self.core.import_flat_f32(flat_params)
        }

        fn forward_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
            self.core.forward_flat(input)
        }
    }
}

/// Live `Candle 0.10` CPU `VarMap` / `Linear` and 2-layer MLP backend bridges (`ST-08`, `AR-14-07`, `14B-1`).
#[cfg(feature = "real-candle")]
pub mod real_candle {
    use candle_core::{DType, Device, Tensor, Var};
    use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
    use swarm_torch_models::backend_bridge::{BackendBridgeError, DevicePlacementV1};
    use swarm_torch_models::safetensors_v1::ModelTensorSchemaV1;

    use crate::{validate_batch_input, validate_finite_f32_slice, CandleModuleOps};

    /// Reject every condition under which `Var::set` could fail after an earlier `Var` in the
    /// same write was already committed: non-contiguous layout, shape, dtype (`F32`), or device
    /// mismatch (`15D-4`). Call it for all `Var`s before the first `set`. `set`'s remaining
    /// shared-storage check cannot trigger here: every source is a fresh `Tensor::from_slice`.
    pub(crate) fn check_var_assignable(
        var: &Var,
        src: &Tensor,
        device: &Device,
    ) -> Result<(), BackendBridgeError> {
        if !var.is_contiguous()
            || var.shape() != src.shape()
            || var.dtype() != DType::F32
            || src.dtype() != DType::F32
            || !var.device().same_device(device)
            || !src.device().same_device(device)
        {
            return Err(BackendBridgeError::SchemaMismatch);
        }
        Ok(())
    }

    /// Live `Candle` affine module (`y = x @ W^T + b`) synchronized transactionally via [`VarMap`]
    /// in canonical `[weight(out_dim * in_dim) | bias(out_dim)]` layout (`AR-14-07`).
    pub struct CandleLinearBridgeV1 {
        schema: ModelTensorSchemaV1,
        in_dim: usize,
        out_dim: usize,
        device: Device,
        varmap: VarMap,
        module: Linear,
    }

    impl std::fmt::Debug for CandleLinearBridgeV1 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CandleLinearBridgeV1")
                .field("in_dim", &self.in_dim)
                .field("out_dim", &self.out_dim)
                .field("device", &self.device)
                .finish()
        }
    }

    impl CandleLinearBridgeV1 {
        /// Construct a new CPU `VarMap`-backed `candle_nn::Linear` matching `for_linear(in_dim, out_dim)`.
        pub fn new(in_dim: usize, out_dim: usize) -> Result<Self, BackendBridgeError> {
            if in_dim == 0 || out_dim == 0 {
                return Err(BackendBridgeError::SchemaMismatch);
            }
            let schema = ModelTensorSchemaV1::for_linear(in_dim, out_dim)
                .map_err(|_| BackendBridgeError::SchemaMismatch)?;
            let device = Device::Cpu;
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let module = linear(in_dim, out_dim, vb.pp("linear"))
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            Ok(Self {
                schema,
                in_dim,
                out_dim,
                device,
                varmap,
                module,
            })
        }

        /// Compute MSE loss `(1/N) * sum((y - target)^2)` and reverse-mode autodiff gradients
        /// in canonical `[weight_grad(out_dim * in_dim) | bias_grad(out_dim)]` order (`AR-14-07`).
        pub fn backward_mse(
            &self,
            input: &[f32],
            target: &[f32],
        ) -> Result<(f32, Vec<f32>), BackendBridgeError> {
            let batch = validate_batch_input(input, self.in_dim)?;
            validate_finite_f32_slice(target, Some(batch * self.out_dim))?;

            let x = Tensor::from_slice(input, (batch, self.in_dim), &self.device)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let t = Tensor::from_slice(target, (batch, self.out_dim), &self.device)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;

            let pred = self
                .module
                .forward(&x)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let loss = pred
                .sub(&t)
                .and_then(|d| d.sqr())
                .and_then(|s| s.mean_all())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let loss_scalar = loss
                .to_scalar::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            validate_finite_f32_slice(&[loss_scalar], Some(1))?;

            let grads = loss
                .backward()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let bias_tensor = self
                .module
                .bias()
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            let b_grad = grads
                .get(bias_tensor)
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed("missing bias gradient".to_string())
                })?
                .flatten_all()
                .and_then(|g| g.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let w_grad = grads
                .get(self.module.weight())
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed(
                        "missing weight gradient".to_string(),
                    )
                })?
                .flatten_all()
                .and_then(|g| g.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;

            let expected = self.out_dim * self.in_dim + self.out_dim;
            let mut flat_grads = Vec::with_capacity(expected);
            flat_grads.extend_from_slice(&w_grad);
            flat_grads.extend_from_slice(&b_grad);
            validate_finite_f32_slice(&flat_grads, Some(expected))?;
            Ok((loss_scalar, flat_grads))
        }
    }

    impl CandleModuleOps for CandleLinearBridgeV1 {
        fn device_placement(&self) -> DevicePlacementV1 {
            DevicePlacementV1::Cpu
        }

        fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
            &self.schema
        }

        fn read_varmap_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
            let bias_vec = self
                .module
                .bias()
                .ok_or(BackendBridgeError::SchemaMismatch)?
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let weight_vec = self
                .module
                .weight()
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let expected = self.out_dim * self.in_dim + self.out_dim;
            let mut out = Vec::with_capacity(expected);
            out.extend_from_slice(&weight_vec);
            out.extend_from_slice(&bias_vec);
            validate_finite_f32_slice(&out, Some(expected))?;
            Ok(out)
        }

        fn write_varmap_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
            let weight_len = self.out_dim * self.in_dim;
            let expected = weight_len + self.out_dim;
            validate_finite_f32_slice(flat_params, Some(expected))?;

            let weight_t = Tensor::from_slice(
                &flat_params[..weight_len],
                (self.out_dim, self.in_dim),
                &self.device,
            )
            .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let bias_t = Tensor::from_slice(&flat_params[weight_len..], self.out_dim, &self.device)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;

            // Validate every VarMap key and `Var::set` precondition BEFORE committing any write.
            let data = self
                .varmap
                .data()
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let bias_var = data
                .get("linear.bias")
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            let weight_var = data
                .get("linear.weight")
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            check_var_assignable(weight_var, &weight_t, &self.device)?;
            check_var_assignable(bias_var, &bias_t, &self.device)?;
            weight_var
                .set(&weight_t)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            bias_var
                .set(&bias_t)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            Ok(())
        }

        fn eval_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
            let batch = validate_batch_input(input, self.in_dim)?;
            let x = Tensor::from_slice(input, (batch, self.in_dim), &self.device)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let out = self
                .module
                .forward(&x)
                .and_then(|y| y.flatten_all())
                .and_then(|y| y.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            validate_finite_f32_slice(&out, Some(batch * self.out_dim))?;
            Ok(out)
        }
    }

    /// Live `Candle` 2-layer ReLU MLP (`y = fc2(relu(fc1(x)))`) synchronized transactionally via [`VarMap`]
    /// in canonical `[W_1 | b_1 | W_2 | b_2]` layout matching [`swarm_torch_models::mlp::MlpModel`] (`14B-1`).
    pub struct CandleMlpBridgeV1 {
        schema: ModelTensorSchemaV1,
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        device: Device,
        varmap: VarMap,
        fc1: Linear,
        fc2: Linear,
    }

    impl std::fmt::Debug for CandleMlpBridgeV1 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CandleMlpBridgeV1")
                .field("in_dim", &self.in_dim)
                .field("hidden_dim", &self.hidden_dim)
                .field("out_dim", &self.out_dim)
                .field("device", &self.device)
                .finish()
        }
    }

    impl CandleMlpBridgeV1 {
        /// Construct a new CPU `VarMap`-backed 2-layer ReLU MLP matching `for_mlp(in_dim, hidden_dim, out_dim)`.
        pub fn new(
            in_dim: usize,
            hidden_dim: usize,
            out_dim: usize,
        ) -> Result<Self, BackendBridgeError> {
            if in_dim == 0 || hidden_dim == 0 || out_dim == 0 {
                return Err(BackendBridgeError::SchemaMismatch);
            }
            let schema = ModelTensorSchemaV1::for_mlp(in_dim, hidden_dim, out_dim)
                .map_err(|_| BackendBridgeError::SchemaMismatch)?;
            let device = Device::Cpu;
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let fc1 = linear(in_dim, hidden_dim, vb.pp("mlp.fc1"))
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let fc2 = linear(hidden_dim, out_dim, vb.pp("mlp.fc2"))
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            Ok(Self {
                schema,
                in_dim,
                hidden_dim,
                out_dim,
                device,
                varmap,
                fc1,
                fc2,
            })
        }

        fn total_elements(&self) -> usize {
            self.hidden_dim * self.in_dim
                + self.hidden_dim
                + self.out_dim * self.hidden_dim
                + self.out_dim
        }

        /// Compute MSE loss `(1/N) * sum((y - target)^2)` and reverse-mode autodiff gradients
        /// in canonical `[dW_1 | db_1 | dW_2 | db_2]` order matching [`swarm_torch_models::mlp::MlpModel`].
        pub fn backward_mse(
            &self,
            input: &[f32],
            target: &[f32],
        ) -> Result<(f32, Vec<f32>), BackendBridgeError> {
            let batch = validate_batch_input(input, self.in_dim)?;
            validate_finite_f32_slice(target, Some(batch * self.out_dim))?;

            let x = Tensor::from_slice(input, (batch, self.in_dim), &self.device)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let t = Tensor::from_slice(target, (batch, self.out_dim), &self.device)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;

            let h = self
                .fc1
                .forward(&x)
                .and_then(|z| z.relu())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let pred = self
                .fc2
                .forward(&h)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let loss = pred
                .sub(&t)
                .and_then(|d| d.sqr())
                .and_then(|s| s.mean_all())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let loss_scalar = loss
                .to_scalar::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            validate_finite_f32_slice(&[loss_scalar], Some(1))?;

            let grads = loss
                .backward()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;

            let w1_grad = grads
                .get(self.fc1.weight())
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed(
                        "missing fc1.weight grad".to_string(),
                    )
                })?
                .flatten_all()
                .and_then(|g| g.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let b1_grad = grads
                .get(self.fc1.bias().ok_or(BackendBridgeError::SchemaMismatch)?)
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed("missing fc1.bias grad".to_string())
                })?
                .flatten_all()
                .and_then(|g| g.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let w2_grad = grads
                .get(self.fc2.weight())
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed(
                        "missing fc2.weight grad".to_string(),
                    )
                })?
                .flatten_all()
                .and_then(|g| g.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let b2_grad = grads
                .get(self.fc2.bias().ok_or(BackendBridgeError::SchemaMismatch)?)
                .ok_or_else(|| {
                    BackendBridgeError::BackendOperationFailed("missing fc2.bias grad".to_string())
                })?
                .flatten_all()
                .and_then(|g| g.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;

            let expected = self.total_elements();
            let mut flat_grads = Vec::with_capacity(expected);
            flat_grads.extend_from_slice(&w1_grad);
            flat_grads.extend_from_slice(&b1_grad);
            flat_grads.extend_from_slice(&w2_grad);
            flat_grads.extend_from_slice(&b2_grad);
            validate_finite_f32_slice(&flat_grads, Some(expected))?;
            Ok((loss_scalar, flat_grads))
        }
    }

    impl CandleModuleOps for CandleMlpBridgeV1 {
        fn device_placement(&self) -> DevicePlacementV1 {
            DevicePlacementV1::Cpu
        }

        fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
            &self.schema
        }

        fn read_varmap_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
            let w1_vec = self
                .fc1
                .weight()
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let b1_vec = self
                .fc1
                .bias()
                .ok_or(BackendBridgeError::SchemaMismatch)?
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let w2_vec = self
                .fc2
                .weight()
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let b2_vec = self
                .fc2
                .bias()
                .ok_or(BackendBridgeError::SchemaMismatch)?
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;

            let expected = self.total_elements();
            let mut out = Vec::with_capacity(expected);
            out.extend_from_slice(&w1_vec);
            out.extend_from_slice(&b1_vec);
            out.extend_from_slice(&w2_vec);
            out.extend_from_slice(&b2_vec);
            validate_finite_f32_slice(&out, Some(expected))?;
            Ok(out)
        }

        fn write_varmap_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
            let w1_len = self.hidden_dim * self.in_dim;
            let b1_end = w1_len + self.hidden_dim;
            let w2_end = b1_end + self.out_dim * self.hidden_dim;
            let expected = w2_end + self.out_dim;
            validate_finite_f32_slice(flat_params, Some(expected))?;

            let w1_t = Tensor::from_slice(
                &flat_params[..w1_len],
                (self.hidden_dim, self.in_dim),
                &self.device,
            )
            .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let b1_t =
                Tensor::from_slice(&flat_params[w1_len..b1_end], self.hidden_dim, &self.device)
                    .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let w2_t = Tensor::from_slice(
                &flat_params[b1_end..w2_end],
                (self.out_dim, self.hidden_dim),
                &self.device,
            )
            .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let b2_t = Tensor::from_slice(&flat_params[w2_end..], self.out_dim, &self.device)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;

            let data = self
                .varmap
                .data()
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let w1_var = data
                .get("mlp.fc1.weight")
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            let b1_var = data
                .get("mlp.fc1.bias")
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            let w2_var = data
                .get("mlp.fc2.weight")
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            let b2_var = data
                .get("mlp.fc2.bias")
                .ok_or(BackendBridgeError::SchemaMismatch)?;
            check_var_assignable(w1_var, &w1_t, &self.device)?;
            check_var_assignable(b1_var, &b1_t, &self.device)?;
            check_var_assignable(w2_var, &w2_t, &self.device)?;
            check_var_assignable(b2_var, &b2_t, &self.device)?;
            w1_var
                .set(&w1_t)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            b1_var
                .set(&b1_t)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            w2_var
                .set(&w2_t)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            b2_var
                .set(&b2_t)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            Ok(())
        }

        fn eval_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
            let batch = validate_batch_input(input, self.in_dim)?;
            let x = Tensor::from_slice(input, (batch, self.in_dim), &self.device)
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let h = self
                .fc1
                .forward(&x)
                .and_then(|z| z.relu())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let out = self
                .fc2
                .forward(&h)
                .and_then(|y| y.flatten_all())
                .and_then(|y| y.to_vec1::<f32>())
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            validate_finite_f32_slice(&out, Some(batch * self.out_dim))?;
            Ok(out)
        }
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

        // Load identical canonical params [w0, w1, bias] into both bridges (`AR-14-07`)
        let weights = [2.0_f32, -1.0, 0.5];
        burn_handle.load_parameters(&weights).unwrap();
        candle_handle.load_parameters(&weights).unwrap();

        assert_eq!(burn_handle.forward(&[3.0, 1.0]).unwrap(), vec![5.5]);
        assert_eq!(candle_handle.forward(&[3.0, 1.0]).unwrap(), vec![5.5]);

        // NaN parameters are rejected before device write, preserving prior parameters
        assert!(burn_handle.load_parameters(&[f32::NAN, 1.0, 1.0]).is_err());
        assert_eq!(burn_handle.parameters(), &weights);
    }

    #[test]
    fn reference_linear_module_fails_closed_and_batches_like_live_bridges() {
        let mut m = ReferenceLinearModule::new(2, 1, DevicePlacementV1::Cpu).unwrap();
        let weights = [2.0_f32, -1.0, 0.5];
        BurnModuleOps::import_flat_f32(&mut m, &weights).unwrap();

        // Module-level writes reject non-finite values on both trait paths without mutating.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                BurnModuleOps::import_flat_f32(&mut m, &[bad, 1.0, 1.0]).unwrap_err(),
                BackendBridgeError::NonFiniteValue
            );
            assert_eq!(
                CandleModuleOps::write_varmap_f32(&mut m, &[1.0, bad, 1.0]).unwrap_err(),
                BackendBridgeError::NonFiniteValue
            );
            assert_eq!(BurnModuleOps::export_flat_f32(&m).unwrap(), weights);
        }
        assert_eq!(
            CandleModuleOps::write_varmap_f32(&mut m, &[1.0, 2.0]).unwrap_err(),
            BackendBridgeError::ParameterCountMismatch {
                expected: 3,
                got: 2
            }
        );
        assert_eq!(CandleModuleOps::read_varmap_f32(&m).unwrap(), weights);

        // `[batch, in_dim]` input evaluates row by row into `[batch, out_dim]`.
        assert_eq!(BurnModuleOps::forward_flat(&m, &[3.0, 1.0]).unwrap(), [5.5]);
        assert_eq!(
            BurnModuleOps::forward_flat(&m, &[3.0, 1.0, 0.0, 2.0]).unwrap(),
            [5.5, -1.5]
        );
        assert_eq!(
            CandleModuleOps::eval_flat(&m, &[3.0, 1.0, 0.0, 2.0]).unwrap(),
            [5.5, -1.5]
        );

        // Ragged, empty, and non-finite inputs fail closed.
        assert!(matches!(
            BurnModuleOps::forward_flat(&m, &[1.0, 2.0, 3.0]),
            Err(BackendBridgeError::ParameterCountMismatch { .. })
        ));
        assert!(matches!(
            BurnModuleOps::forward_flat(&m, &[]),
            Err(BackendBridgeError::ParameterCountMismatch { .. })
        ));
        assert_eq!(
            BurnModuleOps::forward_flat(&m, &[f32::NAN, 1.0]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );

        // Finite inputs whose result overflows `f32` are rejected instead of returned as `inf`.
        BurnModuleOps::import_flat_f32(&mut m, &[f32::MAX, f32::MAX, 0.0]).unwrap();
        assert_eq!(
            BurnModuleOps::forward_flat(&m, &[f32::MAX, f32::MAX]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
    }

    #[cfg(feature = "real-candle")]
    #[test]
    fn candle_var_preflight_rejects_each_set_failure_mode_before_any_write() {
        use crate::real_candle::check_var_assignable;
        use candle_core::{DType, Device, Tensor, Var};

        let dev = Device::Cpu;
        let var = Var::zeros((2, 3), DType::F32, &dev).unwrap();
        let ok = Tensor::ones((2, 3), DType::F32, &dev).unwrap();
        assert!(check_var_assignable(&var, &ok, &dev).is_ok());

        let f64_src = Tensor::ones((2, 3), DType::F64, &dev).unwrap();
        let f64_var = Var::zeros((2, 3), DType::F64, &dev).unwrap();
        let wrong_shape = Tensor::ones((3, 2), DType::F32, &dev).unwrap();
        for (var, src) in [(&var, &f64_src), (&f64_var, &ok), (&var, &wrong_shape)] {
            assert_eq!(
                check_var_assignable(var, src, &dev).unwrap_err(),
                BackendBridgeError::SchemaMismatch
            );
        }

        // These are genuine `Var::set` failure modes, which is why they must be checked for
        // every `Var` before the first write commits.
        assert!(var.set(&f64_src).is_err());
        assert!(var.set(&wrong_shape).is_err());
        assert!(var.set(&ok).is_ok());
    }

    #[cfg(feature = "real-burn")]
    #[test]
    fn real_burn_linear_forward_autodiff_and_transactional_import() {
        let burn_mod = real_burn::BurnLinearBridgeV1::new(3, 2).unwrap();
        let schema = ModelTensorSchemaV1::for_linear(3, 2).unwrap();
        let mut handle = BurnModuleBridgeV1::new(burn_mod)
            .into_handle(Some(schema.schema_hash()))
            .unwrap();

        // Canonical `[weight(2x3 row-major) | bias(2)]` order matching `SwarmTorch::LinearModel` (`AR-14-07`)
        let canonical_params = [0.5f32, -1.25, 0.75, -0.3, 1.4, -0.8, 0.25, -0.65];
        handle.load_parameters(&canonical_params).unwrap();

        // Reject NaN transactionally without clobbering existing weights
        assert_eq!(
            handle
                .load_parameters(&[f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .unwrap_err(),
            swarm_torch_core::Error::NanDetected
        );
        assert_eq!(handle.parameters(), &canonical_params);

        let y = handle.forward(&[1.2f32, -0.4, 2.5]).unwrap();
        // y0 = 0.5*1.2 + (-1.25)*(-0.4) + 0.75*2.5 + 0.25 = 0.6 + 0.5 + 1.875 + 0.25 = 3.225
        // y1 = (-0.3)*1.2 + 1.4*(-0.4) + (-0.8)*2.5 - 0.65 = -0.36 - 0.56 - 2.0 - 0.65 = -3.57
        assert!((y[0] - 3.225).abs() < 1e-5, "got {}", y[0]);
        assert!((y[1] - (-3.57)).abs() < 1e-5, "got {}", y[1]);

        let (loss, grads) = handle
            .backend()
            .inner()
            .backward_mse(&[1.2, -0.4, 2.5], &[1.0, -1.0])
            .unwrap();
        let expected_loss = ((3.225f32 - 1.0).powi(2) + (-3.57f32 + 1.0).powi(2)) * 0.5;
        assert!((loss - expected_loss).abs() < 1e-5);
        assert_eq!(grads.len(), 8);
        // Bias gradients occupy the trailing 2 slots (`[W_grad(6) | b_grad(2)]`):
        // dL/dy0 = y0 - t0 = 2.225; dL/dy1 = y1 - t1 = -2.57 (since 2/N = 2/2 = 1.0)
        assert!((grads[6] - 2.225).abs() < 1e-5);
        assert!((grads[7] - (-2.57)).abs() < 1e-5);
    }

    #[cfg(feature = "real-candle")]
    #[test]
    fn real_candle_linear_forward_backward_and_transactional_varmap_update() {
        let candle_mod = real_candle::CandleLinearBridgeV1::new(3, 2).unwrap();
        let schema = ModelTensorSchemaV1::for_linear(3, 2).unwrap();
        let mut handle = CandleModuleBridgeV1::new(candle_mod)
            .into_handle(Some(schema.schema_hash()))
            .unwrap();

        let canonical_params = [0.5f32, -1.25, 0.75, -0.3, 1.4, -0.8, 0.25, -0.65];
        handle.load_parameters(&canonical_params).unwrap();

        // Reject Inf transactionally without clobbering VarMap
        assert!(handle
            .load_parameters(&[0.0, f32::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .is_err());
        assert_eq!(handle.parameters(), &canonical_params);

        let y = handle.forward(&[1.2f32, -0.4, 2.5]).unwrap();
        assert!((y[0] - 3.225).abs() < 1e-5);
        assert!((y[1] - (-3.57)).abs() < 1e-5);

        let (loss, grads) = handle
            .backend()
            .inner()
            .backward_mse(&[1.2, -0.4, 2.5], &[1.0, -1.0])
            .unwrap();
        let expected_loss = ((3.225f32 - 1.0).powi(2) + (-3.57f32 + 1.0).powi(2)) * 0.5;
        assert!((loss - expected_loss).abs() < 1e-5);
        assert!((grads[6] - 2.225).abs() < 1e-5);
        assert!((grads[7] - (-2.57)).abs() < 1e-5);
    }

    #[cfg(all(feature = "real-burn", feature = "real-candle"))]
    #[test]
    fn cross_framework_safetensors_wb_roundtrip_and_grad_parity_across_swarmtorch_burn_candle() {
        use swarm_torch_models::loss::MseLoss;
        use swarm_torch_models::safetensors_v1::{
            export_safetensors_checkpoint_with_layout_v1,
            import_safetensors_checkpoint_with_layout_v1, ModelFlatTensorLayoutV1,
            SafeTensorsLimitsV1,
        };
        use swarm_torch_models::simple::LinearModel;

        // 1. Construct SwarmTorch LinearModel(3, 2) in flat [W(6) | b(2)] order
        let weights = [0.5f32, -1.25, 0.75, -0.3, 1.4, -0.8];
        let biases = [0.25f32, -0.65];
        let mut flat_wb = Vec::from(weights);
        flat_wb.extend_from_slice(&biases);

        let mut swarm_model = LinearModel::new(3, 2).unwrap();
        swarm_model.load_parameters(&flat_wb).unwrap();

        let layout = ModelFlatTensorLayoutV1::for_linear(3, 2).unwrap();
        let limits = SafeTensorsLimitsV1::default();
        let st_bytes =
            export_safetensors_checkpoint_with_layout_v1(&swarm_model, &layout, &limits).unwrap();

        // 2. Import SafeTensors bytes into live Burn and Candle bridges via canonical `[W | b]` layout!
        let schema = ModelTensorSchemaV1::for_linear(3, 2).unwrap();
        let expected_hash = schema.schema_hash();

        let mut burn_handle =
            BurnModuleBridgeV1::new(real_burn::BurnLinearBridgeV1::new(3, 2).unwrap())
                .into_handle(Some(expected_hash))
                .unwrap();
        let mut candle_handle =
            CandleModuleBridgeV1::new(real_candle::CandleLinearBridgeV1::new(3, 2).unwrap())
                .into_handle(Some(expected_hash))
                .unwrap();

        import_safetensors_checkpoint_with_layout_v1(&mut burn_handle, &layout, &st_bytes, &limits)
            .unwrap();
        import_safetensors_checkpoint_with_layout_v1(
            &mut candle_handle,
            &layout,
            &st_bytes,
            &limits,
        )
        .unwrap();

        // In-memory parameter buffers match `SwarmTorch::LinearModel` 1:1 with zero permutation (`AR-14-07`)!
        assert_eq!(burn_handle.parameters(), swarm_model.parameters());
        assert_eq!(candle_handle.parameters(), swarm_model.parameters());

        // 3. Verify Burn & Candle SafeTensors exports roundtrip bit-for-bit back into SwarmTorch LinearModel!
        let burn_export =
            export_safetensors_checkpoint_with_layout_v1(&burn_handle, &layout, &limits).unwrap();
        let candle_export =
            export_safetensors_checkpoint_with_layout_v1(&candle_handle, &layout, &limits).unwrap();
        assert_eq!(burn_export, st_bytes);
        assert_eq!(candle_export, st_bytes);

        // 4. Verify forward output parity across SwarmTorch, Burn, and Candle
        let input = [1.2f32, -0.4, 2.5];
        let target = [1.0f32, -1.0];
        let swarm_y = swarm_model.forward(&input).unwrap();
        let burn_y = burn_handle.forward(&input).unwrap();
        let candle_y = candle_handle.forward(&input).unwrap();

        for i in 0..2 {
            assert!((swarm_y.as_slice()[i] - burn_y[i]).abs() < 1e-5);
            assert!((swarm_y.as_slice()[i] - candle_y[i]).abs() < 1e-5);
        }

        // 5. Verify MSE loss + 1:1 gradient parity across SwarmTorch backward_into, Burn autodiff, and Candle backprop
        let mut d_preds = [0.0f32; 2];
        let swarm_loss =
            MseLoss::compute_and_grad(swarm_y.as_slice(), &target, &mut d_preds).unwrap();
        let mut swarm_grads_wb = [0.0f32; 8];
        swarm_model
            .backward_into(&input, &d_preds, &mut swarm_grads_wb)
            .unwrap();

        let (burn_loss, burn_grads_wb) = burn_handle
            .backend()
            .inner()
            .backward_mse(&input, &target)
            .unwrap();
        let (candle_loss, candle_grads_wb) = candle_handle
            .backend()
            .inner()
            .backward_mse(&input, &target)
            .unwrap();

        assert!((swarm_loss - burn_loss).abs() < 1e-5);
        assert!((swarm_loss - candle_loss).abs() < 1e-5);

        // Direct 1:1 index comparison without any manual index permutation (`AR-14-07`)!
        for i in 0..8 {
            assert!(
                (swarm_grads_wb[i] - burn_grads_wb[i]).abs() < 1e-5,
                "burn grad mismatch at index {i}: expected {}, got {}",
                swarm_grads_wb[i],
                burn_grads_wb[i]
            );
            assert!(
                (swarm_grads_wb[i] - candle_grads_wb[i]).abs() < 1e-5,
                "candle grad mismatch at index {i}: expected {}, got {}",
                swarm_grads_wb[i],
                candle_grads_wb[i]
            );
        }
    }

    #[cfg(all(feature = "real-burn", feature = "real-candle"))]
    #[test]
    fn mlp_cross_framework_parity_and_federated_trimmed_mean_dp_round() {
        use swarm_torch_core::aggregation::{RobustAggregator, TrimmedMean};
        use swarm_torch_core::crypto::PrivacyAccountantHook;
        use swarm_torch_core::dp::{
            DpAdjacencyV1, DpConversionFormulaV1, DpMechanismProfileV1, DpUnitV1,
            PrivacyAccountingClassificationV1, RdpAccountantV1,
        };
        use swarm_torch_core::traits::{GradientUpdate, SwarmOptimizer};
        use swarm_torch_models::loss::MseLoss;
        use swarm_torch_models::mlp::MlpModel;
        use swarm_torch_models::optim::SgdOptimizer;
        use swarm_torch_models::safetensors_v1::{
            export_safetensors_checkpoint_with_layout_v1,
            import_safetensors_checkpoint_with_layout_v1, ModelFlatTensorLayoutV1,
            SafeTensorsLimitsV1,
        };

        // 1. Initialize 2-layer ReLU MLP (3 -> 4 -> 2; 26 parameters in [W1(12) | b1(4) | W2(8) | b2(2)])
        let mut global_mlp = MlpModel::new(3, 4, 2).unwrap().with_random_init(0xC0FFEE);
        let layout = ModelFlatTensorLayoutV1::for_mlp(3, 4, 2).unwrap();
        let schema = ModelTensorSchemaV1::for_mlp(3, 4, 2).unwrap();
        let limits = SafeTensorsLimitsV1::default();

        let checkpoint_r0 =
            export_safetensors_checkpoint_with_layout_v1(&global_mlp, &layout, &limits).unwrap();

        let mut burn_peer =
            BurnModuleBridgeV1::new(real_burn::BurnMlpBridgeV1::new(3, 4, 2).unwrap())
                .into_handle(Some(schema.schema_hash()))
                .unwrap();
        let mut candle_peer =
            CandleModuleBridgeV1::new(real_candle::CandleMlpBridgeV1::new(3, 4, 2).unwrap())
                .into_handle(Some(schema.schema_hash()))
                .unwrap();

        import_safetensors_checkpoint_with_layout_v1(
            &mut burn_peer,
            &layout,
            &checkpoint_r0,
            &limits,
        )
        .unwrap();
        import_safetensors_checkpoint_with_layout_v1(
            &mut candle_peer,
            &layout,
            &checkpoint_r0,
            &limits,
        )
        .unwrap();

        assert_eq!(burn_peer.parameters(), global_mlp.parameters());
        assert_eq!(candle_peer.parameters(), global_mlp.parameters());

        // 2. Verify forward + backward_mse numerical parity on identical sample across all 3 runtimes
        let x1 = [0.8f32, -0.5, 1.1];
        let y1 = [0.5f32, -0.25];
        let sw_pred = global_mlp.forward(&x1).unwrap();
        let burn_pred = burn_peer.forward(&x1).unwrap();
        let candle_pred = candle_peer.forward(&x1).unwrap();
        for i in 0..2 {
            assert!((sw_pred.as_slice()[i] - burn_pred[i]).abs() < 1e-5);
            assert!((sw_pred.as_slice()[i] - candle_pred[i]).abs() < 1e-5);
        }

        let mut d_out = [0.0f32; 2];
        let sw_loss = MseLoss::compute_and_grad(sw_pred.as_slice(), &y1, &mut d_out).unwrap();
        let mut sw_grads = [0.0f32; 26];
        global_mlp
            .backward_into(&x1, &d_out, &mut sw_grads)
            .unwrap();

        let (burn_loss, burn_grads) = burn_peer.backend().inner().backward_mse(&x1, &y1).unwrap();
        let (candle_loss, candle_grads) = candle_peer
            .backend()
            .inner()
            .backward_mse(&x1, &y1)
            .unwrap();
        assert!((sw_loss - burn_loss).abs() < 1e-5);
        assert!((sw_loss - candle_loss).abs() < 1e-5);
        for i in 0..26 {
            assert!((sw_grads[i] - burn_grads[i]).abs() < 1e-5, "idx {i}");
            assert!((sw_grads[i] - candle_grads[i]).abs() < 1e-5, "idx {i}");
        }

        // 3. Multi-peer federated round: Peer 1 (Burn), Peer 2 (Candle), Peer 3 (SwarmTorch native)
        //    plus a Byzantine outlier peer aggregated via TrimmedMean + RDP privacy accounting!
        let mut opt = SgdOptimizer::<MlpModel, 64>::with_hyperparams(0.15, 0.0, 0.0).unwrap();
        let dp_profile = DpMechanismProfileV1 {
            clipping_radius_tau: 2.0,
            noise_multiplier: 1.2,
            target_delta: 1e-5,
            max_epsilon: 10.0,
            unit: DpUnitV1::ClientLevel,
            adjacency: DpAdjacencyV1::AddRemoveOne,
            conversion_formula: DpConversionFormulaV1::Balle2020Tight,
            classification: PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
        };
        let mut dp_accountant = RdpAccountantV1::new(dp_profile).unwrap();
        let aggregator = TrimmedMean::try_new(0.25).unwrap();

        let updates = vec![
            GradientUpdate {
                sender: [1u8; 32],
                gradients: burn_grads,
                sequence: 1,
                round_id: 1,
            },
            GradientUpdate {
                sender: [2u8; 32],
                gradients: candle_grads,
                sequence: 1,
                round_id: 1,
            },
            GradientUpdate {
                sender: [3u8; 32],
                gradients: sw_grads.to_vec(),
                sequence: 1,
                round_id: 1,
            },
            GradientUpdate {
                sender: [4u8; 32],
                gradients: vec![50.0f32; 26], // Byzantine outlier trimmed!
                sequence: 1,
                round_id: 1,
            },
        ];
        let agg = aggregator.aggregate(&updates).unwrap();
        dp_accountant.record_clipped_round(2.0, 1.0).unwrap();
        assert_eq!(dp_accountant.committed_rounds(), 1);

        opt.step(&mut global_mlp, &agg).unwrap();

        // Export updated global checkpoint and reload into both Burn and Candle peers:
        let checkpoint_r1 =
            export_safetensors_checkpoint_with_layout_v1(&global_mlp, &layout, &limits).unwrap();
        import_safetensors_checkpoint_with_layout_v1(
            &mut burn_peer,
            &layout,
            &checkpoint_r1,
            &limits,
        )
        .unwrap();
        import_safetensors_checkpoint_with_layout_v1(
            &mut candle_peer,
            &layout,
            &checkpoint_r1,
            &limits,
        )
        .unwrap();

        let (burn_loss_after, _) = burn_peer.backend().inner().backward_mse(&x1, &y1).unwrap();
        let (candle_loss_after, _) = candle_peer
            .backend()
            .inner()
            .backward_mse(&x1, &y1)
            .unwrap();
        assert!(
            burn_loss_after < burn_loss,
            "expected MSE loss decrease after federated MLP step: before={burn_loss}, after={burn_loss_after}"
        );
        assert!((burn_loss_after - candle_loss_after).abs() < 1e-5);
    }

    /// `15D-4` T-Agent: the reference module must agree with both live bridges on every import
    /// and forward boundary (same `Ok` values within tolerance, or the same error variant).
    #[cfg(all(feature = "real-burn", feature = "real-candle"))]
    #[test]
    fn reference_linear_module_matches_live_bridges_on_import_and_forward_boundaries() {
        type Outcome = Result<Vec<f32>, BackendBridgeError>;
        // (case, in_dim, out_dim, canonical `[W | b]` params, input)
        type Case = (&'static str, usize, usize, Vec<f32>, Vec<f32>);
        fn agree(case: &str, r: &Outcome, b: &Outcome, c: &Outcome) {
            match (r, b, c) {
                (Ok(r), Ok(b), Ok(c)) => {
                    assert_eq!(
                        (r.len(), b.len()),
                        (c.len(), c.len()),
                        "{case}: output length"
                    );
                    for i in 0..r.len() {
                        assert!(
                            (r[i] - b[i]).abs() <= 1e-5 && (r[i] - c[i]).abs() <= 1e-5,
                            "{case}: value {i} reference={} burn={} candle={}",
                            r[i],
                            b[i],
                            c[i]
                        );
                    }
                }
                (Err(r), Err(b), Err(c)) => {
                    assert!(
                        r == b && r == c,
                        "{case}: ref={r:?} burn={b:?} candle={c:?}"
                    )
                }
                _ => panic!("{case}: reference={r:?} burn={b:?} candle={c:?}"),
            }
        }

        let max = f32::MAX;
        let cases: Vec<Case> = vec![
            (
                "batched [2, 3] -> [2, 2]",
                3,
                2,
                vec![0.5, -1.25, 0.75, -0.3, 1.4, -0.8, 0.25, -0.65],
                vec![1.2, -0.4, 2.5, -0.7, 0.3, 1.1],
            ),
            ("ragged", 2, 1, vec![2.0, -1.0, 0.5], vec![1.0, 2.0, 3.0]),
            ("empty", 2, 1, vec![2.0, -1.0, 0.5], vec![]),
            ("nan input", 2, 1, vec![2.0, -1.0, 0.5], vec![f32::NAN, 1.0]),
            (
                "inf input",
                2,
                1,
                vec![2.0, -1.0, 0.5],
                vec![1.0, f32::INFINITY],
            ),
            (
                "final result overflows f32",
                2,
                1,
                vec![max, max, 0.0],
                vec![max, max],
            ),
            // `x * W` overflows to `inf` in f32 before the bias brings it back into range.
            ("f32 product overflow", 1, 1, vec![max, -max], vec![2.0]),
            // `inf + -inf = NaN` in f32 although the exact dot product is 0.
            (
                "f32 partial-sum overflow",
                2,
                1,
                vec![max, max, 0.0],
                vec![2.0, -2.0],
            ),
        ];

        for (case, in_dim, out_dim, params, input) in cases {
            let mut reference = ReferenceLinearModule::new(in_dim, out_dim, DevicePlacementV1::Cpu)
                .expect("reference module");
            let mut burn = real_burn::BurnLinearBridgeV1::new(in_dim, out_dim).unwrap();
            let mut candle = real_candle::CandleLinearBridgeV1::new(in_dim, out_dim).unwrap();
            BurnModuleOps::import_flat_f32(&mut reference, &params).unwrap();
            BurnModuleOps::import_flat_f32(&mut burn, &params).unwrap();
            CandleModuleOps::write_varmap_f32(&mut candle, &params).unwrap();

            // Rejected imports fail with the same variant on every path and leave params intact.
            let mut nan_params = params.clone();
            nan_params[0] = f32::NAN;
            let mut short_params = params.clone();
            short_params.pop();
            for bad in [&nan_params, &short_params] {
                let r = BurnModuleOps::import_flat_f32(&mut reference, bad).unwrap_err();
                let rc = CandleModuleOps::write_varmap_f32(&mut reference, bad).unwrap_err();
                let b = BurnModuleOps::import_flat_f32(&mut burn, bad).unwrap_err();
                let c = CandleModuleOps::write_varmap_f32(&mut candle, bad).unwrap_err();
                assert!(
                    r == rc && r == b && r == c,
                    "{case}: import {r:?} {rc:?} {b:?} {c:?}"
                );
            }
            assert_eq!(BurnModuleOps::export_flat_f32(&reference).unwrap(), params);
            assert_eq!(BurnModuleOps::export_flat_f32(&burn).unwrap(), params);
            assert_eq!(CandleModuleOps::read_varmap_f32(&candle).unwrap(), params);

            agree(
                case,
                &BurnModuleOps::forward_flat(&reference, &input),
                &BurnModuleOps::forward_flat(&burn, &input),
                &CandleModuleOps::eval_flat(&candle, &input),
            );
            agree(
                case,
                &CandleModuleOps::eval_flat(&reference, &input),
                &BurnModuleOps::forward_flat(&burn, &input),
                &CandleModuleOps::eval_flat(&candle, &input),
            );
        }
    }

    /// `15D-4` T-Agent: a freshly constructed Burn bridge (lazily initialized `Param`s, no import
    /// or export yet) must evaluate one stable parameter set: the one `export_flat_f32` reports.
    #[cfg(feature = "real-burn")]
    #[test]
    fn real_burn_fresh_bridges_evaluate_the_parameters_they_export() {
        fn close(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-5)
        }
        let x = [1.2f32, -0.4, 2.5];
        let t = [0.0f32, 0.0];

        let lin = real_burn::BurnLinearBridgeV1::new(3, 2).unwrap();
        let y1 = BurnModuleOps::forward_flat(&lin, &x).unwrap();
        let y2 = BurnModuleOps::forward_flat(&lin, &x).unwrap();
        let (l1, g1) = lin.backward_mse(&x, &t).unwrap();
        let (l2, g2) = lin.backward_mse(&x, &t).unwrap();
        assert_eq!(y1, y2, "linear forward drew a new parameter set");
        assert_eq!(
            (l1, &g1),
            (l2, &g2),
            "linear backward drew a new parameter set"
        );
        let params = BurnModuleOps::export_flat_f32(&lin).unwrap();
        let mut reference = ReferenceLinearModule::new(3, 2, DevicePlacementV1::Cpu).unwrap();
        BurnModuleOps::import_flat_f32(&mut reference, &params).unwrap();
        let y_ref = BurnModuleOps::forward_flat(&reference, &x).unwrap();
        assert!(close(&y1, &y_ref), "forward {y1:?} vs exported {y_ref:?}");
        // MSE against a zero target: bias gradient is `y` itself (2/N with N = 2).
        assert!(
            close(&g1[6..], &y_ref),
            "bias grad {:?} vs {y_ref:?}",
            &g1[6..]
        );
        assert_eq!(BurnModuleOps::export_flat_f32(&lin).unwrap(), params);

        let mlp = real_burn::BurnMlpBridgeV1::new(3, 4, 2).unwrap();
        let m1 = BurnModuleOps::forward_flat(&mlp, &x).unwrap();
        let m2 = BurnModuleOps::forward_flat(&mlp, &x).unwrap();
        let (ml1, mg1) = mlp.backward_mse(&x, &t).unwrap();
        let (ml2, mg2) = mlp.backward_mse(&x, &t).unwrap();
        assert_eq!(m1, m2, "mlp forward drew a new parameter set");
        assert_eq!(
            (ml1, &mg1),
            (ml2, &mg2),
            "mlp backward drew a new parameter set"
        );
        let mlp_params = BurnModuleOps::export_flat_f32(&mlp).unwrap();
        let mut native = swarm_torch_models::mlp::MlpModel::new(3, 4, 2).unwrap();
        native.load_parameters(&mlp_params).unwrap();
        let m_ref = native.forward(&x).unwrap();
        assert!(
            close(&m1, m_ref.as_slice()),
            "mlp forward {m1:?} vs exported {m_ref:?}"
        );
    }

    /// `15D-4` T-Agent: `forward_flat` / `backward_mse` now snapshot the module under the mutex and
    /// run the autodiff graph outside it, so concurrent readers share `Param` autodiff nodes.
    /// Readers racing each other and a writer alternating two parameter sets must only ever
    /// observe one complete set, with forward, loss, and gradients all from that same set, and
    /// must never deadlock (a watchdog turns a hang into a failure instead of stalling CI).
    #[cfg(feature = "real-burn")]
    #[test]
    fn real_burn_concurrent_forward_backward_and_import_observe_whole_parameter_sets() {
        use std::sync::mpsc::{channel, RecvTimeoutError};
        use std::time::Duration;

        let (done_tx, done_rx) = channel();
        let worker = std::thread::spawn(move || {
            race_whole_parameter_sets();
            let _ = done_tx.send(());
        });
        match done_rx.recv_timeout(Duration::from_secs(60)) {
            Ok(()) => worker.join().unwrap(),
            Err(RecvTimeoutError::Disconnected) => {
                std::panic::resume_unwind(worker.join().unwrap_err())
            }
            Err(RecvTimeoutError::Timeout) => panic!(
                "concurrent forward_flat/backward_mse/import did not finish within 60 s \
                 (deadlock in burn-autodiff graph locking)"
            ),
        }
    }

    #[cfg(feature = "real-burn")]
    fn race_whole_parameter_sets() {
        use std::sync::RwLock;

        type Snapshot = (Vec<f32>, f32, Vec<f32>);
        fn snapshot<M: BurnModuleOps>(
            m: &M,
            backward: impl Fn(&M) -> (f32, Vec<f32>),
            x: &[f32],
        ) -> Snapshot {
            let y = m.forward_flat(x).unwrap();
            let (loss, grads) = backward(m);
            (y, loss, grads)
        }
        fn race<M: BurnModuleOps>(
            mut bridge: M,
            backward: impl Fn(&M) -> (f32, Vec<f32>) + Sync,
            params_a: &[f32],
            params_b: &[f32],
            x: &[f32],
        ) {
            const READERS: usize = 4;
            const ROUNDS: usize = 150;
            bridge.import_flat_f32(params_b).unwrap();
            let expect_b = snapshot(&bridge, &backward, x);
            bridge.import_flat_f32(params_a).unwrap();
            let expect_a = snapshot(&bridge, &backward, x);
            assert_ne!(expect_a, expect_b);

            let shared = RwLock::new(bridge);
            std::thread::scope(|s| {
                for _ in 0..READERS {
                    s.spawn(|| {
                        for _ in 0..ROUNDS {
                            let got = snapshot(&*shared.read().unwrap(), &backward, x);
                            assert!(
                                got == expect_a || got == expect_b,
                                "torn or foreign snapshot: {got:?}"
                            );
                            std::thread::yield_now();
                        }
                    });
                }
                s.spawn(|| {
                    for round in 0..ROUNDS {
                        let next = if round % 2 == 0 { params_b } else { params_a };
                        shared.write().unwrap().import_flat_f32(next).unwrap();
                        let exported = shared.read().unwrap().export_flat_f32().unwrap();
                        assert_eq!(exported, next);
                        std::thread::yield_now();
                    }
                });
            });
        }

        let x = [1.2f32, -0.4, 2.5];
        let lin_a = [0.5f32, -1.25, 0.75, -0.3, 1.4, -0.8, 0.25, -0.65];
        let lin_b: Vec<f32> = lin_a.iter().map(|v| -2.0 * v + 0.125).collect();
        race(
            real_burn::BurnLinearBridgeV1::new(3, 2).unwrap(),
            |m: &real_burn::BurnLinearBridgeV1| m.backward_mse(&x, &[1.0, -1.0]).unwrap(),
            &lin_a,
            &lin_b,
            &x,
        );

        let mlp_a: Vec<f32> = (0..26).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
        let mlp_b: Vec<f32> = (0..26).map(|i| ((i as f32) * 0.91).cos() * 0.4).collect();
        race(
            real_burn::BurnMlpBridgeV1::new(3, 4, 2).unwrap(),
            |m: &real_burn::BurnMlpBridgeV1| m.backward_mse(&x, &[0.5, -0.25]).unwrap(),
            &mlp_a,
            &mlp_b,
            &x,
        );
    }

    #[cfg(feature = "real-burn-wgpu")]
    #[test]
    fn wave16d_wgpu_device_topology_mapping_avoids_global_index_panic_and_rejects_unavailable() {
        use cubecl::device::DeviceId;
        use real_burn_wgpu::{
            build_wgpu_device_inventory_from_type_buckets, resolve_wgpu_device_from_inventory,
            WgpuDevice, WgpuDeviceKindV1, WgpuDeviceSelectorV1,
        };

        // 1. Mixed GPU host: 1 DiscreteGpu(0) + 1 IntegratedGpu(0).
        //    LogicalIndex(1) MUST resolve to IntegratedGpu(0) (per-type ordinal 0, NOT global index 1!).
        let mixed = build_wgpu_device_inventory_from_type_buckets(
            &[DeviceId::new(0, 0)],
            &[DeviceId::new(1, 0)],
            &[],
            &[],
        );
        assert_eq!(mixed.len(), 2);
        let d0 = resolve_wgpu_device_from_inventory(&mixed, WgpuDeviceSelectorV1::LogicalIndex(0))
            .unwrap();
        assert_eq!(d0.device_index, 0);
        assert_eq!(d0.kind, WgpuDeviceKindV1::DiscreteGpu);
        assert_eq!(d0.type_ordinal, 0);
        assert_eq!(d0.wgpu_device, WgpuDevice::DiscreteGpu(0));

        let d1 = resolve_wgpu_device_from_inventory(&mixed, WgpuDeviceSelectorV1::LogicalIndex(1))
            .unwrap();
        assert_eq!(d1.device_index, 1);
        assert_eq!(d1.kind, WgpuDeviceKindV1::IntegratedGpu);
        assert_eq!(d1.type_ordinal, 0);
        assert_eq!(d1.wgpu_device, WgpuDevice::IntegratedGpu(0));

        assert_eq!(
            resolve_wgpu_device_from_inventory(&mixed, WgpuDeviceSelectorV1::IntegratedGpu(0))
                .unwrap(),
            d1
        );
        assert_eq!(
            resolve_wgpu_device_from_inventory(&mixed, WgpuDeviceSelectorV1::IntegratedGpu(1))
                .unwrap_err(),
            BackendBridgeError::DeviceUnavailable
        );
        assert_eq!(
            resolve_wgpu_device_from_inventory(&mixed, WgpuDeviceSelectorV1::LogicalIndex(2))
                .unwrap_err(),
            BackendBridgeError::DeviceUnavailable
        );

        // 2. Integrated-only host: LogicalIndex(0) -> IntegratedGpu(0); DiscreteGpu(0) -> DeviceUnavailable.
        let igpu_only =
            build_wgpu_device_inventory_from_type_buckets(&[], &[DeviceId::new(1, 0)], &[], &[]);
        assert_eq!(igpu_only.len(), 1);
        let ig0 =
            resolve_wgpu_device_from_inventory(&igpu_only, WgpuDeviceSelectorV1::LogicalIndex(0))
                .unwrap();
        assert_eq!(ig0.kind, WgpuDeviceKindV1::IntegratedGpu);
        assert_eq!(ig0.wgpu_device, WgpuDevice::IntegratedGpu(0));
        assert_eq!(
            resolve_wgpu_device_from_inventory(&igpu_only, WgpuDeviceSelectorV1::DiscreteGpu(0))
                .unwrap_err(),
            BackendBridgeError::DeviceUnavailable
        );

        // 3. Multi-discrete (unsorted + dup) + virtual + duplicate CPU + foreign type_id ignored.
        let complex = build_wgpu_device_inventory_from_type_buckets(
            &[
                DeviceId::new(0, 1),
                DeviceId::new(0, 0),
                DeviceId::new(0, 1),
                DeviceId::new(9, 0),
            ],
            &[],
            &[DeviceId::new(2, 0)],
            &[DeviceId::new(3, 0), DeviceId::new(3, 1)],
        );
        assert_eq!(complex.len(), 4);
        assert_eq!(complex[0].wgpu_device, WgpuDevice::DiscreteGpu(0));
        assert_eq!(complex[1].wgpu_device, WgpuDevice::DiscreteGpu(1));
        assert_eq!(complex[2].wgpu_device, WgpuDevice::VirtualGpu(0));
        assert_eq!(complex[3].wgpu_device, WgpuDevice::Cpu);
        assert_eq!(complex[3].device_index, 3);
        assert_eq!(
            resolve_wgpu_device_from_inventory(&complex, WgpuDeviceSelectorV1::Cpu).unwrap(),
            complex[3]
        );
        assert_eq!(
            resolve_wgpu_device_from_inventory(&complex, WgpuDeviceSelectorV1::LogicalIndex(4))
                .unwrap_err(),
            BackendBridgeError::DeviceUnavailable
        );

        // 4. Empty inventory -> every selector fails closed with DeviceUnavailable.
        let empty = build_wgpu_device_inventory_from_type_buckets(&[], &[], &[], &[]);
        assert!(empty.is_empty());
        for sel in [
            WgpuDeviceSelectorV1::LogicalIndex(0),
            WgpuDeviceSelectorV1::DiscreteGpu(0),
            WgpuDeviceSelectorV1::IntegratedGpu(0),
            WgpuDeviceSelectorV1::VirtualGpu(0),
            WgpuDeviceSelectorV1::Cpu,
        ] {
            assert_eq!(
                resolve_wgpu_device_from_inventory(&empty, sel).unwrap_err(),
                BackendBridgeError::DeviceUnavailable
            );
        }
    }

    #[cfg(feature = "real-burn-wgpu")]
    #[test]
    fn wave16d_real_burn_wgpu_linear_and_mlp_match_cpu_forward_backward_and_safetensors() {
        use real_burn_wgpu::{
            enumerate_wgpu_devices, probe_wgpu_device, BurnWgpuLinearBridgeV1, BurnWgpuMlpBridgeV1,
        };
        use swarm_torch_core::traits::SwarmModel;
        use swarm_torch_models::safetensors_v1::{
            export_safetensors_checkpoint_with_layout_v1,
            import_safetensors_checkpoint_with_layout_v1, ModelFlatTensorLayoutV1,
            SafeTensorsLimitsV1,
        };

        // Out-of-range device index and zero-dimension schemas MUST ALWAYS fail closed without panic.
        assert_eq!(
            probe_wgpu_device(u32::MAX).unwrap_err(),
            BackendBridgeError::DeviceUnavailable
        );
        assert_eq!(
            BurnWgpuLinearBridgeV1::new(0, 2, 0).unwrap_err(),
            BackendBridgeError::SchemaMismatch
        );
        assert_eq!(
            BurnWgpuMlpBridgeV1::new(2, 0, 1, 0).unwrap_err(),
            BackendBridgeError::SchemaMismatch
        );
        assert_eq!(
            BurnWgpuLinearBridgeV1::new(2, 1, u32::MAX).unwrap_err(),
            BackendBridgeError::DeviceUnavailable
        );

        let devices = enumerate_wgpu_devices();
        if devices.is_empty() {
            // Headless / no-Vulkan host: constructor must fail closed cleanly with DeviceUnavailable.
            assert_eq!(
                BurnWgpuLinearBridgeV1::new(2, 1, 0).unwrap_err(),
                BackendBridgeError::DeviceUnavailable
            );
            assert_eq!(
                BurnWgpuMlpBridgeV1::new(2, 3, 1, 0).unwrap_err(),
                BackendBridgeError::DeviceUnavailable
            );
            return;
        }

        let desc = probe_wgpu_device(0).unwrap();
        assert_eq!(desc, devices[0]);

        // 1. Linear bridge parity: WGPU GPU vs CPU Burn & SafeTensors roundtrip
        let lin_params = [0.5f32, -1.25, 0.75, -0.3, 1.4, -0.8, 0.25, -0.65];
        let x_batch = [1.2f32, -0.4, 2.5, -0.7, 1.1, 0.3];
        let t_batch = [0.8f32, -0.5, -1.2, 0.9];

        let mut cpu_lin = real_burn::BurnLinearBridgeV1::new(3, 2).unwrap();
        cpu_lin.import_flat_f32(&lin_params).unwrap();
        let cpu_lin_y = cpu_lin.forward_flat(&x_batch).unwrap();
        let (cpu_lin_loss, cpu_lin_grads) = cpu_lin.backward_mse(&x_batch, &t_batch).unwrap();

        let mut gpu_lin = BurnWgpuLinearBridgeV1::new(3, 2, desc.device_index).unwrap();
        assert_eq!(
            gpu_lin.device_placement(),
            DevicePlacementV1::Wgpu {
                device_index: desc.device_index
            }
        );
        assert_eq!(gpu_lin.device_descriptor(), &desc);
        gpu_lin.import_flat_f32(&lin_params).unwrap();
        assert_eq!(gpu_lin.export_flat_f32().unwrap(), lin_params);

        // Transactional NaN/Inf rejection preserves prior GPU parameters
        let mut bad_lin = lin_params;
        bad_lin[3] = f32::NAN;
        assert_eq!(
            gpu_lin.import_flat_f32(&bad_lin).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
        assert_eq!(gpu_lin.export_flat_f32().unwrap(), lin_params);

        let gpu_lin_y = gpu_lin.forward_flat(&x_batch).unwrap();
        let (gpu_lin_loss, gpu_lin_grads) = gpu_lin.backward_mse(&x_batch, &t_batch).unwrap();
        for (g, c) in gpu_lin_y.iter().zip(cpu_lin_y.iter()) {
            assert!(
                (g - c).abs() < 1e-4,
                "linear forward mismatch: gpu={g} cpu={c}"
            );
        }
        assert!(
            (gpu_lin_loss - cpu_lin_loss).abs() < 1e-4,
            "linear loss mismatch: gpu={gpu_lin_loss} cpu={cpu_lin_loss}"
        );
        for (g, c) in gpu_lin_grads.iter().zip(cpu_lin_grads.iter()) {
            assert!(
                (g - c).abs() < 1e-4,
                "linear grad mismatch: gpu={g} cpu={c}"
            );
        }

        // SafeTensors v1 export/import roundtrip through BackendModelHandleV1
        let expected_hash = cpu_lin.tensor_schema().schema_hash();
        let linear_layout = ModelFlatTensorLayoutV1::for_linear(3, 2).unwrap();
        let limits = SafeTensorsLimitsV1::default();
        let gpu_handle = BurnModuleBridgeV1::new(gpu_lin)
            .into_handle(Some(expected_hash))
            .unwrap();
        let st_bytes =
            export_safetensors_checkpoint_with_layout_v1(&gpu_handle, &linear_layout, &limits)
                .unwrap();
        let mut gpu_lin_reload = BurnWgpuLinearBridgeV1::new(3, 2, desc.device_index).unwrap();
        gpu_lin_reload.import_flat_f32(&[0.0; 8]).unwrap();
        let mut reload_handle = BurnModuleBridgeV1::new(gpu_lin_reload)
            .into_handle(Some(expected_hash))
            .unwrap();
        import_safetensors_checkpoint_with_layout_v1(
            &mut reload_handle,
            &linear_layout,
            &st_bytes,
            &limits,
        )
        .unwrap();
        assert_eq!(reload_handle.parameters(), gpu_handle.parameters());
        assert_eq!(
            reload_handle.backend().placement(),
            DevicePlacementV1::Wgpu {
                device_index: desc.device_index
            }
        );

        // 2. 2-layer ReLU MLP parity: WGPU GPU vs CPU Burn + concurrent multi-threaded serialization
        let mlp_params: Vec<f32> = (0..26).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
        let mut cpu_mlp = real_burn::BurnMlpBridgeV1::new(3, 4, 2).unwrap();
        cpu_mlp.import_flat_f32(&mlp_params).unwrap();
        let cpu_mlp_y = cpu_mlp.forward_flat(&x_batch).unwrap();
        let (cpu_mlp_loss, cpu_mlp_grads) = cpu_mlp.backward_mse(&x_batch, &t_batch).unwrap();

        let mut gpu_mlp = BurnWgpuMlpBridgeV1::new(3, 4, 2, desc.device_index).unwrap();
        gpu_mlp.import_flat_f32(&mlp_params).unwrap();
        let gpu_mlp_y = gpu_mlp.forward_flat(&x_batch).unwrap();
        let (gpu_mlp_loss, gpu_mlp_grads) = gpu_mlp.backward_mse(&x_batch, &t_batch).unwrap();
        for (g, c) in gpu_mlp_y.iter().zip(cpu_mlp_y.iter()) {
            assert!(
                (g - c).abs() < 1e-4,
                "mlp forward mismatch: gpu={g} cpu={c}"
            );
        }
        assert!(
            (gpu_mlp_loss - cpu_mlp_loss).abs() < 1e-4,
            "mlp loss mismatch: gpu={gpu_mlp_loss} cpu={cpu_mlp_loss}"
        );
        for (g, c) in gpu_mlp_grads.iter().zip(cpu_mlp_grads.iter()) {
            assert!((g - c).abs() < 1e-4, "mlp grad mismatch: gpu={g} cpu={c}");
        }

        // Concurrent readers on WGPU MLP must serialize safely without burn-autodiff deadlock
        let mlp_ref = &gpu_mlp;
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| {
                    for _ in 0..8 {
                        let y = mlp_ref.forward_flat(&x_batch).unwrap();
                        let (l, g) = mlp_ref.backward_mse(&x_batch, &t_batch).unwrap();
                        assert_eq!(y, gpu_mlp_y);
                        assert_eq!(l, gpu_mlp_loss);
                        assert_eq!(g, gpu_mlp_grads);
                    }
                });
            }
        });
    }

    #[cfg(feature = "real-burn-wgpu")]
    #[test]
    fn wave16d_t_agent_wgpu_filtered_inventory_matches_logical_device_index() {
        use cubecl::device::DeviceId;
        use real_burn_wgpu::{
            build_wgpu_device_inventory_from_type_buckets, resolve_wgpu_device_from_inventory,
            WgpuDevice, WgpuDeviceKindV1, WgpuDeviceSelectorV1,
        };

        // Foreign type_ids (including DeviceType::Other = 4) are ignored in every bucket,
        // and non-zero Cpu index_ids normalize to type_ordinal = 0 with a single Cpu entry.
        let filtered_buckets = build_wgpu_device_inventory_from_type_buckets(
            &[DeviceId::new(4, 0), DeviceId::new(1, 0)],
            &[
                DeviceId::new(0, 0),
                DeviceId::new(1, 2),
                DeviceId::new(1, 0),
            ],
            &[
                DeviceId::new(4, 0),
                DeviceId::new(2, 1),
                DeviceId::new(2, 0),
            ],
            &[
                DeviceId::new(0, 0),
                DeviceId::new(3, 7),
                DeviceId::new(3, 2),
            ],
        );
        assert_eq!(filtered_buckets.len(), 5);
        assert_eq!(
            filtered_buckets[0].wgpu_device,
            WgpuDevice::IntegratedGpu(0)
        );
        assert_eq!(filtered_buckets[0].device_index, 0);
        assert_eq!(
            filtered_buckets[1].wgpu_device,
            WgpuDevice::IntegratedGpu(2)
        );
        assert_eq!(filtered_buckets[1].device_index, 1);
        assert_eq!(filtered_buckets[2].wgpu_device, WgpuDevice::VirtualGpu(0));
        assert_eq!(filtered_buckets[2].device_index, 2);
        assert_eq!(filtered_buckets[3].wgpu_device, WgpuDevice::VirtualGpu(1));
        assert_eq!(filtered_buckets[3].device_index, 3);
        assert_eq!(filtered_buckets[4].wgpu_device, WgpuDevice::Cpu);
        assert_eq!(filtered_buckets[4].type_ordinal, 0);
        assert_eq!(filtered_buckets[4].device_index, 4);

        // Filtered/subset inventory MUST match LogicalIndex against descriptor.device_index,
        // never silently returning a descriptor with a different device_index by slice offset.
        let mixed = build_wgpu_device_inventory_from_type_buckets(
            &[DeviceId::new(0, 0)],
            &[DeviceId::new(1, 0)],
            &[],
            &[],
        );
        let integrated_subset = &mixed[1..]; // contains only device_index == 1
        assert_eq!(
            resolve_wgpu_device_from_inventory(
                integrated_subset,
                WgpuDeviceSelectorV1::LogicalIndex(0),
            )
            .unwrap_err(),
            BackendBridgeError::DeviceUnavailable
        );
        let resolved_sub = resolve_wgpu_device_from_inventory(
            integrated_subset,
            WgpuDeviceSelectorV1::LogicalIndex(1),
        )
        .unwrap();
        assert_eq!(resolved_sub.device_index, 1);
        assert_eq!(resolved_sub.kind, WgpuDeviceKindV1::IntegratedGpu);
        assert_eq!(resolved_sub.wgpu_device, WgpuDevice::IntegratedGpu(0));
    }

    #[cfg(feature = "real-burn-wgpu")]
    #[test]
    fn wave16d_t_agent_wgpu_constructor_validates_full_schema_before_device_probe() {
        use real_burn_wgpu::{BurnWgpuLinearBridgeV1, BurnWgpuMlpBridgeV1, WgpuDeviceSelectorV1};

        // All zero-dimension permutations AND schema overflow / >256 MiB schemas MUST fail closed
        // with SchemaMismatch BEFORE probing WGPU adapters (even with unavailable device_index).
        for (in_dim, out_dim) in [(0, 2), (2, 0), (usize::MAX, 2), (10_000, 10_000)] {
            assert_eq!(
                BurnWgpuLinearBridgeV1::new(in_dim, out_dim, u32::MAX).unwrap_err(),
                BackendBridgeError::SchemaMismatch,
                "linear ({in_dim}, {out_dim}) should reject SchemaMismatch before device probe"
            );
        }
        for (in_dim, hidden_dim, out_dim) in [
            (0, 2, 1),
            (2, 0, 1),
            (2, 3, 0),
            (usize::MAX, 2, 1),
            (10_000, 10_000, 1),
        ] {
            assert_eq!(
                BurnWgpuMlpBridgeV1::new(in_dim, hidden_dim, out_dim, u32::MAX).unwrap_err(),
                BackendBridgeError::SchemaMismatch,
                "mlp ({in_dim}, {hidden_dim}, {out_dim}) should reject SchemaMismatch before device probe"
            );
        }

        // Absent explicit selectors and out-of-range LogicalIndex MUST fail closed with
        // DeviceUnavailable on BOTH Linear and MLP without touching WgpuServer::init.
        for sel in [
            WgpuDeviceSelectorV1::LogicalIndex(u32::MAX),
            WgpuDeviceSelectorV1::DiscreteGpu(9_999),
            WgpuDeviceSelectorV1::IntegratedGpu(9_999),
            WgpuDeviceSelectorV1::VirtualGpu(9_999),
        ] {
            assert_eq!(
                BurnWgpuLinearBridgeV1::new_with_selector(2, 1, sel).unwrap_err(),
                BackendBridgeError::DeviceUnavailable
            );
            assert_eq!(
                BurnWgpuMlpBridgeV1::new_with_selector(2, 3, 1, sel).unwrap_err(),
                BackendBridgeError::DeviceUnavailable
            );
        }
    }

    #[cfg(feature = "real-burn-wgpu")]
    #[test]
    fn wave16d_t_agent_wgpu_live_boundaries_fresh_overflow_safetensors_sgd_and_concurrency() {
        use std::sync::mpsc::{channel, RecvTimeoutError};
        use std::sync::RwLock;
        use std::time::Duration;

        use real_burn_wgpu::{
            enumerate_wgpu_devices, BurnWgpuLinearBridgeV1, BurnWgpuMlpBridgeV1, WgpuDeviceKindV1,
            WgpuDeviceSelectorV1,
        };
        use swarm_torch_core::traits::{SwarmModel, SwarmOptimizer};
        use swarm_torch_models::mlp::MlpModel;
        use swarm_torch_models::optim::SgdOptimizer;
        use swarm_torch_models::safetensors_v1::{
            export_safetensors_checkpoint_with_layout_v1,
            import_safetensors_checkpoint_with_layout_v1, ModelFlatTensorLayoutV1,
            SafeTensorsLimitsV1,
        };

        let devices = enumerate_wgpu_devices();
        if devices.is_empty() {
            return;
        }
        let desc = devices[0].clone();
        let explicit_selector = match desc.kind {
            WgpuDeviceKindV1::DiscreteGpu => WgpuDeviceSelectorV1::DiscreteGpu(desc.type_ordinal),
            WgpuDeviceKindV1::IntegratedGpu => {
                WgpuDeviceSelectorV1::IntegratedGpu(desc.type_ordinal)
            }
            WgpuDeviceKindV1::VirtualGpu => WgpuDeviceSelectorV1::VirtualGpu(desc.type_ordinal),
            WgpuDeviceKindV1::Cpu => WgpuDeviceSelectorV1::Cpu,
        };

        // 1. Explicit selector construction binds the exact descriptor & placement.
        let fresh_lin = BurnWgpuLinearBridgeV1::new_with_selector(3, 2, explicit_selector).unwrap();
        assert_eq!(fresh_lin.device_descriptor(), &desc);
        assert_eq!(
            fresh_lin.device_placement(),
            DevicePlacementV1::Wgpu {
                device_index: desc.device_index
            }
        );

        // 2. Fresh lazily-initialized WGPU Linear & MLP bridges evaluate one stable parameter set
        //    matching export_flat_f32().
        let x = [1.2f32, -0.4, 2.5];
        let t = [0.0f32, 0.0];
        let y1 = fresh_lin.forward_flat(&x).unwrap();
        let y2 = fresh_lin.forward_flat(&x).unwrap();
        let (l1, g1) = fresh_lin.backward_mse(&x, &t).unwrap();
        let (l2, g2) = fresh_lin.backward_mse(&x, &t).unwrap();
        assert_eq!(y1, y2);
        assert_eq!((l1, &g1), (l2, &g2));
        let lin_exported = fresh_lin.export_flat_f32().unwrap();
        let mut ref_lin = ReferenceLinearModule::new(3, 2, DevicePlacementV1::Cpu).unwrap();
        BurnModuleOps::import_flat_f32(&mut ref_lin, &lin_exported).unwrap();
        let y_ref = BurnModuleOps::forward_flat(&ref_lin, &x).unwrap();
        for (a, b) in y1.iter().zip(y_ref.iter()) {
            assert!((a - b).abs() < 1e-4, "fresh wgpu lin {a} vs ref {b}");
        }

        let fresh_mlp = BurnWgpuMlpBridgeV1::new_with_selector(3, 4, 2, explicit_selector).unwrap();
        let m1 = fresh_mlp.forward_flat(&x).unwrap();
        let m2 = fresh_mlp.forward_flat(&x).unwrap();
        let (ml1, mg1) = fresh_mlp.backward_mse(&x, &t).unwrap();
        let (ml2, mg2) = fresh_mlp.backward_mse(&x, &t).unwrap();
        assert_eq!(m1, m2);
        assert_eq!((ml1, &mg1), (ml2, &mg2));
        let mlp_exported = fresh_mlp.export_flat_f32().unwrap();
        let mut native_mlp = MlpModel::new(3, 4, 2).unwrap();
        native_mlp.load_parameters(&mlp_exported).unwrap();
        let m_ref = native_mlp.forward(&x).unwrap();
        for (a, b) in m1.iter().zip(m_ref.as_slice().iter()) {
            assert!((a - b).abs() < 1e-4, "fresh wgpu mlp {a} vs ref {b}");
        }

        // 3. Negative & overflow boundaries on live WGPU Linear & MLP.
        let mut lin = BurnWgpuLinearBridgeV1::new(2, 1, desc.device_index).unwrap();
        let init_lin = [2.0f32, -1.0, 0.5];
        lin.import_flat_f32(&init_lin).unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                lin.import_flat_f32(&[bad, 1.0, 0.5]).unwrap_err(),
                BackendBridgeError::NonFiniteValue
            );
            assert_eq!(
                lin.forward_flat(&[bad, 1.0]).unwrap_err(),
                BackendBridgeError::NonFiniteValue
            );
            assert_eq!(
                lin.backward_mse(&[1.0, 2.0], &[bad]).unwrap_err(),
                BackendBridgeError::NonFiniteValue
            );
        }
        assert_eq!(
            lin.import_flat_f32(&[1.0, 2.0]).unwrap_err(),
            BackendBridgeError::ParameterCountMismatch {
                expected: 3,
                got: 2
            }
        );
        assert_eq!(
            lin.forward_flat(&[]).unwrap_err(),
            BackendBridgeError::ParameterCountMismatch {
                expected: 2,
                got: 0
            }
        );
        assert_eq!(
            lin.forward_flat(&[1.0, 2.0, 3.0]).unwrap_err(),
            BackendBridgeError::ParameterCountMismatch {
                expected: 2,
                got: 3
            }
        );
        assert_eq!(
            lin.backward_mse(&[1.0, 2.0], &[1.0, 2.0]).unwrap_err(),
            BackendBridgeError::ParameterCountMismatch {
                expected: 1,
                got: 2
            }
        );
        assert_eq!(lin.export_flat_f32().unwrap(), init_lin);

        // GPU forward f32 overflow must fail closed with NonFiniteValue.
        lin.import_flat_f32(&[f32::MAX, f32::MAX, 0.0]).unwrap();
        assert_eq!(
            lin.forward_flat(&[f32::MAX, f32::MAX]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
        // Partial-sum overflow (`inf + -inf = NaN`) on GPU must also fail closed.
        assert_eq!(
            lin.forward_flat(&[2.0, -2.0]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
        // Product overflow before bias on GPU (`f32::MAX * 2.0 + (-f32::MAX) = +Inf`).
        let mut lin1 = BurnWgpuLinearBridgeV1::new(1, 1, desc.device_index).unwrap();
        lin1.import_flat_f32(&[f32::MAX, -f32::MAX]).unwrap();
        assert_eq!(
            lin1.forward_flat(&[2.0]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
        // GPU backward loss overflow (finite input/pred/target, squared diff overflows f32).
        lin.import_flat_f32(&[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(
            lin.backward_mse(&[1.0, 0.0], &[1e25]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );
        // GPU backward gradient overflow even when pred (= 0) AND loss (= 1e38) are finite!
        // dL/dW0 = 2 * (-1e19) * 1e25 = -2e44 -> -Inf.
        assert_eq!(
            lin.backward_mse(&[1e25, 0.0], &[1e19]).unwrap_err(),
            BackendBridgeError::NonFiniteValue
        );

        // WGPU MLP transactional import rejection & SafeTensors roundtrip + SgdOptimizer::step.
        let mut mlp = BurnWgpuMlpBridgeV1::new(3, 4, 2, desc.device_index).unwrap();
        let mlp_params: Vec<f32> = (0..26).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
        mlp.import_flat_f32(&mlp_params).unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad_p = mlp_params.clone();
            bad_p[25] = bad;
            assert_eq!(
                mlp.import_flat_f32(&bad_p).unwrap_err(),
                BackendBridgeError::NonFiniteValue
            );
        }
        assert_eq!(
            mlp.import_flat_f32(&mlp_params[..25]).unwrap_err(),
            BackendBridgeError::ParameterCountMismatch {
                expected: 26,
                got: 25
            }
        );
        assert_eq!(mlp.export_flat_f32().unwrap(), mlp_params);

        let mlp_schema = ModelTensorSchemaV1::for_mlp(3, 4, 2).unwrap();
        let mlp_layout = ModelFlatTensorLayoutV1::for_mlp(3, 4, 2).unwrap();
        let limits = SafeTensorsLimitsV1::default();
        let mut mlp_handle = BurnModuleBridgeV1::new(mlp)
            .into_handle(Some(mlp_schema.schema_hash()))
            .unwrap();
        let st_mlp =
            export_safetensors_checkpoint_with_layout_v1(&mlp_handle, &mlp_layout, &limits)
                .unwrap();
        let mut mlp_reload =
            BurnModuleBridgeV1::new(BurnWgpuMlpBridgeV1::new(3, 4, 2, desc.device_index).unwrap())
                .into_handle(Some(mlp_schema.schema_hash()))
                .unwrap();
        import_safetensors_checkpoint_with_layout_v1(
            &mut mlp_reload,
            &mlp_layout,
            &st_mlp,
            &limits,
        )
        .unwrap();
        assert_eq!(mlp_reload.parameters(), mlp_handle.parameters());

        // Live SgdOptimizer::step on WGPU BackendModelHandleV1 reduces MSE loss.
        let x_train = [0.8f32, -0.5, 1.1];
        let y_train = [0.5f32, -0.25];
        let (loss_before, grads_before) = mlp_handle
            .backend()
            .inner()
            .backward_mse(&x_train, &y_train)
            .unwrap();
        let mut opt = SgdOptimizer::<
            BackendModelHandleV1<BurnModuleBridgeV1<BurnWgpuMlpBridgeV1>>,
            64,
        >::with_hyperparams(0.1, 0.0, 0.0)
        .unwrap();
        opt.step(&mut mlp_handle, &grads_before).unwrap();
        let (loss_after, _) = mlp_handle
            .backend()
            .inner()
            .backward_mse(&x_train, &y_train)
            .unwrap();
        assert!(
            loss_after < loss_before,
            "expected WGPU SgdOptimizer::step to decrease loss: before={loss_before} after={loss_after}"
        );

        // 4. Concurrent readers + writer on live WGPU Linear & MLP with deadlock watchdog.
        let device_index = desc.device_index;
        let (done_tx, done_rx) = channel();
        let worker = std::thread::spawn(move || {
            let x_race = [1.2f32, -0.4, 2.5];
            let t_race = [1.0f32, -1.0];
            let lin_a = [0.5f32, -1.25, 0.75, -0.3, 1.4, -0.8, 0.25, -0.65];
            let lin_b: Vec<f32> = lin_a.iter().map(|v| -2.0 * v + 0.125).collect();

            let mut lin_bridge = BurnWgpuLinearBridgeV1::new(3, 2, device_index).unwrap();
            lin_bridge.import_flat_f32(&lin_b).unwrap();
            let expect_b = (
                lin_bridge.forward_flat(&x_race).unwrap(),
                lin_bridge.backward_mse(&x_race, &t_race).unwrap(),
            );
            lin_bridge.import_flat_f32(&lin_a).unwrap();
            let expect_a = (
                lin_bridge.forward_flat(&x_race).unwrap(),
                lin_bridge.backward_mse(&x_race, &t_race).unwrap(),
            );
            assert_ne!(expect_a, expect_b);

            let shared = RwLock::new(lin_bridge);
            std::thread::scope(|s| {
                for _ in 0..3 {
                    s.spawn(|| {
                        for _ in 0..12 {
                            let g = shared.read().unwrap();
                            let got = (
                                g.forward_flat(&x_race).unwrap(),
                                g.backward_mse(&x_race, &t_race).unwrap(),
                            );
                            assert!(got == expect_a || got == expect_b, "torn WGPU linear state");
                            std::thread::yield_now();
                        }
                    });
                }
                s.spawn(|| {
                    for r in 0..12 {
                        let next = if r % 2 == 0 { &lin_b[..] } else { &lin_a[..] };
                        shared.write().unwrap().import_flat_f32(next).unwrap();
                        assert_eq!(shared.read().unwrap().export_flat_f32().unwrap(), next);
                        std::thread::yield_now();
                    }
                });
            });
            let _ = done_tx.send(());
        });
        match done_rx.recv_timeout(Duration::from_secs(60)) {
            Ok(()) => worker.join().unwrap(),
            Err(RecvTimeoutError::Disconnected) => {
                std::panic::resume_unwind(worker.join().unwrap_err())
            }
            Err(RecvTimeoutError::Timeout) => {
                panic!("concurrent WGPU readers+writer timed out (possible burn-autodiff deadlock)")
            }
        }
    }

    #[cfg(feature = "real-burn-wgpu")]
    #[test]
    fn wave16d_concurrent_wgpu_enumeration_and_probes_never_crash() {
        use real_burn_wgpu::{enumerate_wgpu_devices, probe_wgpu_device, BurnWgpuLinearBridgeV1};

        let baseline = enumerate_wgpu_devices();
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..16 {
                        let got = enumerate_wgpu_devices();
                        assert_eq!(got, baseline);
                        assert_eq!(
                            probe_wgpu_device(u32::MAX).unwrap_err(),
                            BackendBridgeError::DeviceUnavailable
                        );
                        assert_eq!(
                            BurnWgpuLinearBridgeV1::new(2, 1, u32::MAX).unwrap_err(),
                            BackendBridgeError::DeviceUnavailable
                        );
                    }
                });
            }
        });
    }
}
