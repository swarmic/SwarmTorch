#![forbid(unsafe_code)]
//! Workspace-excluded tensor backend bridges for `Burn` and `Candle` (`ADR-0024` §4, `W12-01`, `13D-4`, `14B`).
//!
//! **Maturity Tier:** `Integration verified` — live `burn 0.21` (`Autodiff<NdArray<f32>>`) and
//! `candle 0.10` (`VarMap` / `Linear`) linear (`[W | b]`) and 2-layer ReLU MLP (`[W_1 | b_1 | W_2 | b_2]`)
//! bridges enforce canonical [`ModelTensorSchemaV1`] (`model_schema_hash`), [`ModelFlatTensorLayoutV1`]
//! parameter/gradient alignment with `SwarmTorch::LinearModel` and `SwarmTorch::MlpModel`, and
//! transactional `.is_finite()` verification across device synchronization and SafeTensors checkpoints.

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

    fn eval_affine(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
        if input.len() != self.in_dim {
            return Err(BackendBridgeError::ParameterCountMismatch {
                expected: self.in_dim,
                got: input.len(),
            });
        }
        let weight_len = self.out_dim * self.in_dim;
        let weight = &self.params[..weight_len];
        let bias = &self.params[weight_len..];
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

#[cfg(any(feature = "real-burn", feature = "real-candle"))]
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

#[cfg(any(feature = "real-burn", feature = "real-candle"))]
fn validate_batch_input(input: &[f32], in_dim: usize) -> Result<usize, BackendBridgeError> {
    if in_dim == 0 || input.is_empty() || input.len() % in_dim != 0 {
        return Err(BackendBridgeError::ParameterCountMismatch {
            expected: in_dim,
            got: input.len(),
        });
    }
    validate_finite_f32_slice(input, None)?;
    Ok(input.len() / in_dim)
}

/// Live `Burn 0.21` (`Autodiff<NdArray<f32>>`) linear and 2-layer MLP backend bridges (`ST-08`, `AR-14-07`, `14B-1`).
#[cfg(feature = "real-burn")]
pub mod real_burn {
    use std::sync::Mutex;

    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    use burn::module::Param;
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::activation::relu;
    use burn::tensor::{Tensor, TensorData};
    use swarm_torch_models::backend_bridge::{BackendBridgeError, DevicePlacementV1};
    use swarm_torch_models::safetensors_v1::ModelTensorSchemaV1;

    use crate::{validate_batch_input, validate_finite_f32_slice, BurnModuleOps};

    /// CPU autodiff Burn backend used by [`BurnLinearBridgeV1`] and [`BurnMlpBridgeV1`].
    pub type BurnNdArrayBackend = Autodiff<NdArray<f32>>;
    type BurnInnerBackend = NdArray<f32>;

    /// Live `Burn` affine module (`y = x @ W_t + b`) synchronized transactionally in canonical
    /// `[weight(out_dim * in_dim) | bias(out_dim)]` layout matching [`swarm_torch_models::simple::LinearModel`] (`AR-14-07`).
    #[derive(Debug)]
    pub struct BurnLinearBridgeV1 {
        schema: ModelTensorSchemaV1,
        in_dim: usize,
        out_dim: usize,
        device: NdArrayDevice,
        linear: Mutex<Linear<BurnNdArrayBackend>>,
    }

    impl BurnLinearBridgeV1 {
        /// Construct a new CPU autodiff `Burn` linear module matching `for_linear(in_dim, out_dim)`.
        pub fn new(in_dim: usize, out_dim: usize) -> Result<Self, BackendBridgeError> {
            if in_dim == 0 || out_dim == 0 {
                return Err(BackendBridgeError::SchemaMismatch);
            }
            let schema = ModelTensorSchemaV1::for_linear(in_dim, out_dim)
                .map_err(|_| BackendBridgeError::SchemaMismatch)?;
            let device = NdArrayDevice::Cpu;
            let linear = LinearConfig::new(in_dim, out_dim)
                .with_bias(true)
                .init::<BurnNdArrayBackend>(&device);
            Ok(Self {
                schema,
                in_dim,
                out_dim,
                device,
                linear: Mutex::new(linear),
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

            let guard = self
                .linear
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;

            let x = Tensor::<BurnNdArrayBackend, 2>::from_data(
                TensorData::new(input.to_vec(), [batch, self.in_dim]),
                &self.device,
            );
            let t = Tensor::<BurnNdArrayBackend, 2>::from_data(
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
    }

    impl BurnModuleOps for BurnLinearBridgeV1 {
        fn device_placement(&self) -> DevicePlacementV1 {
            DevicePlacementV1::Cpu
        }

        fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
            &self.schema
        }

        fn export_flat_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
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

        fn import_flat_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
            let weight_len = self.out_dim * self.in_dim;
            let expected = weight_len + self.out_dim;
            validate_finite_f32_slice(flat_params, Some(expected))?;

            // Prepare leaf tensors completely before acquiring lock or replacing parameters.
            let inner_weight_t = Tensor::<BurnInnerBackend, 2>::from_data(
                TensorData::new(
                    flat_params[..weight_len].to_vec(),
                    [self.out_dim, self.in_dim],
                ),
                &self.device,
            )
            .transpose();
            let inner_bias = Tensor::<BurnInnerBackend, 1>::from_data(
                TensorData::new(flat_params[weight_len..].to_vec(), [self.out_dim]),
                &self.device,
            );

            let new_weight = Param::from_tensor(
                Tensor::<BurnNdArrayBackend, 2>::from_inner(inner_weight_t).require_grad(),
            );
            let new_bias = Param::from_tensor(
                Tensor::<BurnNdArrayBackend, 1>::from_inner(inner_bias).require_grad(),
            );

            let mut guard = self
                .linear
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            guard.weight = new_weight;
            guard.bias = Some(new_bias);
            Ok(())
        }

        fn forward_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
            let batch = validate_batch_input(input, self.in_dim)?;
            let guard = self
                .linear
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let x = Tensor::<BurnNdArrayBackend, 2>::from_data(
                TensorData::new(input.to_vec(), [batch, self.in_dim]),
                &self.device,
            );
            let out = guard
                .forward(x)
                .to_data()
                .to_vec::<f32>()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(format!("{e:?}")))?;
            validate_finite_f32_slice(&out, Some(batch * self.out_dim))?;
            Ok(out)
        }
    }

    /// Live `Burn` 2-layer ReLU MLP (`y = fc2(relu(fc1(x)))`) synchronized transactionally in
    /// canonical `[W_1 | b_1 | W_2 | b_2]` layout matching [`swarm_torch_models::mlp::MlpModel`] (`14B-1`).
    #[derive(Debug)]
    pub struct BurnMlpBridgeV1 {
        schema: ModelTensorSchemaV1,
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        device: NdArrayDevice,
        layers: Mutex<(Linear<BurnNdArrayBackend>, Linear<BurnNdArrayBackend>)>,
    }

    impl BurnMlpBridgeV1 {
        /// Construct a new CPU autodiff `Burn` 2-layer ReLU MLP matching `for_mlp(in_dim, hidden_dim, out_dim)`.
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
            let device = NdArrayDevice::Cpu;
            let fc1 = LinearConfig::new(in_dim, hidden_dim)
                .with_bias(true)
                .init::<BurnNdArrayBackend>(&device);
            let fc2 = LinearConfig::new(hidden_dim, out_dim)
                .with_bias(true)
                .init::<BurnNdArrayBackend>(&device);
            Ok(Self {
                schema,
                in_dim,
                hidden_dim,
                out_dim,
                device,
                layers: Mutex::new((fc1, fc2)),
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

            let guard = self
                .layers
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let (ref fc1, ref fc2) = *guard;

            let x = Tensor::<BurnNdArrayBackend, 2>::from_data(
                TensorData::new(input.to_vec(), [batch, self.in_dim]),
                &self.device,
            );
            let t = Tensor::<BurnNdArrayBackend, 2>::from_data(
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
    }

    impl BurnModuleOps for BurnMlpBridgeV1 {
        fn device_placement(&self) -> DevicePlacementV1 {
            DevicePlacementV1::Cpu
        }

        fn tensor_schema(&self) -> &ModelTensorSchemaV1 {
            &self.schema
        }

        fn export_flat_f32(&self) -> Result<Vec<f32>, BackendBridgeError> {
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

        fn import_flat_f32(&mut self, flat_params: &[f32]) -> Result<(), BackendBridgeError> {
            let w1_len = self.hidden_dim * self.in_dim;
            let b1_end = w1_len + self.hidden_dim;
            let w2_end = b1_end + self.out_dim * self.hidden_dim;
            let expected = w2_end + self.out_dim;
            validate_finite_f32_slice(flat_params, Some(expected))?;

            let inner_w1_t = Tensor::<BurnInnerBackend, 2>::from_data(
                TensorData::new(
                    flat_params[..w1_len].to_vec(),
                    [self.hidden_dim, self.in_dim],
                ),
                &self.device,
            )
            .transpose();
            let inner_b1 = Tensor::<BurnInnerBackend, 1>::from_data(
                TensorData::new(flat_params[w1_len..b1_end].to_vec(), [self.hidden_dim]),
                &self.device,
            );
            let inner_w2_t = Tensor::<BurnInnerBackend, 2>::from_data(
                TensorData::new(
                    flat_params[b1_end..w2_end].to_vec(),
                    [self.out_dim, self.hidden_dim],
                ),
                &self.device,
            )
            .transpose();
            let inner_b2 = Tensor::<BurnInnerBackend, 1>::from_data(
                TensorData::new(flat_params[w2_end..].to_vec(), [self.out_dim]),
                &self.device,
            );

            let new_w1 = Param::from_tensor(
                Tensor::<BurnNdArrayBackend, 2>::from_inner(inner_w1_t).require_grad(),
            );
            let new_b1 = Param::from_tensor(
                Tensor::<BurnNdArrayBackend, 1>::from_inner(inner_b1).require_grad(),
            );
            let new_w2 = Param::from_tensor(
                Tensor::<BurnNdArrayBackend, 2>::from_inner(inner_w2_t).require_grad(),
            );
            let new_b2 = Param::from_tensor(
                Tensor::<BurnNdArrayBackend, 1>::from_inner(inner_b2).require_grad(),
            );

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

        fn forward_flat(&self, input: &[f32]) -> Result<Vec<f32>, BackendBridgeError> {
            let batch = validate_batch_input(input, self.in_dim)?;
            let guard = self
                .layers
                .lock()
                .map_err(|e| BackendBridgeError::BackendOperationFailed(e.to_string()))?;
            let x = Tensor::<BurnNdArrayBackend, 2>::from_data(
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
            validate_finite_f32_slice(&out, Some(batch * self.out_dim))?;
            Ok(out)
        }
    }
}

/// Live `Candle 0.10` CPU `VarMap` / `Linear` and 2-layer MLP backend bridges (`ST-08`, `AR-14-07`, `14B-1`).
#[cfg(feature = "real-candle")]
pub mod real_candle {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
    use swarm_torch_models::backend_bridge::{BackendBridgeError, DevicePlacementV1};
    use swarm_torch_models::safetensors_v1::ModelTensorSchemaV1;

    use crate::{validate_batch_input, validate_finite_f32_slice, CandleModuleOps};

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

            // Validate both VarMap keys and shapes BEFORE committing either write.
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
            if bias_var.shape() != bias_t.shape() || weight_var.shape() != weight_t.shape() {
                return Err(BackendBridgeError::SchemaMismatch);
            }
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
            if w1_var.shape() != w1_t.shape()
                || b1_var.shape() != b1_t.shape()
                || w2_var.shape() != w2_t.shape()
                || b2_var.shape() != b2_t.shape()
            {
                return Err(BackendBridgeError::SchemaMismatch);
            }
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
}
