//! `SwarmTorchCheckpointV1` SafeTensors profile and canonical `model_schema_hash`
//! (`ADR-0024` §3, `W11-04`).
//!
//! Implements the HuggingFace SafeTensors v1 binary wire format
//! (`[u64_le(header_len) || utf8_json_header || raw_le_tensor_bytes]`) with:
//! - Strict pre-parse resource bounds ([`SafeTensorsLimitsV1`]),
//! - Duplicate-key-rejecting JSON header parser (`F-07`),
//! - Checked shape-product and contiguous non-overlapping `[begin, end]` offset verification,
//! - Canonical length-delimited binary `model_schema_hash` (`swarmtorch.schema.v1`),
//! - Transactional all-or-nothing `.is_finite()` verification before mutating a [`SwarmModel`],
//! - Wave 12 (`W12-03`) [`LoRaAdapterDescriptorV1`] extension metadata for low-rank adapter checkpoints.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use swarm_torch_core::crypto::{sha256_bytes, MODEL_SCHEMA_HASH_DOMAIN_TAG_V1};
use swarm_torch_core::traits::SwarmModel;

/// Canonical metadata format identifier required in `__metadata__.format`.
pub const SWARMTORCH_CHECKPOINT_FORMAT_V1: &str = "swarmtorch.checkpoint.v1";

/// Allowed SafeTensors element data types (`ADR-0024` §3, `W11-04`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum SafeTensorDtypeV1 {
    /// IEEE-754 32-bit float (`"F32"`, 4 bytes per element, default trainable format).
    F32,
    /// IEEE-754 16-bit float (`"F16"`, 2 bytes per element, opt-in frozen/quantized profile).
    F16,
    /// Brain floating-point 16-bit (`"BF16"`, 2 bytes per element, opt-in profile).
    BF16,
    /// Signed 8-bit integer (`"I8"`, 1 byte per element, opt-in quantized profile).
    I8,
}

impl SafeTensorDtypeV1 {
    /// Canonical SafeTensors dtype string (`"F32"`, `"F16"`, `"BF16"`, `"I8"`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::I8 => "I8",
        }
    }

    /// Parse a SafeTensors dtype string.
    pub fn from_str_name(s: &str) -> Result<Self, SafeTensorsError> {
        match s {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "I8" => Ok(Self::I8),
            _ => Err(SafeTensorsError::UnsupportedDtype),
        }
    }

    /// Byte width per element.
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
        }
    }

    /// Canonical 1-byte tag used in `model_schema_hash` (`ADR-0024` §3).
    pub const fn schema_tag(self) -> u8 {
        match self {
            Self::F32 => 0x01,
            Self::F16 => 0x02,
            Self::BF16 => 0x03,
            Self::I8 => 0x04,
        }
    }

    const fn bitmask(self) -> u8 {
        1u8 << (self.schema_tag() - 1)
    }
}

/// Resource and dtype bounds for parsing a `SwarmTorchCheckpointV1` SafeTensors artifact (`ADR-0024` §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafeTensorsLimitsV1 {
    /// Maximum JSON header byte length (default `1_048_576` = 1 MiB).
    pub max_header_bytes: usize,
    /// Maximum number of tensors in a single checkpoint (default `4_096`).
    pub max_tensors: usize,
    /// Maximum byte length of any tensor name or metadata key/value (default `256`).
    pub max_name_bytes: usize,
    /// Maximum tensor rank / number of dimensions (default `8`).
    pub max_rank: usize,
    /// Maximum total raw tensor data buffer size in bytes (default `268_435_456` = 256 MiB).
    pub max_total_data_bytes: usize,
    /// Bitmask of permitted [`SafeTensorDtypeV1`] tags (default: `F32` only).
    pub allowed_dtypes_mask: u8,
}

impl Default for SafeTensorsLimitsV1 {
    fn default() -> Self {
        Self {
            max_header_bytes: 1_048_576,
            max_tensors: 4_096,
            max_name_bytes: 256,
            max_rank: 8,
            max_total_data_bytes: 256 * 1024 * 1024,
            allowed_dtypes_mask: SafeTensorDtypeV1::F32.bitmask(),
        }
    }
}

impl SafeTensorsLimitsV1 {
    /// Enable an additional [`SafeTensorDtypeV1`] in `allowed_dtypes_mask`.
    pub const fn allow_dtype(mut self, dtype: SafeTensorDtypeV1) -> Self {
        self.allowed_dtypes_mask |= dtype.bitmask();
        self
    }

    /// Check whether `dtype` is permitted by this profile.
    pub const fn is_dtype_allowed(&self, dtype: SafeTensorDtypeV1) -> bool {
        (self.allowed_dtypes_mask & dtype.bitmask()) != 0
    }
}

/// Errors returned during `SwarmTorchCheckpointV1` SafeTensors serialization, parsing, or model load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafeTensorsError {
    /// Buffer is shorter than the 8-byte header length prefix or declared header/data span.
    TruncatedBuffer,
    /// JSON header length is 0 or exceeds `max_header_bytes`.
    HeaderTooLarge,
    /// Raw tensor data buffer exceeds `max_total_data_bytes`.
    DataTooLarge,
    /// Header is not valid UTF-8 or violates the SafeTensors JSON object grammar.
    MalformedHeaderJson,
    /// Duplicate key encountered in JSON header (`__metadata__`, tensor name, or descriptor field) (`F-07`).
    DuplicateKey,
    /// Number of tensors exceeds `max_tensors` or tensor is empty.
    TooManyOrEmptyTensors,
    /// Tensor name or metadata string exceeds `max_name_bytes` or is empty.
    InvalidIdentifierLength,
    /// Tensor rank exceeds `max_rank`.
    RankExceeded,
    /// Unrecognized dtype string in header.
    UnsupportedDtype,
    /// Dtype is valid in SafeTensors but disallowed by [`SafeTensorsLimitsV1::allowed_dtypes_mask`].
    DisallowedDtype,
    /// Tensor shape product $\times$ dtype width overflowed `usize`.
    ShapeOverflow,
    /// Tensor `data_offsets: [begin, end]` are inverted, out-of-bounds, non-contiguous, or overlapping.
    InvalidOffsets,
    /// Missing or mismatched `__metadata__.format` or `__metadata__.model_schema_hash`.
    SchemaMismatch,
    /// Non-finite (`NaN` or `±Inf`) parameter value detected during export or transactional import (`F-07`).
    NonFiniteParameter,
    /// Underlying [`SwarmModel::load_parameters`] rejected the parameter slice.
    ModelLoadFailed,
}

impl core::fmt::Display for SafeTensorsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TruncatedBuffer => write!(f, "SafeTensors buffer is truncated"),
            Self::HeaderTooLarge => write!(f, "SafeTensors JSON header is empty or exceeds max_header_bytes"),
            Self::DataTooLarge => write!(f, "SafeTensors data section exceeds max_total_data_bytes"),
            Self::MalformedHeaderJson => write!(f, "malformed SafeTensors UTF-8 JSON header"),
            Self::DuplicateKey => write!(f, "duplicate JSON key in SafeTensors header"),
            Self::TooManyOrEmptyTensors => write!(f, "tensor count is zero or exceeds max_tensors"),
            Self::InvalidIdentifierLength => write!(f, "tensor name or metadata string length out of bounds"),
            Self::RankExceeded => write!(f, "tensor shape rank exceeds max_rank"),
            Self::UnsupportedDtype => write!(f, "unsupported SafeTensors dtype string"),
            Self::DisallowedDtype => write!(f, "SafeTensors dtype is not permitted by profile"),
            Self::ShapeOverflow => write!(f, "tensor shape product overflowed usize"),
            Self::InvalidOffsets => write!(
                f,
                "tensor data_offsets are out-of-bounds, non-contiguous, overlapping, or mismatch shape byte size"
            ),
            Self::SchemaMismatch => write!(
                f,
                "SafeTensors metadata format or model_schema_hash does not match expected schema"
            ),
            Self::NonFiniteParameter => write!(
                f,
                "non-finite (NaN or Inf) parameter rejected before model mutation"
            ),
            Self::ModelLoadFailed => write!(f, "SwarmModel::load_parameters failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SafeTensorsError {}

/// Descriptor for a single named tensor in [`ModelTensorSchemaV1`] (`ADR-0024` §3, `W11-04`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelTensorDescriptorV1 {
    /// Canonical tensor identifier (e.g. `"linear.weight"`, `"linear.bias"`).
    pub name: String,
    /// Element data type.
    pub dtype: SafeTensorDtypeV1,
    /// Whether this tensor participates in federated/swarm gradient updates.
    pub trainable: bool,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
}

impl ModelTensorDescriptorV1 {
    /// Checked element count ($\prod_r \text{shape}_r$).
    pub fn num_elements(&self) -> Result<usize, SafeTensorsError> {
        if self.shape.is_empty() {
            return Err(SafeTensorsError::InvalidOffsets);
        }
        let n = self
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(SafeTensorsError::ShapeOverflow)?;
        if n == 0 {
            return Err(SafeTensorsError::InvalidOffsets);
        }
        Ok(n)
    }

    /// Checked byte length ($\text{num\_elements}() \times \text{dtype.byte\_width}()$).
    pub fn byte_len(&self) -> Result<usize, SafeTensorsError> {
        self.num_elements()?
            .checked_mul(self.dtype.byte_width())
            .ok_or(SafeTensorsError::ShapeOverflow)
    }
}

/// Wave 12 (`W12-03`) extension descriptor for Low-Rank Adaptation (`LoRA`) factorized matrix pairs
/// ($W = W_0 + \frac{\alpha}{r} B \cdot A$) stored inside `SwarmTorchCheckpointV1` artifacts.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LoRaAdapterDescriptorV1 {
    /// Target module prefix (e.g. `"attn.q_proj"`).
    pub target_module: String,
    /// Down-projection tensor name $A \in \mathbb{R}^{r \times d_{\text{in}}}$.
    pub tensor_a_name: String,
    /// Up-projection tensor name $B \in \mathbb{R}^{d_{\text{out}} \times r}$.
    pub tensor_b_name: String,
    /// Adapter rank $r \ge 1$.
    pub rank: u32,
    /// Scaling hyperparameter $\alpha > 0$.
    pub alpha: f32,
}

/// Canonical model tensor schema (`ADR-0024` §3, `W11-04`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelTensorSchemaV1 {
    /// Named tensor descriptors (stored in canonical UTF-8 byte-lexicographic `name` order).
    pub tensors: Vec<ModelTensorDescriptorV1>,
    /// Optional Wave 12 (`W12-03`) LoRA adapter descriptors.
    pub lora_adapters: Vec<LoRaAdapterDescriptorV1>,
}

impl ModelTensorSchemaV1 {
    /// Construct and validate a canonical [`ModelTensorSchemaV1`], sorting tensors by UTF-8 byte
    /// order of `name` and rejecting duplicate or invalid descriptors.
    pub fn try_new(
        mut tensors: Vec<ModelTensorDescriptorV1>,
        limits: &SafeTensorsLimitsV1,
    ) -> Result<Self, SafeTensorsError> {
        if tensors.is_empty() || tensors.len() > limits.max_tensors {
            return Err(SafeTensorsError::TooManyOrEmptyTensors);
        }
        tensors.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));

        let mut total_bytes = 0usize;
        for (i, t) in tensors.iter().enumerate() {
            if t.name.is_empty() || t.name.len() > limits.max_name_bytes || t.name == "__metadata__"
            {
                return Err(SafeTensorsError::InvalidIdentifierLength);
            }
            if i > 0 && tensors[i - 1].name == t.name {
                return Err(SafeTensorsError::DuplicateKey);
            }
            if t.shape.is_empty() || t.shape.len() > limits.max_rank {
                return Err(SafeTensorsError::RankExceeded);
            }
            if !limits.is_dtype_allowed(t.dtype) {
                return Err(SafeTensorsError::DisallowedDtype);
            }
            let b_len = t.byte_len()?;
            total_bytes = total_bytes
                .checked_add(b_len)
                .ok_or(SafeTensorsError::ShapeOverflow)?;
            if total_bytes > limits.max_total_data_bytes {
                return Err(SafeTensorsError::DataTooLarge);
            }
        }

        Ok(Self {
            tensors,
            lora_adapters: Vec::new(),
        })
    }

    /// Convenience schema for [`crate::simple::LinearModel`] (`weight: [output_dim, input_dim]`, `bias: [output_dim]`).
    pub fn for_linear(input_dim: usize, output_dim: usize) -> Result<Self, SafeTensorsError> {
        Self::try_new(
            vec![
                ModelTensorDescriptorV1 {
                    name: "linear.bias".to_string(),
                    dtype: SafeTensorDtypeV1::F32,
                    trainable: true,
                    shape: vec![output_dim],
                },
                ModelTensorDescriptorV1 {
                    name: "linear.weight".to_string(),
                    dtype: SafeTensorDtypeV1::F32,
                    trainable: true,
                    shape: vec![output_dim, input_dim],
                },
            ],
            &SafeTensorsLimitsV1::default(),
        )
    }

    /// Convenience schema for [`crate::mlp::MlpModel`] (`fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`).
    pub fn for_mlp(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Result<Self, SafeTensorsError> {
        Self::try_new(
            vec![
                ModelTensorDescriptorV1 {
                    name: "mlp.fc1.bias".to_string(),
                    dtype: SafeTensorDtypeV1::F32,
                    trainable: true,
                    shape: vec![hidden_dim],
                },
                ModelTensorDescriptorV1 {
                    name: "mlp.fc1.weight".to_string(),
                    dtype: SafeTensorDtypeV1::F32,
                    trainable: true,
                    shape: vec![hidden_dim, input_dim],
                },
                ModelTensorDescriptorV1 {
                    name: "mlp.fc2.bias".to_string(),
                    dtype: SafeTensorDtypeV1::F32,
                    trainable: true,
                    shape: vec![output_dim],
                },
                ModelTensorDescriptorV1 {
                    name: "mlp.fc2.weight".to_string(),
                    dtype: SafeTensorDtypeV1::F32,
                    trainable: true,
                    shape: vec![output_dim, hidden_dim],
                },
            ],
            &SafeTensorsLimitsV1::default(),
        )
    }

    /// Convenience schema for a single flat `F32` parameter vector.
    pub fn single_f32_tensor(name: &str, num_elements: usize) -> Result<Self, SafeTensorsError> {
        Self::try_new(
            vec![ModelTensorDescriptorV1 {
                name: name.to_string(),
                dtype: SafeTensorDtypeV1::F32,
                trainable: true,
                shape: vec![num_elements],
            }],
            &SafeTensorsLimitsV1::default(),
        )
    }

    /// Total number of elements across all tensors in canonical order.
    pub fn total_elements(&self) -> Result<usize, SafeTensorsError> {
        self.tensors.iter().try_fold(0usize, |acc, t| {
            acc.checked_add(t.num_elements()?)
                .ok_or(SafeTensorsError::ShapeOverflow)
        })
    }

    /// Compute the normative length-delimited binary `model_schema_hash` (`ADR-0024` §3, `W11-04`):
    ///
    /// $$\text{SHA-256}\!\left(\texttt{"swarmtorch.schema.v1"} \;\|\; \text{u32\_le}(N_{\text{tensors}}) \;\|\; \bigoplus_{i=1}^{N_{\text{tensors}}} \left(\text{u32\_le}(\text{name\_len}) \;\|\; \text{name} \;\|\; \text{u8}(\text{dtype\_tag}) \;\|\; \text{u8}(\text{trainable}) \;\|\; \text{u32\_le}(\text{rank}) \;\|\; \bigoplus_{r} \text{u64\_le}(\text{dim}_r)\right)\right)$$
    pub fn schema_hash(&self) -> [u8; 32] {
        let mut sorted = self.tensors.clone();
        sorted.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));

        let mut buf = Vec::new();
        buf.extend_from_slice(MODEL_SCHEMA_HASH_DOMAIN_TAG_V1);
        buf.extend_from_slice(&(sorted.len() as u32).to_le_bytes());

        for t in &sorted {
            let name_bytes = t.name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.push(t.dtype.schema_tag());
            buf.push(u8::from(t.trainable));
            buf.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for &dim in &t.shape {
                buf.extend_from_slice(&(dim as u64).to_le_bytes());
            }
        }

        sha256_bytes(&buf)
    }

    /// Lowercase 64-character hex encoding of [`Self::schema_hash`].
    pub fn schema_hash_hex(&self) -> String {
        hex_encode_lower(&self.schema_hash())
    }
}

fn hex_encode_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn push_json_escaped_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Disjoint flat-buffer element range (`[flat_offset .. flat_offset + num_elements)`) mapping
/// a canonical SafeTensors tensor name to its position in [`SwarmModel::parameters()`] (`ST-08`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelTensorFlatRangeV1 {
    /// Canonical tensor name present in [`ModelTensorSchemaV1::tensors`].
    pub name: String,
    /// Starting `f32` element index in [`SwarmModel::parameters()`].
    pub flat_offset: usize,
    /// Exact number of `f32` elements declared by the schema descriptor.
    pub num_elements: usize,
}

/// Validated bidirectional mapping between canonical sorted [`ModelTensorSchemaV1`] tensor
/// entries and a model's contiguous [`SwarmModel::parameters()`] slice (`ST-08`).
///
/// Ensures models whose contiguous parameter buffer follows layer declaration order (such as
/// [`crate::simple::LinearModel`] `[W | b]` or [`crate::mlp::MlpModel`] `[W1 | b1 | W2 | b2]`)
/// serialize each named SafeTensors entry from its true slice rather than alphabetical order.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelFlatTensorLayoutV1 {
    schema: ModelTensorSchemaV1,
    ranges_by_schema_order: Vec<ModelTensorFlatRangeV1>,
    total_elements: usize,
}

impl ModelFlatTensorLayoutV1 {
    /// Construct and strictly verify a disjoint, gap-free mapping covering `[0 .. schema.total_elements())`.
    pub fn try_new(
        schema: ModelTensorSchemaV1,
        explicit_offsets: BTreeMap<String, usize>,
        limits: &SafeTensorsLimitsV1,
    ) -> Result<Self, SafeTensorsError> {
        let validated = ModelTensorSchemaV1::try_new(schema.tensors, limits)?;
        let total_elements = validated.total_elements()?;
        if explicit_offsets.len() != validated.tensors.len() {
            return Err(SafeTensorsError::SchemaMismatch);
        }

        let mut ranges_by_schema_order = Vec::with_capacity(validated.tensors.len());
        let mut intervals = Vec::with_capacity(validated.tensors.len());
        for desc in &validated.tensors {
            let &flat_offset = explicit_offsets
                .get(&desc.name)
                .ok_or(SafeTensorsError::SchemaMismatch)?;
            let num_elements = desc.num_elements()?;
            let flat_end = flat_offset
                .checked_add(num_elements)
                .ok_or(SafeTensorsError::ShapeOverflow)?;
            if flat_end > total_elements {
                return Err(SafeTensorsError::SchemaMismatch);
            }
            ranges_by_schema_order.push(ModelTensorFlatRangeV1 {
                name: desc.name.clone(),
                flat_offset,
                num_elements,
            });
            intervals.push((flat_offset, flat_end));
        }

        intervals.sort_unstable();
        let mut cursor = 0usize;
        for (begin, end) in intervals {
            if begin != cursor {
                return Err(SafeTensorsError::SchemaMismatch);
            }
            cursor = end;
        }
        if cursor != total_elements {
            return Err(SafeTensorsError::SchemaMismatch);
        }

        Ok(Self {
            schema: validated,
            ranges_by_schema_order,
            total_elements,
        })
    }

    /// Construct the identity layout matching canonical sorted [`ModelTensorSchemaV1`] tensor order
    /// (used by backends exporting parameters in sorted schema order).
    pub fn canonical(schema: ModelTensorSchemaV1) -> Result<Self, SafeTensorsError> {
        let validated =
            ModelTensorSchemaV1::try_new(schema.tensors, &SafeTensorsLimitsV1::default())?;
        let mut offsets = BTreeMap::new();
        let mut cursor = 0usize;
        for desc in &validated.tensors {
            offsets.insert(desc.name.clone(), cursor);
            cursor = cursor
                .checked_add(desc.num_elements()?)
                .ok_or(SafeTensorsError::ShapeOverflow)?;
        }
        Self::try_new(validated, offsets, &SafeTensorsLimitsV1::default())
    }

    /// Flat `[W | b]` parameter layout matching [`crate::simple::LinearModelWithCapacity`].
    pub fn for_linear(input_dim: usize, output_dim: usize) -> Result<Self, SafeTensorsError> {
        let schema = ModelTensorSchemaV1::for_linear(input_dim, output_dim)?;
        let weight_count = input_dim
            .checked_mul(output_dim)
            .ok_or(SafeTensorsError::ShapeOverflow)?;
        let mut offsets = BTreeMap::new();
        offsets.insert("linear.weight".to_string(), 0);
        offsets.insert("linear.bias".to_string(), weight_count);
        Self::try_new(schema, offsets, &SafeTensorsLimitsV1::default())
    }

    /// Flat `[W1 | b1 | W2 | b2]` parameter layout matching [`crate::mlp::MlpModelWithCapacity`].
    pub fn for_mlp(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Result<Self, SafeTensorsError> {
        let schema = ModelTensorSchemaV1::for_mlp(input_dim, hidden_dim, output_dim)?;
        let w1_count = input_dim
            .checked_mul(hidden_dim)
            .ok_or(SafeTensorsError::ShapeOverflow)?;
        let b1_offset = w1_count;
        let w2_offset = b1_offset
            .checked_add(hidden_dim)
            .ok_or(SafeTensorsError::ShapeOverflow)?;
        let w2_count = hidden_dim
            .checked_mul(output_dim)
            .ok_or(SafeTensorsError::ShapeOverflow)?;
        let b2_offset = w2_offset
            .checked_add(w2_count)
            .ok_or(SafeTensorsError::ShapeOverflow)?;

        let mut offsets = BTreeMap::new();
        offsets.insert("mlp.fc1.weight".to_string(), 0);
        offsets.insert("mlp.fc1.bias".to_string(), b1_offset);
        offsets.insert("mlp.fc2.weight".to_string(), w2_offset);
        offsets.insert("mlp.fc2.bias".to_string(), b2_offset);
        Self::try_new(schema, offsets, &SafeTensorsLimitsV1::default())
    }

    /// Borrow the underlying canonical sorted [`ModelTensorSchemaV1`].
    pub const fn schema(&self) -> &ModelTensorSchemaV1 {
        &self.schema
    }

    /// Borrow validated flat parameter ranges ordered to match `self.schema().tensors`.
    pub fn ranges(&self) -> &[ModelTensorFlatRangeV1] {
        &self.ranges_by_schema_order
    }

    /// Return the total mapped parameter element count.
    pub const fn total_elements(&self) -> usize {
        self.total_elements
    }
}

/// Serialize a [`SwarmModel`] into a canonical `SwarmTorchCheckpointV1` SafeTensors binary buffer
/// (`[u64_le(header_len) || utf8_json_header || raw_le_tensor_bytes]`) assuming canonical sorted
/// [`ModelTensorSchemaV1`] parameter ordering.
///
/// Models with declaration-ordered flat parameters (`LinearModel` `[W | b]` or `MlpModel`
/// `[W1 | b1 | W2 | b2]`) must use [`export_safetensors_checkpoint_with_layout_v1`].
pub fn export_safetensors_checkpoint_v1<M: SwarmModel>(
    model: &M,
    schema: &ModelTensorSchemaV1,
    limits: &SafeTensorsLimitsV1,
) -> Result<Vec<u8>, SafeTensorsError> {
    let validated = ModelTensorSchemaV1::try_new(schema.tensors.clone(), limits)?;
    let layout = ModelFlatTensorLayoutV1::canonical(validated)?;
    export_safetensors_checkpoint_with_layout_v1(model, &layout, limits)
}

/// Serialize a [`SwarmModel`] into a canonical `SwarmTorchCheckpointV1` SafeTensors binary buffer
/// using an explicit [`ModelFlatTensorLayoutV1`] mapping between tensor names and flat parameter
/// offsets (`ST-08`).
pub fn export_safetensors_checkpoint_with_layout_v1<M: SwarmModel>(
    model: &M,
    layout: &ModelFlatTensorLayoutV1,
    limits: &SafeTensorsLimitsV1,
) -> Result<Vec<u8>, SafeTensorsError> {
    let validated = ModelTensorSchemaV1::try_new(layout.schema.tensors.clone(), limits)?;
    let expected_elements = validated.total_elements()?;
    if expected_elements != layout.total_elements
        || layout.ranges_by_schema_order.len() != validated.tensors.len()
    {
        return Err(SafeTensorsError::SchemaMismatch);
    }
    let params = model.parameters();
    if params.len() != expected_elements {
        return Err(SafeTensorsError::SchemaMismatch);
    }
    for &p in params {
        if !p.is_finite() {
            return Err(SafeTensorsError::NonFiniteParameter);
        }
    }

    let schema_hash_hex = validated.schema_hash_hex();
    let mut header_json = String::new();
    header_json.push_str("{\"__metadata__\":{\"format\":");
    push_json_escaped_str(&mut header_json, SWARMTORCH_CHECKPOINT_FORMAT_V1);
    header_json.push_str(",\"model_schema_hash\":");
    push_json_escaped_str(&mut header_json, &schema_hash_hex);
    header_json.push('}');

    let mut byte_cursor = 0usize;
    for t in &validated.tensors {
        if t.dtype != SafeTensorDtypeV1::F32 {
            return Err(SafeTensorsError::DisallowedDtype);
        }
        let b_len = t.byte_len()?;
        let begin = byte_cursor;
        let end = begin
            .checked_add(b_len)
            .ok_or(SafeTensorsError::ShapeOverflow)?;
        byte_cursor = end;

        header_json.push(',');
        push_json_escaped_str(&mut header_json, &t.name);
        header_json.push_str(":{\"dtype\":");
        push_json_escaped_str(&mut header_json, t.dtype.as_str());
        header_json.push_str(",\"shape\":[");
        for (i, &dim) in t.shape.iter().enumerate() {
            if i > 0 {
                header_json.push(',');
            }
            header_json.push_str(&dim.to_string());
        }
        header_json.push_str("],\"data_offsets\":[");
        header_json.push_str(&begin.to_string());
        header_json.push(',');
        header_json.push_str(&end.to_string());
        header_json.push_str("]}");
    }
    header_json.push('}');

    // Pad JSON header with ASCII spaces (0x20) to an 8-byte boundary per SafeTensors spec.
    while header_json.len() % 8 != 0 {
        header_json.push(' ');
    }
    if header_json.len() > limits.max_header_bytes {
        return Err(SafeTensorsError::HeaderTooLarge);
    }

    let header_len_u64 = header_json.len() as u64;
    let total_len = 8usize
        .checked_add(header_json.len())
        .and_then(|h| h.checked_add(byte_cursor))
        .ok_or(SafeTensorsError::ShapeOverflow)?;

    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&header_len_u64.to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    for (desc, range) in validated.tensors.iter().zip(&layout.ranges_by_schema_order) {
        if desc.name != range.name || range.num_elements != desc.num_elements()? {
            return Err(SafeTensorsError::SchemaMismatch);
        }
        let end = range
            .flat_offset
            .checked_add(range.num_elements)
            .ok_or(SafeTensorsError::ShapeOverflow)?;
        let slice = params
            .get(range.flat_offset..end)
            .ok_or(SafeTensorsError::SchemaMismatch)?;
        for &p in slice {
            out.extend_from_slice(&p.to_le_bytes());
        }
    }
    Ok(out)
}

/// Parsed tensor entry from a SafeTensors JSON header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSafeTensorEntryV1 {
    pub name: String,
    pub dtype: SafeTensorDtypeV1,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Parsed and offset-verified SafeTensors v1 header (`ADR-0024` §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSafeTensorsHeaderV1 {
    pub metadata: BTreeMap<String, String>,
    pub tensors: BTreeMap<String, ParsedSafeTensorEntryV1>,
    pub data_section_offset: usize,
    pub data_section_len: usize,
}

struct JsonHeaderCursor<'a> {
    src: &'a [u8],
    pos: usize,
    limits: SafeTensorsLimitsV1,
}

impl<'a> JsonHeaderCursor<'a> {
    fn new(src: &'a [u8], limits: SafeTensorsLimitsV1) -> Self {
        Self {
            src,
            pos: 0,
            limits,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.src.len() {
            match self.src[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn expect_byte(&mut self, expected: u8) -> Result<(), SafeTensorsError> {
        self.skip_ws();
        if self.pos >= self.src.len() || self.src[self.pos] != expected {
            return Err(SafeTensorsError::MalformedHeaderJson);
        }
        self.pos += 1;
        Ok(())
    }

    fn peek_byte(&mut self) -> Result<u8, SafeTensorsError> {
        self.skip_ws();
        self.src
            .get(self.pos)
            .copied()
            .ok_or(SafeTensorsError::MalformedHeaderJson)
    }

    fn parse_string(&mut self) -> Result<String, SafeTensorsError> {
        self.expect_byte(b'"')?;
        let mut out = Vec::new();
        while self.pos < self.src.len() {
            let b = self.src[self.pos];
            self.pos += 1;
            if b == b'"' {
                let s = core::str::from_utf8(&out)
                    .map_err(|_| SafeTensorsError::MalformedHeaderJson)?
                    .to_string();
                if s.len() > self.limits.max_name_bytes {
                    return Err(SafeTensorsError::InvalidIdentifierLength);
                }
                return Ok(s);
            }
            if b == b'\\' {
                if self.pos >= self.src.len() {
                    return Err(SafeTensorsError::MalformedHeaderJson);
                }
                let esc = self.src[self.pos];
                self.pos += 1;
                match esc {
                    b'"' | b'\\' | b'/' => out.push(esc),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    _ => return Err(SafeTensorsError::MalformedHeaderJson),
                }
            } else if b < 0x20 {
                return Err(SafeTensorsError::MalformedHeaderJson);
            } else {
                out.push(b);
            }
            if out.len() > self.limits.max_name_bytes {
                return Err(SafeTensorsError::InvalidIdentifierLength);
            }
        }
        Err(SafeTensorsError::MalformedHeaderJson)
    }

    fn parse_usize(&mut self) -> Result<usize, SafeTensorsError> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(SafeTensorsError::MalformedHeaderJson);
        }
        // Reject leading zeros on multi-digit integers per RFC 8259.
        if self.pos - start > 1 && self.src[start] == b'0' {
            return Err(SafeTensorsError::MalformedHeaderJson);
        }
        let s = core::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| SafeTensorsError::MalformedHeaderJson)?;
        s.parse::<usize>()
            .map_err(|_| SafeTensorsError::ShapeOverflow)
    }

    fn parse_usize_list(&mut self, max_len: usize) -> Result<Vec<usize>, SafeTensorsError> {
        self.expect_byte(b'[')?;
        let mut items = Vec::new();
        if self.peek_byte()? == b']' {
            self.pos += 1;
            return Ok(items);
        }
        loop {
            if items.len() >= max_len {
                return Err(SafeTensorsError::RankExceeded);
            }
            items.push(self.parse_usize()?);
            match self.peek_byte()? {
                b',' => {
                    self.pos += 1;
                }
                b']' => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(SafeTensorsError::MalformedHeaderJson),
            }
        }
        Ok(items)
    }

    fn parse_metadata_map(&mut self) -> Result<BTreeMap<String, String>, SafeTensorsError> {
        self.expect_byte(b'{')?;
        let mut map = BTreeMap::new();
        if self.peek_byte()? == b'}' {
            self.pos += 1;
            return Ok(map);
        }
        loop {
            let k = self.parse_string()?;
            if k.is_empty() {
                return Err(SafeTensorsError::InvalidIdentifierLength);
            }
            self.expect_byte(b':')?;
            let v = self.parse_string()?;
            if map.insert(k, v).is_some() {
                return Err(SafeTensorsError::DuplicateKey);
            }
            match self.peek_byte()? {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(SafeTensorsError::MalformedHeaderJson),
            }
        }
        Ok(map)
    }

    fn parse_tensor_entry(
        &mut self,
        name: String,
    ) -> Result<ParsedSafeTensorEntryV1, SafeTensorsError> {
        self.expect_byte(b'{')?;
        let mut seen_fields = BTreeSet::new();
        let mut dtype: Option<SafeTensorDtypeV1> = None;
        let mut shape: Option<Vec<usize>> = None;
        let mut offsets: Option<(usize, usize)> = None;

        loop {
            let field = self.parse_string()?;
            if !seen_fields.insert(field.clone()) {
                return Err(SafeTensorsError::DuplicateKey);
            }
            self.expect_byte(b':')?;
            match field.as_str() {
                "dtype" => {
                    let dt_str = self.parse_string()?;
                    let dt = SafeTensorDtypeV1::from_str_name(&dt_str)?;
                    if !self.limits.is_dtype_allowed(dt) {
                        return Err(SafeTensorsError::DisallowedDtype);
                    }
                    dtype = Some(dt);
                }
                "shape" => {
                    let sh = self.parse_usize_list(self.limits.max_rank)?;
                    if sh.is_empty() {
                        return Err(SafeTensorsError::RankExceeded);
                    }
                    shape = Some(sh);
                }
                "data_offsets" => {
                    let offs = self
                        .parse_usize_list(2)
                        .map_err(|_| SafeTensorsError::InvalidOffsets)?;
                    if offs.len() != 2 {
                        return Err(SafeTensorsError::InvalidOffsets);
                    }
                    offsets = Some((offs[0], offs[1]));
                }
                _ => return Err(SafeTensorsError::MalformedHeaderJson),
            }

            match self.peek_byte()? {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(SafeTensorsError::MalformedHeaderJson),
            }
        }

        Ok(ParsedSafeTensorEntryV1 {
            name,
            dtype: dtype.ok_or(SafeTensorsError::MalformedHeaderJson)?,
            shape: shape.ok_or(SafeTensorsError::MalformedHeaderJson)?,
            data_offsets: offsets.ok_or(SafeTensorsError::MalformedHeaderJson)?,
        })
    }
}

/// Parse and validate a SafeTensors v1 binary buffer against `limits` (`ADR-0024` §3, `W11-04`).
///
/// Enforces:
/// 1. Pre-parse bounds on `header_len` and `data_section_len`.
/// 2. Strict duplicate-key rejection on `__metadata__`, tensor names, and tensor fields (`F-07`).
/// 3. Checked shape-product $\times$ dtype width equality with `end - begin`.
/// 4. Exact contiguous, non-overlapping coverage from `0` to `data_section.len()`.
pub fn parse_and_validate_safetensors_v1(
    bytes: &[u8],
    limits: &SafeTensorsLimitsV1,
) -> Result<ParsedSafeTensorsHeaderV1, SafeTensorsError> {
    if bytes.len() < 8 {
        return Err(SafeTensorsError::TruncatedBuffer);
    }
    let header_len_u64 = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    if header_len_u64 == 0 || header_len_u64 > (limits.max_header_bytes as u64) {
        return Err(SafeTensorsError::HeaderTooLarge);
    }
    let header_len = header_len_u64 as usize;
    let data_section_offset = 8usize
        .checked_add(header_len)
        .ok_or(SafeTensorsError::HeaderTooLarge)?;
    if bytes.len() < data_section_offset {
        return Err(SafeTensorsError::TruncatedBuffer);
    }
    let data_section_len = bytes.len() - data_section_offset;
    if data_section_len > limits.max_total_data_bytes {
        return Err(SafeTensorsError::DataTooLarge);
    }

    let header_slice = &bytes[8..data_section_offset];
    let _ =
        core::str::from_utf8(header_slice).map_err(|_| SafeTensorsError::MalformedHeaderJson)?;

    let mut cursor = JsonHeaderCursor::new(header_slice, *limits);
    cursor.expect_byte(b'{')?;
    if cursor.peek_byte()? == b'}' {
        return Err(SafeTensorsError::TooManyOrEmptyTensors);
    }

    let mut seen_top_keys = BTreeSet::new();
    let mut metadata: Option<BTreeMap<String, String>> = None;
    let mut tensors = BTreeMap::new();

    loop {
        let key = cursor.parse_string()?;
        if key.is_empty() {
            return Err(SafeTensorsError::InvalidIdentifierLength);
        }
        if !seen_top_keys.insert(key.clone()) {
            return Err(SafeTensorsError::DuplicateKey);
        }
        cursor.expect_byte(b':')?;

        if key == "__metadata__" {
            metadata = Some(cursor.parse_metadata_map()?);
        } else {
            if tensors.len() >= limits.max_tensors {
                return Err(SafeTensorsError::TooManyOrEmptyTensors);
            }
            let entry = cursor.parse_tensor_entry(key.clone())?;
            tensors.insert(key, entry);
        }

        match cursor.peek_byte()? {
            b',' => cursor.pos += 1,
            b'}' => {
                cursor.pos += 1;
                break;
            }
            _ => return Err(SafeTensorsError::MalformedHeaderJson),
        }
    }

    cursor.skip_ws();
    if cursor.pos != header_slice.len() {
        return Err(SafeTensorsError::MalformedHeaderJson);
    }
    if tensors.is_empty() {
        return Err(SafeTensorsError::TooManyOrEmptyTensors);
    }

    // Validate shape products and contiguous non-overlapping offsets [0..data_section_len].
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(tensors.len());
    for entry in tensors.values() {
        let elems = entry
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(SafeTensorsError::ShapeOverflow)?;
        if elems == 0 {
            return Err(SafeTensorsError::InvalidOffsets);
        }
        let expected_bytes = elems
            .checked_mul(entry.dtype.byte_width())
            .ok_or(SafeTensorsError::ShapeOverflow)?;
        let (begin, end) = entry.data_offsets;
        if begin >= end || end > data_section_len || (end - begin) != expected_bytes {
            return Err(SafeTensorsError::InvalidOffsets);
        }
        spans.push((begin, end));
    }

    spans.sort_by_key(|&(b, _)| b);
    if spans[0].0 != 0 {
        return Err(SafeTensorsError::InvalidOffsets);
    }
    for window in spans.windows(2) {
        if window[0].1 != window[1].0 {
            return Err(SafeTensorsError::InvalidOffsets);
        }
    }
    if spans.last().map(|&(_, e)| e) != Some(data_section_len) {
        return Err(SafeTensorsError::InvalidOffsets);
    }

    Ok(ParsedSafeTensorsHeaderV1 {
        metadata: metadata.unwrap_or_default(),
        tensors,
        data_section_offset,
        data_section_len,
    })
}

/// Transactionally validate and load a `SwarmTorchCheckpointV1` SafeTensors buffer into `model`
/// (`ADR-0024` §3, `W11-04`).
///
/// Guarantees all-or-nothing semantics (`F-07`):
///
/// 1. Validates header bounds, duplicate keys, contiguous offsets, and `model_schema_hash`.
/// 2. Stages all decoded `f32` parameters and verifies `.is_finite()` on **every** element
///    before calling `model.load_parameters(&staged)` so `model` is never partially mutated on error.
///
/// Assumes canonical sorted tensor ordering; models with `[W | b]` or `[W1 | b1 | W2 | b2]` flat
/// buffers must use [`import_safetensors_checkpoint_with_layout_v1`] with
/// [`ModelFlatTensorLayoutV1::for_linear`] or [`ModelFlatTensorLayoutV1::for_mlp`].
pub fn import_safetensors_checkpoint_v1<M: SwarmModel>(
    model: &mut M,
    expected_schema: &ModelTensorSchemaV1,
    bytes: &[u8],
    limits: &SafeTensorsLimitsV1,
) -> Result<[u8; 32], SafeTensorsError> {
    let canonical_schema = ModelTensorSchemaV1::try_new(expected_schema.tensors.clone(), limits)?;
    let layout = ModelFlatTensorLayoutV1::canonical(canonical_schema)?;
    import_safetensors_checkpoint_with_layout_v1(model, &layout, bytes, limits)
}

/// Validate and transactionally import `SwarmTorchCheckpointV1` SafeTensors v1 bytes using an
/// explicit [`ModelFlatTensorLayoutV1`] mapping between tensor names and `model.parameters()` (`ST-08`).
pub fn import_safetensors_checkpoint_with_layout_v1<M: SwarmModel>(
    model: &mut M,
    layout: &ModelFlatTensorLayoutV1,
    bytes: &[u8],
    limits: &SafeTensorsLimitsV1,
) -> Result<[u8; 32], SafeTensorsError> {
    let canonical_schema = ModelTensorSchemaV1::try_new(layout.schema.tensors.clone(), limits)?;
    let parsed = parse_and_validate_safetensors_v1(bytes, limits)?;

    let fmt = parsed
        .metadata
        .get("format")
        .ok_or(SafeTensorsError::SchemaMismatch)?;
    if fmt != SWARMTORCH_CHECKPOINT_FORMAT_V1 {
        return Err(SafeTensorsError::SchemaMismatch);
    }

    let expected_hash = canonical_schema.schema_hash();
    let expected_hash_hex = canonical_schema.schema_hash_hex();
    let got_hash_hex = parsed
        .metadata
        .get("model_schema_hash")
        .ok_or(SafeTensorsError::SchemaMismatch)?;
    if got_hash_hex != &expected_hash_hex {
        return Err(SafeTensorsError::SchemaMismatch);
    }

    if parsed.tensors.len() != canonical_schema.tensors.len()
        || layout.ranges_by_schema_order.len() != canonical_schema.tensors.len()
    {
        return Err(SafeTensorsError::SchemaMismatch);
    }

    let total_elems = canonical_schema.total_elements()?;
    if total_elems != layout.total_elements || total_elems != model.num_parameters() {
        return Err(SafeTensorsError::SchemaMismatch);
    }

    let mut staged_params = vec![0.0f32; total_elems];
    let data_section = &bytes[parsed.data_section_offset..];

    for (desc, range) in canonical_schema
        .tensors
        .iter()
        .zip(&layout.ranges_by_schema_order)
    {
        if desc.name != range.name || range.num_elements != desc.num_elements()? {
            return Err(SafeTensorsError::SchemaMismatch);
        }
        let entry = parsed
            .tensors
            .get(&desc.name)
            .ok_or(SafeTensorsError::SchemaMismatch)?;
        if entry.dtype != desc.dtype || entry.shape != desc.shape {
            return Err(SafeTensorsError::SchemaMismatch);
        }
        if entry.dtype != SafeTensorDtypeV1::F32 {
            return Err(SafeTensorsError::DisallowedDtype);
        }
        let (begin, end) = entry.data_offsets;
        let tensor_bytes = &data_section[begin..end];
        let dest_end = range
            .flat_offset
            .checked_add(range.num_elements)
            .ok_or(SafeTensorsError::ShapeOverflow)?;
        let dest_slice = staged_params
            .get_mut(range.flat_offset..dest_end)
            .ok_or(SafeTensorsError::SchemaMismatch)?;
        if tensor_bytes.len() != dest_slice.len() * 4 {
            return Err(SafeTensorsError::SchemaMismatch);
        }
        for (idx, chunk) in tensor_bytes.chunks_exact(4).enumerate() {
            let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !val.is_finite() {
                return Err(SafeTensorsError::NonFiniteParameter);
            }
            dest_slice[idx] = val;
        }
    }

    model
        .load_parameters(&staged_params)
        .map_err(|_| SafeTensorsError::ModelLoadFailed)?;
    Ok(expected_hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simple::LinearModel;

    #[test]
    fn safetensors_checkpoint_v1_roundtrip_and_canonical_schema_hash() {
        let mut model = LinearModel::new(2, 1).unwrap();
        model.load_parameters(&[1.25, -2.5, 0.75]).unwrap();

        let schema = ModelTensorSchemaV1::for_linear(2, 1).unwrap();
        let limits = SafeTensorsLimitsV1::default();
        let bytes = export_safetensors_checkpoint_v1(&model, &schema, &limits).unwrap();

        // Header length must be a multiple of 8 for 8-byte alignment of data buffer.
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        assert_eq!(header_len % 8, 0);

        let mut restored = LinearModel::new(2, 1).unwrap();
        let loaded_hash =
            import_safetensors_checkpoint_v1(&mut restored, &schema, &bytes, &limits).unwrap();
        assert_eq!(loaded_hash, schema.schema_hash());
        assert_eq!(restored.parameters(), &[1.25, -2.5, 0.75]);
    }

    #[test]
    fn safetensors_rejects_duplicate_json_keys_overlapping_offsets_and_nan_transactionally() {
        let limits = SafeTensorsLimitsV1::default();

        // 1. Duplicate tensor key in JSON header (F-07) must be rejected with DuplicateKey!
        let dup_json = b"{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
        let mut dup_buf = Vec::new();
        dup_buf.extend_from_slice(&(dup_json.len() as u64).to_le_bytes());
        dup_buf.extend_from_slice(dup_json);
        dup_buf.extend_from_slice(&1.0f32.to_le_bytes());
        assert_eq!(
            parse_and_validate_safetensors_v1(&dup_buf, &limits).unwrap_err(),
            SafeTensorsError::DuplicateKey
        );

        // 2. Non-contiguous / gap in data_offsets must be rejected with InvalidOffsets!
        let gap_json = b"{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}}";
        let mut gap_buf = Vec::new();
        gap_buf.extend_from_slice(&(gap_json.len() as u64).to_le_bytes());
        gap_buf.extend_from_slice(gap_json);
        gap_buf.extend_from_slice(&[0u8; 12]);
        assert_eq!(
            parse_and_validate_safetensors_v1(&gap_buf, &limits).unwrap_err(),
            SafeTensorsError::InvalidOffsets
        );

        // 3. Transactional NaN rejection: corrupt the last f32 in a valid checkpoint to NaN
        // and verify the target model's parameters remain completely untouched!
        let mut src_model = LinearModel::new(2, 1).unwrap();
        src_model.load_parameters(&[1.0, 2.0, 3.0]).unwrap();
        let schema = ModelTensorSchemaV1::for_linear(2, 1).unwrap();
        let mut bytes = export_safetensors_checkpoint_v1(&src_model, &schema, &limits).unwrap();
        let len = bytes.len();
        bytes[(len - 4)..len].copy_from_slice(&f32::NAN.to_le_bytes());

        let mut target_model = LinearModel::new(2, 1).unwrap();
        target_model.load_parameters(&[9.0, 8.0, 7.0]).unwrap();
        assert_eq!(
            import_safetensors_checkpoint_v1(&mut target_model, &schema, &bytes, &limits)
                .unwrap_err(),
            SafeTensorsError::NonFiniteParameter
        );
        // Target model must still hold [9.0, 8.0, 7.0]!
        assert_eq!(target_model.parameters(), &[9.0, 8.0, 7.0]);
    }

    fn read_tensor_f32s(bytes: &[u8], parsed: &ParsedSafeTensorsHeaderV1, name: &str) -> Vec<f32> {
        let entry = parsed.tensors.get(name).unwrap();
        let (begin, end) = entry.data_offsets;
        bytes[parsed.data_section_offset + begin..parsed.data_section_offset + end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn layout_for_linear_maps_weights_before_bias_despite_alphabetical_sorting() {
        let mut model = LinearModel::new(3, 2).unwrap();
        // Distinct values across all 6 weights [W] and 2 biases [b]
        let expected_weights = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected_bias = [0.25f32, -0.75];
        let mut flat = Vec::from(expected_weights);
        flat.extend_from_slice(&expected_bias);
        model.load_parameters(&flat).unwrap();

        let layout = ModelFlatTensorLayoutV1::for_linear(3, 2).unwrap();
        let limits = SafeTensorsLimitsV1::default();
        let bytes = export_safetensors_checkpoint_with_layout_v1(&model, &layout, &limits).unwrap();

        // Parse raw SafeTensors header and verify "linear.bias" vs "linear.weight" bytes are NOT swapped!
        let parsed = parse_and_validate_safetensors_v1(&bytes, &limits).unwrap();
        assert_eq!(
            read_tensor_f32s(&bytes, &parsed, "linear.bias"),
            expected_bias
        );
        assert_eq!(
            read_tensor_f32s(&bytes, &parsed, "linear.weight"),
            expected_weights
        );
        // Must preserve identical schema hash as ModelTensorSchemaV1::for_linear(3, 2)
        assert_eq!(
            layout.schema().schema_hash(),
            ModelTensorSchemaV1::for_linear(3, 2).unwrap().schema_hash()
        );

        let mut restored = LinearModel::new(3, 2).unwrap();
        import_safetensors_checkpoint_with_layout_v1(&mut restored, &layout, &bytes, &limits)
            .unwrap();
        assert_eq!(restored.parameters(), &flat[..]);
        let input = [0.5f32, -1.0, 2.0];
        let out_orig = model.forward(&input).unwrap();
        let out_restored = restored.forward(&input).unwrap();
        assert_eq!(out_orig.as_slice(), out_restored.as_slice());
    }

    #[test]
    fn layout_for_mlp_maps_fc1_fc2_weights_and_biases_without_swapping() {
        use crate::mlp::MlpModel;
        let mut mlp = MlpModel::new(2, 3, 2).unwrap();
        // [W1(6) | b1(3) | W2(6) | b2(2)] = 17 distinct floats
        let w1 = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let b1 = [-0.1f32, -0.2, -0.3];
        let w2 = [1.1f32, 1.2, 1.3, 1.4, 1.5, 1.6];
        let b2 = [2.25f32, -3.5];
        let mut flat = Vec::new();
        flat.extend_from_slice(&w1);
        flat.extend_from_slice(&b1);
        flat.extend_from_slice(&w2);
        flat.extend_from_slice(&b2);
        mlp.load_parameters(&flat).unwrap();

        let layout = ModelFlatTensorLayoutV1::for_mlp(2, 3, 2).unwrap();
        let limits = SafeTensorsLimitsV1::default();
        let bytes = export_safetensors_checkpoint_with_layout_v1(&mlp, &layout, &limits).unwrap();

        let parsed = parse_and_validate_safetensors_v1(&bytes, &limits).unwrap();
        assert_eq!(read_tensor_f32s(&bytes, &parsed, "mlp.fc1.weight"), w1);
        assert_eq!(read_tensor_f32s(&bytes, &parsed, "mlp.fc1.bias"), b1);
        assert_eq!(read_tensor_f32s(&bytes, &parsed, "mlp.fc2.weight"), w2);
        assert_eq!(read_tensor_f32s(&bytes, &parsed, "mlp.fc2.bias"), b2);

        let mut restored = MlpModel::new(2, 3, 2).unwrap();
        import_safetensors_checkpoint_with_layout_v1(&mut restored, &layout, &bytes, &limits)
            .unwrap();
        assert_eq!(restored.parameters(), &flat[..]);
    }

    #[test]
    fn layout_try_new_rejects_overlaps_gaps_and_mismatched_names() {
        let schema = ModelTensorSchemaV1::for_linear(2, 2).unwrap();
        let limits = SafeTensorsLimitsV1::default();

        // Overlapping offsets (both start at 0)
        let mut overlap = BTreeMap::new();
        overlap.insert("linear.weight".to_string(), 0);
        overlap.insert("linear.bias".to_string(), 0);
        assert_eq!(
            ModelFlatTensorLayoutV1::try_new(schema.clone(), overlap, &limits).unwrap_err(),
            SafeTensorsError::SchemaMismatch
        );

        // Missing key
        let mut missing = BTreeMap::new();
        missing.insert("linear.weight".to_string(), 0);
        assert_eq!(
            ModelFlatTensorLayoutV1::try_new(schema, missing, &limits).unwrap_err(),
            SafeTensorsError::SchemaMismatch
        );
    }
}
