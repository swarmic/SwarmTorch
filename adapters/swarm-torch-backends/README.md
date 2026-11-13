# `swarm-torch-backends`

Workspace-excluded (`exclude = ["adapters/*"]`) tensor backend adapter crate bridging external deep-learning frameworks (`Burn`, `Candle`) into `swarm_torch_models::backend_bridge::BackendTensorBridge` without adding heavyweight framework crates or CUDA/WGPU dependency trees to root `Cargo.lock` or violating `no_std` guarantees.

## Integration

Implement `BurnModuleOps` or `CandleModuleOps` on your downstream model container (e.g., `burn::module::Module` state mapper or `candle_nn::VarMap`) and wrap it in `BurnModuleBridgeV1` or `CandleModuleBridgeV1` + `BackendModelHandleV1::try_new(...)`. All tensor transfers are checked in canonical UTF-8 sorted `ModelTensorSchemaV1` order with `model_schema_hash` verification and `.is_finite()` gating before device commit.
