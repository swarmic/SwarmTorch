# `swarm-torch-backends`

Workspace-excluded (`exclude = ["adapters/*"]`, MSRV `1.92`, CPU-only — GPU backends remain deferred) tensor backend adapter crate bridging external deep-learning frameworks (`Burn 0.21.0` via `real-burn` `Autodiff<NdArray<f32>>`, `Candle 0.10.2` via `real-candle`) into `swarm_torch_models::backend_bridge::BackendTensorBridge` without adding heavyweight framework crates or CUDA/WGPU dependency trees to root `Cargo.lock` or violating `no_std` guarantees.

## Integration

Use the built-in linear and 2-layer ReLU MLP bridges (`BurnLinearBridgeV1`, `BurnMlpBridgeV1`, `CandleLinearBridgeV1`, `CandleMlpBridgeV1`, plus `ReferenceLinearModule`), or implement `BurnModuleOps` / `CandleModuleOps` on your downstream model container and wrap it in `BurnModuleBridgeV1` or `CandleModuleBridgeV1` + `BackendModelHandleV1::try_new(...)`. All tensor transfers are checked in canonical UTF-8 sorted `ModelTensorSchemaV1` order with `model_schema_hash` verification, pre-set `dtype`/`device`/`shape` checks, and `.is_finite()` gating before commit.
