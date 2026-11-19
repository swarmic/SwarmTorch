# `swarm-torch-backends`

Workspace-excluded (`exclude = ["adapters/*"]`, MSRV `1.92`, CPU-only — GPU backends remain deferred) tensor backend adapter crate bridging external deep-learning frameworks (`Burn 0.21.0` via `real-burn` `Autodiff<NdArray<f32>>`, `Candle 0.10.2` via `real-candle`) into `swarm_torch_models::backend_bridge::BackendTensorBridge` without adding heavyweight framework crates or CUDA/WGPU dependency trees to root `Cargo.lock` or violating `no_std` guarantees.

## Integration

Use the built-in linear and 2-layer ReLU MLP bridges (`BurnLinearBridgeV1`, `BurnMlpBridgeV1`, `CandleLinearBridgeV1`, `CandleMlpBridgeV1`, plus `ReferenceLinearModule`), or implement `BurnModuleOps` / `CandleModuleOps` on your downstream model container and wrap it in `BurnModuleBridgeV1` or `CandleModuleBridgeV1` + `BackendModelHandleV1::try_new(...)`. Flat `f32` parameter/gradient transfers follow canonical declaration layout (`ModelFlatTensorLayoutV1`: `[W | b]` for linear, `[W_1 | b_1 | W_2 | b_2]` for 2-layer MLP) while `ModelTensorSchemaV1` enforces canonical UTF-8 tensor ordering for `model_schema_hash` and SafeTensors interchange, with `.is_finite()` gating before commit (plus pre-set `dtype`/`device`/`shape`/contiguity validation on `real-candle`'s `VarMap` path).
