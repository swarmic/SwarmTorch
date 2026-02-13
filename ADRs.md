# Architecture Decision Records (ADRs)

## Index

### Foundation (Phase 0)

- [ADR-0000: ADR Process and Template](#adr-0000-adr-process-and-template)
- [ADR-0001: Autodiff and Tensor Backend Strategy](#adr-0001-autodiff-and-tensor-backend-strategy)
- [ADR-0002: Crate Topology, Feature Flag Policy, and Embedded Profiles](#adr-0002-crate-topology-feature-flag-policy-and-embedded-profiles)
- [ADR-0003: Runtime Abstraction (Embassy ↔ Tokio)](#adr-0003-runtime-abstraction-embassy--tokio)

### Networking & Distribution (Phase 1)

- [ADR-0004: Network Transport Layer](#adr-0004-network-transport-layer)
- [ADR-0004A: Topology Policy as First-Class Primitive](#adr-0004a-topology-policy-as-first-class-primitive)
- [ADR-0004B: Asynchrony and Staleness Model](#adr-0004b-asynchrony-and-staleness-model)
- [ADR-0005: State Serialization and Message Formats](#adr-0005-state-serialization-and-message-formats)
- [ADR-0006: Identity and Membership Model](#adr-0006-identity-and-membership-model)
- [ADR-0006A: Coordination Protocol (Gossip-Based Eventual Consistency)](#adr-0006a-coordination-protocol-gossip-based-eventual-consistency)

### Security & Robustness (Phase 2)

- [ADR-0007: Robust Aggregation Strategy](#adr-0007-robust-aggregation-strategy)
- [ADR-0007A: Aggregation Validation and Attack Harness](#adr-0007a-aggregation-validation-and-attack-harness)
- [ADR-0008: Threat Model and Trust Boundaries](#adr-0008-threat-model-and-trust-boundaries)
- [ADR-0008A: Identity and Sybil Resistance Boundary](#adr-0008a-identity-and-sybil-resistance-boundary)
- [ADR-0008B: Message Envelope Sender Identity Contract](#adr-0008b-message-envelope-sender-identity-contract)

### Interoperability (Phase 3)

- [ADR-0009: Python Interoperability (PyO3) and Model Portability Contract](#adr-0009-python-interoperability-pyo3-and-model-portability-contract)
- [ADR-0010: Model Portability and ONNX Integration (Inference Interchange)](#adr-0010-model-portability-and-onnx-integration-inference-interchange)

### Operations (Phase 4)

- [ADR-0011: CI/CD Build Matrix and Artifact Strategy](#adr-0011-cicd-build-matrix-and-artifact-strategy)
- [ADR-0012: Observability and Telemetry](#adr-0012-observability-and-telemetry)
- [ADR-0013: Reproducibility and Determinism](#adr-0013-reproducibility-and-determinism)
- [ADR-0014: Benchmark, Dataset, and Release Gates](#adr-0014-benchmark-dataset-and-release-gates)
- [ADR-0016: Run Artifacts and Visualization Surface](#adr-0016-run-artifacts-and-visualization-surface)
- [ADR-0017: Data Pipeline DSL and Asset Model](#adr-0017-data-pipeline-dsl-and-asset-model)
- [ADR-0018: Extension and Execution Policy](#adr-0018-extension-and-execution-policy)

### Acceleration (Phase 5)

- [ADR-0015: GPU Acceleration Strategy (WGPU-first, CUDA Optional)](#adr-0015-gpu-acceleration-strategy-wgpu-first-cuda-optional)

-----

## ADR-0000: ADR Process and Template

**Status:** Accepted  
**Date:** 2025-01-08  
**Deciders:** Core Team

### Context

Architecture decisions need to be documented, traceable, and revisitable. As SwarmTorch evolves from prototype to production, we need a lightweight process that:

- Captures *why* decisions were made, not just *what*
- Allows new contributors to understand historical context
- Enables rational revision when circumstances change
- Prevents rehashing settled debates

### Decision

We adopt Architecture Decision Records (ADRs) following the pattern established by Michael Nygard:

- All ADRs are consolidated in this file (`ADRs.md`)
- Numbered sequentially: `ADR-NNNN`
- Immutable once accepted (superceded by new ADRs, not edited)
- Mandatory sections: Context, Decision, Consequences, Alternatives Considered

**ADR lifecycle states:**

- **Proposed** - Under discussion
- **Accepted** - Active and implemented
- **Deprecated** - Superseded by newer ADR
- **Rejected** - Considered but not adopted

### Template

```markdown
# ADR-NNNN: Title

**Status:** [Proposed | Accepted | Deprecated | Rejected]  
**Date:** YYYY-MM-DD  
**Deciders:** [Names/Roles]  
**Supersedes:** [ADR-XXXX] (if applicable)  
**Superseded by:** [ADR-YYYY] (if deprecated)  

## Context

What is the issue we're facing? What constraints exist?
What are the forces at play (technical, business, social)?

## Decision

What are we doing? Be specific and actionable.

## Consequences

### Positive
- What improves?
- What becomes easier?

### Negative
- What becomes harder?
- What tradeoffs are we accepting?

### Neutral
- What changes that's neither good nor bad?

## Alternatives Considered

### Alternative 1: [Name]
- **Description:** Brief summary
- **Pros:** Advantages
- **Cons:** Disadvantages
- **Why not chosen:** Specific reason

### Alternative 2: [Name]
...

## Implementation Notes

Practical guidance for implementers.

## References
- Links to papers, RFCs, prior art
```

### Consequences

**Positive:**

- Decisions are documented and searchable
- New contributors can understand historical context
- Architectural drift becomes visible
- Prevents “why did we do this?” conversations

**Negative:**

- Overhead of writing ADRs (mitigated by template)
- Risk of ADRs becoming stale if not maintained

**Neutral:**

- ADRs are not design specs; they complement PRDs and code comments

-----

## ADR-0001: Autodiff and Tensor Backend Strategy

**Status:** Accepted  
**Date:** 2025-01-08  
**Deciders:** Core Team

### Context

SwarmTorch requires a tensor computation and automatic differentiation backend. The Rust ecosystem has several options:

|Backend    |Maturity        |no_std Support |Performance       |Ecosystem            |
|-----------|----------------|---------------|------------------|---------------------|
|**Burn**   |Production-ready|✅ Yes (partial)|High              |Growing, modular     |
|**Candle** |Production-ready|❌ No           |Very High         |HuggingFace ecosystem|
|**tch-rs** |Mature          |❌ No           |Highest (LibTorch)|PyTorch interop      |
|**dfdx**   |Experimental    |⚠️ Limited      |Medium            |Compile-time shapes  |
|**ndarray**|Mature          |✅ Yes          |Medium            |No autodiff          |

**Key requirements:**

1. **Embedded support:** Must compile for `no_std` + `alloc` targets (ESP32, STM32)
1. **Portability:** Same model code should work on server and microcontroller
1. **Interoperability:** Need path to PyTorch model import/export
1. **Performance:** Competitive with PyTorch for edge-class devices
1. **Maintainability:** Active development, not a research project

### Decision

**Primary backend: Burn**

- Use Burn as the default tensor/autodiff backend for SwarmTorch
- Leverage Burn’s backend abstraction (WGPU, NdArray, LibTorch, Candle)
- Provide thin wrapper traits for SwarmTorch-specific optimizations

**Secondary support: tch-rs for interop**

- Offer `tch-backend` feature flag for PyTorch model compatibility
- Use for model import/export, not training loops

**Custom primitives where needed:**

- Implement specialized ops (robust aggregation, gradient compression) directly
- Don’t depend on backend for swarm-specific algorithms

### Architecture

```rust
// Core abstraction - backend-agnostic
pub trait SwarmModel<B: Backend> {
    type Input;
    type Output;
    
    fn forward(&self, input: Self::Input) -> Self::Output;
    fn parameters(&self) -> Vec<Param<B>>;
    fn load_state(&mut self, state: ModelState<B>);
}

// Default implementation for Burn models
impl<M: burn::module::Module<B>, B: Backend> SwarmModel<B> for M {
    // Auto-implement for any Burn model
}

// User-facing API is backend-generic
use swarm_torch::prelude::*;

let model = SimpleNet::<NdArray>::new(784, 128, 10);
swarm.train(model, optimizer, data).await?;
```

### Consequences

**Positive:**

- Burn’s modular backend system aligns with our portability goals
- Active development and responsive maintainers
- Growing ecosystem (HuggingFace Candle-Burn bridge exists)
- `no_std` support is a first-class goal for Burn team
- Can swap backends without changing model code

**Negative:**

- Burn is younger than PyTorch/TensorFlow (less battle-tested)
- Some ops may be missing (we’ll contribute upstream)
- Performance may lag LibTorch for some workloads
- Two-backend support increases maintenance burden

**Neutral:**

- Users can bring their own backend via trait implementation
- We’re not locked in; backends are pluggable

### Alternatives Considered

#### Alternative 1: Candle (HuggingFace)

**Pros:**

- Excellent performance (optimized Metal, CUDA, CPU)
- Growing model zoo
- HuggingFace backing

**Cons:**

- No `no_std` support (fundamental design choice)
- Less modular than Burn (monolithic backend)
- Would require separate embedded implementation

**Why not chosen:** Embedded support is non-negotiable for SwarmTorch’s robotics/IoT focus.

#### Alternative 2: tch-rs (LibTorch bindings)

**Pros:**

- Highest performance (native PyTorch)
- Full PyTorch model compatibility
- Mature and stable

**Cons:**

- No `no_std` support (C++ runtime required)
- Large binary size (50MB+ base)
- Python-style API doesn’t leverage Rust type system

**Why not chosen:** Can’t run on embedded targets. We use tch-rs for *interop only*, not primary backend.

#### Alternative 3: Build custom backend

**Pros:**

- Complete control
- Optimized for swarm learning workloads

**Cons:**

- Years of engineering effort
- Unlikely to match mature backends
- Diverts resources from core swarm learning features

**Why not chosen:** Reinventing the wheel. Burn provides 80% of what we need; we contribute the remaining 20%.

### Implementation Notes

**Phase 1 (v0.1):**

- All examples use `burn::backend::NdArray` (pure Rust, no_std compatible)
- Wrap Burn models in `SwarmModel` trait
- Test on x86_64, aarch64, and STM32 simulator

**Phase 2 (v0.2):**

- Add `tch-backend` feature for PyTorch interop
- Implement model import/export (PyTorch ↔ Burn)
- Benchmark performance vs PyTorch DDP

**Phase 3 (v0.3):**

- Contribute missing ops to Burn upstream
- Optimize critical paths (gradient aggregation, communication)
- Explore Burn’s WGPU backend for GPU acceleration

### References

- [Burn Framework](https://github.com/tracel-ai/burn)
- [Candle Framework](https://github.com/huggingface/candle)
- [tch-rs](https://github.com/LaurentMazare/tch-rs)
- [Embassy no_std async](https://embassy.dev/)

-----

## ADR-0002: Crate Topology, Feature Flag Policy, and Embedded Profiles

**Status:** Accepted  
**Date:** 2025-01-08 (Updated 2026-01-09)  
**Deciders:** Core Team

### Context

SwarmTorch must support radically different deployment environments:

- **Servers/desktops:** Full std library, Tokio async, GB of RAM
- **Edge gateways:** Linux/std, limited resources, WiFi/Ethernet
- **Microcontrollers:** no_std, Embassy async, KB of RAM, LoRa/BLE

A monolithic crate cannot satisfy all these constraints. We need a crate structure that:

1. Allows minimal embedded builds (<50KB binary)
1. Shares core logic across std and no_std
1. Prevents accidental dependencies (e.g., `tokio` in no_std build)
1. Provides clear upgrade paths (embedded → edge → server)

**Critical clarification:** "no_std" encompasses multiple distinct profiles with different capabilities and type constraints. We must be explicit about which profile is targeted.

### Decision

**Workspace structure:**

```
swarm-torch/
├── swarm-torch-core/          # no_std compatible, minimal
│   ├── algorithms/            # PSO, ACO, Firefly (pure math)
│   ├── aggregation/           # Robust aggregators (no I/O)
│   └── traits/                # Core abstractions
├── swarm-torch-net/           # Network transport abstraction
│   ├── transports/            # TCP, UDP, BLE, LoRa impls
│   └── protocols/             # Message framing, serialization
├── swarm-torch-runtime/       # Async runtime glue
│   ├── tokio/                 # Server/edge runtime
│   └── embassy/               # Embedded runtime
├── swarm-torch-models/        # Model utilities & zoo
│   ├── burn-integration/      # Burn backend wrappers
│   └── tch-integration/       # PyTorch interop (opt-in)
├── swarm-torch/               # Main crate (re-exports)
└── swarm-torch-python/        # PyO3 bindings (separate)
```

**Explicit Embedded Profiles (First-Class Build Targets):**

| Profile | Feature Flags | Allocator | Types Available | Target Examples |
|---------|---------------|-----------|-----------------|-----------------|
| `embedded_min` | `no_std`, no `alloc` | None | `heapless::Vec<T, N>`, fixed-size arrays | STM32L0, nRF52810 (≤64KB RAM) |
| `embedded_alloc` | `no_std` + `alloc` | Required | `Vec<u8>`, `Box<dyn Trait>` | ESP32, STM32H7 (≥256KB RAM) |
| `edge_std` | `std` | Standard | Full Rust std library | Raspberry Pi, Jetson, Linux gateways |

**Profile-aware type definitions:**

```rust
// swarm-torch-core/src/types.rs

/// Message payload - profile-aware
#[cfg(not(feature = "alloc"))]
pub type MessagePayload = heapless::Vec<u8, 256>;  // Fixed max size

#[cfg(all(feature = "alloc", not(feature = "std")))]
pub type MessagePayload = alloc::vec::Vec<u8>;

#[cfg(feature = "std")]
pub type MessagePayload = Vec<u8>;

/// Gradient buffer - profile-aware  
#[cfg(not(feature = "alloc"))]
pub type GradientBuffer = heapless::Vec<f32, 1024>;  // Max 1024 parameters

#[cfg(feature = "alloc")]
pub type GradientBuffer = Vec<f32>;
```

**Feature flag hierarchy:**

```toml
[features]
# Core features
default = ["std", "tokio-runtime", "burn-backend"]
std = ["swarm-torch-core/std", "swarm-torch-net/std"]
alloc = ["swarm-torch-core/alloc"]

# Runtime selection (mutually exclusive via cfg)
tokio-runtime = ["tokio", "swarm-torch-runtime/tokio"]
embassy-runtime = ["embassy-executor", "swarm-torch-runtime/embassy"]

# Backend selection
burn-backend = ["burn", "swarm-torch-models/burn"]
tch-backend = ["tch", "swarm-torch-models/tch"]

# Transport selection
tcp-transport = ["swarm-torch-net/tcp"]
udp-transport = ["swarm-torch-net/udp"]
ble-transport = ["swarm-torch-net/ble"]
lora-transport = ["swarm-torch-net/lora"]

# Aggregation algorithms
robust-aggregation = ["swarm-torch-core/krum", "swarm-torch-core/bulyan"]

# Python bindings
python = ["pyo3", "swarm-torch-python"]

# Explicit profile presets (convenience)
profile-embedded-min = []  # no_std, no alloc
profile-embedded-alloc = ["alloc"]  # no_std + alloc
profile-edge-std = ["std"]  # full std
```

**Dependency cascade (minimal → maximal):**

```
embedded_min (no_std, no alloc):
  swarm-torch-core (no default features)
  + swarm-torch-net (ble-transport OR lora-transport)
  + swarm-torch-runtime (embassy-runtime)
  Types: heapless::Vec, fixed arrays, no Box<dyn>
  
embedded_alloc (no_std + alloc):
  swarm-torch-core (alloc feature)
  + swarm-torch-net (ble-transport OR lora-transport)  
  + swarm-torch-runtime (embassy-runtime)
  Types: Vec<u8>, Box<dyn Trait>, Arc (via alloc)

edge_std (std, resource-constrained):
  swarm-torch-core (std)
  + swarm-torch-net (tcp-transport, udp-transport)
  + swarm-torch-runtime (tokio-runtime)
  + swarm-torch-models (burn-backend)
  Types: Full std library

Server (full-featured):
  swarm-torch (default features)
  + all transports, all aggregators, all backends
```

**Trait design constraints for embedded compatibility:**

1. **Avoid `async_trait` in core traits** where embedded matters; prefer GATs or explicit `impl Future` types
2. **No `Arc<Mutex<...>>` in core** - use ownership patterns or platform-specific synchronization
3. **Transport trait must work with both** `heapless::Vec` and `Vec` payloads

```rust
// swarm-torch-net/src/traits.rs
// Good: GAT-based trait (works in no_std without async_trait)
pub trait SwarmTransport {
    type SendFuture<'a>: Future<Output = Result<()>> + 'a where Self: 'a;
    type RecvFuture<'a>: Future<Output = Result<(PeerId, MessagePayload)>> + 'a where Self: 'a;
    
    fn send<'a>(&'a self, peer: PeerId, msg: &'a [u8]) -> Self::SendFuture<'a>;
    fn recv<'a>(&'a self) -> Self::RecvFuture<'a>;
}
```

### Consequences

**Positive:**

- Clear separation of concerns (algorithm logic vs I/O vs runtime)
- Minimal embedded builds possible (just `swarm-torch-core`)
- Feature flags prevent accidental bloat
- Easy to add new backends/transports without touching core
- Workspace allows independent versioning (stable core, experimental net)
- **Explicit profiles prevent "it compiles but doesn't work" scenarios**
- **Type system enforces profile constraints at compile time**

**Negative:**

- More complex for contributors (multiple crates to navigate)
- Feature flag combinations explode (must test subset in CI)
- Re-export strategy needed to keep main crate ergonomic
- Documentation must explain crate purposes
- **More complex trait definitions to support both heapless and Vec**

**Neutral:**

- Users importing `swarm-torch` get "batteries included"”
- Embedded users import `swarm-torch-core` directly

### Alternatives Considered

#### Alternative 1: Monolithic crate with feature flags

**Pros:**

- Simpler for contributors (one crate)
- Easier to document

**Cons:**

- Embedded users pull in server dependencies
- Harder to maintain no_std guarantees
- Feature flag hell (all combinations in one crate)

**Why not chosen:** Cannot achieve <50KB embedded builds with monolithic approach.

#### Alternative 2: Separate repositories per target

**Pros:**

- Complete isolation
- Independent versioning

**Cons:**

- Code duplication (algorithms duplicated across repos)
- Nightmare to keep in sync
- Fragments community

**Why not chosen:** Overkill. Workspace provides enough isolation.

#### Alternative 3: Single core crate + platform-specific facades

**Pros:**

- Core logic shared
- Platform differences isolated

**Cons:**

- Facades duplicate API surface
- Confusing for users (which crate to import?)

**Why not chosen:** Workspace + feature flags achieves same goal with less duplication.

### Implementation Notes

**Crate maturity roadmap:**

|Crate                |v0.1          |v0.2          |v0.3    |v1.0  |
|---------------------|--------------|--------------|--------|------|
|`swarm-torch-core`   |✅ Stable      |Stable        |Stable  |Stable|
|`swarm-torch-net`    |⚠️ Experimental|✅ Stable      |Stable  |Stable|
|`swarm-torch-runtime`|⚠️ Experimental|⚠️ Experimental|✅ Stable|Stable|
|`swarm-torch-models` |✅ Stable      |Stable        |Stable  |Stable|
|`swarm-torch`        |⚠️ Alpha       |⚠️ Beta        |✅ RC    |Stable|

**CI must test:**

```bash
# Minimal no_std build
cargo build -p swarm-torch-core --no-default-features --target thumbv7em-none-eabihf

# Minimal std build (no alloc)
cargo build -p swarm-torch-core --no-default-features --features std

# Full desktop build
cargo build -p swarm-torch --all-features

# Each transport in isolation
cargo build -p swarm-torch-net --no-default-features --features tcp-transport
```

**Re-export strategy:**

```rust
// swarm-torch/src/lib.rs
pub use swarm_torch_core::{algorithms, aggregation, traits};

#[cfg(feature = "tokio-runtime")]
pub use swarm_torch_runtime::tokio::*;

#[cfg(feature = "embassy-runtime")]
pub use swarm_torch_runtime::embassy::*;

pub mod prelude {
    pub use crate::algorithms::*;
    pub use crate::traits::*;
    // One-stop import for users
}
```

### References

- [Rust Feature Flags Best Practices](https://doc.rust-lang.org/cargo/reference/features.html)
- [Embassy Documentation](https://embassy.dev/book/)
- [no_std Rust](https://docs.rust-embedded.org/book/intro/no-std.html)

-----

## ADR-0003: Runtime Abstraction (Embassy ↔ Tokio)

**Status:** Accepted  
**Date:** 2025-01-08  
**Deciders:** Core Team

### Context

SwarmTorch requires async I/O for network communication and task coordination. However, the async runtime differs drastically between environments:

- **Servers/Edge (std):** Tokio is the dominant, production-grade runtime
- **Embedded (no_std):** Embassy is purpose-built for microcontrollers

These runtimes have incompatible APIs:

```rust
// Tokio
tokio::spawn(async { /* task */ });
tokio::time::sleep(Duration::from_secs(1)).await;

// Embassy
embassy_executor::Spawner::spawn(/* static task */);
embassy_time::Timer::after(Duration::from_secs(1)).await;
```

We need an abstraction that:

1. Allows SwarmTorch core logic to be runtime-agnostic
1. Compiles efficiently for both runtimes (zero-cost abstraction)
1. Doesn’t force users into a “lowest common denominator” API
1. Enables testing without real async runtimes

### Decision

**Use trait-based abstraction with compile-time dispatch:**

```rust
// swarm-torch-runtime/src/traits.rs
#[cfg_attr(feature = "tokio-runtime", async_trait::async_trait)]
pub trait SwarmRuntime: Send + Sync + 'static {
    type Instant: Ord + Copy;
    type Sleep: Future<Output = ()>;
    
    fn now() -> Self::Instant;
    fn sleep(duration: Duration) -> Self::Sleep;
    fn spawn<F>(&self, future: F) where F: Future<Output = ()> + Send + 'static;
}

// Tokio implementation (std)
#[cfg(feature = "tokio-runtime")]
pub struct TokioRuntime;

#[cfg(feature = "tokio-runtime")]
#[async_trait::async_trait]
impl SwarmRuntime for TokioRuntime {
    type Instant = tokio::time::Instant;
    type Sleep = tokio::time::Sleep;
    
    fn now() -> Self::Instant {
        tokio::time::Instant::now()
    }
    
    fn sleep(duration: Duration) -> Self::Sleep {
        tokio::time::sleep(duration)
    }
    
    fn spawn<F>(&self, future: F) where F: Future<Output = ()> + Send + 'static {
        tokio::spawn(future);
    }
}

// Embassy implementation (no_std)
#[cfg(feature = "embassy-runtime")]
pub struct EmbassyRuntime<'a> {
    spawner: embassy_executor::Spawner,
    _phantom: PhantomData<&'a ()>,
}

#[cfg(feature = "embassy-runtime")]
impl SwarmRuntime for EmbassyRuntime<'_> {
    type Instant = embassy_time::Instant;
    type Sleep = embassy_time::Timer;
    
    fn now() -> Self::Instant {
        embassy_time::Instant::now()
    }
    
    fn sleep(duration: Duration) -> Self::Sleep {
        embassy_time::Timer::after(duration)
    }
    
    fn spawn<F>(&self, future: F) where F: Future<Output = ()> + Send + 'static {
        // Embassy requires static tasks; see implementation notes
        self.spawner.spawn(/* ... */);
    }
}
```

**User-facing API stays simple:**

```rust
use swarm_torch::prelude::*;

#[tokio::main]  // or #[embassy_executor::main]
async fn main() -> Result<()> {
    let swarm = SwarmCluster::builder()
        .runtime(SwarmRuntime::default())  // Auto-selects based on feature flag
        .build()
        .await?;
    
    swarm.train(model, optimizer, data).await?;
    Ok(())
}
```

### Consequences

**Positive:**

- Core logic is runtime-agnostic (can test with mock runtime)
- Zero runtime overhead (trait dispatch is compile-time)
- Users can bring their own runtime via trait implementation
- Clear separation: runtime in `swarm-torch-runtime`, logic in `swarm-torch-core`

**Negative:**

- Embassy’s static task requirements leak into abstraction (see implementation notes)
- `async_trait` adds some complexity (proc macro, trait objects)
- Two code paths to maintain (Tokio vs Embassy)

**Neutral:**

- Most users never touch runtime abstraction (default “just works”)
- Advanced users can customize (e.g., custom schedulers)

### Alternatives Considered

#### Alternative 1: Tokio-only, no embedded support

**Pros:**

- Simpler codebase (one runtime)
- Mature ecosystem

**Cons:**

- Abandons embedded use case (core mission)
- Can’t run on ESP32, STM32, etc.

**Why not chosen:** Embedded support is a primary differentiator for SwarmTorch.

#### Alternative 2: Async agnostic via callbacks

**Pros:**

- No async/await, no runtime dependency
- Maximum portability

**Cons:**

- Callback hell for complex coordination
- No async/await ergonomics
- Hard to compose async operations

**Why not chosen:** Modern Rust expects async/await. Callbacks are a step backwards.

#### Alternative 3: Runtime as generic parameter

```rust
pub struct SwarmCluster<R: SwarmRuntime> { ... }
```

**Pros:**

- Maximum compile-time optimization
- No dynamic dispatch

**Cons:**

- Generic pollution throughout codebase
- Users must thread `<R>` everywhere
- Error messages become unreadable

**Why not chosen:** Ergonomics matter. Most users don’t care about runtime internals.

### Implementation Notes

#### Embassy Static Task Challenge

Embassy requires tasks to be `'static` due to its executor design:

```rust
#[embassy_executor::task]
async fn my_task() { /* ... */ }

spawner.spawn(my_task()).unwrap();
```

**Our solution:** Task pool pattern

```rust
// swarm-torch-runtime/src/embassy/mod.rs
use embassy_executor::Spawner;
use static_cell::StaticCell;

static TASK_POOL: StaticCell<[TaskSlot; 16]> = StaticCell::new();

pub struct EmbassyRuntime<'a> {
    spawner: Spawner,
    task_pool: &'a [TaskSlot],
}

impl EmbassyRuntime<'_> {
    pub fn spawn<F>(&self, future: F) 
    where F: Future<Output = ()> + Send + 'static 
    {
        // Find free slot in pool, spawn there
        // If pool exhausted, return error (not panic)
    }
}
```

**Trade-off:** Limited concurrent tasks on embedded (16-32 typical), but explicit and predictable.

#### Testing with Mock Runtime

```rust
// swarm-torch-runtime/src/mock.rs
pub struct MockRuntime {
    tasks: Arc<Mutex<Vec<BoxFuture<'static, ()>>>>,
}

impl SwarmRuntime for MockRuntime {
    fn spawn<F>(&self, future: F) {
        self.tasks.lock().unwrap().push(Box::pin(future));
    }
}

impl MockRuntime {
    pub async fn run_all_tasks(&self) {
        // Execute all tasks in deterministic order
        // Useful for testing without real async runtime
    }
}
```

#### Performance Considerations

**Tokio (std):**

- Work-stealing scheduler, hundreds of thousands of tasks
- Efficient for bursty network I/O
- ~1-2ms task wake latency

**Embassy (no_std):**

- Cooperative scheduling, ~10-50 tasks
- Optimized for low-power embedded (WFI sleep)
- ~100µs-1ms task wake latency (depends on MCU)

**Design implication:** SwarmTorch cannot assume “infinite tasks”. Algorithms must be task-efficient.

### References

- [Tokio Documentation](https://tokio.rs/)
- [Embassy Book](https://embassy.dev/book/)
- [async-trait](https://docs.rs/async-trait)
- [static_cell](https://docs.rs/static_cell) - For Embassy static allocation

-----

## ADR-0004: Network Transport Layer

**Status:** Accepted  
**Date:** 2025-01-08  
**Deciders:** Core Team

### Context

SwarmTorch must communicate across radically different network environments:

|Environment |Transport   |Bandwidth  |Reliability|Range|Power    |
|------------|------------|-----------|-----------|-----|---------|
|Datacenter  |TCP/Ethernet|Gbps       |99.999%    |km   |N/A      |
|Edge gateway|WiFi        |100+ Mbps  |99%        |100m |Moderate |
|IoT sensors |BLE         |1 Mbps     |90%        |10m  |Ultra-low|
|Remote farm |LoRa        |0.3-50 kbps|80%        |15km |Ultra-low|

No single transport abstraction can hide these differences. Users need:

1. **Unified API** for basic operations (send, recv, broadcast)
1. **Transport-specific tuning** (BLE scan windows, LoRa duty cycles)
1. **Graceful degradation** (fall back from WiFi to LoRa)
1. **Testability** (simulate transports without hardware)

**Key design tension:** Abstraction vs exposure of transport realities.

### Decision

**Two-level abstraction:**

#### Level 1: Core Transport Trait (minimal, portable)

```rust
// swarm-torch-net/src/traits.rs
#[async_trait]
pub trait SwarmTransport: Send + Sync {
    /// Send message to specific peer
    async fn send(&self, peer: PeerId, msg: &[u8]) -> Result<()>;
    
    /// Receive next message (blocking until available)
    async fn recv(&self) -> Result<(PeerId, Vec<u8>)>;
    
    /// Broadcast to all known peers (best-effort)
    async fn broadcast(&self, msg: &[u8]) -> Result<BroadcastStats>;
    
    /// Discover peers in network
    async fn discover(&self) -> Result<Vec<PeerId>>;
    
    /// Transport capabilities (reliability, bandwidth class)
    fn capabilities(&self) -> TransportCapabilities;
}

pub struct TransportCapabilities {
    pub reliability: ReliabilityClass,
    pub bandwidth_class: BandwidthClass,
    pub max_message_size: usize,
    pub supports_multicast: bool,
}

pub enum ReliabilityClass {
    BestEffort,      // LoRa, BLE
    AtLeastOnce,     // UDP with retries
    Reliable,        // TCP
}

pub enum BandwidthClass {
    UltraLow,   // <10 kbps (LoRa)
    Low,        // 10-1000 kbps (BLE)
    Medium,     // 1-100 Mbps (WiFi)
    High,       // >100 Mbps (Ethernet, datacenter)
}
```

#### Level 2: Transport-Specific Extensions (power users)

```rust
// swarm-torch-net/src/transports/lora.rs
pub struct LoRaTransport {
    config: LoRaConfig,
    // ...
}

impl LoRaTransport {
    pub fn set_spreading_factor(&mut self, sf: u8) { /* ... */ }
    pub fn set_duty_cycle_limit(&mut self, percent: f32) { /* ... */ }
    pub async fn estimate_airtime(&self, payload_len: usize) -> Duration { /* ... */ }
}

impl SwarmTransport for LoRaTransport {
    // Implement core trait
}
```

**Multi-transport support:**

```rust
// swarm-torch-net/src/multi.rs
pub struct MultiTransport {
    transports: Vec<(Priority, Box<dyn SwarmTransport>)>,
    fallback_policy: FallbackPolicy,
}

impl MultiTransport {
    pub fn new() -> Self { /* ... */ }
    
    pub fn add<T: SwarmTransport + 'static>(
        mut self, 
        transport: T, 
        priority: Priority
    ) -> Self { /* ... */ }
}

impl SwarmTransport for MultiTransport {
    async fn send(&self, peer: PeerId, msg: &[u8]) -> Result<()> {
        // Try transports in priority order
        for (_, transport) in &self.transports {
            match transport.send(peer, msg).await {
                Ok(()) => return Ok(()),
                Err(e) if e.is_retryable() => continue,
                Err(e) => return Err(e),
            }
        }
        Err(Error::AllTransportsFailed)
    }
}
```

**Message framing and serialization:**

> Historical design sketch from early planning.
> Current canonical envelope contract is ADR-0008B and `swarm-torch-net/src/protocol.rs`.

```rust
// swarm-torch-net/src/framing.rs
pub struct SwarmMessage {
    pub version: u8,
    pub sender: PeerId,
    pub payload: MessagePayload,
    pub signature: Option<Signature>,
}

pub enum MessagePayload {
    GradientUpdate(CompressedGradient),
    ModelState(ModelCheckpoint),
    ConsensusVote(Vote),
    Heartbeat,
    PeerDiscovery,
}

// Serialization uses postcard (no_std compatible, compact)
impl SwarmMessage {
    pub fn serialize(&self) -> Result<Vec<u8>> {
        postcard::to_allocvec(self).map_err(Into::into)
    }
    
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        postcard::from_bytes(bytes).map_err(Into::into)
    }
}
```

### Consequences

**Positive:**

- Core trait is simple and testable (mock transport for unit tests)
- Transport-specific tuning available without polluting abstraction
- Multi-transport is first-class (not bolted on)
- Message format is version-tagged for backward compatibility
- `postcard` serialization is no_std compatible and compact

**Negative:**

- Abstraction hides transport realities (users may make bad assumptions)
- Multi-transport adds complexity (which transport has the peer?)
- Different transports have different failure modes (hard to unify errors)

**Neutral:**

- Users importing a single transport get simple, direct API
- Multi-transport is opt-in complexity

### Alternatives Considered

#### Alternative 1: Single unified transport (hide all differences)

**Pros:**

- Simplest API for users
- No transport-specific knowledge required

**Cons:**

- Can't optimize for LoRa's duty cycle constraints
- Can't leverage high-bandwidth transports
- "Least common denominator" performance

**Why not chosen:** SwarmTorch targets radically different environments; hiding differences is a disservice.

#### Alternative 2: Expose all transports as separate, unrelated types

**Pros:**

- Maximum control per transport
- No abstraction overhead

**Cons:**

- No code sharing between transports
- Users must handle each transport separately
- Can't write generic algorithms

**Why not chosen:** Core trait provides uniformity; extensions provide control.

#### Alternative 3: Use existing networking crate (libp2p, quinn)

**Pros:**

- Mature, battle-tested
- Handles peer discovery, NAT traversal

**Cons:**

- libp2p is complex and heavy (not no_std ready)
- quinn is QUIC-only (not suitable for LoRa/BLE)
- Neither fits embedded constraints

**Why not chosen:** We borrow ideas but build purpose-fit abstractions.

### Implementation Notes

**Transport implementations planned:**

|Transport|Crate                           |no_std|Priority|
|---------|--------------------------------|------|--------|
|TCP      |`tokio::net`                    |❌     |v0.1    |
|UDP      |`tokio::net`                    |❌     |v0.1    |
|WiFi     |Platform-specific (ESP-IDF, etc)|✅     |v0.2    |
|BLE      |`btleplug` / `embassy-ble`      |✅     |v0.2    |
|LoRa     |`embassy-lora`                  |✅     |v0.3    |

**Peer ID design:**

```rust
pub struct PeerId {
    // 32-byte unique identifier
    // Derived from public key for authenticated transports
    // Random for unauthenticated transports
    id: [u8; 32],
}

impl PeerId {
    pub fn from_public_key(key: &PublicKey) -> Self {
        // SHA-256 hash of public key
    }
    
    pub fn random() -> Self {
        // Secure random generation
    }
}
```

**Testing strategy:**

```rust
// swarm-torch-net/src/mock.rs
pub struct MockTransport {
    peers: Arc<Mutex<HashMap<PeerId, VecDeque<Vec<u8>>>>>,
    network: Arc<MockNetwork>,
}

impl MockTransport {
    pub fn new_network(num_peers: usize) -> Vec<Self> {
        // Create interconnected mock transports
        // Messages routed through shared MockNetwork
    }
    
    pub fn set_failure_rate(&mut self, rate: f32) {
        // Simulate lossy networks
    }
    
    pub fn set_latency(&mut self, latency: Duration) {
        // Simulate network delays
    }
}
```

### References

- [postcard serialization](https://docs.rs/postcard)
- [embassy-net](https://docs.embassy.dev/embassy-net)
- [btleplug](https://github.com/deviceplug/btleplug)
- [LoRa Alliance specs](https://lora-alliance.org/)

-----

## ADR-0005: State Serialization and Message Formats

**Status:** Accepted  
**Date:** 2025-01-08  
**Deciders:** Core Team

### Context

SwarmTorch exchanges various data types across the network:

- **Gradient updates:** High-dimensional tensors (potentially millions of floats)
- **Model checkpoints:** Full model state for synchronization
- **Consensus messages:** Votes, proposals, heartbeats
- **Metadata:** Peer info, topology updates, diagnostics

**Key requirements:**

1. **Compact:** LoRa can only send ~200 bytes per message
2. **Fast:** Server nodes exchange updates at high frequency
3. **Portable:** Same format works on no_std and std targets
4. **Versioned:** Protocol evolution without breaking compatibility
5. **Secure:** Prevent malformed messages from causing crashes

### Decision

**Primary serialization: `postcard`**

- Rust-native, no_std compatible, compact binary format
- Based on serde, so integrates with existing Rust ecosystem
- Self-describing optional (schema evolution support)

**Compression for gradients: Custom protocol**

- TopK sparsification (send only top K% of gradient values)
- Randomized sparsification (probabilistic selection)
- Quantization (FP32 → INT8 where appropriate)

**Message envelope:**

> Historical design sketch from early planning.
> Current canonical envelope contract is ADR-0008B and `swarm-torch-net/src/protocol.rs`.

```rust
// swarm-torch-net/src/protocol.rs
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct MessageEnvelope {
    /// Protocol version (major.minor)
    pub version: (u8, u8),
    
    /// Message type discriminator
    pub message_type: MessageType,
    
    /// Sender identity
    pub sender: PeerId,
    
    /// Monotonic sequence number (replay protection)
    pub sequence: u64,
    
    /// Unix timestamp (coarse, for expiry)
    pub timestamp: u32,
    
    /// Payload (type depends on message_type)
    pub payload: Vec<u8>,
    
    /// Optional cryptographic signature
    pub signature: Option<[u8; 64]>,
}

#[derive(Serialize, Deserialize)]
#[repr(u8)]
pub enum MessageType {
    GradientUpdate = 0x01,
    ModelCheckpoint = 0x02,
    ConsensusVote = 0x03,
    Heartbeat = 0x04,
    PeerDiscovery = 0x05,
    TopologyChange = 0x06,
    AggregationResult = 0x07,
}
```

**Gradient compression protocol:**

```rust
// swarm-torch-core/src/compression.rs
#[derive(Serialize, Deserialize)]
pub struct CompressedGradient {
    /// Compression method used
    pub method: CompressionMethod,
    
    /// Original tensor shape
    pub shape: Vec<usize>,
    
    /// Compressed data
    pub data: CompressedData,
}

pub enum CompressionMethod {
    /// No compression (full gradient)
    None,
    
    /// Top-K sparsification (indices + values)
    TopK { k_ratio: f32 },
    
    /// Random sparsification with seed
    RandomSparse { p: f32, seed: u64 },
    
    /// Quantized to INT8 with scale
    Quantized { scale: f32 },
    
    /// Combined: TopK + Quantization
    TopKQuantized { k_ratio: f32, scale: f32 },
}

pub enum CompressedData {
    Dense(Vec<u8>),
    Sparse { indices: Vec<u32>, values: Vec<u8> },
}
```

### Consequences

**Positive:**

- `postcard` is battle-tested in embedded Rust (ESP32, STM32)
- Compression ratios of 10-100x possible for sparse updates
- Versioned envelope allows protocol evolution
- Signature field enables authenticated messages (opt-in)

**Negative:**

- Custom compression means less interoperability with other frameworks
- Quantization introduces approximation errors (documented tradeoffs)
- Schema evolution requires careful handling

**Neutral:**

- JSON/Protobuf available via serde for debugging/interop
- Compression is opt-in (full gradients always supported)

### Alternatives Considered

#### Alternative 1: Protocol Buffers

**Pros:**

- Industry standard, excellent tooling
- Schema evolution built-in
- Cross-language support

**Cons:**

- Large runtime (not no_std compatible without effort)
- Code generation adds build complexity
- Overkill for Rust-to-Rust communication

**Why not chosen:** postcard is Rust-native and no_std ready.

#### Alternative 2: MessagePack

**Pros:**

- Compact binary format
- Wide language support
- Self-describing

**Cons:**

- Less compact than postcard for Rust types
- no_std support varies by implementation

**Why not chosen:** postcard is more idiomatic for Rust.

#### Alternative 3: Cap'n Proto

**Pros:**

- Zero-copy deserialization
- Excellent performance
- Schema evolution

**Cons:**

- Complex implementation
- Code generation required
- Not no_std ready

**Why not chosen:** Complexity outweighs benefits for our use case.

### Implementation Notes

**Bandwidth estimates for MNIST (784→128→10 network):**

|Method                           |Size per update|Compression ratio|
|---------------------------------|---------------|-----------------|
|Full FP32                        |410 KB         |1x               |
|TopK (1%)                        |8 KB           |51x              |
|TopK (1%) + INT8                 |2.1 KB         |195x             |
|Random Sparse (5%) + INT8        |10 KB          |41x              |

**LoRa compatibility:** With TopK (0.1%) + INT8, updates fit in ~200 bytes per LoRa packet.

### References

- [postcard crate](https://docs.rs/postcard)
- [Gradient compression survey](https://arxiv.org/abs/2010.12460)
- [Top-K sparsification](https://arxiv.org/abs/1712.01887)

-----

## ADR-0006: Identity and Membership Model

**Status:** Accepted  
**Date:** 2025-01-08 (Updated 2026-01-09)  
**Deciders:** Core Team

> **⚠️ Terminology Clarification:** This ADR describes **identity management and membership semantics**, NOT "Byzantine consensus" in the classical sense (PBFT/Tendermint-style safety/liveness guarantees). The coordination protocol (gossip-based eventual consistency) is documented separately in **ADR-0006A**. Using "consensus" loosely would create semantic debt.

### Context

SwarmTorch nodes need a coherent identity and membership model to:

- **Authenticate participants:** Verify message origins
- **Manage lifecycle:** Handle join/leave/churn
- **Assign roles:** Differentiate coordinators, contributors, observers
- **Enable reputation:** Track contribution quality over time

**What must converge (invariants):**
- Membership view (eventually consistent peer set)
- Round ID (current training round)
- Aggregated model hash (after round completion)

**What may temporarily diverge:**
- Partial update sets (before quorum)
- Peer lists (during churn)
- Pending gossip messages

**Byzantine goals (what we're defending against):**
- Prevent invalid model adoption
- Prevent update replay (sequence numbers)
- Bound adversarial influence (robust aggregators)

**Key constraints:**

1. Nodes join/leave dynamically (churn)
2. Some nodes may be malicious (Byzantine)
3. Network partitions are expected (intermittent connectivity)
4. No central coordinator (fully decentralized)

### Decision

**Gossip-based eventual consistency with weighted voting:**

```rust
// swarm-torch-core/src/consensus.rs
pub struct GossipConsensus {
    /// Local view of cluster membership
    membership: MembershipView,
    
    /// Pending updates awaiting quorum
    pending: HashMap<RoundId, Vec<(PeerId, Update)>>,
    
    /// Aggregation configuration
    aggregator: Box<dyn RobustAggregator>,
    
    /// Minimum participation for round completion
    quorum_ratio: f32,
}

impl GossipConsensus {
    pub async fn propose_update(&mut self, update: GradientUpdate) -> Result<()> {
        // Gossip update to random peers (fanout)
        // Collect acknowledgments
        // When quorum reached, trigger aggregation
    }
    
    pub async fn on_receive_update(&mut self, from: PeerId, update: GradientUpdate) {
        // Validate update (bounds checking, format)
        // Store in pending buffer
        // Gossip to other peers (with probability)
        // Check if round complete
    }
}
```

**Identity model:**

```rust
// swarm-torch-core/src/identity.rs
pub struct NodeIdentity {
    /// Unique node ID (derived from public key)
    pub id: PeerId,
    
    /// Cryptographic key pair (Ed25519)
    pub key_pair: KeyPair,
    
    /// Attestation (optional, for TEE environments)
    pub attestation: Option<Attestation>,
    
    /// Role in swarm
    pub role: NodeRole,
}

pub enum NodeRole {
    /// Full participant: trains, aggregates, coordinates
    Coordinator,
    
    /// Contributor: trains locally, sends updates
    Contributor,
    
    /// Observer: receives model, doesn't contribute
    Observer,
    
    /// Gateway: bridges networks (e.g., LoRa ↔ WiFi)
    Gateway,
}
```

**Round protocol:**

```
1. EPOCH_START: Coordinators propose new round with round_id
2. TRAINING: Contributors compute local gradients on their data
3. UPDATE: Contributors send signed gradients to coordinators
4. AGGREGATION: Coordinators aggregate using robust aggregator
5. BROADCAST: Aggregated model broadcast to all participants
6. EPOCH_END: Round complete, metrics logged
```

### Consequences

**Positive:**

- Gossip scales well (O(log N) hops to reach all nodes)
- No single point of failure
- Tolerates network partitions (partitions converge when healed)
- Weighted voting allows reputation-based trust

**Negative:**

- Eventual consistency means temporary divergence possible
- Sybil attacks require external identity layer (not solved internally)
- Round synchronization is approximate (clock skew issues)

**Neutral:**

- Nodes can observe cluster state via membership view
- Metrics exposed for debugging (pending updates, round latency)

### Alternatives Considered

#### Alternative 1: Raft consensus

**Pros:**

- Strong consistency guarantees
- Well-understood protocol
- Leader election built-in

**Cons:**

- Requires stable leader (bad for mobile/IoT)
- Not Byzantine tolerant
- Overkill for ML aggregation

**Why not chosen:** Swarm learning needs Byzantine tolerance, not strong consistency.

#### Alternative 2: PBFT (Practical Byzantine Fault Tolerance)

**Pros:**

- Byzantine fault tolerant
- Deterministic finality

**Cons:**

- O(N²) message complexity
- Requires known, stable membership
- Overkill for gradient averaging

**Why not chosen:** Too heavyweight for IoT-scale swarms.

#### Alternative 3: No consensus (async aggregation only)

**Pros:**

- Simplest implementation
- No coordination overhead

**Cons:**

- No round boundaries (hard to measure progress)
- Stale updates mixed with fresh
- No Byzantine defense

**Why not chosen:** Need at least basic coordination for practical training.

### Implementation Notes

**Gossip parameters:**

```rust
pub struct GossipConfig {
    /// Number of peers to gossip to per round
    pub fanout: usize,  // Default: 3
    
    /// Probability of forwarding received messages
    pub forward_probability: f32,  // Default: 0.7
    
    /// Time before message expires
    pub message_ttl: Duration,  // Default: 60s
    
    /// Heartbeat interval
    pub heartbeat_interval: Duration,  // Default: 10s
}
```

**Quorum calculation:**

```rust
impl GossipConsensus {
    fn calculate_quorum(&self) -> usize {
        let active_nodes = self.membership.active_count();
        let byzantine_ratio = self.aggregator.byzantine_tolerance();
        
        // Need > 2f + 1 for Byzantine tolerance where f is malicious fraction
        ((active_nodes as f32) * (1.0 - 2.0 * byzantine_ratio)).ceil() as usize
    }
}
```

### References

- [Gossip protocols survey](https://www.cs.cornell.edu/projects/Quicksilver/public_pdfs/SWIM.pdf)
- [Epidemic algorithms](https://www.cs.cornell.edu/~asdas/research/dsn02-swim.pdf)
- [PBFT paper](https://pmg.csail.mit.edu/papers/osdi99.pdf)

-----

## ADR-0007: Robust Aggregation Strategy

**Status:** Accepted  
**Date:** 2025-01-08  
**Deciders:** Core Team

### Context

In federated/swarm learning, the central aggregation step is vulnerable to Byzantine attacks:

- **Data poisoning:** Malicious nodes train on corrupted data
- **Model poisoning:** Nodes send crafted updates to degrade global model
- **Free-riding:** Nodes send random updates without training

Classical FedAvg (simple averaging) breaks down when even 1-2 malicious nodes participate. SwarmTorch needs **robust aggregation** that tolerates a fraction of Byzantine participants.

**Key research findings:**

- No aggregator is perfectly robust in all settings
- High-dimensional gradients are harder to defend
- Defenses have computational costs
- Multiple layered defenses work better than one

### Decision

**Pluggable aggregator architecture with sensible defaults:**

```rust
// swarm-torch-core/src/aggregation/mod.rs
pub trait RobustAggregator: Send + Sync {
    /// Aggregate multiple gradient updates into one
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<GradientUpdate>;
    
    /// Fraction of Byzantine nodes this aggregator tolerates
    fn byzantine_tolerance(&self) -> f32;
    
    /// Computational complexity class
    fn complexity(&self) -> AggregatorComplexity;
    
    /// Explain why updates were rejected (for telemetry)
    fn rejection_reasons(&self) -> Vec<RejectionReason>;
}

pub enum AggregatorComplexity {
    Linear,      // O(n) - Trimmed Mean
    Quadratic,   // O(n²) - Krum
    Cubic,       // O(n³) - Bulyan
}
```

**Implemented aggregators:**

```rust
// Coordinate-wise median
pub struct CoordinateMedian;

// Trimmed mean (discard top/bottom k%)
pub struct TrimmedMean {
    pub trim_ratio: f32,  // e.g., 0.2 for 20%
}

// Krum (select update closest to others)
pub struct Krum {
    pub num_selected: usize,
    pub num_byzantine: usize,
}

// Multi-Krum (select multiple closest updates, then average)
pub struct MultiKrum {
    pub num_selected: usize,
    pub num_byzantine: usize,
}

// Bulyan (Krum selection + coordinate-wise trimmed mean)
pub struct Bulyan {
    pub num_byzantine: usize,
}
```

**Default strategy:**

```rust
impl Default for RobustAggregation {
    fn default() -> Self {
        // Trimmed Mean is a good balance of:
        // - Computational efficiency (O(n) per coordinate)
        // - Byzantine tolerance (~20% malicious nodes)
        // - Statistical efficiency (minimal variance inflation)
        RobustAggregation::TrimmedMean { trim_ratio: 0.2 }
    }
}
```

### Consequences

**Positive:**

- Users can choose aggregator based on threat model
- Telemetry shows why updates were rejected (debugging)
- Attack simulation harness validates defenses
- Layered defense possible (Krum + outlier detection)

**Negative:**

- No aggregator is perfect (we document limitations)
- Computational overhead scales with defense strength
- High-dimensional models reduce defense effectiveness

**Neutral:**

- Default (Trimmed Mean) is reasonable for most cases
- Security-critical deployments should use Bulyan + monitoring

### Alternatives Considered

#### Alternative 1: No Byzantine defense (trust all nodes)

**Pros:**

- Simplest implementation
- Lowest computational cost
- Maximum model accuracy (no trimming)

**Cons:**

- Single malicious node can poison model
- No defense against compromised devices

**Why not chosen:** SwarmTorch targets adversarial environments (IoT, robotics).

#### Alternative 2: Cryptographic verification only

**Pros:**

- Proves update came from claimed sender
- Prevents impersonation

**Cons:**

- Doesn't prevent malicious valid updates
- Byzantine node with valid key can still attack

**Why not chosen:** Authentication is necessary but not sufficient.

#### Alternative 3: Machine learning-based detection

**Pros:**

- Can learn complex attack patterns
- Adapts to new attacks

**Cons:**

- Requires training data (attacks)
- False positives hurt convergence
- Meta-learning attacks possible

**Why not chosen:** Research area, not production-ready. We provide hooks for future integration.

### Implementation Notes

**Attack simulation harness:**

```rust
// swarm-torch-core/src/testing/attacks.rs
pub enum AttackType {
    /// Flip labels in training data
    LabelFlipping { flip_ratio: f32 },
    
    /// Scale gradients by large factor
    ScalingAttack { scale: f32 },
    
    /// Send random noise instead of gradients
    GaussianNoise { std: f32 },
    
    /// Targeted attack toward specific misclassification
    BackdoorAttack { trigger: Tensor, target_class: usize },
}

pub struct AttackHarness {
    pub attack: AttackType,
    pub byzantine_ratio: f32,
}

impl AttackHarness {
    pub fn inject(&self, updates: &mut [GradientUpdate]) {
        // Apply attack to subset of updates
    }
}
```

**Telemetry integration:**

```rust
pub struct AggregationMetrics {
    pub round_id: u64,
    pub num_updates_received: usize,
    pub num_updates_accepted: usize,
    pub rejection_reasons: HashMap<RejectionReason, usize>,
    pub aggregation_time: Duration,
    pub aggregator_used: String,
}
```

### References

- [Krum: Machine Learning with Adversaries](https://arxiv.org/abs/1703.02757)
- [Bulyan: Distributed Learning with Byzantine Workers](https://arxiv.org/abs/1802.07927)
- [Byzantine-Resilient Distributed Learning Survey](https://arxiv.org/abs/2003.08937)

-----

## ADR-0008: Threat Model and Trust Boundaries

**Status:** Accepted  
**Date:** 2025-01-08  
**Deciders:** Core Team

### Context

SwarmTorch operates in environments where:

- Devices can be physically compromised
- Networks are untrusted (public WiFi, LoRa)
- Some participants may be malicious
- Central authorities may not exist

We need a clear threat model that defines:

1. What attacks we defend against
2. What attacks are out of scope
3. Trust assumptions we make

### Decision

**Threat model:**

| Threat | In Scope | Defense |
|--------|----------|---------|
| Byzantine updates | ✅ Yes | Robust aggregators |
| Message tampering | ✅ Yes | Signatures (Ed25519) |
| Replay attacks | ✅ Yes | Sequence numbers + timestamps |
| Eavesdropping | ✅ Yes | TLS/DTLS for transports |
| Sybil attacks | ⚠️ Partial | Rate limiting, reputation (external identity required) |
| Physical device compromise | ❌ No | Out of scope (TEE integration future work) |
| Side-channel attacks | ❌ No | Out of scope |
| Denial of service | ⚠️ Partial | Rate limiting, but not exhaustive |

**Trust boundaries:**

```
┌─────────────────────────────────────────────────────────┐
│                    TRUSTED ZONE                          │
│  • Local model computation                               │
│  • Local data storage                                    │
│  • Cryptographic key material                            │
├─────────────────────────────────────────────────────────┤
│                  VERIFIED ZONE                           │
│  • Signed messages from authenticated peers              │
│  • Validated gradient updates (bounds checked)           │
│  • Aggregated updates (robust aggregation applied)       │
├─────────────────────────────────────────────────────────┤
│                  UNTRUSTED ZONE                          │
│  • Network traffic (may be intercepted/modified)         │
│  • Peer claims (must be verified)                        │
│  • Raw gradient updates (may be malicious)               │
└─────────────────────────────────────────────────────────┘
```

**Authentication implementation:**

```rust
// swarm-torch-core/src/crypto.rs
use ed25519_dalek::{Keypair, PublicKey, Signature, Signer, Verifier};

pub struct MessageAuth {
    keypair: Keypair,
}

impl MessageAuth {
    pub fn sign(&self, message: &[u8]) -> Signature {
        self.keypair.sign(message)
    }
    
    pub fn verify(public_key: &PublicKey, message: &[u8], signature: &Signature) -> bool {
        public_key.verify(message, signature).is_ok()
    }
}

// Replay protection
pub struct ReplayProtection {
    seen_sequences: LruCache<PeerId, HashSet<u64>>,
    max_clock_skew: Duration,
}

impl ReplayProtection {
    pub fn is_valid(&mut self, envelope: &MessageEnvelope) -> bool {
        // Check timestamp within acceptable window
        // Check sequence number not seen before
        // Record sequence number
    }
}
```

### Consequences

**Positive:**

- Clear documentation of security assumptions
- Layered defenses (authentication + robust aggregation)
- Users understand what's protected and what's not

**Negative:**

- Sybil attacks require external identity (not fully solved)
- Physical compromise out of scope (hardware security is complex)
- Security adds overhead (signatures, encryption)

**Neutral:**

- Ed25519 is fast and no_std compatible
- Security features are opt-in (can disable for trusted networks)

### Implementation Notes

**Secure defaults:**

```rust
pub struct SecurityConfig {
    /// Require message signatures (default: true)
    pub require_signatures: bool,
    
    /// Encrypt network traffic (default: true for TCP/WiFi)
    pub encrypt_transport: bool,
    
    /// Reject updates outside bounds (default: true)
    pub validate_gradients: bool,
    
    /// Maximum clock skew for replay protection
    pub max_clock_skew: Duration,  // Default: 60 seconds
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            require_signatures: true,
            encrypt_transport: true,
            validate_gradients: true,
            max_clock_skew: Duration::from_secs(60),
        }
    }
}
```

**Gradient validation:**

```rust
pub struct GradientValidator {
    /// Maximum L2 norm for gradient updates
    pub max_gradient_norm: f32,
    
    /// Maximum absolute value for any coordinate
    pub max_coordinate_value: f32,
}

impl GradientValidator {
    pub fn validate(&self, gradient: &GradientUpdate) -> Result<(), ValidationError> {
        // Check L2 norm
        // Check for NaN/Inf
        // Check coordinate bounds
    }
}
```

### References

- [Ed25519 specification](https://ed25519.cr.yp.to/)
- [Byzantine threat models in FL](https://arxiv.org/abs/2003.02575)
- [DTLS for constrained devices](https://datatracker.ietf.org/doc/html/rfc6347)

-----

## ADR-0009: Python Interoperability (PyO3) and Model Portability Contract

**Status:** Accepted  
**Date:** 2025-01-08 (Updated 2026-01-09)  
**Deciders:** Core Team

### Context

While SwarmTorch is Rust-native, many ML practitioners work primarily in Python. To maximize adoption, we need Python bindings that:

1. Allow training in Python, deployment in Rust
2. ~~Import PyTorch models into SwarmTorch~~ **See constraint below**
3. Provide Jupyter notebook integration
4. Achieve zero-copy data exchange where possible

> **⚠️ Critical Constraint: Model Zoo, Not Arbitrary Conversion**
> 
> The idea of "reconstruct any PyTorch model architecture into Burn from `state_dict`" is **NOT generally feasible**—you lose structure, ops graph, and semantics. SwarmTorch Python interop is constrained to:
> 
> - **Supported model zoo:** Pre-defined architectures that exist in both PyTorch and Burn
> - **Explicit portability contract:** Documented weight format, op-set, training semantics
> - **SwarmTorch-owned forward/backward:** Python is surface area, not escape hatch
> 
> This is the "model zoo + portability contract" path, chosen for:
> - Stronger portability guarantees (train on edge, deploy on MCU)
> - Better alignment with no_std constraints
> - Security/auditability (constrained surface area)
> - Avoiding "Year 2 trap" of rebuilding PyTorch
> 
> **Who this is for:** Robotics/defense/health/regulated buyers who value a constrained, auditable surface over maximum flexibility.

### Decision

**Use PyO3 for Python bindings with careful API design:**

```rust
// swarm-torch-python/src/lib.rs
use pyo3::prelude::*;
use numpy::PyArrayDyn;

#[pyclass]
pub struct PySwarmCluster {
    inner: SwarmCluster,
}

#[pymethods]
impl PySwarmCluster {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self { inner: SwarmCluster::default() })
    }
    
    fn train(
        &mut self, 
        py: Python<'_>,
        model: PyObject,  // PyTorch model
        data: &PyArrayDyn<f32>,  // NumPy array
    ) -> PyResult<PyObject> {
        // Convert PyTorch model to Burn/internal format
        // Train using Rust backend
        // Return result as PyTorch model
    }
}

#[pymodule]
fn swarm_torch(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySwarmCluster>()?;
    m.add_class::<PyRobustAggregator>()?;
    Ok(())
}
```

**Python API design:**

```python
import swarm_torch
import torch

# Create swarm cluster
cluster = swarm_torch.SwarmCluster(
    topology="gossip",
    transport="tcp",
    aggregation="trimmed_mean",
)

# Train with PyTorch model
model = torch.nn.Sequential(...)
trained = cluster.train(
    model=model,
    data=train_loader,
    rounds=100,
    on_round=lambda r, m: print(f"Round {r}: {m}")
)

# Export back to PyTorch
torch.save(trained.state_dict(), "model.pt")
```

**Model conversion:**

```rust
// swarm-torch-python/src/convert.rs
pub fn pytorch_to_burn(py_model: &PyObject) -> Result<BurnModel> {
    // Extract state_dict from PyTorch model
    // Convert tensor formats (PyTorch → Burn)
    // Reconstruct model architecture
}

pub fn burn_to_pytorch(burn_model: &BurnModel, py: Python) -> PyResult<PyObject> {
    // Convert Burn tensors to PyTorch tensors
    // Create new PyTorch model with state_dict
}
```

### Consequences

**Positive:**

- Python users can adopt SwarmTorch without learning Rust
- Jupyter notebook workflow preserved
- Zero-copy for NumPy arrays (via numpy crate)
- Can leverage existing PyTorch model zoo

**Negative:**

- Python bindings add maintenance burden
- Model conversion may lose some PyTorch features
- GIL limits threading performance (mitigated by releasing GIL during training)

**Neutral:**

- Python bindings are optional (separate crate/feature)
- Performance-critical code remains in Rust

### Alternatives Considered

#### Alternative 1: No Python support

**Pros:**

- Simpler maintenance
- No Python dependencies

**Cons:**

- Limits adoption (most ML practitioners use Python)
- Can't leverage PyTorch ecosystem

**Why not chosen:** Adoption matters; Python bridges gap.

#### Alternative 2: Python subprocess (call Rust CLI)

**Pros:**

- Simple implementation
- No FFI complexity

**Cons:**

- High overhead (process startup)
- No in-memory data sharing
- Clunky API

**Why not chosen:** Poor user experience.

#### Alternative 3: ONNX as interchange

**Pros:**

- Language-agnostic model format
- Wide tool support

**Cons:**

- Doesn't help with training loop
- Additional conversion step

**Why not chosen:** ONNX complements but doesn't replace Python bindings.

### Implementation Notes

**GIL release for long operations:**

```rust
#[pymethods]
impl PySwarmCluster {
    fn train(&mut self, py: Python<'_>, ...) -> PyResult<PyObject> {
        // Release GIL during training to allow Python threads to run
        py.allow_threads(|| {
            self.inner.train(...).map_err(|e| ...)
        })
    }
}
```

**Version compatibility:**

- Support Python 3.9+
- Support PyTorch 2.0+
- Wheels published for Linux (manylinux), macOS, Windows

**Jupyter integration:**

```python
# Rich display for training progress
from swarm_torch import SwarmCluster
from IPython.display import display

cluster = SwarmCluster(...)
with cluster.train_async(model, data) as training:
    display(training.progress_widget())  # Live updates
```

### References

- [PyO3 documentation](https://pyo3.rs/)
- [numpy crate](https://docs.rs/numpy)
- [maturin](https://www.maturin.rs/) (build tool for PyO3)

-----

## ADR-0010: Model Portability and ONNX Integration (Inference Interchange)

**Status:** Accepted  
**Date:** 2025-01-08 (Updated 2026-01-09)  
**Deciders:** Core Team

### Context

Model portability is critical for SwarmTorch's mission:

- Train on server, deploy to edge
- ~~Import pre-trained models from PyTorch/TensorFlow~~ **Via model zoo only (ADR-0009)**
- Export trained models for inference on microcontrollers
- Interoperate with existing ML inference infrastructure

> **⚠️ ONNX Scope Clarification:**
> 
> ONNX is primarily for **inference portability**; **training interchange is NOT a supported path**.
> 
> - ✅ Export SwarmTorch-trained models to ONNX for inference elsewhere
> - ✅ Import ONNX models for inference in SwarmTorch (no training)
> - ❌ Import ONNX → continue training in SwarmTorch (not guaranteed)
> - ❌ Full PyTorch training parity via ONNX (ops/semantics differ)

ONNX (Open Neural Network Exchange) is the de facto standard for model interchange.

### Decision

**ONNX as the primary inference interchange format:**

```rust
// swarm-torch-models/src/onnx.rs
use tract_onnx::prelude::*;

pub struct OnnxModel {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>>,
}

impl OnnxModel {
    /// Load ONNX model from file
    pub fn load(path: &Path) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }
    
    /// Export to ONNX format
    pub fn export(&self, path: &Path) -> Result<()> {
        // Serialize model to ONNX protobuf
    }
    
    /// Convert to Burn model for training
    pub fn to_burn<B: Backend>(&self) -> Result<BurnModel<B>> {
        // Reconstruct Burn model from ONNX graph
    }
}
```

**Quantization for embedded deployment:**

```rust
pub enum QuantizationMode {
    /// No quantization (full FP32)
    None,
    
    /// Static INT8 quantization (calibration required)
    StaticInt8 { calibration_data: CalibrationData },
    
    /// Dynamic INT8 quantization (no calibration)
    DynamicInt8,
    
    /// FP16 (half precision)
    Float16,
}

pub fn quantize_for_embedded(
    model: &OnnxModel, 
    mode: QuantizationMode,
    target: EmbeddedTarget,
) -> Result<QuantizedModel> {
    // Apply quantization
    // Optimize for target (ARM NEON, RISC-V vectors, etc.)
}
```

### Consequences

**Positive:**

- ONNX is widely supported (PyTorch, TensorFlow, etc.)
- tract provides efficient Rust inference
- Quantization enables microcontroller deployment
- Model zoo compatibility (HuggingFace, etc.)

**Negative:**

- Not all ops are supported in ONNX (custom ops need special handling)
- Training through ONNX is limited (primarily for inference)
- Quantization may reduce accuracy

**Neutral:**

- ONNX export is opt-in (requires `onnx` feature)
- Users can use Burn models directly without ONNX

### Implementation Notes

**Supported ONNX operations:**

| Category | Operations |
|----------|------------|
| Core | MatMul, Gemm, Conv, Pool |
| Activation | ReLU, Sigmoid, Tanh, Softmax |
| Normalization | BatchNorm, LayerNorm |
| Shape | Reshape, Transpose, Concat |
| Reduction | ReduceSum, ReduceMean |

**Export pipeline:**

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Burn Model   │───▶│ ONNX Export  │───▶│ Optimization │
└──────────────┘    └──────────────┘    └──────────────┘
                                              │
                    ┌──────────────┐           │
                    │ Quantization │◀──────────┘
                    └──────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    ┌─────────┐    ┌───────────┐    ┌───────────┐
    │ Server  │    │   Edge    │    │ Embedded  │
    │ (FP32)  │    │  (FP16)   │    │  (INT8)   │
    └─────────┘    └───────────┘    └───────────┘
```

### References

- [ONNX specification](https://onnx.ai/)
- [tract inference engine](https://github.com/sonos/tract)
- [Quantization whitepaper](https://arxiv.org/abs/1712.05877)

-----

## ADR-0011: CI/CD Build Matrix and Artifact Strategy

**Status:** Accepted  
**Date:** 2025-01-08  
**Deciders:** Core Team

### Context

SwarmTorch must build and test across:

- Multiple architectures (x86_64, aarch64, armv7, thumbv7em)
- Multiple operating systems (Linux, macOS, Windows)
- Multiple feature flag combinations
- Embedded targets (no host OS)

A comprehensive CI/CD strategy ensures:

1. All supported configurations work
2. Performance regressions are caught
3. Releases are automated and reproducible

### Decision

**GitHub Actions as primary CI with multi-tier strategy:**

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  # Tier 1: Fast checks on every commit
  quick-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo check --all-features
      - run: cargo fmt --check
      - run: cargo clippy --all-features -- -D warnings
      
  # Tier 2: Full test suite on supported platforms
  test-matrix:
    needs: quick-check
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta]
        features: [default, "std", "alloc", "burn-backend", "tch-backend"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@${{ matrix.rust }}
      - run: cargo test --features ${{ matrix.features }}
      
  # Tier 3: Cross-compilation for embedded
  embedded-check:
    needs: quick-check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: thumbv7em-none-eabihf, aarch64-unknown-linux-gnu
      - run: cargo build -p swarm-torch-core --target thumbv7em-none-eabihf --no-default-features
      - run: cargo build -p swarm-torch --target aarch64-unknown-linux-gnu
      
  # Tier 4: Benchmarks on main only
  benchmarks:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo bench --features bench -- --save-baseline main
      - uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: target/criterion
```

**Release process:**

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags: ['v*']

jobs:
  publish-crates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo publish -p swarm-torch-core
      - run: cargo publish -p swarm-torch-net
      - run: cargo publish -p swarm-torch-runtime
      - run: cargo publish -p swarm-torch-models
      - run: cargo publish -p swarm-torch
      
  build-wheels:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release --out dist
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: dist
```

### Consequences

**Positive:**

- All supported configurations tested automatically
- Regressions caught before merge
- Release process is automated and reproducible
- Benchmark history tracked

**Negative:**

- CI time increases with matrix size
- Embedded testing requires hardware-in-the-loop (not in CI)
- Windows/macOS runners incur cost

**Neutral:**

- GitHub Actions is standard for Rust projects
- Can migrate to other CI (GitLab, Buildkite) if needed

### Implementation Notes

**Feature flag combinations to test:**

```
# Minimum viable combinations (not full matrix)
- default (std + tokio + burn)
- std only
- alloc only (no std)
- embassy-runtime (no tokio)
- tch-backend (PyTorch interop)
- all-features
```

**Performance regression detection:**

```yaml
# Use criterion-compare-action for benchmark changes
- uses: boa-dev/criterion-compare-action@v3
  with:
    branchName: ${{ github.base_ref }}
    threshold: 10%  # Fail if >10% regression
```

### References

- [GitHub Actions documentation](https://docs.github.com/actions)
- [Rust CI best practices](https://matklad.github.io/2021/09/04/fast-rust-builds.html)
- [maturin for Python wheels](https://www.maturin.rs/)

-----

## ADR-0012: Observability and Telemetry

**Status:** Accepted  
**Date:** 2025-01-08  
**Deciders:** Core Team

### Context

Debugging distributed swarm learning is notoriously difficult:

- Updates may be rejected silently
- Network issues obscure convergence problems
- Byzantine nodes are hard to identify
- Performance varies across heterogeneous hardware

SwarmTorch needs observability that:

1. Exposes internal state without impacting performance
2. Works on embedded (no alloc telemetry)
3. Integrates with standard tools (Prometheus, OpenTelemetry)
4. Enables debugging Byzantine defenses

### Decision

**Layered telemetry architecture:**

```rust
// swarm-torch-core/src/telemetry.rs
pub trait SwarmTelemetry {
    /// Record a metric value
    fn record_metric(&self, name: &str, value: f64, tags: &[(&str, &str)]);
    
    /// Record a span (timing)
    fn record_span(&self, name: &str, duration: Duration, tags: &[(&str, &str)]);
    
    /// Record an event
    fn record_event(&self, name: &str, tags: &[(&str, &str)]);
}

// No-op implementation for embedded/minimal builds
#[cfg(not(feature = "telemetry"))]
pub struct NullTelemetry;

#[cfg(not(feature = "telemetry"))]
impl SwarmTelemetry for NullTelemetry {
    fn record_metric(&self, _: &str, _: f64, _: &[(&str, &str)]) {}
    fn record_span(&self, _: &str, _: Duration, _: &[(&str, &str)]) {}
    fn record_event(&self, _: &str, _: &[(&str, &str)]) {}
}

// OpenTelemetry implementation for full builds
#[cfg(feature = "opentelemetry")]
pub struct OtelTelemetry {
    meter: opentelemetry::metrics::Meter,
    tracer: opentelemetry::trace::Tracer,
}
```

**Key metrics exposed:**

| Metric | Type | Description |
|--------|------|-------------|
| `swarm.round.duration` | Histogram | Time per training round |
| `swarm.updates.received` | Counter | Number of gradient updates received |
| `swarm.updates.rejected` | Counter | Number of updates rejected (with reason tag) |
| `swarm.aggregation.time` | Histogram | Time spent in aggregation |
| `swarm.network.bytes_sent` | Counter | Network bytes transmitted |
| `swarm.network.bytes_recv` | Counter | Network bytes received |
| `swarm.model.loss` | Gauge | Current model loss |
| `swarm.peers.active` | Gauge | Number of active peers |

**Structured logging:**

```rust
// swarm-torch-core/src/logging.rs
use tracing::{info, warn, error, instrument};

#[instrument(skip(updates))]
pub fn aggregate_updates(updates: &[GradientUpdate]) -> Result<GradientUpdate> {
    info!(num_updates = updates.len(), "Starting aggregation");
    
    // ... aggregation logic ...
    
    if rejected_count > 0 {
        warn!(
            rejected = rejected_count,
            reason = ?rejection_reasons,
            "Some updates were rejected"
        );
    }
    
    Ok(aggregated)
}
```

### Consequences

**Positive:**

- Standard integrations (Prometheus, Grafana, Jaeger)
- Zero-cost for embedded (compile-time feature gate)
- Byzantine defense debugging via rejection reasons
- Performance profiling built-in

**Negative:**

- Telemetry adds code complexity
- Metrics overhead in hot paths (mitigated by sampling)

**Neutral:**

- Users can implement custom `SwarmTelemetry` for other backends
- Default telemetry is off for minimal builds

### Implementation Notes

**Prometheus example:**

```rust
// swarm-torch/examples/prometheus-metrics.rs
use prometheus::{Registry, Counter, Histogram};

let registry = Registry::new();
let swarm = SwarmCluster::builder()
    .telemetry(PrometheusTelemetry::new(&registry))
    .build()
    .await?;

// Expose /metrics endpoint
warp::serve(warp::path("metrics").map(move || {
    let encoder = prometheus::TextEncoder::new();
    encoder.encode_to_string(&registry.gather()).unwrap()
}))
.run(([0, 0, 0, 0], 9090))
.await;
```

**Embedded (defmt) integration:**

```rust
#[cfg(feature = "defmt")]
impl SwarmTelemetry for DefmtTelemetry {
    fn record_metric(&self, name: &str, value: f64, _tags: &[(&str, &str)]) {
        defmt::info!("{}: {}", name, value);
    }
}
```

### References

- [OpenTelemetry Rust](https://opentelemetry.io/docs/instrumentation/rust/)
- [tracing crate](https://docs.rs/tracing)
- [defmt](https://defmt.ferrous-systems.com/) (embedded logging)

-----

## ADR-0013: Reproducibility and Determinism

**Status:** Accepted  
**Date:** 2025-01-08  
**Deciders:** Core Team

### Context

Machine learning reproducibility is a known challenge:

- Floating-point non-associativity
- Random number generator state
- Parallel execution order
- Hardware-specific optimizations

For SwarmTorch, reproducibility is complicated by:

- Distributed execution across heterogeneous nodes
- Asynchronous message arrival
- Byzantine nodes with non-deterministic behavior

### Decision

**Best-effort reproducibility with clear documentation:**

```rust
// swarm-torch-core/src/determinism.rs
pub struct DeterminismConfig {
    /// Global random seed for reproducibility
    pub seed: Option<u64>,
    
    /// Force deterministic aggregation order
    pub deterministic_aggregation: bool,
    
    /// Disable parallel execution
    pub single_threaded: bool,
    
    /// Use deterministic floating-point (may reduce performance)
    pub deterministic_float: bool,
}

impl DeterminismConfig {
    /// Maximum reproducibility (for debugging/testing)
    pub fn strict() -> Self {
        Self {
            seed: Some(42),
            deterministic_aggregation: true,
            single_threaded: true,
            deterministic_float: true,
        }
    }
    
    /// Default (best performance, limited reproducibility)
    pub fn default() -> Self {
        Self {
            seed: None,
            deterministic_aggregation: false,
            single_threaded: false,
            deterministic_float: false,
        }
    }
}
```

**Audit trail for training runs:**

```rust
// swarm-torch-core/src/audit.rs
#[derive(Serialize, Deserialize)]
pub struct TrainingAudit {
    /// Unique run identifier
    pub run_id: Uuid,
    
    /// Timestamp of run start
    pub started_at: DateTime<Utc>,
    
    /// Configuration used
    pub config: SwarmConfig,
    
    /// Random seeds used
    pub seeds: HashMap<String, u64>,
    
    /// SwarmTorch version
    pub version: String,
    
    /// Rust compiler version
    pub rustc_version: String,
    
    /// Per-round metadata
    pub rounds: Vec<RoundAudit>,
}

#[derive(Serialize, Deserialize)]
pub struct RoundAudit {
    pub round_id: u64,
    pub participants: Vec<PeerId>,
    pub updates_received: usize,
    pub updates_accepted: usize,
    pub aggregation_method: String,
    pub model_checksum: [u8; 32],  // SHA-256 of model state
    pub duration: Duration,
}
```

### Consequences

**Positive:**

- Strict mode enables debugging
- Audit trail enables post-hoc analysis
- Checksum verification catches divergence
- Clear documentation of reproducibility limits

**Negative:**

- Strict mode significantly reduces performance
- Distributed execution is inherently non-deterministic
- Some hardware has non-deterministic float ops

**Neutral:**

- Users choose tradeoff (performance vs reproducibility)
- Audit trail is opt-in (storage overhead)

### Implementation Notes

**Deterministic aggregation:**

```rust
impl<A: RobustAggregator> GossipConsensus<A> {
    fn aggregate_deterministic(&self, updates: &[GradientUpdate]) -> GradientUpdate {
        // Sort updates by (peer_id, sequence_number)
        let mut sorted = updates.to_vec();
        sorted.sort_by_key(|u| (u.sender, u.sequence));
        
        // Aggregate in deterministic order
        self.aggregator.aggregate(&sorted)
    }
}
```

**Model checksum:**

```rust
pub fn model_checksum(model: &dyn SwarmModel) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    
    for param in model.parameters() {
        // Hash each parameter in canonical order
        hasher.update(param.name.as_bytes());
        hasher.update(&param.data.to_le_bytes());
    }
    
    hasher.finalize().into()
}
```

**Compliance mode (HIPAA/GDPR):**

```rust
pub struct ComplianceConfig {
    /// Retain audit trails for N days
    pub audit_retention_days: u32,
    
    /// Encrypt audit trail at rest
    pub encrypt_audit: bool,
    
    /// Log access to training data
    pub log_data_access: bool,
}
```

### References

- [ML Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [Floating-Point Determinism](https://randomascii.wordpress.com/2013/07/16/floating-point-determinism/)

-----

## ADR-0004A: Topology Policy as First-Class Primitive

**Status:** Accepted  
**Date:** 2026-01-09  
**Deciders:** Core Team

### Context

Topology—how nodes are connected and communicate—materially affects both convergence speed and communication overhead. In decentralized federated learning, topology is not a detail; it is a core design parameter.

**Research shows:**
- Ring topologies reduce communication but slow convergence
- Mesh topologies converge faster but have O(N²) communication
- Hierarchical topologies balance both with tunable depth
- Dynamic rewiring can adapt to changing conditions

SwarmTorch needs topology as a **first-class primitive**, not an afterthought.

### Decision

**Topology policy is configurable and adaptive:**

```rust
// swarm-torch-core/src/topology.rs
pub enum Topology {
    /// Full mesh: everyone connected to everyone
    FullMesh,
    
    /// Ring: each node connected to k neighbors
    Ring { neighbors: usize },
    
    /// Hierarchical: tree structure with aggregators at each level
    Hierarchical { 
        depth: usize,
        branching_factor: usize,
    },
    
    /// Directed DAG: custom dependency graph
    DAG { edges: Vec<(PeerId, PeerId)> },
    
    /// Random gossip: probabilistic connections
    Gossip { fanout: usize },
}

pub trait TopologyPolicy: Send + Sync {
    fn neighbors(&self, node: PeerId) -> Vec<PeerId>;
    fn should_rewire(&self, metrics: &TopologyMetrics) -> bool;
    fn rewire(&mut self, metrics: &TopologyMetrics);
}

pub struct TopologyMetrics {
    pub round_latency_ms: f32,
    pub convergence_rate: f32,
    pub messages_per_round: usize,
    pub partition_detected: bool,
}
```

**Static vs adaptive rewiring:**

```rust
pub enum RewiringPolicy {
    Static,
    AdaptiveThreshold {
        latency_threshold_ms: f32,
        check_interval: Duration,
    },
    PeriodicRandom {
        interval: Duration,
        change_ratio: f32,
    },
}
```

### Consequences

**Positive:**
- Topology affects convergence—now we can measure and tune it
- Different deployments can choose appropriate topology
- Adaptive rewiring enables self-healing

**Negative:**
- More configuration surface area
- Rewiring logic adds complexity

**Neutral:**
- Default is `Gossip { fanout: 3 }`

### Validation Required

- Benchmark runner outputs comms + convergence curves for ≥3 topologies
- Topology A/B comparison in CI

### References

- [Decentralized Learning topologies survey](https://arxiv.org/abs/2006.07350)

-----

## ADR-0004B: Asynchrony and Staleness Model

**Status:** Accepted  
**Date:** 2026-01-09  
**Deciders:** Core Team

### Context

Real distributed systems are asynchronous. A purely synchronous training model blocks on the slowest participant—unacceptable for edge/IoT scenarios.

**Key challenges:**
- Stale updates degrade model quality
- Unbounded staleness can diverge training
- Must balance freshness vs. waiting cost

### Decision

**Bounded asynchronous training with staleness-aware aggregation:**

```rust
// swarm-torch-core/src/staleness.rs
pub struct StalenessPolicy {
    pub max_staleness: u64,
    pub decay: StalenessDecay,
    pub min_updates: usize,
    pub timeout: Duration,
}

pub enum StalenessDecay {
    None,
    Linear,
    Exponential { tau: f32 },
    Threshold { cutoff: u64 },
}

impl Default for StalenessPolicy {
    fn default() -> Self {
        Self {
            max_staleness: 5,
            decay: StalenessDecay::Exponential { tau: 2.0 },
            min_updates: 3,
            timeout: Duration::from_secs(30),
        }
    }
}
```

### Consequences

**Positive:**
- Training proceeds despite slow/offline nodes
- Configurable tradeoff between freshness and throughput

**Negative:**
- Stale updates reduce convergence quality
- Version tracking adds overhead

### Validation Required

- Dropout + slow-node simulation: training must not diverge
- Staleness metrics reported

### References

- [Staleness-Aware Async SGD](https://arxiv.org/abs/1511.05950)

-----

## ADR-0006A: Coordination Protocol (Gossip-Based Eventual Consistency)

**Status:** Accepted  
**Date:** 2026-01-09  
**Deciders:** Core Team

> **Note:** This ADR documents the **coordination mechanism**, NOT "Byzantine consensus" (PBFT/Tendermint-style). It is gossip-based eventual consistency.

### Context

SwarmTorch needs lightweight coordination to:
- Disseminate gradient updates across the swarm
- Establish round boundaries for training epochs
- Achieve quorum for aggregation

**Explicit non-goals:**
- Total ordering (not needed for gradient averaging)
- Finality guarantees (eventual consistency is sufficient)
- Sybil resistance (delegated to identity layer)

### Decision

**Gossip-based dissemination with quorum-based rounds:**

```rust
pub struct GossipCoordinator {
    membership: MembershipView,
    pending: HashMap<RoundId, Vec<(PeerId, GradientUpdate)>>,
    aggregator: Box<dyn RobustAggregator>,
    quorum_ratio: f32,
    gossip_config: GossipConfig,
}

pub struct GossipConfig {
    pub fanout: usize,  // Default: 3
    pub forward_probability: f32,  // Default: 0.7
    pub message_ttl: Duration,  // Default: 60s
    pub heartbeat_interval: Duration,  // Default: 10s
}
```

**Round protocol:**
1. ROUND_PROPOSED: Coordinator proposes new round_id
2. TRAINING: Contributors compute local gradients
3. UPDATE_GOSSIP: Updates gossip through the swarm
4. QUORUM_REACHED: When sufficient updates collected
5. AGGREGATION: Robust aggregator combines updates
6. MODEL_BROADCAST: New model gossips to all
7. ROUND_COMPLETE: Metrics logged

### Consequences

**Positive:**
- Gossip scales well: O(log N) message hops
- No single point of failure
- Partitions heal naturally

**Negative:**
- Eventual consistency means temporary state divergence
- Clock skew affects round synchronization

### References

- [SWIM Protocol](https://www.cs.cornell.edu/projects/Quicksilver/public_pdfs/SWIM.pdf)

-----

## ADR-0007A: Aggregation Validation and Attack Harness

**Status:** Accepted  
**Date:** 2026-01-09  
**Deciders:** Core Team

### Context

Byzantine-robust aggregators have theoretical guarantees that depend on assumptions. We need reproducible robustness testing before each release.

### Decision

**Ship a reproducible robustness harness:**

```rust
pub struct AttackHarness {
    aggregator: Box<dyn RobustAggregator>,
    scenarios: Vec<AttackScenario>,
}

pub enum AttackType {
    RandomNoise { std_dev: f32 },
    SignFlip,
    Scaling { factor: f32 },
    LabelFlip { flip_ratio: f32 },
    ModelReplacement,
    ALittleIsEnough { perturbation_bound: f32 },
}

pub struct AttackScenario {
    pub attack: AttackType,
    pub byzantine_ratio: f32,
    pub num_honest: usize,
    pub num_rounds: usize,
}
```

**Release gate integration:**
- Mandatory scenarios must pass before release
- Robustness report published per release

### Consequences

**Positive:**
- No silent robustness regressions
- Documented attack surface

**Negative:**
- Harness maintenance overhead

### References

- [Byzantine-Robust Distributed Learning](https://arxiv.org/abs/2006.07350)

-----

## ADR-0008A: Identity and Sybil Resistance Boundary

**Status:** Accepted  
**Date:** 2026-01-09  
**Deciders:** Core Team

### Context

Sybil attacks can trivially break voting-based systems. SwarmTorch cannot solve Sybil resistance alone.

### Decision

**SwarmTorch assumes identity attestation is provided externally:**

```
SwarmTorch does NOT:
  - Issue identities
  - Validate real-world identity
  - Solve Sybil resistance

SwarmTorch DOES:
  - Authenticate message signatures (Ed25519)
  - Track per-identity rate limits
  - Maintain reputation scores
  - Provide hooks for external identity systems
```

**Identity provider trait:**

```rust
pub trait IdentityProvider: Send + Sync {
    fn validate(&self, peer: &PeerId) -> Result<ValidationResult>;
    fn is_revoked(&self, peer: &PeerId) -> bool;
    fn reputation(&self, peer: &PeerId) -> Option<f32>;
}

/// Built-in: accepts any valid signature
pub struct SignatureOnlyProvider;

/// Integration with Swarmic Network
pub struct SwarmicNetworkProvider {
    // References: SWARMIC_NETWORK_WHITE_PAPER_v12.2.md
}
```

### Consequences

**Positive:**
- Clear scope boundary
- Pluggable identity layer

**Negative:**
- Sybil resistance NOT solved by SwarmTorch alone

### References

- SWARMIC_NETWORK_WHITE_PAPER_v12.2.md (internal)

-----

## ADR-0008B: Message Envelope Sender Identity Contract

**Status:** Accepted  
**Date:** 2026-02-13  
**Deciders:** Core Team  
**Supersedes:** Implicit sender semantics from ADR-0008 message envelope examples

### Context

`MessageEnvelope.sender` must participate directly in Ed25519 signature verification.
Using hashed identity bytes (for example `PeerId::from_public_key`) in the sender field
breaks this cryptographic binding and causes deterministic verification failures.

### Decision

`MessageEnvelope.sender` **MUST** contain the sender's Ed25519 public key bytes
(raw 32-byte key, not a hashed peer identifier).

API contract:

- Primary constructor: `MessageEnvelope::new_with_public_key([u8; 32], ...)`
- Deprecated compatibility constructor: `MessageEnvelope::new(PeerId, ...)`
- Verification path: `verify_authenticated()` validates signatures against raw sender bytes

### Consequences

**Positive:**
- Prevents identity-model ambiguity at callsites
- Preserves direct cryptographic binding between sender field and signature
- Removes a common integration trap (`PeerId::from_public_key` bytes in envelope sender)

**Negative:**
- Existing `MessageEnvelope::new(PeerId, ...)` callsites should migrate to
  `new_with_public_key(...)` for explicit semantics

### Migration

Before (ambiguous/misuse-prone):

```rust
let sender = keypair.peer_id();
let envelope = MessageEnvelope::new(sender, MessageType::Heartbeat, payload);
```

After (explicit and correct):

```rust
let envelope = MessageEnvelope::new_with_public_key(
    *keypair.public_key(),
    MessageType::Heartbeat,
    payload,
);
```

### References

- `swarm-torch-net/src/protocol.rs`
- `swarm-torch-core/src/crypto.rs`
- `swarm-torch-core/src/traits.rs`

-----

## ADR-0014: Benchmark, Dataset, and Release Gates

**Status:** Accepted  
**Date:** 2026-01-09  
**Deciders:** Core Team

### Context

Without rigorous release gates, quality erodes. We need blocking gates.

### Decision

**No minor release ships unless all gates pass:**

| Gate | Description | Blocking? |
|------|-------------|-----------|
| `convergence-benchmark` | Training converges on MNIST/CIFAR-10 | ✅ Yes |
| `topology-ab-test` | ≥3 topologies tested | ✅ Yes |
| `staleness-dropout` | 30% dropout, bounded staleness | ✅ Yes |
| `robustness-harness` | Attack harness passes | ✅ Yes |
| `embedded-build` | Builds for `thumbv7em-none-eabihf` | ✅ Yes |
| `memory-profile` | No leaks in 1000-round simulation | ✅ Yes |

**Release manifest versioned and published.**

### Consequences

**Positive:**
- No silent regressions
- Scientific credibility

**Negative:**
- Slower release velocity

-----

## ADR-0015: GPU Acceleration Strategy (WGPU-first, CUDA Optional)

**Status:** Accepted  
**Date:** 2026-01-09  
**Deciders:** Core Team

### Context

GPU acceleration is essential, but SwarmTorch has constraints:
- Portability across vendors
- Embedded targets have no GPU
- CUDA introduces non-determinism

### Decision

**WGPU as primary GPU backend; CUDA as optional:**

```rust
pub enum GpuBackend {
    Cpu,
    Wgpu,
    #[cfg(feature = "cuda")]
    Cuda,
}
```

**CUDA policy:**
1. No CUDA in `swarm-torch-core` (must be no_std)
2. CUDA behind feature flag
3. Deterministic mode available
4. Dynamic loading (user provides CUDA libs)

```toml
[features]
default = ["wgpu-backend"]
wgpu-backend = ["wgpu", "burn/wgpu"]
cuda = ["cudarc", "burn/cuda"]
```

### Consequences

**Positive:**
- WGPU provides broad GPU support
- Core crates remain GPU-free and no_std

**Negative:**
- Two code paths to maintain
- WGPU may lag CUDA in performance

### References

- [WGPU](https://wgpu.rs/)
- [cudarc](https://docs.rs/cudarc)

-----

## ADR-0016: Run Artifacts and Visualization Surface

**Status:** Proposed  
**Date:** 2026-02-08  
**Deciders:** Core Team

### Context

SwarmTorch’s target environments (edge, intermittent networks, and adversarial settings) make debugging and trustworthiness hard:

- Live dashboards are brittle when links are unreliable
- Embedded targets can’t host heavy observability stacks
- Early UI decisions create long-term coupling and attack surface

SwarmTorch also explicitly targets “real work” that happens *before* training:

- Data ingest, wrangling, validation, splitting, and materialization
- Dataset lineage and provenance
- Quality diagnostics and schema drift detection

We already require telemetry primitives (ADR-0012), but we do not yet define a **stable, offline-first contract** that unifies *data wrangling* and *training* into one analysis/compliance/visualization surface.

### Decision

Adopt an **artifact-first** surface:

- SwarmTorch runs **SHOULD** emit a versioned **Run Artifact Bundle** (directory or archive).
- Notebooks and dashboards **MUST** treat the bundle as the primary interface (read-only).
- Live telemetry (Prometheus/OpenTelemetry/etc.) **MAY** be supported as an export view of the same underlying events/metrics.

**Single event model across data + training**

SwarmTorch uses one internal observability model:

- **Span** = execution of a node in the run graph (dataset stage, training round, aggregation step, export step)
- **Event** = discrete fact (schema inferred, outlier rejected, retry, checkpoint saved)
- **Attributes** = structured tags (`dataset_id`, `asset_key`, `peer_id`, `transport_kind`, `bytes_sent`, etc.)

Run artifacts are a persisted view of spans/events/metrics.

**Canonical IDs and namespaces (OTel-compatible, not OTel-dependent)**

Rule: **SwarmTorch defines its own persisted signal schema; it is OTel-compatible, not OTel-dependent.**

Concretely:

- SwarmTorch artifacts (`spans.*`, `events.*`, `metrics.*`) use a SwarmTorch-defined schema.
- Exporters **MAY** translate those signals to OpenTelemetry/OTLP, but SwarmTorch runs and artifacts **MUST NOT** require an OpenTelemetry SDK.

**ID sizes and encoding:**

- `trace_id`: 16 bytes (32 lowercase hex chars), all-zero is invalid.
- `span_id`: 8 bytes (16 lowercase hex chars), all-zero is invalid.
- `run_id`: 16 bytes; by default, `run_id == trace_id` for the run root trace.

These sizes are chosen to be compatible with W3C Trace Context and OpenTelemetry IDs.

**Attribute namespace convention:**

- SwarmTorch-defined attributes **MUST** use the `swarmtorch.*` prefix.
- Exporters **MAY** additionally emit selected OpenTelemetry semantic-convention keys where they fit, but SwarmTorch’s on-disk contract remains `swarmtorch.*`-namespaced.

**Trace context propagation (where it applies):**

- SwarmTorch transports are message-oriented; trace context is carried as explicit envelope/metadata fields (not HTTP headers).
- Across cross-process boundaries (coordinator <-> gateway <-> worker), trace context **SHOULD** be propagated.
- For `embedded_min` / constrained `no_std` nodes, trace context propagation is **OPTIONAL** and may be omitted to save bytes/complexity.
  - If omitted, the coordinator/gateway **MUST** create a linking span and correlation attributes so events can be stitched into the run trace.

**`graph.json` is a real DAG spec (not a visualization dump)**

Invariant: `graph.json` is the **source of truth** for run structure; any visualization is derived from it.

`graph.json` **MUST** be sufficient to reconstruct the run graph semantics:

- Node id, op type
- Inputs/outputs as dataset “asset keys”
- Parameters + deterministic hash
- Resource requirements (cpu/mem), cache/materialization policy
- Optional trust/sandbox flags for non-core operations

**Minimum artifact bundle layout (schema v1)**

```
runs/<run_id>/
  manifest.json          # File list + SHA-256 hashes (tamper-evidence)
  run.json               # Run metadata + config + versions
  graph.json             # Unified data + training DAG (semantics-first)
  spans.ndjson           # Append-only span records (portable baseline)
  events.ndjson          # Append-only event records (portable baseline)
  metrics.ndjson         # Metrics time-series (portable baseline)
  datasets/
    registry.json        # Dataset IDs, versions/fingerprints, schemas, locations, license/PII tags
    lineage.json         # inputs -> transforms -> outputs edges (asset lineage)
    materializations.ndjson  # Stage outputs (rows/bytes/timing/cache/quality) - baseline
  artifacts/             # Optional: checkpoints, plots, reports, etc.
```

**Format policy:**

- `*.json` / `*.ndjson` are the portability baseline.
- For `edge_std` coordinator/gateway roles, columnar formats (Arrow/Parquet) are **SHOULD** and preferred for “facts at rest”:
  - `metrics.parquet`, `events.parquet`, `spans.parquet`, `datasets/materializations.parquet`
- Every file format includes `schema_version` and is forwards-extendable.

**Security and privacy policy for artifacts:**

- Artifacts **MUST NOT** contain raw training data or raw dataset rows by default.
- Artifacts **MAY** include small samples only when explicitly enabled, size-limited, and clearly marked as potentially sensitive.
- Dataset payloads **SHOULD** be referenced by URI/pointer + fingerprint/hash by default, not embedded.
- Artifacts **MUST** support redaction of sensitive fields (IDs, IPs, keys, PII).
- `manifest.json` **MUST** include cryptographic hashes for all files; signing is a separate, optional layer (see `SECURITY.md`).

**Manifest hashing strategy (normative)**

- Hash algorithm: **SHA-256** of raw file bytes.
- Addressing: path-addressed within the run bundle (relative paths under `runs/<run_id>/`).
- Content-addressed storage MAY be used as an internal optimization, but the canonical contract is the path-addressed manifest.
- Partial bundles: supported. `manifest.json` MUST distinguish required vs optional entries, and MUST represent missing entries explicitly (with expected hash/size and optional external URI for re-hydration).

**Dataset “asset” conventions**

- Dataset and intermediate outputs are treated as named assets (`dataset://<namespace>/<name>` or equivalent).
- Materializations record the asset key + fingerprint + schema hash, plus quality/timing stats.

**Phased visualization roadmap (non-normative):**

1. Phase 0: artifact bundle + telemetry events (no UI required)
2. Phase 1: notebook-native viewers (Rust kernel first; optional thin Python helper)
3. Phase 2: local web dashboard that reads the same artifacts (offline-first)
4. Phase 3: Tauri/Iced wrapper once UX stabilizes (packaging only)

### Consequences

**Positive:**
- Stabilizes the interface between core library and visualization tooling
- Enables offline debugging, replay, and audit trails
- Keeps attack surface narrow (no forced Python runtime)
- Makes “observability” a product feature, not an afterthought

**Negative:**
- Requires schema/version discipline and compatibility testing
- Adds storage and I/O overhead (mitigated via sampling + configurable sinks)

### Alternatives Considered

**Alternative 1: Start with a full desktop app**
- **Pros:** Polished UX early
- **Cons:** High product risk; architecture churn; packaging dominates schedule
- **Why not chosen:** Locks us into UI choices before the workflow contract is stable

**Alternative 2: Live-only telemetry (no artifacts)**
- **Pros:** Simple operational story for servers
- **Cons:** Poor for offline/edge; hard to reproduce/forensically debug; weak auditability
- **Why not chosen:** SwarmTorch’s environments are intermittently connected by design

**Alternative 3: Python-first experiment tracking (MLflow-only)**
- **Pros:** Immediate ecosystem reuse
- **Cons:** Expands integration surface area; increases operational complexity on constrained nodes
- **Why not chosen:** Conflicts with Rust-first portability and minimized attack surface goals

### Implementation Notes

- Introduce a `RunArtifactSink` (std) and `RunEventEmitter` (`no_std` + `alloc`) abstraction.
- Keep artifact emission optional, but make it the “golden path” for debugging and demos.
- Add conformance tests: “given a run, produced bundle validates against schema v1”.
- Prefer mapping SwarmTorch signals to OpenTelemetry semantic conventions when exporting (OTel-compatible, not OTel-dependent). On-disk artifact attributes remain `swarmtorch.*`-namespaced.
- Prefer “core ops in Rust” for pipeline execution. Extensions (if supported) should be sandboxed by default (e.g. WASM), and any non-sandboxed extension must be explicitly enabled and marked in artifacts/UI.

### References

- ADR-0012 (Observability and Telemetry)
- [OpenTelemetry](https://opentelemetry.io/)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
- [Apache Parquet](https://parquet.apache.org/)

-----

## ADR-0017: Data Pipeline DSL and Asset Model

**Status:** Proposed  
**Date:** 2026-02-08  
**Deciders:** Core Team

### Context

SwarmTorch’s product value depends on treating data wrangling and training as one coherent, inspectable run:

- Ingest, transforms, validation, splitting, caching, and materializations are first-class
- Lineage must be reconstructable offline from run artifacts (ADR-0016)
- Execution must remain Rust-first and auditable, with a narrow attack surface

We need a stable, portable representation for “what ran” and “what it produced” that does not assume a specific UI, database, or orchestration framework.

### Decision

Define a SwarmTorch-native **pipeline graph DSL** and **asset model**, serialized via `graph.json` and the `datasets/*` artifact set described in ADR-0016.

**Invariant: `graph.json` is executable specification**

`graph.json` is the source of truth for run structure and semantics. Any visualization is derived from it.

**Assets**

- Assets are named, durable data products and intermediates produced by pipeline nodes.
- Asset keys are stable identifiers: `dataset://<namespace>/<name>` (string form).
- Every asset instance **MUST** have a fingerprint (content hash and/or deterministic recipe hash) to support caching and reproducibility.

**Graph semantics (minimum)**

`graph.json` **MUST** be sufficient to reconstruct semantics and lineage:

- `nodes[]`:
  - `node_key` (human-stable string; e.g. `prep/clean_users`)
  - `node_id` (stable addressing id derived from `node_key`)
  - `op_kind` (enum string): `data`, `train`, `comms`, `governance`, `system`
  - `op_type` (enum string): e.g. `ingest`, `transform`, `validate`, `split`, `stats`, `persist`, `train`, `aggregate`, `eval`, `export`, `broadcast`, `vote`, `redact`, ...
  - `inputs[]` / `outputs[]` as asset keys
  - `params` (JSON object) + `op_hash` (deterministic hash of op code/config)
  - `resources` (cpu/mem hints; optional accelerator hints)
  - `cache_policy` and `materialization_policy`
  - `execution_trust` (e.g. `core`, `sandboxed_extension`, `unsafe_extension`)
  - `output_refs[]` (optional): how to locate outputs (bundle path or external URI + hash)
- `edges[]` (optional if inputs/outputs fully define edges)

**Stable node IDs and deterministic hashing**

To avoid “hash edge cases” and preserve portability:

- SwarmTorch separates **node identity** from **materialization content**.
- `node_id` identifies the *operation definition* (what should run), not the produced bytes.
- Materializations carry *content fingerprints* (what did run and what it produced).

Minimum hashing rules (normative):

- Hash algorithm: SHA-256.
- Node id derivation: `node_id` MUST be derived from `node_key` as:
  - `node_id = sha256(node_key_bytes)[0..16]` displayed as 32 lowercase hex chars.
- Node identity hash (`node_def_hash`) MUST be computed from:
  - `op_kind`, `op_type`
  - normalized `params` (key-order independent)
  - declared `inputs[]`/`outputs[]` asset keys
  - an explicit `code_ref` (crate name/version and/or explicit op version)
- Node identity hash MUST be computed over a canonical binary encoding (not JSON bytes):
  - `node_def_hash = sha256(postcard(NodeDefCanonicalV1))`
  - map-like fields MUST use deterministic ordering (`BTreeMap` or equivalent).
- Node identity hash MUST NOT include:
  - timestamps, machine paths, hostnames, random seeds (unless explicitly part of params), or runtime-only counters
- Cache key SHOULD be computed from:
  - `cache_key = hash(node_def_hash + input_asset_fingerprints + execution_profile)`
- Content fingerprints MUST be recorded per output materialization (content hash and/or deterministic recipe hash, as applicable).

**Artifact addressing**

Outputs are referenced either:

- in-bundle (path-addressed): `bundle://artifacts/<node_id>/<name>` (relative path + manifest hash), or
- externally (pointer-addressed): `uri` + `expected_sha256` + size metadata

**Untrusted inputs (data trust boundary)**

- Dataset sources MUST be classified (e.g. `trusted` vs `untrusted`) in `datasets/registry.json`.
- If a node reads from an `untrusted` location, the run artifacts MUST record this as an unsafe surface (even if the op implementation is “core”).

**Dataset registry and lineage artifacts**

- `datasets/registry.json` records datasets and assets used/produced:
  - IDs, fingerprints, schemas, locations (logical/URI), licensing flags, PII classification tags, and source-trust classification
- `datasets/lineage.json` records input -> transform -> output edges (asset lineage)
- `datasets/materializations.*` records one row per stage output:
  - row counts, null stats, schema hash, storage bytes, duration, cache hit/miss, optional quality summary

**Fingerprint v0 (metadata-first, no raw rows required)**

To keep fingerprints stable and computable without reading full datasets:

- `source_fingerprint_v0` = `sha256(postcard(normalized SourceDescriptorV0))`
  - normalized source descriptor includes: URI, content type, auth mode marker, optional etag/version
- `schema_hash_v0` = `sha256(postcard(normalized SchemaDescriptorV0))`
  - when available, prefer a canonical schema representation (e.g., canonical Arrow schema form)
- `recipe_hash_v0` = `sha256(postcard({ node_def_hash, upstream_fingerprints }))`
  - computed from the operation definition plus upstream fingerprints (no payload bytes)
- `dataset_fingerprint_v0` = `sha256(postcard({ source_fingerprint_v0, schema_hash_v0, recipe_hash_v0 }))`

If a full content hash is available (e.g., content-addressed store), it MAY be recorded separately, but v0 fingerprinting MUST remain computable without raw rows by default.

**Placeholder hashing (canonical rules, alpha.6+)**

When optional components are absent, the following placeholder hashes MUST be used:

- `no_schema_hash_v0()` = `sha256(postcard("no_schema_v0"))`
  - Used when `SchemaDescriptorV0` is absent
- `derived_source_fingerprint_v0(asset_key)` = `sha256(postcard("derived_v0:{asset_key}"))`
  - Used for derived outputs (no source descriptor); salted with `asset_key` to prevent collision when a single node produces multiple outputs with identical schemas
- `root_source_v0` = `sha256(postcard("root_source_v0"))`
  - Used for source datasets without an explicit `SourceDescriptorV0` (rare; prefer explicit sources)

These placeholders are implemented in `swarm-torch-core::dataops` and MUST be used by all session/artifact emitters to ensure fingerprint consistency.

**“Facts at rest” formats**

- For `edge_std` builds, Arrow + Parquet are the preferred formats for analytics-grade facts:
  - `metrics.parquet`, `events.parquet`, `spans.parquet`, `datasets/materializations.parquet`
- JSON/NDJSON remain the baseline for portability and debugging.

### Consequences

**Positive:**
- Makes data wrangling a first-class part of SwarmTorch’s system model
- Enables offline inspection and “run replay” debugging without a UI dependency
- Creates a stable interface for notebook tooling and future dashboards

**Negative:**
- Requires schema/version discipline and compatibility tests
- Adds work up front to define deterministic hashing, caching, and materialization semantics

### Alternatives Considered

**Alternative 1: Rely on external orchestrators (Dagster/Airflow/etc.)**
- **Pros:** Mature tooling
- **Cons:** Forces UI/storage assumptions; increases integration surface; weak embedded story
- **Why not chosen:** SwarmTorch needs a stable artifact-first contract independent of orchestration/runtime

**Alternative 2: Only emit ad-hoc telemetry and logs**
- **Pros:** Fast to ship
- **Cons:** Hard to build reliable lineage/materialization tooling; encourages drift and bespoke viewers
- **Why not chosen:** Lacks a portable contract for analysis and compliance

**Alternative 3: Adopt DVC/MLflow formats directly**
- **Pros:** Familiar concepts
- **Cons:** Pulls in ecosystem assumptions (central tracking, Python-heavy workflows)
- **Why not chosen:** SwarmTorch should lift the concepts, not import the entire stack

### Implementation Notes

- Start with coordinator/gateway-only execution for the pipeline DSL; embedded participants only emit bounded spans/events/metrics.
- Keep the built-in op set small (high leverage primitives): `ingest`, `infer_schema`, `validate`, `map/filter`, `join`, `split`, `sample`, `persist`, `stats`.
  - Anything beyond this SHOULD be implemented as an extension governed by ADR-0018.
- Use deterministic hashing:
  - `node_def_hash` = “this operation definition”
  - `content_hash` = “this exact produced output”
  - `cache_key` = “can we reuse a previous materialization for these inputs”
- Treat PII classification tags as mandatory metadata for any dataset that may include user data.

### References

- ADR-0016 (Run Artifacts and Visualization Surface)
- [Apache Arrow](https://arrow.apache.org/)
- [Apache Parquet](https://parquet.apache.org/)
- [OpenLineage](https://openlineage.io/)
- [Dagster Assets](https://docs.dagster.io/concepts/assets/software-defined-assets)
- [DVC](https://dvc.org/)

-----

## ADR-0018: Extension and Execution Policy

**Status:** Proposed  
**Date:** 2026-02-08  
**Deciders:** Core Team

### Context

SwarmTorch needs extensibility for real-world pipelines, but must preserve:

- A narrow, auditable execution surface (especially for data-wrangling automation)
- Portability (`no_std` participants; constrained gateways; offline-first)
- Clear trust boundaries that can be reflected in artifacts and UI

Unconstrained “arbitrary code execution” (especially Python-forward) is a security and operability footgun in the environments SwarmTorch targets.

### Decision

Define an explicit execution policy for pipeline ops:

1. **Core ops are Rust-implemented by default** and are the recommended path.
2. **Extensions are sandboxed-by-default**, with WASM-first as the intended mechanism (when implemented).
3. Any **non-sandboxed extension** is explicit opt-in and **MUST** be marked as an unsafe surface in:
   - `graph.json` node metadata (`execution_trust`)
   - run artifacts (events/materializations)
   - notebook/dashboard renderers

**Policy requirements (normative):**

- Sandbox execution **MUST** be resource-bounded (time/memory) and deny network by default.
- “Sandboxed” means (minimum):
  - no network by default
  - filesystem allowlist (default: read-only access to explicitly provided inputs; no arbitrary writes outside the run bundle)
  - environment-variable allowlist
  - no host process execution
  - explicit timeouts and memory limits
- Extension loading **MUST** be allowlist-based (no implicit fetch).
- If an extension is used, the run artifacts **MUST** record:
  - extension identifier/version
  - hash/fingerprint of the extension module
  - sandbox mode (sandboxed vs unsafe)

**Unsafe opt-in recording (normative):**

- Any `unsafe_extension` node **MUST** be explicitly marked in:
  - `graph.json` (`execution_trust`)
  - `events.*` (an explicit event or attribute indicating unsafe surface)
  - `datasets/materializations.*` (materializations from unsafe nodes are flagged)
- The run bundle **SHOULD** include a summary list in `run.json` (e.g. `unsafe_surfaces[]`) to make unsafe usage obvious without scanning events.

### Consequences

**Positive:**
- Preserves the Rust-first, low-attack-surface posture
- Keeps notebook and dashboard viewers honest about trust boundaries
- Allows safe extensibility without committing to Python as the substrate

**Negative:**
- Implementing a robust sandbox is non-trivial
- Some users will prefer “just run Python”; this remains possible only as an explicit, unsafe opt-in

### Alternatives Considered

**Alternative 1: Python-first extensions**
- **Pros:** Large ecosystem
- **Cons:** Large trusted surface; difficult on constrained targets; packaging complexity
- **Why not chosen:** Conflicts with portability and security posture

**Alternative 2: Native dynamic libraries/plugins**
- **Pros:** High performance
- **Cons:** Full native code execution; weak isolation; supply-chain risk
- **Why not chosen:** Hard to secure and hard to make portable

**Alternative 3: No extension mechanism**
- **Pros:** Minimal surface area
- **Cons:** Limits adoption and forces forks/bespoke patches for real pipelines
- **Why not chosen:** Users need controlled extensibility

### Implementation Notes

- When introducing an extension host, prefer a dedicated crate and feature flag (default off).
- Keep pipeline execution deterministic where possible; record non-determinism explicitly in artifacts.
- Treat Python interoperability (ADR-0009) as a separate, optional boundary; do not make it the default pipeline execution story.

### References

- ADR-0017 (Data Pipeline DSL and Asset Model)
- ADR-0016 (Run Artifacts and Visualization Surface)
- [WASI](https://wasi.dev/)
