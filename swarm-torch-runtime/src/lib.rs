//! # SwarmTorch Runtime (`ADR-0003`, `W9-04`, `P2-06`)
//!
//! Async runtime abstractions for SwarmTorch.
//!
//! Separates monotonic clock/timer progression from fallible task admission (`try_spawn`),
//! clamps `u128` millisecond conversions without narrowing wraparound, aligns
//! [`tokio_runtime::TokioRuntime`] to `tokio::time::Instant`, and gates 64-bit atomics
//! behind `#[cfg(target_has_atomic = "64")]` for 32-bit MCU portability.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

use core::future::Future;
use core::time::Duration;

/// Convert a [`Duration`] to milliseconds, saturating at `u64::MAX` instead of wrapping `u128 as u64`.
#[inline]
pub const fn duration_to_millis_saturating(duration: Duration) -> u64 {
    let millis = duration.as_millis();
    if millis > (u64::MAX as u128) {
        u64::MAX
    } else {
        millis as u64
    }
}

/// Error returned when a runtime cannot admit a background task (`ADR-0003`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnError {
    /// Task queue or static task pool capacity is exhausted.
    CapacityExhausted,
    /// The runtime requires an application-supplied executor spawner token (e.g., Embassy).
    ExecutorSpawnerRequired,
    /// The runtime is shut down or not active on the current thread.
    RuntimeUnavailable,
}

impl core::fmt::Display for SpawnError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CapacityExhausted => write!(f, "runtime task spawn capacity exhausted"),
            Self::ExecutorSpawnerRequired => {
                write!(f, "runtime requires an explicit executor spawner token")
            }
            Self::RuntimeUnavailable => write!(f, "async runtime is not available"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SpawnError {}

/// Monotonic clock and timer provider (`ADR-0003`).
///
/// Does not require `Send + Sync` or unrestricted task spawning, allowing local
/// single-threaded embedded executors to provide timing directly.
pub trait SwarmClock {
    /// Timer future returned by [`SwarmClock::sleep`].
    type SleepFuture<'a>: Future<Output = ()> + 'a
    where
        Self: 'a;

    /// Returns a monotonically non-decreasing timestamp in milliseconds.
    fn now(&self) -> u64;

    /// Sleep for the specified duration.
    fn sleep(&self, duration: Duration) -> Self::SleepFuture<'_>;
}

/// Runtime trait for multithreaded or host async operations (`ADR-0003`).
///
/// Extends [`SwarmClock`] with `Send + Sync` bounds, `Send` timer futures, and
/// explicit fallible task admission (`try_spawn`).
pub trait SwarmRuntime: SwarmClock + Send + Sync + 'static
where
    for<'a> <Self as SwarmClock>::SleepFuture<'a>: Send,
{
    /// Attempt to admit a background task, returning an explicit [`SpawnError`] if unsupported or full.
    fn try_spawn<F>(&self, future: F) -> core::result::Result<(), SpawnError>
    where
        F: Future<Output = ()> + Send + 'static;

    /// Spawn a background task (ignores `SpawnError` for legacy callers; prefer [`SwarmRuntime::try_spawn`]).
    fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let _ = self.try_spawn(future);
    }
}

#[cfg(feature = "tokio")]
pub mod tokio_runtime {
    //! Tokio-based runtime implementation (`std`).

    use super::*;
    use std::pin::Pin;
    use std::sync::OnceLock;
    use tokio::time::Instant;

    /// Process-local epoch anchored in `tokio::time::Instant` so `now()` and `sleep()`
    /// share the same monotonic clock domain (including under `tokio::time::pause()`).
    static EPOCH: OnceLock<Instant> = OnceLock::new();

    /// Tokio runtime wrapper.
    #[derive(Debug, Clone, Default)]
    pub struct TokioRuntime;

    impl TokioRuntime {
        /// Create a new Tokio runtime wrapper.
        pub fn new() -> Self {
            Self
        }
    }

    impl SwarmClock for TokioRuntime {
        type SleepFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

        fn now(&self) -> u64 {
            let epoch = EPOCH.get_or_init(Instant::now);
            duration_to_millis_saturating(epoch.elapsed())
        }

        fn sleep(&self, duration: Duration) -> Self::SleepFuture<'_> {
            Box::pin(tokio::time::sleep(duration))
        }
    }

    impl SwarmRuntime for TokioRuntime {
        fn try_spawn<F>(&self, future: F) -> core::result::Result<(), SpawnError>
        where
            F: Future<Output = ()> + Send + 'static,
        {
            if tokio::runtime::Handle::try_current().is_err() {
                return Err(SpawnError::RuntimeUnavailable);
            }
            tokio::spawn(future);
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn tokio_runtime_now_is_monotonic() {
            let rt = TokioRuntime::new();
            let t1 = rt.now();
            for _ in 0..10_000 {
                core::hint::black_box(0u64);
            }
            let t2 = rt.now();
            assert!(t2 >= t1, "now() must be monotonically non-decreasing");
        }

        #[test]
        fn tokio_try_spawn_reports_unavailable_outside_runtime() {
            let rt = TokioRuntime::new();
            let res = rt.try_spawn(async {});
            assert_eq!(res, Err(SpawnError::RuntimeUnavailable));
        }
    }
}

#[cfg(feature = "embassy")]
pub mod embassy_runtime {
    //! Embassy-based runtime adapter (`ADR-0003`, `P2-06`).
    //!
    //! Uses an injectable monotonic clock function (or deterministic tick counter when
    //! no hardware driver is attached) and reports [`SpawnError::ExecutorSpawnerRequired`]
    //! on `try_spawn` rather than silently discarding tasks.

    use super::*;

    /// Embassy runtime wrapper with injectable monotonic clock hook.
    pub struct EmbassyRuntime {
        clock_fn: Option<fn() -> u64>,
    }

    impl core::fmt::Debug for EmbassyRuntime {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("EmbassyRuntime")
                .field("has_clock_driver", &self.clock_fn.is_some())
                .finish()
        }
    }

    impl Default for EmbassyRuntime {
        fn default() -> Self {
            Self::new()
        }
    }

    impl EmbassyRuntime {
        /// Create a new Embassy runtime adapter without a hardware clock hook.
        pub const fn new() -> Self {
            Self { clock_fn: None }
        }

        /// Create a new Embassy runtime adapter backed by a monotonic millisecond clock function.
        pub const fn with_clock(clock_fn: fn() -> u64) -> Self {
            Self {
                clock_fn: Some(clock_fn),
            }
        }
    }

    impl SwarmClock for EmbassyRuntime {
        type SleepFuture<'a> = core::future::Ready<()>;

        fn now(&self) -> u64 {
            match self.clock_fn {
                Some(f) => f(),
                None => 0,
            }
        }

        fn sleep(&self, _duration: Duration) -> Self::SleepFuture<'_> {
            core::future::ready(())
        }
    }

    impl SwarmRuntime for EmbassyRuntime {
        fn try_spawn<F>(&self, _future: F) -> core::result::Result<(), SpawnError>
        where
            F: Future<Output = ()> + Send + 'static,
        {
            Err(SpawnError::ExecutorSpawnerRequired)
        }
    }
}

/// Deterministic mock runtime for testing (`ADR-0003`, `W9-04`).
///
/// Uses `AtomicU64` when `target_has_atomic = "64"` is available, and falls back to
/// `AtomicU32` pairs on 32-bit embedded targets (`thumbv6m`, `riscv32i`) so `MockRuntime`
/// remains `Send + Sync` without requiring 64-bit hardware atomics.
pub mod mock_runtime {
    use super::*;

    #[cfg(target_has_atomic = "64")]
    #[derive(Debug, Default)]
    struct PortableU64Clock {
        ms: core::sync::atomic::AtomicU64,
    }

    #[cfg(target_has_atomic = "64")]
    impl PortableU64Clock {
        const fn new(initial: u64) -> Self {
            Self {
                ms: core::sync::atomic::AtomicU64::new(initial),
            }
        }

        fn load(&self) -> u64 {
            use core::sync::atomic::Ordering;
            self.ms.load(Ordering::SeqCst)
        }

        fn saturating_add(&self, delta: u64) {
            use core::sync::atomic::Ordering;
            let _ = self
                .ms
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |cur| {
                    Some(cur.saturating_add(delta))
                });
        }
    }

    #[cfg(not(target_has_atomic = "64"))]
    #[derive(Debug, Default)]
    struct PortableU64Clock {
        lo: core::sync::atomic::AtomicU32,
        hi: core::sync::atomic::AtomicU32,
    }

    #[cfg(not(target_has_atomic = "64"))]
    impl PortableU64Clock {
        const fn new(initial: u64) -> Self {
            Self {
                lo: core::sync::atomic::AtomicU32::new(initial as u32),
                hi: core::sync::atomic::AtomicU32::new((initial >> 32) as u32),
            }
        }

        fn load(&self) -> u64 {
            use core::sync::atomic::Ordering;
            loop {
                let hi1 = self.hi.load(Ordering::Acquire);
                let lo = self.lo.load(Ordering::Acquire);
                let hi2 = self.hi.load(Ordering::Acquire);
                if hi1 == hi2 {
                    return ((hi1 as u64) << 32) | (lo as u64);
                }
            }
        }

        fn saturating_add(&self, delta: u64) {
            use core::sync::atomic::Ordering;
            let next = self.load().saturating_add(delta);
            self.hi.store((next >> 32) as u32, Ordering::Release);
            self.lo.store(next as u32, Ordering::Release);
        }
    }

    /// Mock runtime for deterministic testing without real async timers.
    #[derive(Debug, Default)]
    pub struct MockRuntime {
        current_time_ms: PortableU64Clock,
    }

    impl MockRuntime {
        /// Create a new mock runtime starting at `0 ms`.
        pub const fn new() -> Self {
            Self {
                current_time_ms: PortableU64Clock::new(0),
            }
        }

        /// Advance the mock clock by `duration`, saturating at `u64::MAX` milliseconds.
        pub fn advance(&self, duration: Duration) {
            let delta = duration_to_millis_saturating(duration);
            self.current_time_ms.saturating_add(delta);
        }
    }

    impl SwarmClock for MockRuntime {
        type SleepFuture<'a> = core::future::Ready<()>;

        fn now(&self) -> u64 {
            self.current_time_ms.load()
        }

        fn sleep(&self, duration: Duration) -> Self::SleepFuture<'_> {
            self.advance(duration);
            core::future::ready(())
        }
    }

    impl SwarmRuntime for MockRuntime {
        fn try_spawn<F>(&self, _future: F) -> core::result::Result<(), SpawnError>
        where
            F: Future<Output = ()> + Send + 'static,
        {
            Err(SpawnError::ExecutorSpawnerRequired)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mock_runtime_clamps_u128_duration_and_saturates_monotonic_clock() {
            let rt = MockRuntime::new();
            assert_eq!(rt.now(), 0);
            rt.advance(Duration::from_millis(250));
            assert_eq!(rt.now(), 250);

            // Duration::MAX has > u64::MAX milliseconds; must saturate at u64::MAX without wraparound.
            rt.advance(Duration::MAX);
            assert_eq!(rt.now(), u64::MAX);

            // Additional advance must stay at u64::MAX.
            rt.advance(Duration::from_millis(10));
            assert_eq!(rt.now(), u64::MAX);
        }
    }
}

/// Construct the Tokio runtime adapter explicitly.
#[cfg(feature = "tokio")]
pub fn default_runtime() -> tokio_runtime::TokioRuntime {
    tokio_runtime::TokioRuntime::new()
}

/// Construct the Embassy runtime adapter when `tokio` is not enabled.
#[cfg(all(feature = "embassy", not(feature = "tokio")))]
pub fn default_runtime() -> embassy_runtime::EmbassyRuntime {
    embassy_runtime::EmbassyRuntime::new()
}
