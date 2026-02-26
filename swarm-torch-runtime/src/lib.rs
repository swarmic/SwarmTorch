//! # SwarmTorch Runtime
//!
//! Async runtime abstractions for SwarmTorch.
//!
//! This crate provides a unified interface for async operations across:
//! - **Tokio**: Implemented for server/edge deployments (std)
//! - **Embassy**: Placeholder adapter for embedded microcontrollers (no_std path is partial)
//!
//! Current conformance posture:
//! - Tokio path is validated on Rust 1.75.
//! - Embassy path is experimental and not part of the Rust 1.75 conformance gate.
//!
//! ## Feature Flags
//!
//! - `tokio` (default): Use Tokio runtime
//! - `embassy`: Enable embassy placeholder runtime adapter (experimental)

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

use core::future::Future;
use core::time::Duration;

/// Runtime trait for async operations
pub trait SwarmRuntime: Send + Sync + 'static {
    /// Returns a monotonically non-decreasing timestamp in milliseconds.
    ///
    /// The epoch is **implementation-defined** (not necessarily UNIX epoch).
    /// Callers must not interpret this value as wall-clock time.
    /// Implementations must guarantee the value never decreases between calls.
    fn now(&self) -> u64;

    /// Sleep for the specified duration
    fn sleep(&self, duration: Duration) -> impl Future<Output = ()> + Send;

    /// Spawn a task (if supported by the runtime)
    fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static;
}

#[cfg(feature = "tokio")]
pub mod tokio_runtime {
    //! Tokio-based runtime implementation

    use super::*;
    use std::sync::OnceLock;
    use std::time::Instant;

    /// Process-local epoch for monotonic clock derivation.
    /// Initialized on first call to `TokioRuntime::now()`.
    static EPOCH: OnceLock<Instant> = OnceLock::new();

    /// Tokio runtime wrapper
    #[derive(Debug, Clone, Default)]
    pub struct TokioRuntime;

    impl TokioRuntime {
        /// Create a new Tokio runtime wrapper
        pub fn new() -> Self {
            Self
        }
    }

    impl SwarmRuntime for TokioRuntime {
        /// Returns monotonically non-decreasing milliseconds since process-local epoch.
        ///
        /// **Contract**: The returned value is **not** UNIX epoch wall-clock time.
        /// It is derived from `std::time::Instant` and is guaranteed to never
        /// decrease between calls, even across NTP step-backs.
        ///
        /// Consumers needing absolute wall-clock time must use `SystemTime` directly.
        fn now(&self) -> u64 {
            let epoch = EPOCH.get_or_init(Instant::now);
            epoch.elapsed().as_millis() as u64
        }

        async fn sleep(&self, duration: Duration) {
            tokio::time::sleep(duration).await;
        }

        fn spawn<F>(&self, future: F)
        where
            F: Future<Output = ()> + Send + 'static,
        {
            tokio::spawn(future);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn tokio_runtime_now_is_monotonic() {
            let rt = TokioRuntime::new();
            let t1 = rt.now();
            // Busy-spin briefly to ensure elapsed time
            for _ in 0..10_000 {
                core::hint::black_box(0u64);
            }
            let t2 = rt.now();
            assert!(t2 >= t1, "now() must be monotonically non-decreasing");
        }
    }
}

#[cfg(feature = "embassy")]
pub mod embassy_runtime {
    //! Embassy-based runtime implementation for embedded

    use super::*;

    /// Embassy runtime wrapper
    pub struct EmbassyRuntime {
        // Embassy spawner would go here
        _private: (),
    }

    impl Default for EmbassyRuntime {
        fn default() -> Self {
            Self::new()
        }
    }

    impl EmbassyRuntime {
        /// Create a new Embassy runtime wrapper
        pub fn new() -> Self {
            Self { _private: () }
        }
    }

    impl SwarmRuntime for EmbassyRuntime {
        fn now(&self) -> u64 {
            // Embassy clock integration is not yet wired — returning zero
            // would silently falsify every timestamp in the protocol layer.
            // Fail-fast until a real embassy_time::Instant driver is available.
            unimplemented!(
                "EmbassyRuntime::now() is a placeholder; \
                 wire embassy_time::Instant before using this runtime"
            )
        }

        async fn sleep(&self, duration: Duration) {
            // Would use embassy_time::Timer::after(duration).await
            let _ = duration;
        }

        fn spawn<F>(&self, _future: F)
        where
            F: Future<Output = ()> + Send + 'static,
        {
            // Embassy requires static tasks - this is a simplified placeholder
            // Real implementation would use spawner.spawn()
        }
    }
}

/// Mock runtime for testing
pub mod mock_runtime {
    use super::*;

    /// Mock runtime for testing without real async
    #[derive(Debug, Default)]
    pub struct MockRuntime {
        current_time_ms: core::sync::atomic::AtomicU64,
    }

    impl MockRuntime {
        /// Create a new mock runtime
        pub fn new() -> Self {
            Self {
                current_time_ms: core::sync::atomic::AtomicU64::new(0),
            }
        }

        /// Advance the mock clock
        pub fn advance(&self, duration: Duration) {
            use core::sync::atomic::Ordering;
            self.current_time_ms
                .fetch_add(duration.as_millis() as u64, Ordering::SeqCst);
        }
    }

    impl SwarmRuntime for MockRuntime {
        fn now(&self) -> u64 {
            use core::sync::atomic::Ordering;
            self.current_time_ms.load(Ordering::SeqCst)
        }

        async fn sleep(&self, _duration: Duration) {
            // Mock sleep does nothing
        }

        fn spawn<F>(&self, _future: F)
        where
            F: Future<Output = ()> + Send + 'static,
        {
            // Mock spawn does nothing
        }
    }
}

/// Get the default runtime based on features
#[cfg(feature = "tokio")]
pub fn default_runtime() -> tokio_runtime::TokioRuntime {
    tokio_runtime::TokioRuntime::new()
}

#[cfg(all(feature = "embassy", not(feature = "tokio")))]
pub fn default_runtime() -> embassy_runtime::EmbassyRuntime {
    embassy_runtime::EmbassyRuntime::new()
}
