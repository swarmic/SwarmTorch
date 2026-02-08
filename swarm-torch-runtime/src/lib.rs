//! # SwarmTorch Runtime
//!
//! Async runtime abstractions for SwarmTorch.
//!
//! This crate provides a unified interface for async operations across:
//! - **Tokio**: For server and edge deployments (std)
//! - **Embassy**: For embedded microcontrollers (no_std)
//!
//! ## Feature Flags
//!
//! - `tokio` (default): Use Tokio runtime
//! - `embassy`: Use Embassy runtime for embedded

#![cfg_attr(not(feature = "std"), no_std)]

use core::future::Future;
use core::time::Duration;

/// Runtime trait for async operations
pub trait SwarmRuntime: Send + Sync + 'static {
    /// Get the current instant
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
        fn now(&self) -> u64 {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
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

    impl EmbassyRuntime {
        /// Create a new Embassy runtime wrapper
        pub fn new() -> Self {
            Self { _private: () }
        }
    }

    impl SwarmRuntime for EmbassyRuntime {
        fn now(&self) -> u64 {
            // Would use embassy_time::Instant::now()
            0
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
            self.current_time_ms.fetch_add(
                duration.as_millis() as u64,
                Ordering::SeqCst,
            );
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
