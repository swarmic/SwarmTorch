//! Bounded deduplicated waiter helpers for evented `swarm-torch-net` transports (`ST-06`).

use core::task::Waker;
use std::collections::VecDeque;
#[cfg(feature = "tcp-transport")]
use std::sync::Arc;
#[cfg(feature = "tcp-transport")]
use std::task::Wake;

/// Hard bound on distinct queued wakers per virtual endpoint queue.
const MAX_QUEUED_WAITERS: usize = 256;

/// Safe no-op waker target used only for synchronous non-blocking polls on promoted Tokio sockets.
#[cfg(feature = "tcp-transport")]
pub(crate) struct NoopWake;

#[cfg(feature = "tcp-transport")]
impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

/// Bounded deduplicated FIFO of async task [`Waker`] handles.
#[derive(Debug, Default)]
pub(crate) struct WaiterQueue {
    waiters: VecDeque<Waker>,
}

impl WaiterQueue {
    /// Register `waker` if an equivalent handle (`Waker::will_wake`) is not already queued.
    pub(crate) fn register(&mut self, waker: &Waker) {
        if self
            .waiters
            .iter()
            .any(|existing| existing.will_wake(waker))
        {
            return;
        }
        if self.waiters.len() >= MAX_QUEUED_WAITERS {
            let _ = self.waiters.pop_front();
        }
        self.waiters.push_back(waker.clone());
    }

    /// Drain all registered wakers so callers can invoke [`Waker::wake`] after dropping locks.
    pub(crate) fn take_all(&mut self) -> Vec<Waker> {
        self.waiters.drain(..).collect()
    }
}

/// Wake every waker in `wakers`. Must be called with transport mutexes released.
pub(crate) fn wake_all(wakers: Vec<Waker>) {
    for waker in wakers {
        waker.wake();
    }
}
