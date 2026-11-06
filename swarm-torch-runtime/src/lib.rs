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

/// Deterministic mock runtime for testing (`ADR-0003`, `W9-04`, `W10-03`).
///
/// Uses `AtomicU64` when `target_has_atomic = "64"` is available, and falls back to
/// `AtomicU32` pairs on 32-bit embedded targets (`thumbv6m`, `riscv32i`) so `MockRuntime`
/// remains `Send + Sync` without requiring 64-bit hardware atomics.
///
/// When compiled with `feature = "std"`, `MockRuntime` also provides a bounded,
/// deterministic FIFO task queue and deadline-ordered timer waker registry using
/// `std::task::Wake` (`#![forbid(unsafe_code)]`).
pub mod mock_runtime {
    use super::*;

    #[cfg(feature = "std")]
    use std::collections::{BTreeMap, BTreeSet, VecDeque};
    #[cfg(feature = "std")]
    use std::pin::Pin;
    #[cfg(feature = "std")]
    use std::sync::{Arc, Mutex};
    #[cfg(feature = "std")]
    use std::task::{Context, Poll, Wake, Waker};

    /// Default maximum number of concurrent spawned tasks in [`MockRuntime`].
    pub const DEFAULT_MAX_MOCK_TASKS: usize = 256;

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
            // Explicit CAS loop: `fetch_update` is deprecated in favour of
            // `try_update` on newer toolchains, but `try_update` is unavailable
            // on the 1.75 MSRV. This loop is semantically identical on both.
            let mut cur = self.ms.load(Ordering::SeqCst);
            loop {
                match self.ms.compare_exchange_weak(
                    cur,
                    cur.saturating_add(delta),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(observed) => cur = observed,
                }
            }
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

    #[cfg(feature = "std")]
    type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

    #[cfg(feature = "std")]
    struct MockSchedulerInner {
        max_tasks: usize,
        deterministic_timers: bool,
        next_task_id: u64,
        next_timer_id: u64,
        active_task_count: usize,
        tasks: BTreeMap<u64, BoxFuture>,
        runnable_queue: VecDeque<u64>,
        runnable_set: BTreeSet<u64>,
        timers: BTreeMap<(u64, u64), Waker>,
    }

    #[cfg(feature = "std")]
    impl MockSchedulerInner {
        fn new(max_tasks: usize, deterministic_timers: bool) -> Self {
            Self {
                max_tasks,
                deterministic_timers,
                next_task_id: 1,
                next_timer_id: 1,
                active_task_count: 0,
                tasks: BTreeMap::new(),
                runnable_queue: VecDeque::new(),
                runnable_set: BTreeSet::new(),
                timers: BTreeMap::new(),
            }
        }

        fn enqueue_runnable(&mut self, task_id: u64) {
            if self.runnable_set.insert(task_id) {
                self.runnable_queue.push_back(task_id);
            }
        }
    }

    #[cfg(feature = "std")]
    struct MockTaskWaker {
        task_id: u64,
        inner: Arc<Mutex<MockSchedulerInner>>,
    }

    #[cfg(feature = "std")]
    impl Wake for MockTaskWaker {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            if let Ok(mut guard) = self.inner.lock() {
                guard.enqueue_runnable(self.task_id);
            }
        }
    }

    /// Future returned by [`MockRuntime::sleep`].
    pub struct MockSleepFuture<'a> {
        runtime: &'a MockRuntime,
        deadline_ms: u64,
        #[cfg(feature = "std")]
        timer_id: Option<u64>,
        #[cfg(feature = "std")]
        deterministic: bool,
        completed: bool,
    }

    impl<'a> Future for MockSleepFuture<'a> {
        type Output = ();

        fn poll(
            mut self: core::pin::Pin<&mut Self>,
            #[cfg_attr(not(feature = "std"), allow(unused_variables))] cx: &mut core::task::Context<
                '_,
            >,
        ) -> core::task::Poll<Self::Output> {
            if self.completed {
                return core::task::Poll::Ready(());
            }
            #[cfg(feature = "std")]
            if self.deterministic {
                if self.runtime.now() >= self.deadline_ms {
                    if let Some(tid) = self.timer_id.take() {
                        if let Some(ref sched) = self.runtime.scheduler {
                            if let Ok(mut guard) = sched.lock() {
                                guard.timers.remove(&(self.deadline_ms, tid));
                            }
                        }
                    }
                    self.completed = true;
                    return Poll::Ready(());
                }
                if let Some(ref sched) = self.runtime.scheduler {
                    if let Ok(mut guard) = sched.lock() {
                        let tid = match self.timer_id {
                            Some(id) => id,
                            None => {
                                let id = guard.next_timer_id;
                                guard.next_timer_id = guard.next_timer_id.saturating_add(1);
                                self.timer_id = Some(id);
                                id
                            }
                        };
                        guard
                            .timers
                            .insert((self.deadline_ms, tid), cx.waker().clone());
                    }
                }
                return Poll::Pending;
            }

            #[cfg(not(feature = "std"))]
            {
                let _ = (self.runtime.now(), self.deadline_ms);
            }
            self.completed = true;
            core::task::Poll::Ready(())
        }
    }

    /// Mock runtime for deterministic testing without real async timers (`ADR-0003`, `W10-03`).
    pub struct MockRuntime {
        current_time_ms: PortableU64Clock,
        #[cfg(feature = "std")]
        scheduler: Option<Arc<Mutex<MockSchedulerInner>>>,
    }

    impl core::fmt::Debug for MockRuntime {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("MockRuntime")
                .field("current_time_ms", &self.now())
                .finish()
        }
    }

    impl Default for MockRuntime {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MockRuntime {
        /// Create a new mock runtime starting at `0 ms` with default task capacity (`256`).
        #[cfg(feature = "std")]
        pub fn new() -> Self {
            Self {
                current_time_ms: PortableU64Clock::new(0),
                scheduler: Some(Arc::new(Mutex::new(MockSchedulerInner::new(
                    DEFAULT_MAX_MOCK_TASKS,
                    false,
                )))),
            }
        }

        /// Create a new mock runtime starting at `0 ms` (`no_std`).
        #[cfg(not(feature = "std"))]
        pub const fn new() -> Self {
            Self {
                current_time_ms: PortableU64Clock::new(0),
            }
        }

        /// Create a mock runtime with deterministic deadline-ordered timer futures and
        /// configurable task queue capacity (`ADR-0003`, `W10-03`).
        #[cfg(feature = "std")]
        pub fn with_deterministic_scheduler(max_tasks: usize) -> Self {
            Self {
                current_time_ms: PortableU64Clock::new(0),
                scheduler: Some(Arc::new(Mutex::new(MockSchedulerInner::new(
                    max_tasks, true,
                )))),
            }
        }

        /// Advance the mock clock by `duration`, saturating at `u64::MAX` milliseconds,
        /// and wake any expired timers in deterministic `(deadline_ms, timer_id)` order.
        pub fn advance(&self, duration: Duration) {
            let delta = duration_to_millis_saturating(duration);
            self.current_time_ms.saturating_add(delta);
            #[cfg(feature = "std")]
            self.wake_expired_timers();
        }

        #[cfg(feature = "std")]
        fn wake_expired_timers(&self) {
            let now = self.now();
            let Some(ref sched) = self.scheduler else {
                return;
            };
            let wakers: std::vec::Vec<Waker> = if let Ok(mut guard) = sched.lock() {
                let expired_keys: std::vec::Vec<(u64, u64)> = guard
                    .timers
                    .keys()
                    .copied()
                    .take_while(|(deadline_ms, _)| *deadline_ms <= now)
                    .collect();
                let mut out = std::vec::Vec::with_capacity(expired_keys.len());
                for key in expired_keys {
                    if let Some(w) = guard.timers.remove(&key) {
                        out.push(w);
                    }
                }
                out
            } else {
                return;
            };
            for waker in wakers {
                waker.wake();
            }
        }

        /// Poll all currently runnable tasks in deterministic FIFO order until the runnable
        /// queue drains or `max_steps` is reached, returning the number of task polls executed.
        #[cfg(feature = "std")]
        pub fn poll_ready_tasks(&self) -> usize {
            self.wake_expired_timers();
            let Some(ref sched) = self.scheduler else {
                return 0;
            };
            let mut polls = 0usize;
            let max_steps = 10_000usize;
            while polls < max_steps {
                let next_item = {
                    let Ok(mut guard) = sched.lock() else {
                        break;
                    };
                    let Some(task_id) = guard.runnable_queue.pop_front() else {
                        break;
                    };
                    guard.runnable_set.remove(&task_id);
                    guard.tasks.remove(&task_id).map(|fut| (task_id, fut))
                };
                let Some((task_id, mut fut)) = next_item else {
                    continue;
                };
                let waker = Waker::from(Arc::new(MockTaskWaker {
                    task_id,
                    inner: Arc::clone(sched),
                }));
                let mut cx = Context::from_waker(&waker);
                polls += 1;
                match fut.as_mut().poll(&mut cx) {
                    Poll::Ready(()) => {
                        if let Ok(mut guard) = sched.lock() {
                            guard.active_task_count = guard.active_task_count.saturating_sub(1);
                            guard.runnable_set.remove(&task_id);
                        }
                    }
                    Poll::Pending => {
                        if let Ok(mut guard) = sched.lock() {
                            guard.tasks.insert(task_id, fut);
                        }
                    }
                }
                self.wake_expired_timers();
            }
            polls
        }

        /// Advance the mock clock by `duration` and poll all newly ready tasks to quiescence.
        #[cfg(feature = "std")]
        pub fn advance_and_run(&self, duration: Duration) -> usize {
            self.advance(duration);
            self.poll_ready_tasks()
        }

        /// Number of unfinished tasks currently registered in the scheduler.
        #[cfg(feature = "std")]
        pub fn active_task_count(&self) -> usize {
            self.scheduler
                .as_ref()
                .and_then(|s| s.lock().ok().map(|g| g.active_task_count))
                .unwrap_or(0)
        }
    }

    impl SwarmClock for MockRuntime {
        type SleepFuture<'a> = MockSleepFuture<'a>;

        fn now(&self) -> u64 {
            self.current_time_ms.load()
        }

        fn sleep(&self, duration: Duration) -> Self::SleepFuture<'_> {
            let delta = duration_to_millis_saturating(duration);
            let deadline_ms = self.now().saturating_add(delta);
            #[cfg(feature = "std")]
            {
                let deterministic = self
                    .scheduler
                    .as_ref()
                    .and_then(|s| s.lock().ok().map(|g| g.deterministic_timers))
                    .unwrap_or(false);
                if !deterministic {
                    self.advance(duration);
                }
                MockSleepFuture {
                    runtime: self,
                    deadline_ms,
                    timer_id: None,
                    deterministic,
                    completed: false,
                }
            }
            #[cfg(not(feature = "std"))]
            {
                self.advance(duration);
                MockSleepFuture {
                    runtime: self,
                    deadline_ms,
                    completed: false,
                }
            }
        }
    }

    impl SwarmRuntime for MockRuntime {
        #[cfg(feature = "std")]
        fn try_spawn<F>(&self, future: F) -> core::result::Result<(), SpawnError>
        where
            F: Future<Output = ()> + Send + 'static,
        {
            let Some(ref sched) = self.scheduler else {
                return Err(SpawnError::ExecutorSpawnerRequired);
            };
            let mut guard = sched.lock().map_err(|_| SpawnError::RuntimeUnavailable)?;
            if guard.max_tasks == 0 {
                return Err(SpawnError::CapacityExhausted);
            }
            if guard.active_task_count >= guard.max_tasks {
                return Err(SpawnError::CapacityExhausted);
            }
            let task_id = guard.next_task_id;
            guard.next_task_id = guard.next_task_id.saturating_add(1);
            guard.active_task_count += 1;
            guard.tasks.insert(task_id, Box::pin(future));
            guard.enqueue_runnable(task_id);
            Ok(())
        }

        #[cfg(not(feature = "std"))]
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

        #[cfg(feature = "std")]
        #[test]
        fn mock_runtime_deterministic_scheduler_orders_timers_and_enforces_capacity() {
            let rt = Arc::new(MockRuntime::with_deterministic_scheduler(2));
            let events = Arc::new(Mutex::new(std::vec::Vec::<u32>::new()));

            let rt1 = Arc::clone(&rt);
            let ev1 = Arc::clone(&events);
            rt.try_spawn(async move {
                rt1.sleep(Duration::from_millis(50)).await;
                ev1.lock().unwrap().push(50);
            })
            .unwrap();

            let rt2 = Arc::clone(&rt);
            let ev2 = Arc::clone(&events);
            rt.try_spawn(async move {
                rt2.sleep(Duration::from_millis(20)).await;
                ev2.lock().unwrap().push(20);
            })
            .unwrap();

            // Third spawn exceeds max_tasks == 2 -> CapacityExhausted!
            assert_eq!(rt.try_spawn(async {}), Err(SpawnError::CapacityExhausted));

            // Initial poll registers both timers without completing either.
            rt.poll_ready_tasks();
            assert!(events.lock().unwrap().is_empty());
            assert_eq!(rt.active_task_count(), 2);

            // Advance 25ms -> task 2 (20ms) wakes and finishes; task 1 (50ms) stays pending.
            rt.advance_and_run(Duration::from_millis(25));
            assert_eq!(*events.lock().unwrap(), std::vec![20]);
            assert_eq!(rt.active_task_count(), 1);

            // Advance another 30ms (total 55ms) -> task 1 (50ms) wakes and finishes.
            rt.advance_and_run(Duration::from_millis(30));
            assert_eq!(*events.lock().unwrap(), std::vec![20, 50]);
            assert_eq!(rt.active_task_count(), 0);
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
