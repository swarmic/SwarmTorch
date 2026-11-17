//! # SwarmTorch Runtime (`ADR-0003`, `W9-04`, `P2-06`)
//!
//! Async runtime abstractions for SwarmTorch.
//!
//! Separates monotonic clock/timer progression from fallible task admission (`try_spawn`),
//! clamps `u128` millisecond conversions without narrowing wraparound, aligns
//! [`tokio_runtime::TokioRuntime`] to `tokio::time::Instant`, and uses `portable-atomic`
//! (`fallback` + `critical-section`) when `#[cfg(not(target_has_atomic = "64"))]` (`Decision A6`).

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
    /// Handle representing an admitted task on the concrete runtime executor.
    type TaskHandle: Send + 'static;

    /// Attempt to admit a background task, returning an explicit [`SpawnError`] if unsupported or full.
    fn try_spawn<F>(&self, future: F) -> core::result::Result<Self::TaskHandle, SpawnError>
    where
        F: Future<Output = ()> + Send + 'static;
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
        type TaskHandle = tokio::task::JoinHandle<()>;

        fn try_spawn<F>(&self, future: F) -> core::result::Result<Self::TaskHandle, SpawnError>
        where
            F: Future<Output = ()> + Send + 'static,
        {
            if tokio::runtime::Handle::try_current().is_err() {
                return Err(SpawnError::RuntimeUnavailable);
            }
            Ok(tokio::spawn(future))
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
            assert!(matches!(res, Err(SpawnError::RuntimeUnavailable)));
        }

        #[tokio::test]
        async fn tokio_try_spawn_returns_joinable_task_handle() {
            let rt = TokioRuntime::new();
            let flag = std::sync::Arc::new(core::sync::atomic::AtomicBool::new(false));
            let flag_clone = std::sync::Arc::clone(&flag);
            let handle = rt
                .try_spawn(async move {
                    flag_clone.store(true, core::sync::atomic::Ordering::SeqCst);
                })
                .expect("TokioRuntime::try_spawn inside runtime must succeed");
            handle.await.expect("spawned task must finish cleanly");
            assert!(flag.load(core::sync::atomic::Ordering::SeqCst));
        }
    }
}

#[cfg(feature = "embassy")]
pub mod embassy_runtime {
    //! Embassy-based runtime adapter (`ADR-0003`, `P2-06`, `ST-07`).
    //!
    //! Wires [`SwarmClock`] directly to [`embassy_time::Instant`] (or an explicit hardware clock
    //! hook when provided) and [`embassy_time::Timer`], and supports static task admission through
    //! [`EmbassyRuntime::try_spawn_task`] backed by [`embassy_executor::SendSpawner`]. Dynamic
    //! [`SwarmRuntime::try_spawn`] cannot synthesize Embassy static `'static` task pools and
    //! returns [`SpawnError::ExecutorSpawnerRequired`] rather than discarding tasks.

    use super::*;

    fn duration_to_embassy(duration: Duration) -> embassy_time::Duration {
        let millis = duration_to_millis_saturating(duration);
        if millis >= embassy_time::Duration::MAX.as_millis() {
            embassy_time::Duration::MAX
        } else {
            embassy_time::Duration::from_millis(millis)
        }
    }

    /// Embassy runtime wrapper supporting [`embassy_time`] timers and [`embassy_executor::SendSpawner`].
    #[derive(Clone)]
    pub struct EmbassyRuntime {
        clock_fn: Option<fn() -> u64>,
        spawner: Option<embassy_executor::SendSpawner>,
    }

    impl core::fmt::Debug for EmbassyRuntime {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("EmbassyRuntime")
                .field("has_clock_driver", &self.clock_fn.is_some())
                .field("has_spawner", &self.spawner.is_some())
                .finish()
        }
    }

    impl Default for EmbassyRuntime {
        fn default() -> Self {
            Self::new()
        }
    }

    impl EmbassyRuntime {
        /// Create a new Embassy runtime adapter backed by [`embassy_time`] clocks/timers without an attached executor spawner.
        pub const fn new() -> Self {
            Self {
                clock_fn: None,
                spawner: None,
            }
        }

        /// Create a new Embassy runtime adapter backed by an explicit monotonic millisecond clock function.
        pub const fn with_clock(clock_fn: fn() -> u64) -> Self {
            Self {
                clock_fn: Some(clock_fn),
                spawner: None,
            }
        }

        /// Create an Embassy runtime adapter attached to an [`embassy_executor::SendSpawner`].
        pub const fn with_spawner(spawner: embassy_executor::SendSpawner) -> Self {
            Self {
                clock_fn: None,
                spawner: Some(spawner),
            }
        }

        /// Create an Embassy runtime adapter with both an [`embassy_executor::SendSpawner`] and custom clock hook.
        pub const fn with_spawner_and_clock(
            spawner: embassy_executor::SendSpawner,
            clock_fn: fn() -> u64,
        ) -> Self {
            Self {
                clock_fn: Some(clock_fn),
                spawner: Some(spawner),
            }
        }

        /// Attempt to spawn a static `#[embassy_executor::task]` into the attached [`embassy_executor::SendSpawner`].
        ///
        /// Accepts a constructor closure so [`embassy_executor::SpawnToken`] is never instantiated
        /// when no spawner is configured (preventing [`embassy_executor::SpawnToken`] drop panics).
        pub fn try_spawn_task<S: Send>(
            &self,
            make_token: impl FnOnce() -> embassy_executor::SpawnToken<S>,
        ) -> core::result::Result<(), SpawnError> {
            let Some(ref spawner) = self.spawner else {
                return Err(SpawnError::ExecutorSpawnerRequired);
            };
            spawner.spawn(make_token()).map_err(|err| match err {
                embassy_executor::SpawnError::Busy => SpawnError::CapacityExhausted,
            })
        }
    }

    impl SwarmClock for EmbassyRuntime {
        type SleepFuture<'a> = embassy_time::Timer;

        fn now(&self) -> u64 {
            match self.clock_fn {
                Some(f) => f(),
                None => embassy_time::Instant::now().as_millis(),
            }
        }

        fn sleep(&self, duration: Duration) -> Self::SleepFuture<'_> {
            embassy_time::Timer::after(duration_to_embassy(duration))
        }
    }

    impl SwarmRuntime for EmbassyRuntime {
        type TaskHandle = ();

        fn try_spawn<F>(&self, _future: F) -> core::result::Result<Self::TaskHandle, SpawnError>
        where
            F: Future<Output = ()> + Send + 'static,
        {
            Err(SpawnError::ExecutorSpawnerRequired)
        }
    }

    #[cfg(all(test, feature = "embassy-std-test"))]
    mod tests {
        use super::*;

        #[test]
        fn duration_to_embassy_saturates_safely_at_max() {
            let clamped = duration_to_embassy(Duration::MAX);
            assert_eq!(clamped, embassy_time::Duration::MAX);
            let small = duration_to_embassy(Duration::from_millis(25));
            assert_eq!(small.as_millis(), 25);
        }

        #[test]
        fn embassy_runtime_without_spawner_rejects_dynamic_spawn_and_task_closure() {
            let rt = EmbassyRuntime::new();
            assert_eq!(
                rt.try_spawn(async {}),
                Err(SpawnError::ExecutorSpawnerRequired)
            );
            assert_eq!(
                rt.try_spawn_task(embassy_executor::SpawnToken::<()>::new_failed),
                Err(SpawnError::ExecutorSpawnerRequired)
            );
        }

        #[tokio::test]
        async fn embassy_runtime_clock_and_timer_progress_under_std_driver() {
            let rt = EmbassyRuntime::new();
            let t0 = rt.now();
            rt.sleep(Duration::from_millis(5)).await;
            let t1 = rt.now();
            assert!(
                t1 >= t0,
                "Embassy Instant clock must progress monotonically across Timer::after"
            );
        }
    }
}

/// Mock runtime module for deterministic testing (`ADR-0003`, `W9-04`, `W10-03`).
///
/// Uses `core::sync::atomic::AtomicU64` when `target_has_atomic = "64"` and
/// `portable_atomic::AtomicU64` (`fallback` + `critical-section`) on 32-bit targets
/// lacking hardware 64-bit atomics (`Decision A6`), avoiding non-atomic RMW races.
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

    /// Default maximum number of pending deterministic timers retained in [`MockRuntime`].
    pub const DEFAULT_MAX_MOCK_TIMERS: usize = 4_096;

    #[cfg(target_has_atomic = "64")]
    use core::sync::atomic::{AtomicU64 as InnerAtomicU64, Ordering as AtomicU64Ordering};
    #[cfg(not(target_has_atomic = "64"))]
    use portable_atomic::{AtomicU64 as InnerAtomicU64, Ordering as AtomicU64Ordering};

    #[derive(Debug, Default)]
    struct PortableU64Clock {
        ms: InnerAtomicU64,
    }

    impl PortableU64Clock {
        const fn new(initial: u64) -> Self {
            Self {
                ms: InnerAtomicU64::new(initial),
            }
        }

        fn load(&self) -> u64 {
            self.ms.load(AtomicU64Ordering::SeqCst)
        }

        fn saturating_add(&self, delta: u64) {
            let mut cur = self.ms.load(AtomicU64Ordering::SeqCst);
            loop {
                match self.ms.compare_exchange_weak(
                    cur,
                    cur.saturating_add(delta),
                    AtomicU64Ordering::SeqCst,
                    AtomicU64Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(observed) => cur = observed,
                }
            }
        }
    }

    #[cfg(feature = "std")]
    type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

    #[cfg(feature = "std")]
    struct MockSchedulerInner {
        max_tasks: usize,
        max_timers: usize,
        deterministic_timers: bool,
        next_task_id: u64,
        next_timer_id: u64,
        active_task_count: usize,
        rejected_timer_registrations: u64,
        tasks: BTreeMap<u64, BoxFuture>,
        in_flight_tasks: BTreeSet<u64>,
        rewake_pending: BTreeSet<u64>,
        runnable_queue: VecDeque<u64>,
        runnable_set: BTreeSet<u64>,
        timers: BTreeMap<(u64, u64), Waker>,
    }

    #[cfg(feature = "std")]
    impl MockSchedulerInner {
        fn new(max_tasks: usize, max_timers: usize, deterministic_timers: bool) -> Self {
            Self {
                max_tasks,
                max_timers,
                deterministic_timers,
                next_task_id: 1,
                next_timer_id: 1,
                active_task_count: 0,
                rejected_timer_registrations: 0,
                tasks: BTreeMap::new(),
                in_flight_tasks: BTreeSet::new(),
                rewake_pending: BTreeSet::new(),
                runnable_queue: VecDeque::new(),
                runnable_set: BTreeSet::new(),
                timers: BTreeMap::new(),
            }
        }

        fn enqueue_runnable(&mut self, task_id: u64) {
            if self.in_flight_tasks.contains(&task_id) {
                // Defer re-enqueue until the currently executing poll returns the future,
                // preventing another worker thread from consuming the wake and dropping it.
                self.rewake_pending.insert(task_id);
                return;
            }
            if !self.tasks.contains_key(&task_id) {
                return;
            }
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
                if let Some(ref sched) = self.runtime.scheduler {
                    if let Ok(mut guard) = sched.lock() {
                        // Lock `sched` BEFORE re-reading `runtime.now()` (`SeqCst`). Since
                        // `advance()` stores `current_time_ms` BEFORE locking `sched` in
                        // `wake_expired_timers()`, no wakeup can slip through TOCTOU between
                        // reading `now` and inserting the timer.
                        if self.runtime.now() >= self.deadline_ms {
                            if let Some(tid) = self.timer_id.take() {
                                guard.timers.remove(&(self.deadline_ms, tid));
                            }
                            drop(guard);
                            self.completed = true;
                            return Poll::Ready(());
                        }
                        if let Some(tid) = self.timer_id {
                            guard
                                .timers
                                .insert((self.deadline_ms, tid), cx.waker().clone());
                        } else if guard.timers.len() >= guard.max_timers {
                            guard.rejected_timer_registrations =
                                guard.rejected_timer_registrations.saturating_add(1);
                        } else if let Some(next_id) = guard.next_timer_id.checked_add(1) {
                            let tid = guard.next_timer_id;
                            guard.next_timer_id = next_id;
                            self.timer_id = Some(tid);
                            guard
                                .timers
                                .insert((self.deadline_ms, tid), cx.waker().clone());
                        } else {
                            guard.rejected_timer_registrations =
                                guard.rejected_timer_registrations.saturating_add(1);
                        }
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

    #[cfg(feature = "std")]
    impl Drop for MockSleepFuture<'_> {
        fn drop(&mut self) {
            if self.deterministic {
                if let Some(tid) = self.timer_id.take() {
                    if let Some(ref sched) = self.runtime.scheduler {
                        if let Ok(mut guard) = sched.lock() {
                            guard.timers.remove(&(self.deadline_ms, tid));
                        }
                    }
                }
            }
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
                    DEFAULT_MAX_MOCK_TIMERS,
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
            Self::with_deterministic_scheduler_limits(max_tasks, DEFAULT_MAX_MOCK_TIMERS)
        }

        /// Create a mock runtime with deterministic deadline-ordered timer futures and explicit
        /// bounded `max_tasks` and `max_timers` capacities.
        #[cfg(feature = "std")]
        pub fn with_deterministic_scheduler_limits(max_tasks: usize, max_timers: usize) -> Self {
            Self {
                current_time_ms: PortableU64Clock::new(0),
                scheduler: Some(Arc::new(Mutex::new(MockSchedulerInner::new(
                    max_tasks, max_timers, true,
                )))),
            }
        }

        /// Number of pending deterministic timers currently registered in the scheduler.
        #[cfg(feature = "std")]
        pub fn pending_timer_count(&self) -> usize {
            self.scheduler
                .as_ref()
                .and_then(|s| s.lock().ok().map(|g| g.timers.len()))
                .unwrap_or(0)
        }

        /// Total number of new timer registrations rejected due to timer capacity or ID limits.
        #[cfg(feature = "std")]
        pub fn rejected_timer_registrations(&self) -> u64 {
            self.scheduler
                .as_ref()
                .and_then(|s| s.lock().ok().map(|g| g.rejected_timer_registrations))
                .unwrap_or(0)
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
            let Some(ref sched) = self.scheduler else {
                return;
            };
            let wakers: std::vec::Vec<Waker> = if let Ok(mut guard) = sched.lock() {
                if guard.timers.is_empty() {
                    return;
                }
                let now = self.now();
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
                    if guard.in_flight_tasks.contains(&task_id) {
                        guard.rewake_pending.insert(task_id);
                        None
                    } else if let Some(fut) = guard.tasks.remove(&task_id) {
                        guard.in_flight_tasks.insert(task_id);
                        Some((task_id, fut))
                    } else {
                        None
                    }
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
                            guard.in_flight_tasks.remove(&task_id);
                            guard.rewake_pending.remove(&task_id);
                            guard.active_task_count = guard.active_task_count.saturating_sub(1);
                            guard.runnable_set.remove(&task_id);
                        }
                    }
                    Poll::Pending => {
                        if let Ok(mut guard) = sched.lock() {
                            guard.in_flight_tasks.remove(&task_id);
                            guard.tasks.insert(task_id, fut);
                            if guard.rewake_pending.remove(&task_id) {
                                guard.enqueue_runnable(task_id);
                            }
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

    /// Handle identifying a task admitted to [`MockRuntime`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MockTaskHandle {
        id: u64,
    }

    impl MockTaskHandle {
        /// Monotonic task identifier assigned at admission time.
        pub const fn id(&self) -> u64 {
            self.id
        }
    }

    impl SwarmRuntime for MockRuntime {
        type TaskHandle = MockTaskHandle;

        #[cfg(feature = "std")]
        fn try_spawn<F>(&self, future: F) -> core::result::Result<Self::TaskHandle, SpawnError>
        where
            F: Future<Output = ()> + Send + 'static,
        {
            let Some(ref sched) = self.scheduler else {
                return Err(SpawnError::ExecutorSpawnerRequired);
            };
            let mut guard = sched.lock().map_err(|_| SpawnError::RuntimeUnavailable)?;
            if guard.max_tasks == 0 || guard.active_task_count >= guard.max_tasks {
                drop(guard);
                return Err(SpawnError::CapacityExhausted);
            }
            let Some(next_id) = guard.next_task_id.checked_add(1) else {
                drop(guard);
                return Err(SpawnError::CapacityExhausted);
            };
            let task_id = guard.next_task_id;
            guard.next_task_id = next_id;
            guard.active_task_count += 1;
            guard.tasks.insert(task_id, Box::pin(future));
            guard.enqueue_runnable(task_id);
            Ok(MockTaskHandle { id: task_id })
        }

        #[cfg(not(feature = "std"))]
        fn try_spawn<F>(&self, _future: F) -> core::result::Result<Self::TaskHandle, SpawnError>
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

        #[cfg(feature = "std")]
        #[test]
        fn mock_sleep_future_drop_cancels_timer_and_enforces_max_timers() {
            use core::sync::atomic::{AtomicUsize, Ordering};

            struct CountWake(AtomicUsize);
            impl Wake for CountWake {
                fn wake(self: Arc<Self>) {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
            }

            let rt = MockRuntime::with_deterministic_scheduler_limits(8, 2);
            let woke = Arc::new(CountWake(AtomicUsize::new(0)));
            let waker = Waker::from(Arc::clone(&woke));
            let mut cx = Context::from_waker(&waker);

            let mut f1 = Box::pin(rt.sleep(Duration::from_millis(50)));
            let mut f2 = Box::pin(rt.sleep(Duration::from_millis(50)));
            let mut f3 = Box::pin(rt.sleep(Duration::from_millis(50)));

            assert!(matches!(f1.as_mut().poll(&mut cx), Poll::Pending));
            assert!(matches!(f2.as_mut().poll(&mut cx), Poll::Pending));
            assert_eq!(rt.pending_timer_count(), 2);
            assert_eq!(rt.rejected_timer_registrations(), 0);

            // Re-polling an already registered future updates in place without consuming extra capacity.
            assert!(matches!(f1.as_mut().poll(&mut cx), Poll::Pending));
            assert_eq!(rt.pending_timer_count(), 2);
            assert_eq!(rt.rejected_timer_registrations(), 0);

            // A 3rd distinct sleep future exceeds `max_timers = 2` -> bounded reject!
            assert!(matches!(f3.as_mut().poll(&mut cx), Poll::Pending));
            assert_eq!(rt.pending_timer_count(), 2);
            assert_eq!(rt.rejected_timer_registrations(), 1);

            // Dropping `f1` before deadline cancels its timer and reclaims capacity immediately.
            drop(f1);
            assert_eq!(rt.pending_timer_count(), 1);

            assert!(matches!(f3.as_mut().poll(&mut cx), Poll::Pending));
            assert_eq!(rt.pending_timer_count(), 2);

            rt.advance(Duration::from_millis(60));
            assert_eq!(rt.pending_timer_count(), 0);
            assert_eq!(woke.0.load(Ordering::SeqCst), 2);
            assert!(matches!(f2.as_mut().poll(&mut cx), Poll::Ready(())));
            assert!(matches!(f3.as_mut().poll(&mut cx), Poll::Ready(())));
        }

        #[cfg(feature = "std")]
        #[test]
        fn mock_runtime_preserves_mid_poll_and_concurrent_wakeups_without_toctou_loss() {
            use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
            use std::sync::Barrier;

            struct SelfRewakeOnce {
                polled: Arc<AtomicUsize>,
                done: Arc<AtomicBool>,
            }
            impl Future for SelfRewakeOnce {
                type Output = ();
                fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
                    let prev = self.polled.fetch_add(1, Ordering::SeqCst);
                    if prev == 0 {
                        // Mid-poll wake: while another thread calls poll_ready_tasks(), this wake
                        // MUST defer via `rewake_pending` and never be lost!
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    } else {
                        self.done.store(true, Ordering::SeqCst);
                        Poll::Ready(())
                    }
                }
            }

            let rt = Arc::new(MockRuntime::with_deterministic_scheduler(32));
            let polled = Arc::new(AtomicUsize::new(0));
            let done = Arc::new(AtomicBool::new(false));
            rt.try_spawn(SelfRewakeOnce {
                polled: Arc::clone(&polled),
                done: Arc::clone(&done),
            })
            .unwrap();

            // Stress concurrent poll_ready_tasks() across two threads so a queued mid-poll wake cannot drop.
            let barrier = Arc::new(Barrier::new(2));
            let rt_bg = Arc::clone(&rt);
            let b_bg = Arc::clone(&barrier);
            let t = std::thread::spawn(move || {
                b_bg.wait();
                for _ in 0..8 {
                    let _ = rt_bg.poll_ready_tasks();
                }
            });
            barrier.wait();
            for _ in 0..8 {
                let _ = rt.poll_ready_tasks();
            }
            t.join().unwrap();
            assert!(
                done.load(Ordering::SeqCst),
                "mid-poll wake must not be dropped across concurrent poll_ready_tasks"
            );
            assert_eq!(rt.active_task_count(), 0);

            // Hammer `MockSleepFuture::poll` vs concurrent `advance(1ms)` after fixing `deadline_ms`
            // to prove zero poll/advance TOCTOU wakeup loss across all thread interleavings.
            let clock_rt = Arc::new(MockRuntime::with_deterministic_scheduler(64));
            for i in 0..50u64 {
                let c_rt = Arc::clone(&clock_rt);
                let finished = Arc::new(AtomicBool::new(false));
                let fin_task = Arc::clone(&finished);
                let b2 = Arc::new(Barrier::new(2));
                let b2_task = Arc::clone(&b2);
                clock_rt
                    .try_spawn(async move {
                        let sleeper = c_rt.sleep(Duration::from_millis(1));
                        b2_task.wait();
                        sleeper.await;
                        fin_task.store(true, Ordering::SeqCst);
                    })
                    .unwrap();

                let adv_rt = Arc::clone(&clock_rt);
                let b2_adv = Arc::clone(&b2);
                let adv = std::thread::spawn(move || {
                    b2_adv.wait();
                    adv_rt.advance(Duration::from_millis(1));
                });
                let _ = clock_rt.poll_ready_tasks();
                adv.join().unwrap();
                let _ = clock_rt.poll_ready_tasks();
                assert!(
                    finished.load(Ordering::SeqCst),
                    "iteration {i}: timer wakeup lost to advance/poll TOCTOU race"
                );
            }
            assert_eq!(clock_rt.active_task_count(), 0);
            assert_eq!(clock_rt.pending_timer_count(), 0);
        }

        #[cfg(feature = "std")]
        #[test]
        fn mock_runtime_clock_concurrent_saturating_add_preserves_all_increments() {
            let rt = Arc::new(MockRuntime::new());
            let mut handles = std::vec::Vec::new();
            for _ in 0..4 {
                let r = Arc::clone(&rt);
                handles.push(std::thread::spawn(move || {
                    for _ in 0..250 {
                        r.advance(Duration::from_millis(1));
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
            assert_eq!(rt.now(), 1_000);
        }

        #[cfg(feature = "std")]
        #[test]
        fn mock_sleep_future_repoll_after_completion_zero_limits_and_same_deadline_fifo() {
            use core::sync::atomic::{AtomicUsize, Ordering};

            struct NoopWake;
            impl Wake for NoopWake {
                fn wake(self: Arc<Self>) {}
            }

            let waker = Waker::from(Arc::new(NoopWake));
            let mut cx = Context::from_waker(&waker);

            // (0, 0) zero limits: non-zero sleep fails capacity check; zero sleep completes without slot.
            let zero_rt = MockRuntime::with_deterministic_scheduler_limits(0, 0);
            assert_eq!(
                zero_rt.try_spawn(async {}),
                Err(SpawnError::CapacityExhausted)
            );
            assert_eq!(zero_rt.active_task_count(), 0);

            let mut pending_sleep = Box::pin(zero_rt.sleep(Duration::from_millis(10)));
            assert!(matches!(
                pending_sleep.as_mut().poll(&mut cx),
                Poll::Pending
            ));
            assert_eq!(zero_rt.pending_timer_count(), 0);
            assert_eq!(zero_rt.rejected_timer_registrations(), 1);
            drop(pending_sleep);

            let mut zero_sleep = Box::pin(zero_rt.sleep(Duration::ZERO));
            assert!(matches!(zero_sleep.as_mut().poll(&mut cx), Poll::Ready(())));
            assert_eq!(zero_rt.pending_timer_count(), 0);
            assert_eq!(zero_rt.rejected_timer_registrations(), 1);
            // Re-polling after completion must stay Ready(()) without consuming timer capacity.
            assert!(matches!(zero_sleep.as_mut().poll(&mut cx), Poll::Ready(())));
            assert_eq!(zero_rt.rejected_timer_registrations(), 1);

            // Re-polling a completed timer when capacity = 1 is occupied by another pending timer.
            let cap1_rt = MockRuntime::with_deterministic_scheduler_limits(4, 1);
            let mut done_sleep = Box::pin(cap1_rt.sleep(Duration::from_millis(10)));
            assert!(matches!(done_sleep.as_mut().poll(&mut cx), Poll::Pending));
            cap1_rt.advance(Duration::from_millis(10));
            assert!(matches!(done_sleep.as_mut().poll(&mut cx), Poll::Ready(())));

            let mut occupier = Box::pin(cap1_rt.sleep(Duration::from_millis(50)));
            assert!(matches!(occupier.as_mut().poll(&mut cx), Poll::Pending));
            assert_eq!(cap1_rt.pending_timer_count(), 1);

            assert!(matches!(done_sleep.as_mut().poll(&mut cx), Poll::Ready(())));
            assert_eq!(cap1_rt.pending_timer_count(), 1);
            assert_eq!(cap1_rt.rejected_timer_registrations(), 0);
            drop(occupier);
            assert_eq!(cap1_rt.pending_timer_count(), 0);

            // Shared deadline_ms uniqueness via `(deadline_ms, timer_id)` and FIFO order.
            let order_rt = Arc::new(MockRuntime::with_deterministic_scheduler_limits(8, 8));
            let order = Arc::new(Mutex::new(std::vec::Vec::<u32>::new()));
            let completed = Arc::new(AtomicUsize::new(0));
            for idx in 1..=3u32 {
                let r = Arc::clone(&order_rt);
                let o = Arc::clone(&order);
                let c = Arc::clone(&completed);
                order_rt
                    .try_spawn(async move {
                        r.sleep(Duration::from_millis(50)).await;
                        o.lock().unwrap().push(idx);
                        c.fetch_add(1, Ordering::SeqCst);
                    })
                    .unwrap();
            }
            assert_eq!(order_rt.poll_ready_tasks(), 3);
            assert_eq!(order_rt.pending_timer_count(), 3);
            order_rt.advance_and_run(Duration::from_millis(50));
            assert_eq!(*order.lock().unwrap(), std::vec![1, 2, 3]);
            assert_eq!(completed.load(Ordering::SeqCst), 3);
            assert_eq!(order_rt.pending_timer_count(), 0);
            assert_eq!(order_rt.active_task_count(), 0);
        }

        #[cfg(feature = "std")]
        #[test]
        fn mock_runtime_wake_during_ready_and_after_completion_and_rejected_drop() {
            use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

            struct DropObserved {
                rt: Arc<MockRuntime>,
                drops: Arc<AtomicUsize>,
            }
            impl Drop for DropObserved {
                fn drop(&mut self) {
                    // Verify scheduler lock is NOT held when rejected future drops.
                    let _ = self.rt.active_task_count();
                    self.drops.fetch_add(1, Ordering::SeqCst);
                }
            }

            let rt = Arc::new(MockRuntime::with_deterministic_scheduler_limits(1, 4));
            let drops = Arc::new(AtomicUsize::new(0));

            // 1. Mid-poll wake right before returning Poll::Ready(()) must not leave zombie queue state.
            let stashed_waker: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
            let sw_clone = Arc::clone(&stashed_waker);
            let polls = Arc::new(AtomicUsize::new(0));
            let p_clone = Arc::clone(&polls);
            let gated = Arc::new(AtomicBool::new(false));
            let g_clone = Arc::clone(&gated);

            struct TwoPhaseSelfWake {
                polls: Arc<AtomicUsize>,
                gated: Arc<AtomicBool>,
                stashed: Arc<Mutex<Option<Waker>>>,
            }
            impl Future for TwoPhaseSelfWake {
                type Output = ();
                fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
                    self.polls.fetch_add(1, Ordering::SeqCst);
                    *self.stashed.lock().unwrap() = Some(cx.waker().clone());
                    if !self.gated.load(Ordering::SeqCst) {
                        Poll::Pending
                    } else {
                        // Wake mid-poll AND immediately finish Ready(())!
                        cx.waker().wake_by_ref();
                        Poll::Ready(())
                    }
                }
            }

            let handle = rt
                .try_spawn(TwoPhaseSelfWake {
                    polls: p_clone,
                    gated: g_clone,
                    stashed: sw_clone,
                })
                .unwrap();
            assert_eq!(handle.id(), 1);

            // While task 1 is active (max_tasks = 1), spawning a 2nd task must reject and drop cleanly.
            let d1 = DropObserved {
                rt: Arc::clone(&rt),
                drops: Arc::clone(&drops),
            };
            assert_eq!(
                rt.try_spawn(async move {
                    let _keep = d1;
                }),
                Err(SpawnError::CapacityExhausted)
            );
            assert_eq!(drops.load(Ordering::SeqCst), 1);

            assert_eq!(rt.poll_ready_tasks(), 1);
            assert_eq!(rt.active_task_count(), 1);

            gated.store(true, Ordering::SeqCst);
            let w = stashed_waker.lock().unwrap().take().unwrap();
            w.wake_by_ref();
            assert_eq!(rt.poll_ready_tasks(), 1);
            assert_eq!(polls.load(Ordering::SeqCst), 2);
            assert_eq!(rt.active_task_count(), 0);
            // Mid-poll wake before Poll::Ready must be cleared cleanly.
            assert_eq!(rt.poll_ready_tasks(), 0);

            // Firing a stashed Waker AFTER task completion must not enqueue dead task IDs.
            w.wake();
            {
                let guard = rt.scheduler.as_ref().unwrap().lock().unwrap();
                assert!(guard.runnable_queue.is_empty());
                assert!(guard.runnable_set.is_empty());
                assert!(guard.rewake_pending.is_empty());
            }
            assert_eq!(rt.poll_ready_tasks(), 0);
        }

        #[cfg(feature = "std")]
        #[test]
        fn mock_runtime_id_overflow_guards_reject_without_wraparound() {
            use core::sync::atomic::{AtomicUsize, Ordering};

            struct NoopWake;
            impl Wake for NoopWake {
                fn wake(self: Arc<Self>) {}
            }
            struct DropFlag(Arc<AtomicUsize>);
            impl Drop for DropFlag {
                fn drop(&mut self) {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
            }

            let rt = MockRuntime::with_deterministic_scheduler_limits(4, 4);
            let waker = Waker::from(Arc::new(NoopWake));
            let mut cx = Context::from_waker(&waker);

            // Existing timer (id 1) and task (id 1) must stay intact when counters hit u64::MAX.
            let mut first_timer = Box::pin(rt.sleep(Duration::from_millis(50)));
            assert!(matches!(first_timer.as_mut().poll(&mut cx), Poll::Pending));
            let first_task = rt
                .try_spawn(async {})
                .expect("first task id 1 must succeed");
            assert_eq!(first_task.id(), 1);

            {
                let mut guard = rt.scheduler.as_ref().unwrap().lock().unwrap();
                guard.next_timer_id = u64::MAX;
                guard.next_task_id = u64::MAX;
            }

            let mut overflow_timer = Box::pin(rt.sleep(Duration::from_millis(50)));
            assert!(matches!(
                overflow_timer.as_mut().poll(&mut cx),
                Poll::Pending
            ));
            assert_eq!(rt.pending_timer_count(), 1);
            assert_eq!(rt.rejected_timer_registrations(), 1);

            let drop_count = Arc::new(AtomicUsize::new(0));
            let flag = DropFlag(Arc::clone(&drop_count));
            assert_eq!(
                rt.try_spawn(async move {
                    let _keep = flag;
                }),
                Err(SpawnError::CapacityExhausted)
            );
            assert_eq!(drop_count.load(Ordering::SeqCst), 1);
            assert_eq!(rt.active_task_count(), 1);

            {
                let guard = rt.scheduler.as_ref().unwrap().lock().unwrap();
                assert_eq!(guard.next_timer_id, u64::MAX);
                assert_eq!(guard.next_task_id, u64::MAX);
                assert!(guard.timers.contains_key(&(50, 1)));
                assert!(guard.tasks.contains_key(&1));
            }
            drop(overflow_timer);
            drop(first_timer);
            assert_eq!(rt.pending_timer_count(), 0);
        }

        #[cfg(not(feature = "std"))]
        #[test]
        fn mock_runtime_no_std_try_spawn_fails_and_drops_cleanly() {
            use core::pin::Pin;
            use core::sync::atomic::{AtomicBool, Ordering};
            use core::task::{Context, Poll};

            static DROPPED: AtomicBool = AtomicBool::new(false);
            struct DropFuture(&'static AtomicBool);
            impl Future for DropFuture {
                type Output = ();
                fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
                    Poll::Ready(())
                }
            }
            impl Drop for DropFuture {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::SeqCst);
                }
            }

            DROPPED.store(false, Ordering::SeqCst);
            let rt = MockRuntime::new();
            let res = rt.try_spawn(DropFuture(&DROPPED));
            assert_eq!(res, Err(SpawnError::ExecutorSpawnerRequired));
            assert!(DROPPED.load(Ordering::SeqCst));
        }

        #[cfg(all(not(feature = "std"), feature = "alloc"))]
        #[test]
        fn mock_runtime_no_std_alloc_sleep_future_completes_once_and_idempotent_on_repoll() {
            extern crate alloc;
            use alloc::boxed::Box;
            use alloc::sync::Arc;
            use alloc::task::Wake;
            use core::task::{Context, Poll, Waker};

            struct NoopWake;
            impl Wake for NoopWake {
                fn wake(self: Arc<Self>) {}
            }

            let rt = MockRuntime::new();
            let waker = Waker::from(Arc::new(NoopWake));
            let mut cx = Context::from_waker(&waker);

            let mut fut = Box::pin(rt.sleep(Duration::from_millis(25)));
            assert_eq!(rt.now(), 25);
            assert!(matches!(fut.as_mut().poll(&mut cx), Poll::Ready(())));
            assert!(matches!(fut.as_mut().poll(&mut cx), Poll::Ready(())));
            assert_eq!(rt.now(), 25);

            let mut max_fut = Box::pin(rt.sleep(Duration::MAX));
            assert_eq!(rt.now(), u64::MAX);
            assert!(matches!(max_fut.as_mut().poll(&mut cx), Poll::Ready(())));
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
