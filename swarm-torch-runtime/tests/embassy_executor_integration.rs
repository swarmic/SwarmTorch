#![cfg(feature = "embassy-std-test")]

use core::time::Duration;
use std::sync::mpsc;

use swarm_torch_runtime::embassy_runtime::EmbassyRuntime;
use swarm_torch_runtime::{SpawnError, SwarmClock};

#[embassy_executor::task]
async fn sleep_and_report_worker(rt: EmbassyRuntime, tx: mpsc::SyncSender<(u64, u64)>) {
    let t0 = rt.now();
    rt.sleep(Duration::from_millis(10)).await;
    let t1 = rt.now();
    let _ = tx.send((t0, t1));
}

#[test]
fn embassy_executor_spawns_via_send_spawner_runs_timer_and_reports_busy_capacity() {
    let (res_tx, res_rx) = mpsc::sync_channel(1);
    let (busy_tx, busy_rx) = mpsc::sync_channel(1);

    std::thread::spawn(move || {
        let executor: &'static mut embassy_executor::Executor =
            Box::leak(Box::new(embassy_executor::Executor::new()));
        executor.run(move |spawner| {
            let rt = EmbassyRuntime::with_spawner(spawner.make_send());
            let first = rt.try_spawn_task(|| sleep_and_report_worker(rt.clone(), res_tx.clone()));
            assert_eq!(first, Ok(()));

            let second = rt.try_spawn_task(|| sleep_and_report_worker(rt.clone(), res_tx));
            let _ = busy_tx.send(second);
        });
    });

    let second_spawn = busy_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("Embassy Executor init must report second spawn result");
    assert_eq!(second_spawn, Err(SpawnError::CapacityExhausted));

    let (t0, t1) = res_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("Embassy task timer must wake and finish");
    assert!(
        t1 >= t0,
        "Embassy clock must be monotonic across Timer::after sleep (t0={t0}, t1={t1})"
    );
}
