//! Bounded non-blocking security audit ring and drop/quarantine diagnostics (`Decision A8`).
//!
//! Collects transport-level rejections, Slowloris partial-frame evictions, UDP conflict
//! quarantines, and Windows/OS `WSAECONNRESET` ICMP reset diagnostics into fixed-capacity,
//! schema-validated [`EventRecord`] entries without performing blocking I/O on socket poll paths.
//!
//! Under overload floods, noisy informational drop events are window rate-limited while critical
//! safety, Slowloris-eviction, capacity-abort, and quarantine events remain protected from
//! noisy eviction (`ST-13`, `Decision A8`).

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use swarm_torch_core::observe::{
    validate_event_record, AttrMap, AttrValue, EventRecord, RunEventEmitter, TraceId,
    MAX_ATTR_VALUE_STR_LEN, OBSERVE_SCHEMA_V1,
};
use swarm_torch_core::traits::PeerId;

/// Default maximum retained [`EventRecord`] entries in the bounded transport audit ring.
pub const DEFAULT_MAX_TRANSPORT_AUDIT_EVENTS: usize = 256;

/// Overload rate-limit window duration for high-volume noisy drop events.
pub const DEFAULT_AUDIT_OVERLOAD_WINDOW: Duration = Duration::from_millis(100);

/// Maximum noisy non-critical audit records admitted per [`DEFAULT_AUDIT_OVERLOAD_WINDOW`].
pub const DEFAULT_MAX_NOISY_EVENTS_PER_WINDOW: usize = 16;

/// Snapshot of bounded security/drop/quarantine counters exposed by TCP and UDP transports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TransportDiagnosticsSnapshotV1 {
    /// Frames or datagrams dropped due to malformed framing, CRC mismatch, or size violations.
    pub dropped_malformed_frames: u64,
    /// Socket connections dropped due to read/accept I/O failures.
    pub dropped_connection_errors: u64,
    /// Idle TCP connections reclaimed due to idle timeout or connection-table pressure.
    pub evicted_idle_connections: u64,
    /// Stalled unauthenticated/partial-frame TCP connections evicted (Slowloris defense).
    pub evicted_stalled_partial_connections: u64,
    /// Incoming connections rejected after bounded capacity could not yield another slot.
    pub rejected_incoming_connections_full: u64,
    /// Transient `WSAECONNRESET` (`ConnectionReset`), `Interrupted`, or ICMP soft resets isolated.
    pub icmp_port_unreachable_resets: u64,
    /// UDP chunk transfers or observed sources flagged/quarantined on conflicting chunks.
    pub quarantined_udp_events: u64,
    /// Total validated [`EventRecord`] entries recorded into the bounded audit ring.
    pub audited_events_recorded: u64,
    /// Noisy events suppressed by overload rate limiting or critical-slot preservation.
    pub suppressed_overload_events: u64,
    /// Current number of buffered [`EventRecord`] entries awaiting drain/flush.
    pub retained_audit_events: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuditSeverity {
    /// Safety, quarantine, Slowloris eviction, or capacity-abort records retained preferentially.
    Critical,
    /// High-frequency malformed datagram/frame or ICMP soft-reset diagnostics subject to rate limiting.
    Noisy,
}

#[derive(Debug, Clone)]
struct AuditEntry {
    severity: AuditSeverity,
    record: EventRecord,
}

#[derive(Debug)]
struct AuditInner {
    capacity: usize,
    next_seq: u64,
    window_start: Instant,
    window_noisy_count: usize,
    counters: TransportDiagnosticsSnapshotV1,
    events: VecDeque<AuditEntry>,
}

impl AuditInner {
    fn make_room_for_severity(&mut self, severity: AuditSeverity) -> bool {
        if self.events.len() < self.capacity {
            return true;
        }
        match severity {
            AuditSeverity::Critical => {
                if let Some(pos) = self
                    .events
                    .iter()
                    .position(|e| matches!(e.severity, AuditSeverity::Noisy))
                {
                    self.events.remove(pos);
                } else {
                    self.events.pop_front();
                }
                true
            }
            AuditSeverity::Noisy => {
                let noisy_count = self
                    .events
                    .iter()
                    .filter(|e| matches!(e.severity, AuditSeverity::Noisy))
                    .count();
                let max_noisy_retained = self.capacity / 2;
                if noisy_count > 0 && noisy_count >= max_noisy_retained {
                    if let Some(pos) = self
                        .events
                        .iter()
                        .position(|e| matches!(e.severity, AuditSeverity::Noisy))
                    {
                        self.events.remove(pos);
                        return true;
                    }
                }
                false
            }
        }
    }
}

/// Thread-safe, bounded transport audit ring and counter bank (`Decision A8`).
#[derive(Debug)]
pub(crate) struct TransportAuditBufferV1 {
    trace_id: TraceId,
    inner: Mutex<AuditInner>,
}

impl TransportAuditBufferV1 {
    pub(crate) fn new(local_peer: PeerId) -> Self {
        Self::with_capacity(local_peer, DEFAULT_MAX_TRANSPORT_AUDIT_EVENTS)
    }

    pub(crate) fn with_capacity(local_peer: PeerId, capacity: usize) -> Self {
        let mut trace_bytes = [0u8; 16];
        trace_bytes.copy_from_slice(&local_peer.as_bytes()[..16]);
        if trace_bytes.iter().all(|b| *b == 0) {
            trace_bytes = [
                0x53, 0x54, 0x4E, 0x45, 0x54, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
                0x0A, 0x0B,
            ];
        }
        Self {
            trace_id: TraceId::from_bytes(trace_bytes),
            inner: Mutex::new(AuditInner {
                capacity: capacity.max(2),
                next_seq: 1,
                window_start: Instant::now(),
                window_noisy_count: 0,
                counters: TransportDiagnosticsSnapshotV1::default(),
                events: VecDeque::new(),
            }),
        }
    }

    pub(crate) fn record_event(
        &self,
        severity: AuditSeverity,
        name: &str,
        transport: &str,
        source: Option<String>,
        reason: &str,
        counter_fn: impl FnOnce(&mut TransportDiagnosticsSnapshotV1),
    ) {
        let now = Instant::now();
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        counter_fn(&mut guard.counters);

        if matches!(severity, AuditSeverity::Noisy) {
            if now.saturating_duration_since(guard.window_start) >= DEFAULT_AUDIT_OVERLOAD_WINDOW {
                guard.window_start = now;
                guard.window_noisy_count = 0;
            }
            if guard.window_noisy_count >= DEFAULT_MAX_NOISY_EVENTS_PER_WINDOW {
                guard.counters.suppressed_overload_events =
                    guard.counters.suppressed_overload_events.saturating_add(1);
                return;
            }
            guard.window_noisy_count += 1;
        }

        if !guard.make_room_for_severity(severity) {
            guard.counters.suppressed_overload_events =
                guard.counters.suppressed_overload_events.saturating_add(1);
            return;
        }

        let seq = guard.next_seq;
        guard.next_seq = guard.next_seq.saturating_add(1);

        let mut attrs = AttrMap::new();
        attrs.insert(
            "swarmtorch.net.transport".to_string(),
            AttrValue::Str(transport.to_string()),
        );
        attrs.insert("swarmtorch.net.seq".to_string(), AttrValue::U64(seq));
        attrs.insert(
            "swarmtorch.net.severity".to_string(),
            AttrValue::Str(
                match severity {
                    AuditSeverity::Critical => "critical",
                    AuditSeverity::Noisy => "noisy",
                }
                .to_string(),
            ),
        );
        attrs.insert(
            "swarmtorch.net.reason".to_string(),
            AttrValue::Str(truncate_attr_value(reason)),
        );
        if let Some(src) = source {
            attrs.insert(
                "swarmtorch.net.source".to_string(),
                AttrValue::Str(truncate_attr_value(&src)),
            );
        }

        let ts_unix_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|d| u64::try_from(d.as_nanos()).ok())
            .unwrap_or(0);

        let record = EventRecord {
            schema_version: OBSERVE_SCHEMA_V1,
            ts_unix_nanos,
            trace_id: self.trace_id,
            span_id: None,
            name: name.to_string(),
            attrs,
        };
        if validate_event_record(&record).is_ok() {
            guard.events.push_back(AuditEntry { severity, record });
            guard.counters.audited_events_recorded =
                guard.counters.audited_events_recorded.saturating_add(1);
        }
        guard.counters.retained_audit_events = guard.events.len();
    }

    pub(crate) fn snapshot(&self) -> TransportDiagnosticsSnapshotV1 {
        let Ok(guard) = self.inner.lock() else {
            return TransportDiagnosticsSnapshotV1::default();
        };
        let mut snap = guard.counters;
        snap.retained_audit_events = guard.events.len();
        snap
    }

    pub(crate) fn drain_records(&self) -> Vec<EventRecord> {
        let Ok(mut guard) = self.inner.lock() else {
            return Vec::new();
        };
        let drained = guard.events.drain(..).map(|e| e.record).collect();
        guard.counters.retained_audit_events = 0;
        drained
    }

    pub(crate) fn flush_to_emitter<E: RunEventEmitter>(
        &self,
        emitter: &E,
    ) -> core::result::Result<usize, E::Error> {
        let entries: Vec<AuditEntry> = {
            let Ok(mut guard) = self.inner.lock() else {
                return Ok(0);
            };
            let v: Vec<AuditEntry> = guard.events.drain(..).collect();
            guard.counters.retained_audit_events = 0;
            v
        };

        let mut emitted = 0usize;
        for (idx, entry) in entries.iter().enumerate() {
            if let Err(err) = emitter.emit_event(&entry.record) {
                if let Ok(mut guard) = self.inner.lock() {
                    for remaining in entries[idx..].iter().rev() {
                        if guard.make_room_for_severity(remaining.severity) {
                            guard.events.push_front(remaining.clone());
                        }
                    }
                    guard.counters.retained_audit_events = guard.events.len();
                }
                return Err(err);
            }
            emitted += 1;
        }
        Ok(emitted)
    }
}

fn truncate_attr_value(raw: &str) -> String {
    let max_chars = MAX_ATTR_VALUE_STR_LEN.min(256);
    if raw.len() <= max_chars {
        raw.to_string()
    } else {
        raw.chars().take(max_chars).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use swarm_torch_core::observe::{MetricRecord, SpanRecord};

    struct FailAfterOneEmitter {
        count: AtomicUsize,
        emitted: Mutex<Vec<EventRecord>>,
    }

    impl RunEventEmitter for FailAfterOneEmitter {
        type Error = &'static str;

        fn emit_span(&self, _span: &SpanRecord) -> core::result::Result<(), Self::Error> {
            Ok(())
        }

        fn emit_event(&self, event: &EventRecord) -> core::result::Result<(), Self::Error> {
            if self.count.fetch_add(1, Ordering::Relaxed) >= 1 {
                return Err("sink full");
            }
            self.emitted.lock().unwrap().push(event.clone());
            Ok(())
        }

        fn emit_metric(&self, _metric: &MetricRecord) -> core::result::Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn audit_buffer_rate_limits_noisy_floods_and_retains_critical_events() {
        let buf = TransportAuditBufferV1::with_capacity(PeerId::new([0u8; 32]), 8);
        for idx in 0..4u64 {
            buf.record_event(
                AuditSeverity::Critical,
                "swarmtorch.net.tcp.slowloris_evicted",
                "tcp",
                Some(format!("127.0.0.1:{}", 1000 + idx)),
                "stalled",
                |c| {
                    c.evicted_stalled_partial_connections += 1;
                },
            );
        }
        for idx in 0..100u64 {
            buf.record_event(
                AuditSeverity::Noisy,
                "swarmtorch.net.udp.datagram_rejected",
                "udp",
                Some(format!("127.0.0.1:{}", 2000 + idx)),
                "bad crc",
                |c| {
                    c.dropped_malformed_frames += 1;
                },
            );
        }

        let snap = buf.snapshot();
        assert_eq!(snap.evicted_stalled_partial_connections, 4);
        assert_eq!(snap.dropped_malformed_frames, 100);
        assert!(
            snap.suppressed_overload_events >= 100 - DEFAULT_MAX_NOISY_EVENTS_PER_WINDOW as u64
        );
        assert!(snap.retained_audit_events <= 8);

        let drained = buf.drain_records();
        let critical_count = drained
            .iter()
            .filter(|ev| ev.name == "swarmtorch.net.tcp.slowloris_evicted")
            .count();
        assert_eq!(
            critical_count, 4,
            "noisy overload flood must never evict critical safety records"
        );
        for ev in &drained {
            validate_event_record(ev).unwrap();
            assert!(ev.trace_id.is_valid());
        }
        assert_eq!(buf.snapshot().retained_audit_events, 0);
    }

    #[test]
    fn flush_to_emitter_requeues_unemitted_events_on_sink_failure() {
        let buf = TransportAuditBufferV1::with_capacity(PeerId::new([0x22; 32]), 8);
        for idx in 0..3u64 {
            buf.record_event(
                AuditSeverity::Critical,
                "swarmtorch.net.udp.source_quarantined",
                "udp",
                Some(format!("127.0.0.1:{}", 3000 + idx)),
                "conflict",
                |c| {
                    c.quarantined_udp_events += 1;
                },
            );
        }
        let emitter = FailAfterOneEmitter {
            count: AtomicUsize::new(0),
            emitted: Mutex::new(Vec::new()),
        };
        assert_eq!(buf.flush_to_emitter(&emitter), Err("sink full"));
        assert_eq!(emitter.emitted.lock().unwrap().len(), 1);
        assert_eq!(buf.drain_records().len(), 2);
    }

    #[test]
    fn audit_buffer_critical_evicts_noisy_before_older_critical_and_noisy_never_evicts_critical() {
        let buf = TransportAuditBufferV1::with_capacity(PeerId::new([0x33; 32]), 4);

        // Fill capacity=4 with 2 Critical + 2 Noisy.
        for idx in 0..2u64 {
            buf.record_event(
                AuditSeverity::Critical,
                "swarmtorch.net.tcp.slowloris_evicted",
                "tcp",
                Some(format!("127.0.0.1:{}", 1000 + idx)),
                "initial critical",
                |c| c.evicted_stalled_partial_connections += 1,
            );
        }
        for idx in 0..2u64 {
            buf.record_event(
                AuditSeverity::Noisy,
                "swarmtorch.net.udp.datagram_rejected",
                "udp",
                Some(format!("127.0.0.1:{}", 2000 + idx)),
                "noisy-1",
                |c| c.dropped_malformed_frames += 1,
            );
        }

        // Adding a 3rd Critical must evict an older Noisy slot, preserving both earlier Criticals.
        buf.record_event(
            AuditSeverity::Critical,
            "swarmtorch.net.udp.recv_error",
            "udp",
            None,
            "critical-3",
            |c| c.dropped_connection_errors += 1,
        );
        // Now ring has 3 Critical + 1 Noisy (`max_noisy_retained = 4 / 2 = 2`).
        // A new Noisy event must be suppressed rather than evicting any Critical record!
        buf.record_event(
            AuditSeverity::Noisy,
            "swarmtorch.net.udp.datagram_rejected",
            "udp",
            None,
            "noisy-suppressed",
            |c| c.dropped_malformed_frames += 1,
        );
        assert_eq!(buf.snapshot().suppressed_overload_events, 1);

        // A 4th Critical evicts the last remaining Noisy slot; all 4 Criticals survive!
        buf.record_event(
            AuditSeverity::Critical,
            "swarmtorch.net.tcp.slowloris_evicted",
            "tcp",
            None,
            "critical-4",
            |c| c.evicted_stalled_partial_connections += 1,
        );
        let drained = buf.drain_records();
        assert_eq!(drained.len(), 4);
        assert!(drained.iter().all(|r| {
            r.attrs.get("swarmtorch.net.severity") == Some(&AttrValue::Str("critical".to_string()))
        }));
    }

    #[test]
    fn flush_to_emitter_rollback_evicts_concurrent_noisy_to_preserve_critical() {
        struct InjectNoisyOnFailEmitter<'a> {
            buf: &'a TransportAuditBufferV1,
        }
        impl RunEventEmitter for InjectNoisyOnFailEmitter<'_> {
            type Error = &'static str;
            fn emit_span(&self, _span: &SpanRecord) -> core::result::Result<(), Self::Error> {
                Ok(())
            }
            fn emit_event(&self, _event: &EventRecord) -> core::result::Result<(), Self::Error> {
                for i in 0..4u64 {
                    self.buf.record_event(
                        AuditSeverity::Noisy,
                        "swarmtorch.net.udp.datagram_rejected",
                        "udp",
                        Some(format!("127.0.0.1:{}", 4000 + i)),
                        "concurrent-noisy",
                        |c| c.dropped_malformed_frames += 1,
                    );
                }
                Err("sink busy")
            }
            fn emit_metric(&self, _metric: &MetricRecord) -> core::result::Result<(), Self::Error> {
                Ok(())
            }
        }

        let buf = TransportAuditBufferV1::with_capacity(PeerId::new([0x33; 32]), 4);
        buf.record_event(
            AuditSeverity::Critical,
            "swarmtorch.net.tcp.slowloris_evicted",
            "tcp",
            None,
            "critical-must-survive",
            |c| c.evicted_stalled_partial_connections += 1,
        );
        let emitter = InjectNoisyOnFailEmitter { buf: &buf };
        assert_eq!(buf.flush_to_emitter(&emitter), Err("sink busy"));

        let retained = buf.drain_records();
        assert!(retained
            .iter()
            .any(|e| e.name == "swarmtorch.net.tcp.slowloris_evicted"));
    }

    #[test]
    fn audit_buffer_noisy_fifo_replacement_and_critical_slot_preservation() {
        let buf = TransportAuditBufferV1::with_capacity(PeerId::new([0x44; 32]), 4);
        for idx in 0..5u64 {
            buf.record_event(
                AuditSeverity::Noisy,
                "swarmtorch.net.udp.datagram_rejected",
                "udp",
                Some(format!("127.0.0.1:{}", 5000 + idx)),
                "noisy-fifo",
                |c| c.dropped_malformed_frames += 1,
            );
        }
        let snap = buf.snapshot();
        assert_eq!(snap.dropped_malformed_frames, 5);
        assert_eq!(snap.audited_events_recorded, 5);
        assert_eq!(snap.suppressed_overload_events, 0);
        assert_eq!(snap.retained_audit_events, 4);
        let retained = buf.drain_records();
        let seqs: Vec<u64> = retained
            .iter()
            .map(|r| match r.attrs.get("swarmtorch.net.seq") {
                Some(AttrValue::U64(v)) => *v,
                other => panic!("unexpected seq attr: {other:?}"),
            })
            .collect();
        assert_eq!(seqs, vec![2, 3, 4, 5]);

        // With capacity=4 holding 3 Critical + 1 Noisy (`max_noisy_retained = 2`),
        // a new Noisy event is suppressed (`noisy_count = 1 < 2`) and all 3 Criticals survive.
        for _ in 0..3 {
            buf.record_event(
                AuditSeverity::Critical,
                "swarmtorch.net.tcp.slowloris_evicted",
                "tcp",
                None,
                "crit",
                |c| c.evicted_stalled_partial_connections += 1,
            );
        }
        buf.record_event(
            AuditSeverity::Noisy,
            "swarmtorch.net.udp.datagram_rejected",
            "udp",
            None,
            "noisy-first",
            |c| c.dropped_malformed_frames += 1,
        );
        buf.record_event(
            AuditSeverity::Noisy,
            "swarmtorch.net.udp.datagram_rejected",
            "udp",
            None,
            "noisy-must-suppress",
            |c| c.dropped_malformed_frames += 1,
        );
        assert_eq!(buf.snapshot().suppressed_overload_events, 1);
        let kept = buf.drain_records();
        assert_eq!(kept.len(), 4);
        assert_eq!(
            kept.iter()
                .filter(|r| r.name == "swarmtorch.net.tcp.slowloris_evicted")
                .count(),
            3
        );
    }
}
