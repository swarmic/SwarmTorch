//! End-to-end subprocess CLI & stdio integration tests for `swarm_torch_report` and
//! `swarm_torch_mcp` (`ST-15A`).

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use swarm_torch::artifacts::{ArtifactWriteProfile, DataOpsSession, LockAcquisitionPolicy};
use swarm_torch_core::crypto::KeyPair;
use swarm_torch_core::observe::{AttrMap, MetricRecord, RunId, TraceId};
use swarm_torch_core::run_graph::GraphV1;

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "swarm_torch_cli_bin_{name}_{}_{}",
        std::process::id(),
        nanos
    ))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

fn create_signed_test_bundle(name: &str, signer_seed: u8) -> (PathBuf, PathBuf, String) {
    let base = temp_dir(name);
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([signer_seed; 16]);
    let session = DataOpsSession::create(
        &base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();
    let run_dir = session.sink().bundle().run_dir().to_path_buf();
    session.sink().write_graph(&GraphV1::default()).unwrap();
    session
        .sink()
        .bundle()
        .append_metric(&MetricRecord {
            schema_version: 1,
            ts_unix_nanos: 1_000,
            trace_id: TraceId::from_bytes([0x11; 16]),
            span_id: None,
            name: "loss".to_string(),
            value: 0.42,
            unit: None,
            attrs: AttrMap::new(),
        })
        .unwrap();
    let signer = KeyPair::from_seed([signer_seed; 32]).unwrap();
    let pk_hex = hex_lower(signer.public_key());
    session.commit_and_close(Some(&signer)).unwrap();
    (base, run_dir, pk_hex)
}

#[test]
fn report_cli_enforces_exact_xor_trust_flags_and_verifies_signed_bundles() {
    let report_bin = env!("CARGO_BIN_EXE_swarm_torch_report");
    let (base, run_dir, pk_hex) = create_signed_test_bundle("report_xor", 0x51);
    let out_html = base.join("report.html");
    let out_json = base.join("report.json");

    // 1. Neither trust flag -> exit 2 (no silent Permissive fallback!).
    let neither = Command::new(report_bin)
        .arg(&run_dir)
        .arg(&out_html)
        .output()
        .unwrap();
    assert_eq!(neither.status.code(), Some(2));
    assert!(!out_html.exists());

    // 2. Both trust flags -> exit 2 (XNOR regression guard: no silent downgrade to Permissive!).
    let both = Command::new(report_bin)
        .arg(&run_dir)
        .arg(&out_html)
        .arg("--public-key-hex")
        .arg(&pk_hex)
        .arg("--legacy-permissive-untrusted")
        .output()
        .unwrap();
    assert_eq!(both.status.code(), Some(2));
    assert!(!out_html.exists());

    // 3. Wrong public key -> non-zero exit (fail closed).
    let wrong_pk = "00".repeat(32);
    let wrong = Command::new(report_bin)
        .arg(&run_dir)
        .arg(&out_html)
        .arg("--public-key-hex")
        .arg(&wrong_pk)
        .output()
        .unwrap();
    assert!(!wrong.status.success());
    assert!(!out_html.exists());

    // 4. Strict valid key -> exit 0 and renders verified HTML + JSON.
    let ok = Command::new(report_bin)
        .arg(&run_dir)
        .arg(&out_html)
        .arg("--public-key-hex")
        .arg(&pk_hex)
        .arg("--json-out")
        .arg(&out_json)
        .output()
        .unwrap();
    assert!(
        ok.status.success(),
        "strict report failed: {}",
        String::from_utf8_lossy(&ok.stderr)
    );
    assert!(out_html.exists());
    assert!(out_json.exists());
    let html = fs::read_to_string(&out_html).unwrap();
    assert!(html.contains("VerifiedStrict"));

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn mcp_cli_enforces_required_strict_args_and_serves_stdio_jsonrpc() {
    let mcp_bin = env!("CARGO_BIN_EXE_swarm_torch_mcp");
    let (base, run_dir, pk_hex) = create_signed_test_bundle("mcp_stdio", 0x52);

    // 1. Missing --public-key-hex -> exit 2.
    let missing_pk = Command::new(mcp_bin)
        .arg("--run-dir")
        .arg(&run_dir)
        .output()
        .unwrap();
    assert_eq!(missing_pk.status.code(), Some(2));

    // 2. Spawn live subprocess over stdin/stdout pipes and execute initialize + tools/list + resources/read.
    let mut child = Command::new(mcp_bin)
        .arg("--run-dir")
        .arg(&run_dir)
        .arg("--public-key-hex")
        .arg(&pk_hex)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let run_id_hex = hex_lower(&[0x52; 16]);
    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{}}}}"#
        )
        .unwrap();
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{{}}}}"#
        )
        .unwrap();
        writeln!(
            stdin,
            "{{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"resources/read\",\"params\":{{\"uri\":\"swarmtorch://runs/{run_id_hex}/manifest\"}}}}"
        )
        .unwrap();
        stdin.flush().unwrap();
    }
    // Drop stdin to signal clean EOF.
    drop(child.stdin.take());

    let stdout = child.stdout.take().unwrap();
    let lines: Vec<String> = BufReader::new(stdout).lines().map(|l| l.unwrap()).collect();
    let status = child.wait().unwrap();
    assert!(status.success(), "mcp child exited with {status:?}");
    assert_eq!(lines.len(), 3, "expected 3 JSON-RPC responses: {lines:?}");

    let r1: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(r1["id"], 1);
    assert!(r1["result"]["serverInfo"]["name"].is_string());

    let r2: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
    assert_eq!(r2["id"], 2);
    assert!(r2["result"]["tools"].is_array());

    let r3: serde_json::Value = serde_json::from_str(&lines[2]).unwrap();
    assert_eq!(r3["id"], 3);
    assert!(r3["result"]["contents"].is_array());

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn report_and_mcp_clis_fail_closed_on_tampered_bundle_and_bad_flags() {
    let report_bin = env!("CARGO_BIN_EXE_swarm_torch_report");
    let mcp_bin = env!("CARGO_BIN_EXE_swarm_torch_mcp");
    let (base, run_dir, pk_hex) = create_signed_test_bundle("tampered_cli", 0x53);

    // Tamper a manifest-covered file on disk after signing.
    fs::write(run_dir.join("metrics.ndjson"), b"{\"tampered\":true}\n").unwrap();

    // 1. `swarm_torch_report` rejects no-args with exit 2 and tampered bundle with non-zero exit (no output written).
    let no_args = Command::new(report_bin).output().unwrap();
    assert_eq!(no_args.status.code(), Some(2));

    let out_html = base.join("should_not_exist.html");
    let out_json = base.join("should_not_exist.json");
    let tampered_report = Command::new(report_bin)
        .arg(&run_dir)
        .arg(&out_html)
        .arg("--public-key-hex")
        .arg(&pk_hex)
        .arg("--json-out")
        .arg(&out_json)
        .output()
        .unwrap();
    assert!(!tampered_report.status.success());
    assert!(!out_html.exists());
    assert!(!out_json.exists());

    // 2. `swarm_torch_mcp` rejects invalid `--public-key-hex` and unknown flags with exit 2.
    let bad_hex = Command::new(mcp_bin)
        .arg("--run-dir")
        .arg(&run_dir)
        .arg("--public-key-hex")
        .arg("not-hex")
        .output()
        .unwrap();
    assert_eq!(bad_hex.status.code(), Some(2));

    let unknown_flag = Command::new(mcp_bin).arg("--unknown").output().unwrap();
    assert_eq!(unknown_flag.status.code(), Some(2));

    // 3. `swarm_torch_mcp` spawned on tampered signed bundle returns JSON-RPC verification error `-32001`.
    let mut child = Command::new(mcp_bin)
        .arg("--run-dir")
        .arg(&run_dir)
        .arg("--public-key-hex")
        .arg(&pk_hex)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"resources/list","params":{{}}}}"#
        )
        .unwrap();
        stdin.flush().unwrap();
    }
    drop(child.stdin.take());

    let stdout = child.stdout.take().unwrap();
    let lines: Vec<String> = BufReader::new(stdout).lines().map(|l| l.unwrap()).collect();
    let status = child.wait().unwrap();
    assert!(status.success());
    assert_eq!(lines.len(), 1);
    let resp: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(resp["id"], 1);
    assert_eq!(resp["error"]["code"], -32001);

    // 4. Both CLI binaries exit 0 on `--help` and exit 2 on uppercase hex or duplicate CLI flags.
    assert_eq!(
        Command::new(report_bin)
            .arg("--help")
            .output()
            .unwrap()
            .status
            .code(),
        Some(0)
    );
    assert_eq!(
        Command::new(mcp_bin)
            .arg("--help")
            .output()
            .unwrap()
            .status
            .code(),
        Some(0)
    );
    assert_eq!(
        Command::new(report_bin)
            .arg(&run_dir)
            .arg(&out_html)
            .arg("--public-key-hex")
            .arg("AA".repeat(32))
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        Command::new(mcp_bin)
            .arg("--run-dir")
            .arg(&run_dir)
            .arg("--run-dir")
            .arg(&run_dir)
            .arg("--public-key-hex")
            .arg(&pk_hex)
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        Command::new(mcp_bin)
            .arg("--run-dir")
            .arg(&run_dir)
            .arg("--public-key-hex")
            .arg(&pk_hex)
            .arg("--public-key-hex")
            .arg(&pk_hex)
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        Command::new(report_bin)
            .arg(&run_dir)
            .arg(&out_html)
            .arg("--public-key-hex")
            .arg(&pk_hex)
            .arg("--json-out")
            .arg(&out_json)
            .arg("--json-out")
            .arg(&out_json)
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        Command::new(report_bin)
            .arg("-h")
            .output()
            .unwrap()
            .status
            .code(),
        Some(0)
    );
    assert_eq!(
        Command::new(mcp_bin)
            .arg("-h")
            .output()
            .unwrap()
            .status
            .code(),
        Some(0)
    );

    let _ = fs::remove_dir_all(&base);
}
