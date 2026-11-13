//! Zero-port read-only strict MCP inspector binary (`stdio` JSON-RPC 2.0).

use std::env;
use std::io;
use std::path::PathBuf;
use std::process;

use swarm_torch::mcp_inspector::McpInspectorServerV1;

fn parse_pubkey_hex(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!(
            "expected 64-character Ed25519 public key hex, got {} chars",
            hex.len()
        ));
    }
    let bytes = hex.as_bytes();
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("invalid hex byte 0x{b:02x}")),
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let mut run_dir: Option<PathBuf> = None;
    let mut pubkey_hex: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--run-dir" => {
                let Some(val) = args.next() else {
                    eprintln!("error: --run-dir requires a path value");
                    process::exit(2);
                };
                run_dir = Some(PathBuf::from(val));
            }
            "--public-key-hex" => {
                let Some(val) = args.next() else {
                    eprintln!("error: --public-key-hex requires a 64-char hex value");
                    process::exit(2);
                };
                pubkey_hex = Some(val);
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: swarm_torch_mcp --run-dir <path> --public-key-hex <64-hex-chars>"
                );
                return;
            }
            other => {
                eprintln!("error: unsupported argument '{other}'");
                process::exit(2);
            }
        }
    }

    let Some(run_dir) = run_dir else {
        eprintln!("error: missing required --run-dir <path>");
        process::exit(2);
    };
    let Some(pubkey_hex) = pubkey_hex else {
        eprintln!("error: missing required --public-key-hex <64-hex-chars>");
        process::exit(2);
    };

    let pubkey = match parse_pubkey_hex(&pubkey_hex) {
        Ok(pk) => pk,
        Err(msg) => {
            eprintln!("error: invalid --public-key-hex: {msg}");
            process::exit(2);
        }
    };

    let server = match McpInspectorServerV1::strict_with_key(run_dir, pubkey) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("error: failed to initialize strict MCP inspector: {err}");
            process::exit(1);
        }
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    if let Err(err) = server.serve_stdio(stdin.lock(), stdout.lock()) {
        eprintln!("error: MCP inspector stdio loop failed: {err}");
        process::exit(1);
    }
}
