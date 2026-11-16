//! Zero-port read-only strict MCP inspector binary (`stdio` JSON-RPC 2.0).

use std::env;
use std::io;
use std::path::PathBuf;
use std::process;

use swarm_torch::artifacts::parse_ed25519_public_key_hex;
use swarm_torch::mcp_inspector::McpInspectorServerV1;

#[derive(Debug, PartialEq, Eq)]
struct McpCliConfig {
    run_dir: PathBuf,
    public_key: [u8; 32],
}

#[derive(Debug, PartialEq, Eq)]
enum McpCliParseError {
    HelpRequested,
    Invalid(String),
}

fn parse_mcp_cli_args(args: &[String]) -> Result<McpCliConfig, McpCliParseError> {
    let mut run_dir: Option<PathBuf> = None;
    let mut pubkey_hex: Option<String> = None;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--run-dir" => {
                if run_dir.is_some() {
                    return Err(McpCliParseError::Invalid(
                        "--run-dir may only be specified once".to_string(),
                    ));
                }
                let Some(val) = args.get(i + 1) else {
                    return Err(McpCliParseError::Invalid(
                        "--run-dir requires a path value".to_string(),
                    ));
                };
                run_dir = Some(PathBuf::from(val));
                i += 2;
            }
            "--public-key-hex" => {
                if pubkey_hex.is_some() {
                    return Err(McpCliParseError::Invalid(
                        "--public-key-hex may only be specified once".to_string(),
                    ));
                }
                let Some(val) = args.get(i + 1) else {
                    return Err(McpCliParseError::Invalid(
                        "--public-key-hex requires a 64-char hex value".to_string(),
                    ));
                };
                pubkey_hex = Some(val.clone());
                i += 2;
            }
            "--help" | "-h" => return Err(McpCliParseError::HelpRequested),
            other => {
                return Err(McpCliParseError::Invalid(format!(
                    "unsupported argument '{other}'"
                )));
            }
        }
    }

    let Some(run_dir) = run_dir else {
        return Err(McpCliParseError::Invalid(
            "missing required --run-dir <path>".to_string(),
        ));
    };
    let Some(pubkey_hex) = pubkey_hex else {
        return Err(McpCliParseError::Invalid(
            "missing required --public-key-hex <64-hex-chars>".to_string(),
        ));
    };
    let public_key = parse_ed25519_public_key_hex(&pubkey_hex)
        .map_err(|msg| McpCliParseError::Invalid(format!("invalid --public-key-hex: {msg}")))?;

    Ok(McpCliConfig {
        run_dir,
        public_key,
    })
}

fn main() {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    let cfg = match parse_mcp_cli_args(&raw_args) {
        Ok(c) => c,
        Err(McpCliParseError::HelpRequested) => {
            eprintln!("Usage: swarm_torch_mcp --run-dir <path> --public-key-hex <64-hex-chars>");
            return;
        }
        Err(McpCliParseError::Invalid(msg)) => {
            eprintln!("error: {msg}");
            process::exit(2);
        }
    };

    let server = match McpInspectorServerV1::strict_with_key(cfg.run_dir, cfg.public_key) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_requires_strict_run_dir_and_valid_64_hex_pubkey() {
        let valid_pk = "aa".repeat(32);
        let parsed = parse_mcp_cli_args(&[
            "--run-dir".to_string(),
            "runs/r1".to_string(),
            "--public-key-hex".to_string(),
            valid_pk,
        ])
        .unwrap();
        assert_eq!(parsed.run_dir, PathBuf::from("runs/r1"));
        assert_eq!(parsed.public_key, [0xaa; 32]);

        assert!(matches!(
            parse_mcp_cli_args(&["--run-dir".to_string(), "runs/r1".to_string()]),
            Err(McpCliParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_mcp_cli_args(&[
                "--run-dir".to_string(),
                "runs/r1".to_string(),
                "--public-key-hex".to_string(),
                "zzzz".to_string(),
            ]),
            Err(McpCliParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_mcp_cli_args(&["--help".to_string()]),
            Err(McpCliParseError::HelpRequested)
        ));
        assert!(matches!(
            parse_mcp_cli_args(&[
                "--run-dir".to_string(),
                "r1".to_string(),
                "--run-dir".to_string(),
                "r2".to_string(),
                "--public-key-hex".to_string(),
                "aa".repeat(32),
            ]),
            Err(McpCliParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_mcp_cli_args(&[
                "--run-dir".to_string(),
                "r1".to_string(),
                "--public-key-hex".to_string(),
                "AA".repeat(32),
            ]),
            Err(McpCliParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_mcp_cli_args(&[
                "--run-dir".to_string(),
                "r1".to_string(),
                "--public-key-hex".to_string(),
                "aa".repeat(32),
                "--public-key-hex".to_string(),
                "bb".repeat(32),
            ]),
            Err(McpCliParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_mcp_cli_args(&["-h".to_string()]),
            Err(McpCliParseError::HelpRequested)
        ));
    }
}
