use std::path::PathBuf;
use swarm_torch::artifacts::{parse_ed25519_public_key_hex, TrustPolicy};

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReportCliConfig {
    run_dir: PathBuf,
    out_path: PathBuf,
    json_out: Option<PathBuf>,
    policy: TrustPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CliParseError {
    HelpRequested,
    Invalid(String),
}

fn usage_text() -> &'static str {
    concat!(
        "Usage: swarm_torch_report <run_dir> [out_path] (--public-key-hex <64-hex> | --legacy-permissive-untrusted) [--json-out <path>]\n\n",
        "Options:\n",
        "  --public-key-hex <64-hex>        Verify manifest Ed25519 signature against trusted public key (TrustPolicy::Strict)\n",
        "  --legacy-permissive-untrusted    Explicitly allow unsigned/untrusted bundles (TrustPolicy::LegacyPermissive)\n",
        "  --json-out <path>                Also write pretty-printed JSON to <path>\n\n",
        "Example:\n",
        "  swarm_torch_report runs/<run_id> report.html --public-key-hex <64-hex>\n",
        "  swarm_torch_report runs/<run_id> report.html --legacy-permissive-untrusted --json-out report.json"
    )
}

fn print_usage_and_exit() -> ! {
    eprintln!("{}", usage_text());
    std::process::exit(2);
}

fn parse_report_cli_args(args: &[String]) -> Result<ReportCliConfig, CliParseError> {
    if args.is_empty() {
        return Err(CliParseError::Invalid(
            "missing required arguments (pass --help for usage)".to_string(),
        ));
    }

    let mut run_dir: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut trusted_keys: Vec<[u8; 32]> = Vec::new();
    let mut legacy_permissive = false;

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--json-out" {
            if i + 1 >= args.len() {
                return Err(CliParseError::Invalid(
                    "--json-out requires a path argument".to_string(),
                ));
            }
            if json_out.is_some() {
                return Err(CliParseError::Invalid(
                    "--json-out may only be specified once".to_string(),
                ));
            }
            json_out = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else if arg == "--public-key-hex" {
            if i + 1 >= args.len() {
                return Err(CliParseError::Invalid(
                    "--public-key-hex requires a 64-character hex argument".to_string(),
                ));
            }
            let pk = parse_ed25519_public_key_hex(&args[i + 1]).map_err(|msg| {
                CliParseError::Invalid(format!("invalid --public-key-hex: {msg}"))
            })?;
            trusted_keys.push(pk);
            i += 2;
        } else if arg == "--legacy-permissive-untrusted" {
            legacy_permissive = true;
            i += 1;
        } else if arg == "--help" || arg == "-h" {
            return Err(CliParseError::HelpRequested);
        } else if arg.starts_with('-') {
            return Err(CliParseError::Invalid(format!("unknown flag: {arg}")));
        } else if run_dir.is_none() {
            run_dir = Some(PathBuf::from(arg));
            i += 1;
        } else if out_path.is_none() {
            out_path = Some(PathBuf::from(arg));
            i += 1;
        } else {
            return Err(CliParseError::Invalid(format!(
                "unexpected argument: {arg}"
            )));
        }
    }

    let policy = match (trusted_keys.is_empty(), legacy_permissive) {
        (false, false) => TrustPolicy::Strict {
            trusted_public_keys: trusted_keys,
            expected_run_id: None,
            min_signed_unix_nanos: None,
        },
        (true, true) => TrustPolicy::LegacyPermissive,
        (true, false) | (false, true) => {
            return Err(CliParseError::Invalid(
                "specify exactly one trust mode: --public-key-hex <64-hex> (Strict) OR --legacy-permissive-untrusted"
                    .to_string(),
            ));
        }
    };

    let run_dir = run_dir.ok_or_else(|| {
        CliParseError::Invalid("missing required positional argument <run_dir>".to_string())
    })?;
    let out_path = out_path.unwrap_or_else(|| PathBuf::from("report.html"));

    Ok(ReportCliConfig {
        run_dir,
        out_path,
        json_out,
        policy,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = match parse_report_cli_args(&args) {
        Ok(cfg) => cfg,
        Err(CliParseError::HelpRequested) => {
            eprintln!("{}", usage_text());
            return;
        }
        Err(CliParseError::Invalid(msg)) => {
            eprintln!("error: {msg}");
            print_usage_and_exit();
        }
    };

    if let Err(e) = swarm_torch::report::generate_report_with_policy(
        &cfg.run_dir,
        &cfg.out_path,
        cfg.json_out.as_ref(),
        &cfg.policy,
    ) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }

    println!("{}", cfg.out_path.display());
    if let Some(jp) = cfg.json_out {
        println!("{}", jp.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|v| (*v).to_string()).collect()
    }

    #[test]
    fn cli_rejects_omitted_trust_mode_and_conflicting_both_modes() {
        // Neither trust flag passed MUST fail closed (regression for AR-14-02 XNOR inversion).
        let neither = parse_report_cli_args(&s(&["runs/demo", "out.html"])).unwrap_err();
        assert!(matches!(neither, CliParseError::Invalid(_)));

        // Both strict public key and legacy permissive MUST fail closed.
        let key_hex = "ab".repeat(32);
        let both = parse_report_cli_args(&s(&[
            "runs/demo",
            "out.html",
            "--public-key-hex",
            &key_hex,
            "--legacy-permissive-untrusted",
        ]))
        .unwrap_err();
        assert!(matches!(both, CliParseError::Invalid(_)));
    }

    #[test]
    fn cli_accepts_strict_and_legacy_modes_exclusively() {
        let key_hex = "ab".repeat(32);
        let strict_cfg = parse_report_cli_args(&s(&[
            "runs/demo",
            "--public-key-hex",
            &key_hex,
            "--json-out",
            "out.json",
        ]))
        .expect("strict args should parse");
        assert_eq!(strict_cfg.out_path, PathBuf::from("report.html"));
        assert_eq!(strict_cfg.json_out, Some(PathBuf::from("out.json")));
        assert!(matches!(strict_cfg.policy, TrustPolicy::Strict { .. }));

        let legacy_cfg = parse_report_cli_args(&s(&[
            "runs/demo",
            "custom.html",
            "--legacy-permissive-untrusted",
        ]))
        .expect("legacy args should parse");
        assert_eq!(legacy_cfg.out_path, PathBuf::from("custom.html"));
        assert_eq!(legacy_cfg.policy, TrustPolicy::LegacyPermissive);
    }

    #[test]
    fn cli_rejects_missing_values_bad_hex_and_unknown_flags() {
        assert!(matches!(
            parse_report_cli_args(&s(&["runs/demo", "--public-key-hex"])),
            Err(CliParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_report_cli_args(&s(&["runs/demo", "--json-out"])),
            Err(CliParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_report_cli_args(&s(&["runs/demo", "--public-key-hex", "deadbeef"])),
            Err(CliParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_report_cli_args(&s(&[
                "runs/demo",
                "--unknown-flag",
                "--legacy-permissive-untrusted"
            ])),
            Err(CliParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_report_cli_args(&s(&["--legacy-permissive-untrusted"])),
            Err(CliParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_report_cli_args(&s(&[
                "runs/demo",
                "--legacy-permissive-untrusted",
                "--json-out",
                "a.json",
                "--json-out",
                "b.json",
            ])),
            Err(CliParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_report_cli_args(&s(&["--help"])),
            Err(CliParseError::HelpRequested)
        ));
        assert!(matches!(
            parse_report_cli_args(&s(&["-h"])),
            Err(CliParseError::HelpRequested)
        ));
    }
}
