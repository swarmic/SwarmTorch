use std::path::PathBuf;
use swarm_torch::artifacts::TrustPolicy;

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

fn usage() -> ! {
    eprintln!(
        "Usage: swarm_torch_report <run_dir> [out_path] (--public-key-hex <64-hex> | --legacy-permissive-untrusted) [--json-out <path>]"
    );
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --public-key-hex <64-hex>        Verify manifest Ed25519 signature against trusted public key (TrustPolicy::Strict)");
    eprintln!("  --legacy-permissive-untrusted    Explicitly allow unsigned/untrusted bundles (TrustPolicy::LegacyPermissive)");
    eprintln!("  --json-out <path>                Also write pretty-printed JSON to <path>");
    eprintln!();
    eprintln!("Example:");
    eprintln!("  swarm_torch_report runs/<run_id> report.html --public-key-hex <64-hex>");
    eprintln!("  swarm_torch_report runs/<run_id> report.html --legacy-permissive-untrusted --json-out report.json");
    std::process::exit(2);
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }

    let mut run_dir: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut trusted_keys: Vec<[u8; 32]> = Vec::new();
    let mut legacy_permissive = false;

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--json-out" {
            if i + 1 >= args.len() {
                eprintln!("error: --json-out requires a path argument");
                usage();
            }
            json_out = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else if args[i] == "--public-key-hex" {
            if i + 1 >= args.len() {
                eprintln!("error: --public-key-hex requires a 64-character hex argument");
                usage();
            }
            match parse_pubkey_hex(&args[i + 1]) {
                Ok(pk) => trusted_keys.push(pk),
                Err(msg) => {
                    eprintln!("error: invalid --public-key-hex: {msg}");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if args[i] == "--legacy-permissive-untrusted" {
            legacy_permissive = true;
            i += 1;
        } else if args[i] == "--help" || args[i] == "-h" {
            usage();
        } else if run_dir.is_none() {
            run_dir = Some(PathBuf::from(&args[i]));
            i += 1;
        } else if out_path.is_none() {
            out_path = Some(PathBuf::from(&args[i]));
            i += 1;
        } else {
            eprintln!("error: unexpected argument: {}", args[i]);
            usage();
        }
    }

    if trusted_keys.is_empty() == legacy_permissive {
        eprintln!(
            "error: specify exactly one trust mode: --public-key-hex <64-hex> (Strict) OR --legacy-permissive-untrusted"
        );
        usage();
    }

    let policy = if legacy_permissive {
        TrustPolicy::LegacyPermissive
    } else {
        TrustPolicy::Strict {
            trusted_public_keys: trusted_keys,
            expected_run_id: None,
            min_signed_unix_nanos: None,
        }
    };

    let run_dir = run_dir.unwrap_or_else(|| usage());
    let out_path = out_path.unwrap_or_else(|| PathBuf::from("report.html"));

    if let Err(e) = swarm_torch::report::generate_report_with_policy(
        &run_dir,
        &out_path,
        json_out.as_ref(),
        &policy,
    ) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }

    println!("{}", out_path.display());
    if let Some(jp) = json_out {
        println!("{}", jp.display());
    }
}
