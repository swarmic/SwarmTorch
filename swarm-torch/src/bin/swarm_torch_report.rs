use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("Usage: swarm_torch_report <run_dir> [out_path] [--json-out <path>]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --json-out <path>  Also write pretty-printed JSON to <path>");
    eprintln!();
    eprintln!("Example:");
    eprintln!("  swarm_torch_report runs/<run_id> report.html");
    eprintln!("  swarm_torch_report runs/<run_id> report.html --json-out report.json");
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

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--json-out" {
            if i + 1 >= args.len() {
                eprintln!("error: --json-out requires a path argument");
                usage();
            }
            json_out = Some(PathBuf::from(&args[i + 1]));
            i += 2;
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

    let run_dir = run_dir.unwrap_or_else(|| usage());
    let out_path = out_path.unwrap_or_else(|| PathBuf::from("report.html"));

    if let Err(e) = swarm_torch::report::generate_report(&run_dir, &out_path, json_out.as_ref()) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }

    println!("{}", out_path.display());
    if let Some(jp) = json_out {
        println!("{}", jp.display());
    }
}
