use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("Usage: swarm_torch_report <run_dir> [out_path]");
    eprintln!();
    eprintln!("Example:");
    eprintln!("  swarm_torch_report runs/<run_id> report.html");
    std::process::exit(2);
}

fn main() {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        usage();
    }

    let run_dir = PathBuf::from(args.remove(0));
    let out_path = if args.is_empty() {
        PathBuf::from("report.html")
    } else {
        PathBuf::from(args.remove(0))
    };

    if !args.is_empty() {
        usage();
    }

    if let Err(e) = swarm_torch::report::generate_report_html(&run_dir, &out_path) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }

    println!("{}", out_path.display());
}
