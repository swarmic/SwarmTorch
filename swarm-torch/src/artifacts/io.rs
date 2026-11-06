use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use super::lock::BUNDLE_LOCK_FILE_NAME;

/// Maximum permitted byte size for `manifest.json` (`4 MiB`).
pub const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
/// Default maximum permitted byte size for a single artifact file (`64 MiB`).
pub const MAX_SINGLE_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
/// Default maximum total bundle byte size loaded into `VerifiedBundleV1` (`256 MiB`).
pub const MAX_TOTAL_BUNDLE_BYTES: u64 = 256 * 1024 * 1024;
/// Maximum byte length of a single NDJSON line (`512 KiB`).
pub const MAX_NDJSON_LINE_BYTES: usize = 512 * 1024;
/// Maximum number of lines accepted in a single NDJSON artifact (`250,000`).
pub const MAX_NDJSON_LINES: usize = 250_000;

static STAGING_SEQ: AtomicU64 = AtomicU64::new(1);

pub(crate) fn ensure_file(path: &Path) -> io::Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let f = File::create(path)?;
    f.sync_all()?;
    sync_parent_dir(path)?;
    Ok(())
}

pub(crate) fn write_json_pretty_atomic<T: serde::Serialize>(
    path: &Path,
    value: &T,
) -> io::Result<()> {
    let json = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    atomic_write(path, &json)
}

pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> io::Result<T> {
    let bytes = read_bytes_bounded(path, MAX_SINGLE_ARTIFACT_BYTES)?;
    parse_json_slice(&bytes)
}

pub(crate) fn parse_json_slice<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> io::Result<T> {
    serde_json::from_slice(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub(crate) fn parse_ndjson_slice_bounded<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    max_line_bytes: usize,
    max_lines: usize,
) -> io::Result<Vec<T>> {
    let mut out = Vec::new();
    for (line_idx, raw_line) in bytes.split(|&b| b == b'\n').enumerate() {
        let line_bytes = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        if line_bytes.iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        if line_bytes.len() > max_line_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ndjson line {} length {} exceeds maximum {}",
                    line_idx + 1,
                    line_bytes.len(),
                    max_line_bytes
                ),
            ));
        }
        if out.len() >= max_lines {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("ndjson line count exceeds maximum {max_lines}"),
            ));
        }
        let v = serde_json::from_slice::<T>(line_bytes).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid ndjson at line {}: {}", line_idx + 1, e),
            )
        })?;
        out.push(v);
    }
    Ok(out)
}

/// Read a file into memory enforcing a hard pre-allocation and streaming byte cap (`max_bytes`).
pub(crate) fn read_bytes_bounded(path: &Path, max_bytes: u64) -> io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let meta_len = file.metadata()?.len();
    if meta_len > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "file {} size {} exceeds maximum {} bytes",
                path.display(),
                meta_len,
                max_bytes
            ),
        ));
    }
    let cap = (meta_len as usize).min(64 * 1024);
    let mut out = Vec::with_capacity(cap);
    let mut limited = file.take(max_bytes.saturating_add(1));
    limited.read_to_end(&mut out)?;
    if (out.len() as u64) > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "file {} grew beyond maximum {} bytes during read",
                path.display(),
                max_bytes
            ),
        ));
    }
    Ok(out)
}

/// Read a file in a single streaming pass while verifying `expected_bytes` and computing SHA-256.
pub(crate) fn read_and_hash_bounded(
    path: &Path,
    expected_bytes: u64,
    max_bytes: u64,
) -> io::Result<(Vec<u8>, [u8; 32])> {
    if expected_bytes > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "expected size {} for {} exceeds maximum {} bytes",
                expected_bytes,
                path.display(),
                max_bytes
            ),
        ));
    }
    let file = File::open(path)?;
    let meta_len = file.metadata()?.len();
    if meta_len != expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "size mismatch for {} (expected {}, got {})",
                path.display(),
                expected_bytes,
                meta_len
            ),
        ));
    }

    let mut hasher = Sha256::new();
    let mut out = Vec::with_capacity((expected_bytes as usize).min(1024 * 1024));
    let mut limited = file.take(expected_bytes.saturating_add(1));
    let mut buf = [0u8; 8192];
    loop {
        let n = limited.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        out.extend_from_slice(&buf[..n]);
        if (out.len() as u64) > expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "size mismatch for {} (expected {}, read > {})",
                    path.display(),
                    expected_bytes,
                    expected_bytes
                ),
            ));
        }
    }

    if (out.len() as u64) != expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "size mismatch for {} (expected {}, read {})",
                path.display(),
                expected_bytes,
                out.len()
            ),
        ));
    }

    let digest = hasher.finalize();
    let mut sha = [0u8; 32];
    sha.copy_from_slice(&digest[..]);
    Ok((out, sha))
}

pub(crate) fn append_ndjson<T: serde::Serialize>(path: &Path, record: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let line = serde_json::to_string(record).map_err(io::Error::other)?;
    let mut buf = line.into_bytes();
    buf.push(b'\n');
    file.write_all(&buf)?;
    file.flush()?;
    Ok(())
}

pub(crate) fn is_transient_or_lock_file_name(file_name: &str) -> bool {
    file_name == BUNDLE_LOCK_FILE_NAME
        || file_name.ends_with(".tmp")
        || file_name.starts_with(".staging.")
}

pub(crate) fn collect_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ty = entry.file_type()?;
        if ty.is_dir() {
            collect_files_recursive(&path, out)?;
        } else if ty.is_file() {
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(is_transient_or_lock_file_name)
                .unwrap_or(false)
            {
                continue;
            }
            out.push(path);
        }
    }
    Ok(())
}

pub(crate) fn rel_path_string(path: &Path, base: &Path) -> io::Result<String> {
    let rel = path.strip_prefix(base).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "path is not within bundle root",
        )
    })?;

    let mut parts = Vec::new();
    for c in rel.components() {
        parts.push(c.as_os_str().to_string_lossy().to_string());
    }
    Ok(parts.join("/"))
}

pub(crate) fn sha256_file(path: &Path) -> io::Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest[..]);
    Ok(out)
}

#[cfg(unix)]
pub(crate) fn sync_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        let dir = File::open(parent)?;
        dir.sync_all()?;
    }
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn sync_parent_dir(_path: &Path) -> io::Result<()> {
    Ok(())
}

fn next_unique_staging_path(target_path: &Path) -> io::Result<PathBuf> {
    let file_name = target_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid file name"))?;
    let seq = STAGING_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let tmp_name = format!(".staging.{file_name}.{pid}.{seq}.tmp");
    Ok(target_path.with_file_name(tmp_name))
}

/// Write `bytes` to a unique staging file, check `sync_all()`, rename to `path`, and sync parent dir.
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = next_unique_staging_path(path)?;
    let write_res = (|| -> io::Result<()> {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        f.write_all(bytes)?;
        f.flush()?;
        f.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }

    if let Err(e) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }

    sync_parent_dir(path)?;
    Ok(())
}

/// Write-once content-addressed blob publication under `runs/<run_id>/blobs/sha256/<hex_digest>`
/// (`ADR-0023` §2.2).
///
/// - Verifies `SHA-256(bytes) == expected_sha256`.
/// - If the target CAS object already exists on disk, verifies its exact byte length and SHA-256
///   digest before returning `Ok(blob_path)` (never overwriting an existing CAS object, and
///   failing closed with `InvalidData` if the existing file is corrupt).
/// - Otherwise writes to a unique `.staging.*` file, calls `sync_all()?`, and renames into
///   `blobs/sha256/<hex_digest>`.
pub(crate) fn write_once_cas_blob(
    run_dir: &Path,
    expected_sha256: &[u8; 32],
    bytes: &[u8],
) -> io::Result<PathBuf> {
    let actual_digest = Sha256::digest(bytes);
    if actual_digest.as_slice() != expected_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "CAS blob content does not match expected SHA-256 digest",
        ));
    }

    let hex_digest = hex_lower(expected_sha256);
    let cas_dir = run_dir.join("blobs").join("sha256");
    fs::create_dir_all(&cas_dir)?;
    let blob_path = cas_dir.join(&hex_digest);

    if blob_path.exists() {
        let (_, disk_sha) = read_and_hash_bounded(
            &blob_path,
            bytes.len() as u64,
            (bytes.len() as u64).max(MAX_SINGLE_ARTIFACT_BYTES),
        )?;
        if &disk_sha != expected_sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "existing CAS blob {} failed SHA-256 verification",
                    hex_digest
                ),
            ));
        }
        return Ok(blob_path);
    }

    let tmp_path = next_unique_staging_path(&blob_path)?;
    let write_res = (|| -> io::Result<()> {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        f.write_all(bytes)?;
        f.flush()?;
        f.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }

    // Use hard_link + unlink where supported so an existing target is never overwritten,
    // falling back to rename if hard_link is unsupported on the filesystem.
    match fs::hard_link(&tmp_path, &blob_path) {
        Ok(()) => {
            let _ = fs::remove_file(&tmp_path);
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            let _ = fs::remove_file(&tmp_path);
            let (_, disk_sha) = read_and_hash_bounded(
                &blob_path,
                bytes.len() as u64,
                (bytes.len() as u64).max(MAX_SINGLE_ARTIFACT_BYTES),
            )?;
            if &disk_sha != expected_sha256 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "concurrent CAS blob {} failed SHA-256 verification",
                        hex_digest
                    ),
                ));
            }
            return Ok(blob_path);
        }
        Err(_) => {
            if let Err(rename_err) = fs::rename(&tmp_path, &blob_path) {
                let _ = fs::remove_file(&tmp_path);
                return Err(rename_err);
            }
        }
    }

    sync_parent_dir(&blob_path)?;
    Ok(blob_path)
}

pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

pub(crate) fn decode_hex_exact<const N: usize>(s: &str) -> io::Result<[u8; N]> {
    if s.len() != N * 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected {}-char hex string, got {} chars", N * 2, s.len()),
        ));
    }
    let bytes = s.as_bytes();
    let mut out = [0u8; N];
    for i in 0..N {
        let hi = match bytes[i * 2] {
            b'0'..=b'9' => bytes[i * 2] - b'0',
            b'a'..=b'f' => bytes[i * 2] - b'a' + 10,
            b'A'..=b'F' => bytes[i * 2] - b'A' + 10,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid hex character",
                ))
            }
        };
        let lo = match bytes[i * 2 + 1] {
            b'0'..=b'9' => bytes[i * 2 + 1] - b'0',
            b'a'..=b'f' => bytes[i * 2 + 1] - b'a' + 10,
            b'A'..=b'F' => bytes[i * 2 + 1] - b'A' + 10,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid hex character",
                ))
            }
        };
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}
