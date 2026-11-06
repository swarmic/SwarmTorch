//! Cross-process exclusive bundle lockfile guard (`ADR-0023` §3.2, `P2-02`, `W10-02`).
//!
//! Enforces mutual exclusion across `RunArtifactBundle` mutations using kernel-atomic
//! `OpenOptions::create_new(true)` (`O_CREAT | O_EXCL` on POSIX, `CREATE_NEW` on Windows)
//! and a retained open [`std::fs::File`] handle under `#![forbid(unsafe_code)]`.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Canonical lockfile name inside `runs/<run_id>/`.
pub const BUNDLE_LOCK_FILE_NAME: &str = ".bundle.lock";

static LEASE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Policy controlling how [`BundleLockGuard::acquire`] handles lock contention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockAcquisitionPolicy {
    /// Immediately return [`BundleLockError::BundleBusy`] if `.bundle.lock` exists.
    FailFast,
    /// Poll until `timeout` elapses before returning [`BundleLockError::BundleBusy`].
    BoundedWait {
        /// Maximum total wait duration.
        timeout: Duration,
        /// Sleep interval between acquisition attempts.
        poll_interval: Duration,
    },
}

impl Default for LockAcquisitionPolicy {
    fn default() -> Self {
        Self::BoundedWait {
            timeout: Duration::from_millis(2_000),
            poll_interval: Duration::from_millis(5),
        }
    }
}

/// Error returned when acquiring a [`BundleLockGuard`].
#[derive(Debug)]
pub enum BundleLockError {
    /// Another writer currently holds `.bundle.lock`.
    BundleBusy {
        /// Path to the contested `.bundle.lock` file.
        lock_path: PathBuf,
    },
    /// Underlying filesystem error.
    Io(io::Error),
}

impl core::fmt::Display for BundleLockError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BundleBusy { lock_path } => {
                write!(
                    f,
                    "bundle is locked by another writer: {}",
                    lock_path.display()
                )
            }
            Self::Io(e) => write!(f, "bundle lock I/O error: {e}"),
        }
    }
}

impl std::error::Error for BundleLockError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BundleBusy { .. } => None,
            Self::Io(e) => Some(e),
        }
    }
}

impl From<BundleLockError> for io::Error {
    fn from(err: BundleLockError) -> Self {
        match err {
            BundleLockError::BundleBusy { lock_path } => io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "bundle is locked by another writer: {}",
                    lock_path.display()
                ),
            ),
            BundleLockError::Io(e) => e,
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LockMetadataV1 {
    schema_version: u32,
    pid: u32,
    acquired_unix_nanos: u64,
    lease_nonce: u64,
}

/// RAII guard for the cross-process exclusive `.bundle.lock` file (`ADR-0023` §3.2).
///
/// Retains the open [`File`] handle while alive and removes `.bundle.lock` on [`Drop`].
#[derive(Debug)]
pub struct BundleLockGuard {
    lock_path: PathBuf,
    file: Option<File>,
    lease_nonce: u64,
}

impl BundleLockGuard {
    /// Acquire an exclusive lock on `run_dir/.bundle.lock` according to `policy`.
    pub fn acquire(run_dir: &Path, policy: LockAcquisitionPolicy) -> Result<Self, BundleLockError> {
        let lock_path = run_dir.join(BUNDLE_LOCK_FILE_NAME);
        let start = Instant::now();

        loop {
            match Self::try_create_lock(&lock_path) {
                Ok(guard) => return Ok(guard),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => match policy {
                    LockAcquisitionPolicy::FailFast => {
                        return Err(BundleLockError::BundleBusy { lock_path });
                    }
                    LockAcquisitionPolicy::BoundedWait {
                        timeout,
                        poll_interval,
                    } => {
                        if start.elapsed() >= timeout {
                            return Err(BundleLockError::BundleBusy { lock_path });
                        }
                        thread::sleep(poll_interval.max(Duration::from_millis(1)));
                    }
                },
                Err(e) => return Err(BundleLockError::Io(e)),
            }
        }
    }

    fn try_create_lock(lock_path: &Path) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(lock_path)?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let acquired_unix_nanos = now.as_nanos().min(u64::MAX as u128) as u64;
        let lease_nonce = LEASE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let meta = LockMetadataV1 {
            schema_version: 1,
            pid: std::process::id(),
            acquired_unix_nanos,
            lease_nonce,
        };
        let payload = serde_json::to_vec(&meta).map_err(io::Error::other)?;
        if let Err(e) = file.write_all(&payload).and_then(|()| file.sync_all()) {
            drop(file);
            let _ = fs::remove_file(lock_path);
            return Err(e);
        }

        Ok(Self {
            lock_path: lock_path.to_path_buf(),
            file: Some(file),
            lease_nonce,
        })
    }

    /// Path to the active `.bundle.lock` file.
    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }

    /// Monotonic process-local lease nonce written to the lockfile.
    pub const fn lease_nonce(&self) -> u64 {
        self.lease_nonce
    }
}

impl Drop for BundleLockGuard {
    fn drop(&mut self) {
        // Close the open file handle before unlinking so Windows allows removal.
        drop(self.file.take());
        let _ = fs::remove_file(&self.lock_path);
    }
}
