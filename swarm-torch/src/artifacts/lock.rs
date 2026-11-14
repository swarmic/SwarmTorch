//! Cross-process exclusive bundle lockfile guard (`ADR-0023` §3.2, `P2-02`, `ST-05`).
//!
//! Enforces mutual exclusion across `RunArtifactBundle` writers using kernel advisory/mandatory
//! locking via [`fs4::FileExt::try_lock`] (`flock` on Unix, `LockFileEx` on Windows) over a
//! **persistent** inode-safe `.bundle.lock` handle under `#![forbid(unsafe_code)]`.
//!
//! Unlike `create_new` + `Drop::remove_file`, `.bundle.lock` is never unlinked on release,
//! eliminating two-inode unlink/recreate races and guaranteeing automatic kernel cleanup if a
//! writer crashes (`SIGKILL` / abnormal process termination).

use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fs4::{FileExt, TryLockError};

/// Canonical lockfile name inside `runs/<run_id>/`.
pub const BUNDLE_LOCK_FILE_NAME: &str = ".bundle.lock";

static LEASE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Policy controlling how [`BundleLockGuard::acquire`] handles lock contention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockAcquisitionPolicy {
    /// Immediately return [`BundleLockError::BundleBusy`] if `.bundle.lock` is held.
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

/// RAII guard for the cross-process exclusive `.bundle.lock` file (`ADR-0023` §3.2, `ST-05`).
///
/// Retains an exclusively locked persistent [`File`] handle while alive, calling
/// [`fs4::FileExt::unlock`] on [`Drop`] without unlinking `.bundle.lock`.
#[derive(Debug)]
pub struct BundleLockGuard {
    lock_path: PathBuf,
    file: Option<File>,
    lease_nonce: u64,
}

impl BundleLockGuard {
    /// Acquire an exclusive kernel lock on `run_dir/.bundle.lock` according to `policy`.
    pub fn acquire(run_dir: &Path, policy: LockAcquisitionPolicy) -> Result<Self, BundleLockError> {
        let lock_path = run_dir.join(BUNDLE_LOCK_FILE_NAME);
        let start = Instant::now();

        loop {
            match Self::try_acquire_lock(&lock_path) {
                Ok(guard) => return Ok(guard),
                Err(BundleLockError::BundleBusy { .. }) => match policy {
                    LockAcquisitionPolicy::FailFast => {
                        return Err(BundleLockError::BundleBusy { lock_path });
                    }
                    LockAcquisitionPolicy::BoundedWait {
                        timeout,
                        poll_interval,
                    } => {
                        let elapsed = start.elapsed();
                        if elapsed >= timeout {
                            return Err(BundleLockError::BundleBusy { lock_path });
                        }
                        let remaining = timeout.saturating_sub(elapsed);
                        let sleep_for = poll_interval.max(Duration::from_millis(1)).min(remaining);
                        thread::sleep(sleep_for);
                    }
                },
                Err(e) => return Err(e),
            }
        }
    }

    fn try_acquire_lock(lock_path: &Path) -> Result<Self, BundleLockError> {
        // Never pass `.truncate(true)` here: truncation before holding the kernel lock would
        // clobber an active holder's diagnostic metadata without mutual exclusion.
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .map_err(BundleLockError::Io)?;

        match FileExt::try_lock(&file) {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(BundleLockError::BundleBusy {
                    lock_path: lock_path.to_path_buf(),
                });
            }
            Err(TryLockError::Error(e)) => return Err(BundleLockError::Io(e)),
        }

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
        let payload =
            serde_json::to_vec(&meta).map_err(|e| BundleLockError::Io(io::Error::other(e)))?;

        // Write diagnostic metadata strictly AFTER holding the exclusive kernel lock.
        let write_res: io::Result<()> = (|| {
            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&payload)?;
            file.write_all(b"\n")?;
            file.flush()?;
            file.sync_all()?;
            Ok(())
        })();
        if let Err(e) = write_res {
            let _ = FileExt::unlock(&file);
            return Err(BundleLockError::Io(e));
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
        if let Some(file) = self.file.take() {
            let _ = FileExt::unlock(&file);
            drop(file);
        }
    }
}
