//! Cross-process compaction lock: `<data_dir>/compact.lock`.
//!
//! The protocol is the one every released version follows, because servers of
//! older versions keep running against the same data dir: the lock is the
//! file's existence, it is created exclusively and removed on release, and a
//! lock whose mtime is older than [`LOCK_STALE_SECS`] is taken to belong to a
//! dead process and may be removed and retaken.
//!
//! That protocol alone had three holes, all closed on the holder's side:
//!
//! * a pass longer than the stale age (a full re-embed on a busy machine) was
//!   taken over while it ran: a holder now refreshes the mtime from a
//!   heartbeat thread for as long as it holds the lock;
//! * release removed the file without looking at it, so a holder whose lock
//!   had been taken over deleted its successor's lock: release now removes
//!   the file only if it still carries this holder's token;
//! * a killed holder left the file behind for the stale age. On Windows the
//!   holder keeps the file open, delete-on-close and without delete sharing:
//!   the system removes it the moment the process ends, however it ends, and
//!   nobody else can remove it before that (an older server's takeover
//!   attempt and its unconditional remove on release both fail). Elsewhere
//!   the stale age still applies.

use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// Age past which a compaction lock is considered abandoned.
pub const LOCK_STALE_SECS: u64 = 15 * 60;

/// How often a holder refreshes the lock's mtime.
const HEARTBEAT: Duration = Duration::from_secs(60);

const LOCK_FILE: &str = "compact.lock";

/// Held for one compaction pass (shard merge, segment merge, full re-embed)
/// across every process sharing the data dir. Dropping releases it.
pub struct CompactLock {
    path: PathBuf,
    /// What this holder wrote into the file: `<pid> <nonce>` (older versions
    /// write the pid alone and never read it).
    token: String,
    file: Option<File>,
    /// The system removes the file when the last handle closes (Windows).
    auto_delete: bool,
    beat: Option<Heartbeat>,
}

impl CompactLock {
    /// `None` when another live process holds the lock.
    pub fn try_acquire(data_dir: &Path) -> Option<CompactLock> {
        Self::try_acquire_with(data_dir, true, HEARTBEAT)
    }

    /// [`try_acquire`](Self::try_acquire), waiting up to `wait` for a holder
    /// to finish (commands a person runs wait; servers skip and look again
    /// later).
    pub fn acquire(data_dir: &Path, wait: Duration) -> Option<CompactLock> {
        let deadline = Instant::now() + wait;
        loop {
            if let Some(lock) = Self::try_acquire(data_dir) {
                return Some(lock);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            std::thread::sleep(left.min(Duration::from_millis(250)));
        }
    }

    /// Whether some process holds a lock that is not stale: a cheap look for
    /// callers that would otherwise start a process only to find it taken.
    pub fn is_held(data_dir: &Path) -> bool {
        !is_stale(&data_dir.join(LOCK_FILE))
    }

    /// `os_lock: false` skips the Windows handle (the lock is then the bare
    /// protocol plus heartbeat and owner check, as on other systems).
    fn try_acquire_with(data_dir: &Path, os_lock: bool, heartbeat: Duration) -> Option<CompactLock> {
        let path = data_dir.join(LOCK_FILE);
        for _ in 0..2 {
            match create(&path, os_lock) {
                Ok((mut file, auto_delete)) => {
                    let nonce = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map_or(0, |d| d.as_nanos());
                    let token = format!("{} {nonce:x}", std::process::id());
                    let _ = file.write_all(token.as_bytes());
                    let _ = file.flush();
                    let beat = Heartbeat::start(&file, heartbeat);
                    return Some(CompactLock { path, token, file: Some(file), auto_delete, beat });
                }
                Err(_) => {
                    if !is_stale(&path) {
                        return None;
                    }
                    // fails while a holder of this version lives (Windows)
                    if std::fs::remove_file(&path).is_ok() {
                        tracing::warn!(path = %path.display(), "took over a stale compaction lock");
                    }
                }
            }
        }
        None
    }
}

/// No lock file, or one whose mtime is older than [`LOCK_STALE_SECS`].
fn is_stale(path: &Path) -> bool {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_none_or(|age| age.as_secs() > LOCK_STALE_SECS)
}

/// Create the lock file exclusively; the flag says whether the system
/// removes it when the handle closes.
#[cfg(windows)]
fn create(path: &Path, os_lock: bool) -> std::io::Result<(File, bool)> {
    use std::os::windows::fs::OpenOptionsExt as _;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
    if os_lock {
        // readable by others (the token), not writable and not removable
        let held = OpenOptions::new()
            .write(true)
            .create_new(true)
            .share_mode(FILE_SHARE_READ)
            .custom_flags(FILE_FLAG_DELETE_ON_CLOSE)
            .open(path);
        match held {
            Ok(f) => return Ok((f, true)),
            Err(e) if path.exists() => return Err(e),
            // a filesystem without these semantics: the bare protocol
            Err(_) => {}
        }
    }
    OpenOptions::new().write(true).create_new(true).open(path).map(|f| (f, false))
}

#[cfg(not(windows))]
fn create(path: &Path, _os_lock: bool) -> std::io::Result<(File, bool)> {
    OpenOptions::new().write(true).create_new(true).open(path).map(|f| (f, false))
}

impl Drop for CompactLock {
    fn drop(&mut self) {
        // the heartbeat's handle goes first: the file lives until the last one
        drop(self.beat.take());
        drop(self.file.take());
        if self.auto_delete {
            return;
        }
        // only our own lock: one that was taken over (we were suspended past
        // the stale age) belongs to its new holder
        if std::fs::read_to_string(&self.path).is_ok_and(|s| s == self.token) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// The thread that keeps a held lock's mtime fresh. Parked on a condvar
/// between beats; dropping it stops and joins the thread.
struct Heartbeat {
    stop: Arc<(Mutex<bool>, Condvar)>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Heartbeat {
    /// `None` when the handle cannot be duplicated or no thread can be
    /// started: the lock is then as good as an older version's.
    fn start(file: &File, every: Duration) -> Option<Heartbeat> {
        let file = file.try_clone().ok()?;
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let shared = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("compact-lock".into())
            .spawn(move || {
                let (flag, wake) = &*shared;
                let mut stopped = flag.lock().unwrap_or_else(|e| e.into_inner());
                while !*stopped {
                    let (guard, timeout) = wake.wait_timeout(stopped, every).unwrap_or_else(|e| e.into_inner());
                    stopped = guard;
                    if !*stopped && timeout.timed_out() {
                        let _ = file.set_modified(SystemTime::now());
                    }
                }
            })
            .ok()?;
        Some(Heartbeat { stop, thread: Some(thread) })
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        let (flag, wake) = &*self.stop;
        *flag.lock().unwrap_or_else(|e| e.into_inner()) = true;
        wake.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lock_path(dir: &Path) -> PathBuf {
        dir.join(LOCK_FILE)
    }

    fn age(path: &Path, secs: u64) {
        let f = OpenOptions::new().write(true).open(path).unwrap();
        f.set_modified(SystemTime::now() - Duration::from_secs(secs)).unwrap();
    }

    #[test]
    fn one_holder_at_a_time_and_release_frees_it() {
        let tmp = tempfile::tempdir().unwrap();
        let a = CompactLock::try_acquire(tmp.path()).expect("free lock");
        assert!(CompactLock::is_held(tmp.path()));
        assert!(CompactLock::try_acquire(tmp.path()).is_none(), "held lock was handed out twice");
        // an older version reads nothing but the mtime; ours starts with the pid
        let text = std::fs::read_to_string(lock_path(tmp.path())).unwrap();
        assert_eq!(text.split(' ').next(), Some(std::process::id().to_string().as_str()));
        drop(a);
        assert!(!lock_path(tmp.path()).exists(), "release must remove the file");
        assert!(!CompactLock::is_held(tmp.path()));
        assert!(CompactLock::try_acquire(tmp.path()).is_some());
    }

    #[test]
    fn acquire_waits_for_the_holder() {
        let tmp = tempfile::tempdir().unwrap();
        let a = CompactLock::try_acquire(tmp.path()).unwrap();
        assert!(CompactLock::acquire(tmp.path(), Duration::from_millis(50)).is_none());
        let dir = tmp.path().to_path_buf();
        let waiter = std::thread::spawn(move || CompactLock::acquire(&dir, Duration::from_secs(20)).is_some());
        std::thread::sleep(Duration::from_millis(300));
        drop(a);
        assert!(waiter.join().unwrap(), "the waiter must get the lock once it is released");
    }

    /// A lock left by a dead process of an older version (pid only, no
    /// heartbeat) is taken over once it is older than the stale age, not
    /// before.
    #[test]
    fn a_legacy_lock_is_taken_over_only_when_stale() {
        let tmp = tempfile::tempdir().unwrap();
        let p = lock_path(tmp.path());
        std::fs::write(&p, "4242").unwrap();
        assert!(CompactLock::try_acquire(tmp.path()).is_none(), "a fresh legacy lock is someone's");
        age(&p, LOCK_STALE_SECS + 60);
        let l = CompactLock::try_acquire(tmp.path()).expect("stale lock is taken over");
        assert_ne!(std::fs::read_to_string(&p).unwrap(), "4242");
        drop(l);
        assert!(!p.exists());
    }

    /// Release removes the file only when it is still this holder's: after
    /// a takeover the file is the new holder's lock.
    #[test]
    fn release_leaves_a_lock_that_was_taken_over() {
        let tmp = tempfile::tempdir().unwrap();
        let p = lock_path(tmp.path());
        // the bare protocol, as on systems without the Windows handle
        let a = CompactLock::try_acquire_with(tmp.path(), false, HEARTBEAT).unwrap();
        // what a process that judged the lock stale does: remove, create its own
        std::fs::remove_file(&p).unwrap();
        std::fs::write(&p, "777 feed").unwrap();
        drop(a);
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "777 feed", "release removed the successor's lock");
    }

    #[test]
    fn heartbeat_keeps_the_mtime_fresh() {
        let tmp = tempfile::tempdir().unwrap();
        let p = lock_path(tmp.path());
        let l = CompactLock::try_acquire_with(tmp.path(), true, Duration::from_millis(40)).unwrap();
        let first = std::fs::metadata(&p).unwrap().modified().unwrap();
        let mut moved = false;
        for _ in 0..100 {
            std::thread::sleep(Duration::from_millis(50));
            if std::fs::metadata(&p).unwrap().modified().unwrap() > first {
                moved = true;
                break;
            }
        }
        assert!(moved, "the holder never refreshed the lock's mtime");
        drop(l);
        assert!(!p.exists());
    }

    /// While a holder of this version lives nobody can remove its lock: not
    /// an older server that judges it stale, not an older holder's
    /// unconditional remove on release.
    #[cfg(windows)]
    #[test]
    fn a_held_lock_cannot_be_removed_by_others() {
        let tmp = tempfile::tempdir().unwrap();
        let p = lock_path(tmp.path());
        let l = CompactLock::try_acquire(tmp.path()).unwrap();
        assert!(std::fs::remove_file(&p).is_err(), "the lock file was removable while held");
        assert!(p.exists());
        assert!(OpenOptions::new().write(true).open(&p).is_err(), "the lock file was writable by others");
        assert!(std::fs::read_to_string(&p).is_ok(), "others must be able to read the token");
        // what an older version does when the mtime looks stale: remove and
        // create its own; both must fail while the holder lives
        assert!(OpenOptions::new().write(true).create_new(true).open(&p).is_err());
        drop(l);
        assert!(!p.exists(), "the system removes the file with the holder's handle");
    }
}
