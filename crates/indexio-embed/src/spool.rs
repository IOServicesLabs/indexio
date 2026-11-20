//! Scratch storage for the segment writers.
//!
//! A segment file is laid out once every row is known: the metadata block
//! precedes the rows and its length comes last, the BM25 term dictionary
//! precedes the postings it points into. The writers used to keep every
//! section in memory until then, which on a rebuild is the whole corpus
//! (7.8 GB of f32 rows at 945k rows, before the copies made to lay the file
//! out). A [`Spool`] takes those bytes instead: in memory while they are few
//! (an incremental append never touches the disk for them), in an unnamed
//! file of the segment's directory past a limit, read back once when the
//! file is assembled.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

/// Buffer of a spool's file, and of the copy back out of it.
const BUF: usize = 1 << 20;

/// A file in `dir` that is gone once its handle closes, however the process
/// ends: unlinked at creation on Unix, delete-on-close on Windows. There it
/// keeps a `.tmp` name while it is open, which every segment listing skips
/// and no other process can open; should one outlive its process after all,
/// the stale-file sweep of the segment directories removes it.
pub(crate) fn scratch_file(dir: &Path, tag: &str) -> io::Result<File> {
    static SEQ: AtomicU32 = AtomicU32::new(0);
    fs::create_dir_all(dir)?;
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let name = format!(".{tag}.{ms}.{}.{}.tmp", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed));
    let path = dir.join(name);
    let mut opts = fs::OpenOptions::new();
    opts.read(true).write(true).create_new(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x0000_0100;
        const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
        opts.custom_flags(FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE).share_mode(0);
    }
    let file = opts.open(&path)?;
    #[cfg(not(windows))]
    let _ = fs::remove_file(&path);
    Ok(file)
}

/// Bytes written once, in order, and read back: held in memory up to a
/// limit, in a scratch file past it.
pub(crate) struct Spool {
    dir: PathBuf,
    tag: &'static str,
    limit: usize,
    mem: Vec<u8>,
    file: Option<BufWriter<File>>,
    len: u64,
}

impl Spool {
    /// A spool whose scratch file, if it comes to one, lives in `dir`.
    pub fn new(dir: &Path, tag: &'static str, limit: usize) -> Self {
        Spool { dir: dir.to_path_buf(), tag, limit, mem: Vec::new(), file: None, len: 0 }
    }

    /// Bytes written so far.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the bytes have left memory for a scratch file.
    #[cfg(test)]
    pub fn on_disk(&self) -> bool {
        self.file.is_some()
    }

    pub fn write(&mut self, bytes: &[u8]) -> io::Result<()> {
        if let Some(f) = &mut self.file {
            f.write_all(bytes)?;
        } else if self.mem.len() + bytes.len() <= self.limit {
            self.mem.extend_from_slice(bytes);
        } else {
            let mut f = BufWriter::with_capacity(BUF, scratch_file(&self.dir, self.tag)?);
            f.write_all(&self.mem)?;
            f.write_all(bytes)?;
            self.mem = Vec::new();
            self.file = Some(f);
        }
        self.len += bytes.len() as u64;
        Ok(())
    }

    /// The `buf.len()` bytes written at offset `at`.
    pub fn read_at(&mut self, at: u64, buf: &mut [u8]) -> io::Result<()> {
        let past_end = || io::Error::new(io::ErrorKind::UnexpectedEof, "spool: read past the bytes written");
        if at.checked_add(buf.len() as u64).is_none_or(|end| end > self.len) {
            return Err(past_end());
        }
        match &mut self.file {
            None => {
                let at = at as usize;
                buf.copy_from_slice(&self.mem[at..at + buf.len()]);
            }
            Some(f) => {
                f.flush()?;
                let file = f.get_mut();
                file.seek(SeekFrom::Start(at))?;
                file.read_exact(buf)?;
                file.seek(SeekFrom::Start(self.len))?;
            }
        }
        Ok(())
    }

    /// Everything written, in order.
    pub fn reader(&mut self) -> io::Result<SpoolReader<'_>> {
        match &mut self.file {
            None => Ok(SpoolReader::Mem(&self.mem)),
            Some(f) => {
                f.flush()?;
                let file = f.get_mut();
                file.seek(SeekFrom::Start(0))?;
                Ok(SpoolReader::File(BufReader::with_capacity(BUF, file).take(self.len)))
            }
        }
    }

    /// Copy everything written into `out`. The spool is spent afterwards
    /// (its file is left at its end, nothing more is meant to be written).
    pub fn copy_to(&mut self, out: &mut dyn Write) -> io::Result<()> {
        let len = self.len;
        match self.reader()? {
            SpoolReader::Mem(mem) => out.write_all(mem),
            SpoolReader::File(mut file) => {
                let copied = io::copy(&mut file, out)?;
                if copied != len {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "spool: scratch file is shorter than what was written"));
                }
                Ok(())
            }
        }
    }
}

/// A spool's bytes, front to back.
pub(crate) enum SpoolReader<'a> {
    Mem(&'a [u8]),
    File(io::Take<BufReader<&'a mut File>>),
}

impl Read for SpoolReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            SpoolReader::Mem(mem) => mem.read(buf),
            SpoolReader::File(file) => file.read(buf),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(n: usize, seed: u8) -> Vec<u8> {
        (0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect()
    }

    #[test]
    fn a_spool_reads_back_what_was_written_in_memory_and_on_disk() {
        let tmp = tempfile::tempdir().unwrap();
        for limit in [0usize, 10, 1000, 1 << 20] {
            let mut spool = Spool::new(tmp.path(), "t", limit);
            let mut want = Vec::new();
            for (i, n) in [0usize, 1, 7, 300, 5000, 2, 0, 90_000].into_iter().enumerate() {
                let b = bytes(n, i as u8);
                spool.write(&b).unwrap();
                want.extend_from_slice(&b);
                // a row written earlier, read while writing goes on
                if want.len() >= 300 {
                    let mut got = vec![0u8; 293];
                    spool.read_at(5, &mut got).unwrap();
                    assert_eq!(got, want[5..298]);
                }
            }
            assert_eq!(spool.len(), want.len() as u64);
            assert_eq!(spool.on_disk(), want.len() > limit, "limit {limit}");
            // the end, and one byte past it
            let mut tail = [0u8; 16];
            spool.read_at(want.len() as u64 - 16, &mut tail).unwrap();
            assert_eq!(tail, want[want.len() - 16..]);
            assert!(spool.read_at(want.len() as u64 - 15, &mut tail).is_err());
            // writes after a read land at the end
            spool.write(b"tail").unwrap();
            want.extend_from_slice(b"tail");
            let mut out = Vec::new();
            spool.copy_to(&mut out).unwrap();
            assert_eq!(out, want, "limit {limit}");
            let mut again = Vec::new();
            spool.reader().unwrap().read_to_end(&mut again).unwrap();
            assert_eq!(again, want);
        }
    }

    #[test]
    fn a_scratch_file_leaves_nothing_behind() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("made-on-demand");
        let mut f = scratch_file(&dir, "t").unwrap();
        f.write_all(b"scratch").unwrap();
        f.seek(SeekFrom::Start(0)).unwrap();
        let mut back = String::new();
        f.read_to_string(&mut back).unwrap();
        assert_eq!(back, "scratch");
        // nothing a segment listing or a reader could take for a segment
        for e in fs::read_dir(&dir).unwrap().flatten() {
            assert_eq!(e.path().extension().and_then(|x| x.to_str()), Some("tmp"));
        }
        drop(f);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0, "the scratch file outlived its handle");
    }
}
