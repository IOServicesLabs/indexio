//! Shared by the benchmark and the memory tests: real source text cut into
//! chunks, and a synthetic vocabulary of any size.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

const EXTS: [&str; 6] = ["rs", "py", "ts", "js", "toml", "md"];
const CHUNK_LINES: usize = 20;

fn source_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if p.is_dir() {
            if !(name.starts_with('.') || name.starts_with("target") || name == "node_modules") {
                source_files(&p, out);
            }
        } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| EXTS.contains(&e)) {
            out.push(p);
        }
    }
}

/// `header + "\n" + body` chunks of [`CHUNK_LINES`] lines of every source
/// file under `root`, in path order: the shape the pipeline hands to the
/// embedder.
pub fn chunks(root: &Path) -> Vec<String> {
    let mut files = Vec::new();
    source_files(root, &mut files);
    let mut out = Vec::new();
    for f in files {
        let Ok(text) = std::fs::read_to_string(&f) else { continue };
        let rel = f.strip_prefix(root).unwrap_or(&f).to_string_lossy().replace('\\', "/");
        let lines: Vec<&str> = text.lines().collect();
        for (i, part) in lines.chunks(CHUNK_LINES).enumerate() {
            out.push(format!("{rel} lines {}-{}\n{}", i * CHUNK_LINES + 1, i * CHUNK_LINES + part.len(), part.join("\n")));
        }
    }
    out
}

/// `INDEXIO_BENCH_DIR`, or this workspace's `crates/`.
pub fn bench_root() -> PathBuf {
    std::env::var_os("INDEXIO_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."))
}

const SYLLABLES: [&str; 20] =
    ["ka", "to", "mi", "ra", "se", "lu", "no", "vi", "da", "pe", "xo", "zu", "be", "fi", "go", "hy", "ja", "qu", "wi", "cy"];

/// A distinct lowercase word per `n` (one token for the rindex tokenizer).
pub fn word(mut n: usize) -> String {
    let mut w = String::new();
    let mut digits = 0;
    loop {
        w.push_str(SYLLABLES[n % 20]);
        n /= 20;
        digits += 1;
        if n == 0 && digits >= 3 {
            return w;
        }
    }
}

/// Deterministic xorshift64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> usize {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as usize
    }
}

/// `n` texts of `words` words each (a header line, then the body): a
/// quarter of the words come from 300 frequent ones, the rest from
/// `distinct` rare ones, so contexts fill up and bigrams repeat.
pub fn synthetic_texts(n: usize, words: usize, distinct: usize, seed: u64) -> Vec<String> {
    let mut rng = Rng(seed.max(1));
    (0..n)
        .map(|_| {
            let mut t = String::new();
            for i in 0..words {
                let w = if rng.next() % 4 == 0 { word(rng.next() % 300) } else { word(300 + rng.next() % distinct) };
                t.push_str(&w);
                t.push(if i == 3 { '\n' } else { ' ' });
            }
            t
        })
        .collect()
}
