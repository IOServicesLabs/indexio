//! ShardWriter: accumulates docs, then builds the immutable shard file.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use indexio_types::codec::{encode_doc_postings, write_varint};
use indexio_types::{DocMeta, ExtractedArtifact};
use fst::MapBuilder;
use roaring::RoaringBitmap;
use tracing::{debug, info, warn};

use crate::format::{self, kind, put_u16, put_u32, put_u64, DOC_REC_LEN, TOMBSTONES_LEN};
use crate::ShardMeta;

const ZSTD_LEVEL: i32 = 3;
/// Docs threshold above which a zstd dictionary is trained (SPEC).
const DICT_MIN_DOCS: usize = 100;
/// Target dictionary size (SPEC: 112 KiB), clamped to what the sample set
/// can support so training does not fail on small corpora.
const DICT_TARGET: usize = 112 * 1024;
const GRAM_STATS_TOP: usize = 10_000;

struct PendingDoc {
    meta: DocMeta,
    content: Vec<u8>,
    art: ExtractedArtifact,
    dead: bool,
}

/// Writes `shard-<ulid>.cidx.tmp` in `dir`; `finish()` renames atomically.
///
/// Duplicate `(repo_id, path)` within one shard: **replace-by-latest** —
/// the earlier doc is dropped from the shard (the docid returned for it
/// becomes invalid). Blob dedup across different paths is NOT done here
/// (ingest handles that via the CAS).
pub struct ShardWriter {
    tmp_path: PathBuf,
    final_path: PathBuf,
    docs: Vec<PendingDoc>,
    seen: HashMap<(u32, String), usize>,
    created_unix: u64,
    finished: bool,
}

/// STRINGS blob builder with interning. `off` values are relative to the
/// blob start (the blob sits after the repo table in the STRINGS section).
#[derive(Default)]
struct StringBlob {
    blob: Vec<u8>,
    intern: HashMap<Vec<u8>, u32>,
}

impl StringBlob {
    fn intern_bytes(&mut self, s: &[u8]) -> u32 {
        if let Some(&off) = self.intern.get(s) {
            return off;
        }
        let off = self.blob.len() as u32;
        self.blob.extend_from_slice(s);
        self.intern.insert(s.to_vec(), off);
        off
    }

    /// Intern a NUL-terminated string (scope/caller payloads store only an
    /// offset, so the terminator marks the end).
    fn intern_cstr(&mut self, s: &str) -> u32 {
        let clean = s.split('\0').next().unwrap_or("");
        let mut key = clean.as_bytes().to_vec();
        key.push(0);
        if let Some(&off) = self.intern.get(&key) {
            return off;
        }
        let off = self.blob.len() as u32;
        self.blob.extend_from_slice(&key);
        self.intern.insert(key, off);
        off
    }
}

fn to_io(e: impl std::fmt::Display) -> io::Error {
    crate::invalid(e.to_string())
}

fn compress(content: &[u8], dict: Option<&[u8]>) -> io::Result<Vec<u8>> {
    match dict {
        None => zstd::stream::encode_all(content, ZSTD_LEVEL),
        Some(d) => {
            let mut enc = zstd::stream::Encoder::with_dictionary(Vec::new(), ZSTD_LEVEL, d)?;
            enc.write_all(content)?;
            enc.finish()
        }
    }
}

impl ShardWriter {
    pub fn new(dir: &Path) -> io::Result<Self> {
        let id = ulid::Ulid::new();
        let tmp_path = dir.join(format!("shard-{id}.cidx.tmp"));
        let final_path = dir.join(format!("shard-{id}.cidx"));
        // Reserve the temp path now (also proves the dir is writable).
        File::create(&tmp_path)?;
        let created_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        debug!(?tmp_path, "shard writer created");
        Ok(ShardWriter {
            tmp_path,
            final_path,
            docs: Vec::new(),
            seen: HashMap::new(),
            created_unix,
            finished: false,
        })
    }

    /// Override creation time (used by merge so the compound shard orders
    /// as old as its newest input shard).
    pub(crate) fn set_created_unix(&mut self, t: u64) {
        self.created_unix = t;
    }

    pub fn add_doc(
        &mut self,
        meta: &DocMeta,
        content: &[u8],
        art: &ExtractedArtifact,
    ) -> io::Result<u32> {
        let key = (meta.repo_id, meta.path.clone());
        if let Some(&i) = self.seen.get(&key) {
            self.docs[i].dead = true; // replace-by-latest
        }
        let idx = self.docs.len();
        self.seen.insert(key, idx);
        self.docs.push(PendingDoc {
            meta: meta.clone(),
            content: content.to_vec(),
            art: art.clone(),
            dead: false,
        });
        Ok(idx as u32)
    }

    pub fn finish(mut self, repos: &[String]) -> io::Result<PathBuf> {
        let live: Vec<PendingDoc> = std::mem::take(&mut self.docs)
            .into_iter()
            .filter(|d| !d.dead)
            .collect();
        let n = live.len();

        // ---- strings blob + posting maps --------------------------------
        let mut strings = StringBlob::default();
        let mut repo_offs: Vec<(u32, u32)> = Vec::with_capacity(repos.len());
        for r in repos {
            let off = strings.intern_bytes(r.as_bytes());
            repo_offs.push((off, r.len() as u32));
        }

        // name -> (docid, line, kind, scope_off); callee -> (docid, caller_off, line)
        let mut ngram_map: BTreeMap<Vec<u8>, Vec<u32>> = BTreeMap::new();
        let mut sym_map: BTreeMap<Vec<u8>, Vec<(u32, u32, u8, u32)>> = BTreeMap::new();
        let mut call_map: BTreeMap<Vec<u8>, Vec<(u32, u32, u32)>> = BTreeMap::new();
        let mut path_offs: Vec<(u32, u32)> = Vec::with_capacity(n);
        let mut total_raw: u64 = 0;

        for (i, d) in live.iter().enumerate() {
            let docid = i as u32;
            total_raw += d.meta.raw_len as u64;
            let poff = strings.intern_bytes(d.meta.path.as_bytes());
            path_offs.push((poff, d.meta.path.len() as u32));
            for gram in &d.art.ngrams {
                ngram_map.entry(gram.clone()).or_default().push(docid);
            }
            for s in &d.art.symbols {
                let soff = strings.intern_cstr(&s.scope);
                sym_map
                    .entry(s.name.as_bytes().to_vec())
                    .or_default()
                    .push((docid, s.line, s.kind.as_u8(), soff));
            }
            for c in &d.art.calls {
                let coff = strings.intern_cstr(&c.caller);
                call_map
                    .entry(c.callee.as_bytes().to_vec())
                    .or_default()
                    .push((docid, coff, c.line));
            }
        }

        // ---- gram stats (top-10k by doc frequency) -----------------------
        let mut stats: Vec<(&Vec<u8>, u64)> =
            ngram_map.iter().map(|(g, v)| (g, v.len() as u64)).collect();
        stats.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let gram_stats: Vec<(String, u64)> = stats
            .into_iter()
            .take(GRAM_STATS_TOP)
            .map(|(g, c)| (String::from_utf8_lossy(g).into_owned(), c))
            .collect();

        // ---- zstd dictionary (>100 docs) ---------------------------------
        let dict: Option<Vec<u8>> = if n > DICT_MIN_DOCS {
            let samples: Vec<&[u8]> = live
                .iter()
                .take(512)
                .map(|d| &d.content[..d.content.len().min(64 * 1024)])
                .collect();
            let total: usize = samples.iter().map(|s| s.len()).sum();
            let size = DICT_TARGET.min(total / 10).max(1024);
            match zstd::dict::from_samples(&samples, size) {
                Ok(d) => Some(d),
                Err(e) => {
                    warn!(error = %e, "zstd dict training failed; writing shard without dict");
                    None
                }
            }
        } else {
            None
        };

        // ---- CONTENT section --------------------------------------------
        let mut content_sect: Vec<u8> = Vec::new();
        let mut content_offs: Vec<(u64, u32)> = Vec::with_capacity(n);
        for d in &live {
            let frame = compress(&d.content, dict.as_deref())?;
            content_offs.push((content_sect.len() as u64, frame.len() as u32));
            content_sect.extend_from_slice(&frame);
        }

        // ---- DOCS section ------------------------------------------------
        let mut docs_sect: Vec<u8> = Vec::with_capacity(n * DOC_REC_LEN);
        for (i, d) in live.iter().enumerate() {
            let (poff, plen) = path_offs[i];
            let (coff, clen) = content_offs[i];
            docs_sect.extend_from_slice(&d.meta.blob.0);
            put_u32(&mut docs_sect, d.meta.repo_id);
            put_u32(&mut docs_sect, poff);
            put_u32(&mut docs_sect, plen);
            put_u16(&mut docs_sect, d.meta.lang.as_u16());
            put_u16(&mut docs_sect, 0); // flags
            put_u64(&mut docs_sect, coff);
            put_u32(&mut docs_sect, clen);
            put_u32(&mut docs_sect, d.meta.raw_len);
        }

        // ---- STRINGS section ---------------------------------------------
        let mut strings_sect: Vec<u8> = Vec::new();
        put_u32(&mut strings_sect, repos.len() as u32);
        for (off, len) in &repo_offs {
            put_u32(&mut strings_sect, *off);
            put_u32(&mut strings_sect, *len);
        }
        strings_sect.extend_from_slice(&strings.blob);

        // ---- NGRAM / SYM / CALL FST + POST sections ----------------------
        let (ngram_fst, ngram_post) = build_ngram_index(&ngram_map)?;
        let (sym_fst, sym_post) = build_sym_index(&sym_map)?;
        let (call_fst, call_post) = build_call_index(&call_map)?;

        // ---- TOMBSTONES (empty bitmap, pre-sized 64KiB) ------------------
        let mut tomb = Vec::with_capacity(TOMBSTONES_LEN);
        RoaringBitmap::new().serialize_into(&mut tomb)?;
        tomb.resize(TOMBSTONES_LEN, 0);

        // ---- META ---------------------------------------------------------
        let meta = ShardMeta {
            repos: repos.to_vec(),
            doc_count: n as u64,
            total_raw_bytes: total_raw,
            gram_stats,
            zstd_dict: dict,
            created: format::unix_to_iso(self.created_unix),
        };
        let meta_bytes = serde_json::to_vec(&meta).map_err(to_io)?;

        // ---- assemble file ------------------------------------------------
        let sections: [(u32, &[u8]); 11] = [
            (kind::DOCS, &docs_sect),
            (kind::STRINGS, &strings_sect),
            (kind::CONTENT, &content_sect),
            (kind::NGRAM_FST, &ngram_fst),
            (kind::NGRAM_POST, &ngram_post),
            (kind::SYM_FST, &sym_fst),
            (kind::SYM_POST, &sym_post),
            (kind::CALL_FST, &call_fst),
            (kind::CALL_POST, &call_post),
            (kind::TOMBSTONES, &tomb),
            (kind::META, &meta_bytes),
        ];

        let mut offset =
            (format::HEADER_LEN + format::DIR_ENTRY_LEN * sections.len()) as u64;
        let mut dir_entries: Vec<(u32, u64, u64)> = Vec::with_capacity(sections.len());
        for (k, data) in &sections {
            offset = format::align8(offset);
            dir_entries.push((*k, offset, data.len() as u64));
            offset += data.len() as u64;
        }

        let header = format::encode_header(&format::Header {
            section_count: sections.len() as u32,
            doc_count: n as u64,
            created_unix: self.created_unix,
            flags: 0,
        });

        let file = File::create(&self.tmp_path)?; // truncates the placeholder
        let mut w = BufWriter::new(file);
        w.write_all(&header)?;
        for (k, off, len) in &dir_entries {
            put_u32_buf(&mut w, *k)?;
            put_u32_buf(&mut w, 0)?; // pad
            put_u64_buf(&mut w, *off)?;
            put_u64_buf(&mut w, *len)?;
        }
        let mut pos = (format::HEADER_LEN + format::DIR_ENTRY_LEN * sections.len()) as u64;
        const ZEROS: [u8; 8] = [0; 8];
        for ((_, data), (_, off, _)) in sections.iter().zip(dir_entries.iter()) {
            while pos < *off {
                let n = (*off - pos).min(8) as usize;
                w.write_all(&ZEROS[..n])?;
                pos += n as u64;
            }
            w.write_all(data)?;
            pos += data.len() as u64;
        }
        w.flush()?;
        w.get_ref().sync_all()?;
        drop(w);

        fs::rename(&self.tmp_path, &self.final_path)?;
        self.finished = true;
        info!(path = ?self.final_path, docs = n, "shard written");
        Ok(self.final_path.clone())
    }
}

fn put_u32_buf(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn put_u64_buf(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn build_ngram_index(map: &BTreeMap<Vec<u8>, Vec<u32>>) -> io::Result<(Vec<u8>, Vec<u8>)> {
    let mut post: Vec<u8> = Vec::new();
    let mut b = MapBuilder::memory();
    for (gram, docs) in map {
        let payload = encode_doc_postings(docs);
        let off = post.len() as u64;
        put_u32(&mut post, payload.len() as u32);
        post.extend_from_slice(&payload);
        b.insert(gram, off).map_err(to_io)?;
    }
    Ok((b.into_inner().map_err(to_io)?, post))
}

/// SYM payload per entry: varint(doc_delta), varint(line), u8 kind,
/// varint(scope_off into STRINGS blob) — SPEC.
fn build_sym_index(
    map: &BTreeMap<Vec<u8>, Vec<(u32, u32, u8, u32)>>,
) -> io::Result<(Vec<u8>, Vec<u8>)> {
    let mut post: Vec<u8> = Vec::new();
    let mut b = MapBuilder::memory();
    for (name, entries) in map {
        let mut payload = Vec::new();
        let mut prev = 0u32;
        for (docid, line, kind, scope_off) in entries {
            write_varint(&mut payload, (*docid - prev) as u64);
            prev = *docid;
            write_varint(&mut payload, *line as u64);
            payload.push(*kind);
            write_varint(&mut payload, *scope_off as u64);
        }
        let off = post.len() as u64;
        put_u32(&mut post, payload.len() as u32);
        post.extend_from_slice(&payload);
        b.insert(name, off).map_err(to_io)?;
    }
    Ok((b.into_inner().map_err(to_io)?, post))
}

/// CALL payload per entry: varint(doc_delta), varint(caller_off into
/// STRINGS blob), varint(line) — SPEC.
fn build_call_index(
    map: &BTreeMap<Vec<u8>, Vec<(u32, u32, u32)>>,
) -> io::Result<(Vec<u8>, Vec<u8>)> {
    let mut post: Vec<u8> = Vec::new();
    let mut b = MapBuilder::memory();
    for (callee, entries) in map {
        let mut payload = Vec::new();
        let mut prev = 0u32;
        for (docid, caller_off, line) in entries {
            write_varint(&mut payload, (*docid - prev) as u64);
            prev = *docid;
            write_varint(&mut payload, *caller_off as u64);
            write_varint(&mut payload, *line as u64);
        }
        let off = post.len() as u64;
        put_u32(&mut post, payload.len() as u32);
        post.extend_from_slice(&payload);
        b.insert(callee, off).map_err(to_io)?;
    }
    Ok((b.into_inner().map_err(to_io)?, post))
}

impl Drop for ShardWriter {
    fn drop(&mut self) {
        if !self.finished {
            let _ = fs::remove_file(&self.tmp_path);
        }
    }
}
