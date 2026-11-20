//! Merge of a run of shards into one compound shard, in bounded memory.
//!
//! The compound is byte for byte the file the first merge wrote, which
//! rebuilt every doc in memory (its content, and a gram list of one heap
//! block per (doc, gram) inverted from the postings) and handed them to
//! [`ShardWriter`](crate::ShardWriter): 2.4 GB of heap to merge a 120 MB
//! shard. Here nothing is held per gram or per content byte:
//!
//! - docs are renumbered up front (a `u32` per input doc);
//! - STRINGS is rebuilt in the order the writer interned it (per doc: its
//!   path, the scopes of its symbols, the callers of its calls), from the
//!   string offsets of the symbol and call postings (8 bytes per posting
//!   entry of one input shard at a time);
//! - contents are decompressed and recompressed one doc at a time, straight
//!   into the output file;
//! - posting lists are merged key by key over the inputs' FSTs (which are
//!   sorted, like the map the writer built), docids renumbered; the FST is
//!   streamed into the output file and the postings into a spill file that
//!   is appended after it.
//!
//! Peak heap is the dictionary training set (at most 512 docs of 64 KiB),
//! or the largest doc, plus the tables above.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use fst::{MapBuilder, Streamer};
use indexio_types::codec::{encode_doc_postings, write_varint, PostingCursor};
use indexio_types::SymbolKind;
use tracing::info;

use crate::format::{self, kind, put_u16, put_u32, put_u64, DOC_REC_LEN};
use crate::shard::{decode_call_payload, decode_sym_payload};
use crate::writer::{self, to_io, StringBlob, DICT_MIN_DOCS, DICT_SAMPLES, DICT_SAMPLE_BYTES};
use crate::{invalid, Shard, ShardMeta};

/// `remap` entry of a doc that is not kept.
const GONE: u32 = u32::MAX;

/// A compound shard written by [`write_compound`].
pub(crate) struct Compound {
    pub path: PathBuf,
    /// Per doc of the compound: (index into the merged run, docid there).
    /// Lets tombstones that land on an input during the merge be forwarded.
    pub origin: Vec<(usize, u32)>,
}

/// One input shard and where its docs go.
struct Source<'a> {
    shard: &'a Shard,
    /// Index of the shard in the run (newest first).
    run_idx: usize,
    /// Shard-local repo id -> repo id in the compound.
    repo_ids: Vec<u32>,
    /// Docid here -> docid in the compound, [`GONE`] for a doc that is
    /// tombstoned or superseded within the run.
    remap: Vec<u32>,
    /// Offset of a scope or caller string in this shard's STRINGS blob ->
    /// its offset in the compound's.
    strings: HashMap<u32, u32>,
}

impl Source<'_> {
    fn new_id(&self, docid: u32) -> Option<u32> {
        self.remap.get(docid as usize).copied().filter(|&d| d != GONE)
    }
}

/// The output file, written front to back; its head (header, directory and
/// DOCS) is filled in last.
struct Out {
    w: BufWriter<File>,
    pos: u64,
}

impl Write for Out {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.w.write(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.w.flush()
    }
}

impl Out {
    fn zeros(&mut self, mut n: u64) -> io::Result<()> {
        const ZEROS: [u8; 4096] = [0; 4096];
        while n > 0 {
            let k = n.min(ZEROS.len() as u64) as usize;
            self.write_all(&ZEROS[..k])?;
            n -= k as u64;
        }
        Ok(())
    }

    /// Pad to the next section boundary; the offset the section starts at.
    fn align(&mut self) -> io::Result<u64> {
        self.zeros(format::align8(self.pos) - self.pos)?;
        Ok(self.pos)
    }

    /// Write `data` as the next section; its (offset, length).
    fn section(&mut self, data: &[u8]) -> io::Result<(u64, u64)> {
        let off = self.align()?;
        self.write_all(data)?;
        Ok((off, data.len() as u64))
    }
}

/// A posting section on its way to the output file: its FST comes first in
/// the file and is only complete after the last posting list.
struct Spill {
    file: BufWriter<File>,
    len: u64,
}

impl Spill {
    fn create(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;
        Ok(Spill { file: BufWriter::new(file), len: 0 })
    }

    fn push(&mut self, data: &[u8]) -> io::Result<()> {
        self.file.write_all(data)?;
        self.len += data.len() as u64;
        Ok(())
    }

    /// Append everything pushed to `out` and start over.
    fn drain_into(&mut self, out: &mut Out) -> io::Result<u64> {
        self.file.flush()?;
        let len = self.len;
        let f = self.file.get_mut();
        f.seek(SeekFrom::Start(0))?;
        let copied = io::copy(&mut f.take(len), out)?;
        if copied != len {
            return Err(invalid("merge: posting spill file is short"));
        }
        let f = self.file.get_mut();
        f.set_len(0)?;
        f.seek(SeekFrom::Start(0))?;
        self.len = 0;
        Ok(len)
    }
}

/// Removes a temp file when the merge is over, however it ends (the output
/// has been renamed by then if it succeeded).
struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Merge `run` (adjacent shards of a set, newest first) into one compound
/// shard in `out_dir`: tombstoned docs dropped, the newest doc of a
/// (repo, path) kept, the rest as [`ShardWriter`](crate::ShardWriter) would
/// have written the kept docs in oldest-shard-first order. The compound is
/// dated by its newest input. The inputs are left as they are.
pub(crate) fn write_compound(run: &[&Shard], out_dir: &Path) -> io::Result<Compound> {
    // ---- repo table: the union, oldest shard first -------------------------
    let mut repos: Vec<String> = Vec::new();
    let mut repo_of: HashMap<&str, u32> = HashMap::new();
    let mut sources: Vec<Source> = Vec::with_capacity(run.len());
    for (run_idx, &shard) in run.iter().enumerate().rev() {
        let repo_ids = shard
            .meta()
            .repos
            .iter()
            .map(|name| {
                *repo_of.entry(name.as_str()).or_insert_with(|| {
                    repos.push(name.clone());
                    repos.len() as u32 - 1
                })
            })
            .collect();
        sources.push(Source { shard, run_idx, repo_ids, remap: Vec::new(), strings: HashMap::new() });
    }

    // ---- kept docs: not tombstoned, the newest of its (repo, path) ---------
    {
        let mut seen: HashSet<(&str, &[u8])> = HashSet::new();
        for s in sources.iter_mut().rev() {
            // newest first
            let shard = s.shard;
            let tomb = shard.tombstones();
            s.remap = vec![GONE; shard.doc_count() as usize];
            for docid in 0..shard.doc_count() as u32 {
                if tomb.contains(docid) {
                    continue;
                }
                let Some((rid, path)) = shard.doc_key(docid) else { continue };
                let repo = shard.meta().repos.get(rid as usize).map_or("", |r| r.as_str());
                if seen.insert((repo, path)) {
                    s.remap[docid as usize] = 0; // kept; numbered below
                }
            }
        }
    }
    // Oldest shard first for a stable doc order in the compound.
    let mut origin: Vec<(usize, u32)> = Vec::new();
    for s in sources.iter_mut() {
        for (docid, slot) in s.remap.iter_mut().enumerate() {
            if *slot != GONE {
                *slot = origin.len() as u32;
                origin.push((s.run_idx, docid as u32));
            }
        }
    }
    let n = origin.len();
    if n >= GONE as usize {
        return Err(invalid("merge: too many docs for one shard"));
    }
    let created_unix = run.iter().map(|s| s.created_unix()).max().unwrap_or(0);

    // ---- DOCS (content frames still open) and STRINGS -----------------------
    let mut strings = StringBlob::default();
    let repo_offs: Vec<(u32, u32)> = repos.iter().map(|r| (strings.intern_bytes(r.as_bytes()), r.len() as u32)).collect();
    let mut docs_sect: Vec<u8> = Vec::with_capacity(n * DOC_REC_LEN);
    let mut total_raw: u64 = 0;
    for s in sources.iter_mut() {
        let shard = s.shard;
        let (repo_ids, remap, moved) = (&s.repo_ids, &s.remap, &mut s.strings);
        // (docid in the compound, string offset here) of every symbol and
        // call of a kept doc, per doc in the order the writer interned
        // them: by name, then as listed
        let mut scopes = string_refs(shard, remap, kind::SYM_FST, kind::SYM_POST, |p| {
            decode_sym_payload(p).into_iter().map(|(d, _, _, off)| (d, off)).collect()
        });
        let mut callers = string_refs(shard, remap, kind::CALL_FST, kind::CALL_POST, |p| {
            decode_call_payload(p).into_iter().map(|(d, off, _)| (d, off)).collect()
        });
        scopes.sort_by_key(|r| r.0); // stable: the order within a doc stays
        callers.sort_by_key(|r| r.0);
        let (mut si, mut ci) = (0, 0);
        for (docid, &new) in remap.iter().enumerate() {
            if new == GONE {
                continue;
            }
            let dm = shard.doc(docid as u32).ok_or_else(|| invalid("merge: doc record missing"))?;
            let repo_id = *repo_ids.get(dm.repo_id as usize).ok_or_else(|| invalid("merge: repo_id not mapped"))?;
            let path_off = strings.intern_bytes(dm.path.as_bytes());
            docs_sect.extend_from_slice(&dm.blob.0);
            put_u32(&mut docs_sect, repo_id);
            put_u32(&mut docs_sect, path_off);
            put_u32(&mut docs_sect, dm.path.len() as u32);
            put_u16(&mut docs_sect, dm.lang.as_u16());
            put_u16(&mut docs_sect, 0); // flags
            put_u64(&mut docs_sect, 0); // content_off: set when the frame is written
            put_u32(&mut docs_sect, 0); // content_len: likewise
            put_u32(&mut docs_sect, dm.raw_len);
            total_raw += dm.raw_len as u64;
            for refs in [(&scopes, &mut si), (&callers, &mut ci)] {
                let (list, at) = refs;
                while let Some(&(_, off)) = list.get(*at).filter(|r| r.0 == new) {
                    moved.entry(off).or_insert_with(|| strings.intern_cstr(&shard.cstr_at(off).unwrap_or_default()));
                    *at += 1;
                }
            }
        }
    }
    let mut strings_sect: Vec<u8> = Vec::with_capacity(4 + repo_offs.len() * 8 + strings.blob.len());
    put_u32(&mut strings_sect, repos.len() as u32);
    for (off, len) in &repo_offs {
        put_u32(&mut strings_sect, *off);
        put_u32(&mut strings_sect, *len);
    }
    strings_sect.extend_from_slice(&strings.blob);
    drop(strings);

    // ---- zstd dictionary (>100 docs), from the first docs -------------------
    let dict: Option<Vec<u8>> = if n > DICT_MIN_DOCS {
        let mut data: Vec<u8> = Vec::new();
        let mut sizes: Vec<usize> = Vec::new();
        for &(run_idx, docid) in origin.iter().take(DICT_SAMPLES) {
            let content = run[run_idx].content(docid)?;
            let take = content.len().min(DICT_SAMPLE_BYTES);
            data.extend_from_slice(&content[..take]);
            sizes.push(take);
        }
        writer::train_dict(&data, &sizes)
    } else {
        None
    };

    // ---- the file, front to back --------------------------------------------
    let (tmp_path, final_path) = writer::shard_paths(out_dir);
    let _tmp = RemoveOnDrop(tmp_path.clone());
    let spill_path = final_path.with_extension("post.tmp");
    let _spill = RemoveOnDrop(spill_path.clone());
    let mut out = Out { w: BufWriter::with_capacity(1 << 16, File::create(&tmp_path)?), pos: 0 };
    let mut spill = Spill::create(&spill_path)?;
    let mut sections: Vec<(u32, u64, u64)> = Vec::with_capacity(SECTIONS);
    let head_len = (format::HEADER_LEN + format::DIR_ENTRY_LEN * SECTIONS) as u64;

    // DOCS: its place is known, its content offsets are not yet
    let docs_off = format::align8(head_len);
    sections.push((kind::DOCS, docs_off, docs_sect.len() as u64));
    out.zeros(docs_off + docs_sect.len() as u64)?;

    let (off, len) = out.section(&strings_sect)?;
    sections.push((kind::STRINGS, off, len));
    drop(strings_sect);

    // CONTENT: one doc in memory at a time
    let content_off = out.align()?;
    for (new, &(run_idx, docid)) in origin.iter().enumerate() {
        let content = run[run_idx].content(docid)?; // raw bytes, recompressed with the new dictionary
        let frame = writer::compress(&content, dict.as_deref())?;
        let rec = new * DOC_REC_LEN;
        docs_sect[rec + 32..rec + 40].copy_from_slice(&(out.pos - content_off).to_le_bytes());
        docs_sect[rec + 40..rec + 44].copy_from_slice(&(frame.len() as u32).to_le_bytes());
        out.write_all(&frame)?;
    }
    sections.push((kind::CONTENT, content_off, out.pos - content_off));

    // NGRAM: docids only (positions an older shard may still carry are
    // dropped here, so a compaction also shrinks older shards)
    let mut ids: Vec<u32> = Vec::new();
    posting_sections(&sources, (kind::NGRAM_FST, kind::NGRAM_POST), &mut out, &mut spill, &mut sections, |hits, payload| {
        ids.clear();
        for &(i, off) in hits {
            let s = &sources[i];
            let Some(list) = s.shard.posting_payload(kind::NGRAM_POST, off) else { continue };
            let mut cursor = PostingCursor::new(list);
            while let Some(docid) = cursor.next_docid() {
                ids.extend(s.new_id(docid));
            }
        }
        if !ids.is_empty() {
            if ids.windows(2).any(|w| w[0] > w[1]) {
                return Err(invalid("merge: n-gram posting list out of order"));
            }
            payload.extend_from_slice(&encode_doc_postings(&ids));
        }
        Ok(())
    })?;
    drop(ids);

    // SYM: varint(doc_delta), varint(line), u8 kind, varint(scope_off)
    posting_sections(&sources, (kind::SYM_FST, kind::SYM_POST), &mut out, &mut spill, &mut sections, |hits, payload| {
        let mut prev = 0u32;
        for &(i, off) in hits {
            let s = &sources[i];
            let Some(list) = s.shard.posting_payload(kind::SYM_POST, off) else { continue };
            for (docid, line, k, scope_off) in decode_sym_payload(list) {
                let Some(new) = s.new_id(docid) else { continue };
                let scope = *s.strings.get(&scope_off).ok_or_else(|| invalid("merge: scope string not mapped"))?;
                let delta = new.checked_sub(prev).ok_or_else(|| invalid("merge: symbol posting list out of order"))?;
                prev = new;
                write_varint(payload, delta as u64);
                write_varint(payload, line as u64);
                payload.push(SymbolKind::from_u8(k).as_u8());
                write_varint(payload, scope as u64);
            }
        }
        Ok(())
    })?;

    // CALL: varint(doc_delta), varint(caller_off), varint(line)
    posting_sections(&sources, (kind::CALL_FST, kind::CALL_POST), &mut out, &mut spill, &mut sections, |hits, payload| {
        let mut prev = 0u32;
        for &(i, off) in hits {
            let s = &sources[i];
            let Some(list) = s.shard.posting_payload(kind::CALL_POST, off) else { continue };
            for (docid, caller_off, line) in decode_call_payload(list) {
                let Some(new) = s.new_id(docid) else { continue };
                let caller = *s.strings.get(&caller_off).ok_or_else(|| invalid("merge: caller string not mapped"))?;
                let delta = new.checked_sub(prev).ok_or_else(|| invalid("merge: call posting list out of order"))?;
                prev = new;
                write_varint(payload, delta as u64);
                write_varint(payload, caller as u64);
                write_varint(payload, line as u64);
            }
        }
        Ok(())
    })?;
    drop(spill);

    let (off, len) = out.section(&writer::empty_tombstones()?)?;
    sections.push((kind::TOMBSTONES, off, len));

    let meta = ShardMeta {
        repos,
        doc_count: n as u64,
        total_raw_bytes: total_raw,
        gram_stats: Vec::new(),
        zstd_dict: dict,
        created: format::unix_to_iso(created_unix),
    };
    let (off, len) = out.section(&serde_json::to_vec(&meta).map_err(to_io)?)?;
    sections.push((kind::META, off, len));
    debug_assert_eq!(sections.len(), SECTIONS);

    // ---- the head: header, directory, DOCS -----------------------------------
    let mut head: Vec<u8> = Vec::with_capacity(docs_off as usize + docs_sect.len());
    head.extend_from_slice(&format::encode_header(&format::Header {
        section_count: sections.len() as u32,
        doc_count: n as u64,
        created_unix,
        flags: 0,
    }));
    for (k, off, len) in &sections {
        put_u32(&mut head, *k);
        put_u32(&mut head, 0); // pad
        put_u64(&mut head, *off);
        put_u64(&mut head, *len);
    }
    head.resize(docs_off as usize, 0);
    head.extend_from_slice(&docs_sect);
    out.flush()?;
    let mut file = out.w.into_inner().map_err(|e| e.into_error())?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&head)?;
    file.sync_all()?;
    drop(file);

    fs::rename(&tmp_path, &final_path)?;
    info!(path = ?final_path, docs = n, "compound shard written");
    Ok(Compound { path: final_path, origin })
}

/// Sections of a shard file.
const SECTIONS: usize = 11;

/// (docid in the compound, string offset) of every entry `entries` decodes
/// from the posting lists of one FST section of `shard`, kept docs only, in
/// key order and then as listed.
fn string_refs(
    shard: &Shard,
    remap: &[u32],
    fst_kind: u32,
    post_kind: u32,
    entries: impl Fn(&[u8]) -> Vec<(u32, u32)>,
) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let Some(Ok(map)) = shard.section_slice(fst_kind).map(fst::Map::new) else {
        return out;
    };
    let mut stream = map.stream();
    while let Some((_, off)) = stream.next() {
        let Some(list) = shard.posting_payload(post_kind, off) else { continue };
        for (docid, string_off) in entries(list) {
            if let Some(&new) = remap.get(docid as usize).filter(|&&d| d != GONE) {
                out.push((new, string_off));
            }
        }
    }
    out
}

/// Write one FST section and its posting section. The keys are the union
/// of the inputs' keys in order; `merged(hits, payload)` appends the posting
/// payload of one key, given the `(source index, posting offset)` of every
/// input that has the key, oldest shard first. A key whose payload stays
/// empty (none of its docs is kept) is left out.
fn posting_sections(
    sources: &[Source],
    (fst_kind, post_kind): (u32, u32),
    out: &mut Out,
    spill: &mut Spill,
    sections: &mut Vec<(u32, u64, u64)>,
    mut merged: impl FnMut(&[(usize, u64)], &mut Vec<u8>) -> io::Result<()>,
) -> io::Result<()> {
    // an input whose FST does not open contributes nothing, as before
    let maps: Vec<(usize, fst::Map<&[u8]>)> = sources
        .iter()
        .enumerate()
        .filter_map(|(i, s)| Some((i, fst::Map::new(s.shard.section_slice(fst_kind)?).ok()?)))
        .collect();
    let mut op = fst::map::OpBuilder::new();
    for (_, map) in &maps {
        op = op.add(map);
    }
    let mut union = op.union();

    let fst_off = out.align()?;
    let mut builder = MapBuilder::new(&mut *out).map_err(to_io)?;
    let mut hits: Vec<(usize, u64)> = Vec::new();
    let mut payload: Vec<u8> = Vec::new();
    while let Some((key, values)) = union.next() {
        // the union lists a key's values by value, not by input
        hits.clear();
        hits.extend(values.iter().map(|v| (maps[v.index].0, v.value)));
        hits.sort_unstable();
        payload.clear();
        merged(&hits, &mut payload)?;
        if payload.is_empty() {
            continue;
        }
        builder.insert(key, spill.len).map_err(to_io)?;
        spill.push(&(payload.len() as u32).to_le_bytes())?;
        spill.push(&payload)?;
    }
    builder.finish().map_err(to_io)?;
    sections.push((fst_kind, fst_off, out.pos - fst_off));

    let post_off = out.align()?;
    let post_len = spill.drain_into(out)?;
    sections.push((post_kind, post_off, post_len));
    Ok(())
}

/// The merge up to 0.1.14, kept to show that [`write_compound`] writes the
/// same file: every kept doc is rebuilt in memory and handed to the writer.
#[cfg(test)]
pub(crate) fn write_compound_in_memory(run: &[&Shard], out_dir: &Path) -> io::Result<Compound> {
    use crate::ShardWriter;
    use indexio_types::{CallRec, ExtractedArtifact, SymbolRec};

    // Union repo string tables (oldest shard first), remap repo_ids.
    let mut repos: Vec<String> = Vec::new();
    let mut repo_ids: HashMap<String, u32> = HashMap::new();
    let mut repo_map: HashMap<(usize, u32), u32> = HashMap::new();
    for si in (0..run.len()).rev() {
        for (rid, name) in run[si].meta().repos.iter().enumerate() {
            let next = repos.len() as u32;
            let nid = *repo_ids.entry(name.clone()).or_insert_with(|| {
                repos.push(name.clone());
                next
            });
            repo_map.insert((si, rid as u32), nid);
        }
    }

    // Kept docs per merged shard: skip tombstoned, newest-wins on
    // (repo, path) within the merged group.
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut keep: HashMap<usize, Vec<bool>> = HashMap::new();
    for (si, shard) in run.iter().enumerate() {
        // newest first
        let mut kv = vec![false; shard.doc_count() as usize];
        for docid in 0..shard.doc_count() as u32 {
            if shard.tombstones().contains(docid) {
                continue;
            }
            let Some(dm) = shard.doc(docid) else { continue };
            let repo = shard.meta().repos.get(dm.repo_id as usize).cloned().unwrap_or_default();
            if seen.insert((repo, dm.path)) {
                kv[docid as usize] = true;
            }
        }
        keep.insert(si, kv);
    }

    let mut writer = ShardWriter::new(out_dir)?;
    writer.set_created_unix(run.iter().map(|s| s.created_unix()).max().unwrap_or(0));

    let mut origin: Vec<(usize, u32)> = Vec::new();
    for si in (0..run.len()).rev() {
        let shard = run[si];
        let kv = &keep[&si];
        let kept_ids: Vec<u32> = (0..shard.doc_count() as u32).filter(|d| kv[*d as usize]).collect();
        if kept_ids.is_empty() {
            continue;
        }
        origin.extend(kept_ids.iter().map(|&d| (si, d)));
        let mut local_of: HashMap<u32, usize> = HashMap::new();
        let mut arts: Vec<ExtractedArtifact> = Vec::with_capacity(kept_ids.len());
        for (local, &docid) in kept_ids.iter().enumerate() {
            local_of.insert(docid, local);
            arts.push(ExtractedArtifact::default());
        }
        for (gram, off) in shard.fst_entries(kind::NGRAM_FST) {
            let Some(payload) = shard.posting_payload(kind::NGRAM_POST, off) else { continue };
            let mut cursor = PostingCursor::new(payload);
            while let Some(docid) = cursor.next_docid() {
                if let Some(&local) = local_of.get(&docid) {
                    arts[local].ngrams.push(gram.clone());
                }
            }
        }
        for (name, off) in shard.fst_entries(kind::SYM_FST) {
            let Some(payload) = shard.posting_payload(kind::SYM_POST, off) else { continue };
            let name = String::from_utf8_lossy(&name).into_owned();
            for (docid, line, k, scope_off) in decode_sym_payload(payload) {
                if let Some(&local) = local_of.get(&docid) {
                    arts[local].symbols.push(SymbolRec {
                        name: name.clone(),
                        kind: SymbolKind::from_u8(k),
                        line,
                        col: 0, // col is not stored in the shard format
                        scope: shard.cstr_at(scope_off).unwrap_or_default(),
                    });
                }
            }
        }
        for (callee, off) in shard.fst_entries(kind::CALL_FST) {
            let Some(payload) = shard.posting_payload(kind::CALL_POST, off) else { continue };
            let callee = String::from_utf8_lossy(&callee).into_owned();
            for (docid, caller_off, line) in decode_call_payload(payload) {
                if let Some(&local) = local_of.get(&docid) {
                    arts[local].calls.push(CallRec {
                        callee: callee.clone(),
                        caller: shard.cstr_at(caller_off).unwrap_or_default(),
                        line,
                    });
                }
            }
        }
        for (local, &docid) in kept_ids.iter().enumerate() {
            let mut dm = shard.doc(docid).ok_or_else(|| invalid("merge: doc record missing"))?;
            dm.repo_id = *repo_map.get(&(si, dm.repo_id)).ok_or_else(|| invalid("merge: repo_id not mapped"))?;
            let content = shard.content(docid)?;
            let mut art = std::mem::take(&mut arts[local]);
            art.lang = dm.lang;
            art.raw_len = dm.raw_len;
            writer.add_doc(&dm, &content, &art)?;
        }
    }
    let path = writer.finish(&repos)?;
    Ok(Compound { path, origin })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ShardSet, ShardWriter};
    use indexio_types::{BlobId, CallRec, DocMeta, ExtractedArtifact, Lang, SymbolRec};
    use std::collections::BTreeSet;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
        fn pick<'a>(&mut self, of: &[&'a str]) -> &'a str {
            of[self.below(of.len() as u64) as usize]
        }
    }

    const WORDS: [&str; 24] = [
        "fn", "let", "merge", "shard", "posting", "docid", "tombstone", "writer", "reader", "größe", "世界", "héllo", "struct", "impl", "return",
        "compound", "window", "delta", "index", "search", "result", "\t", "{}", "();",
    ];
    const NAMES: [&str; 12] = ["open", "close", "merge", "Shard", "ShardSet", "new", "len", "größe", "write_all", "", "next", "run"];
    const SCOPES: [&str; 6] = ["", "Shard", "ShardSet", "mod_a::Inner", "größe", "impl Write for Out"];
    const REPOS: [&str; 4] = ["alpha", "beta", "gamma", ""];

    fn grams(content: &[u8]) -> Vec<Vec<u8>> {
        let mut set: BTreeSet<Vec<u8>> = BTreeSet::new();
        for w in content.windows(3) {
            set.insert(w.to_vec());
        }
        set.into_iter().collect()
    }

    fn doc(rng: &mut Lcg, repo_id: u32, path: String, words: u64) -> (DocMeta, Vec<u8>, ExtractedArtifact) {
        let mut text = String::new();
        for i in 0..words {
            text.push_str(rng.pick(&WORDS));
            text.push(if i % 9 == 8 { '\n' } else { ' ' });
        }
        // some docs carry a number of their own: grams no other doc has
        if rng.below(3) == 0 {
            text.push_str(&format!("{}\n", rng.next()));
        }
        let content = text.into_bytes();
        let mut ngrams = grams(&content);
        if rng.below(8) == 0 && !ngrams.is_empty() {
            // an artifact that lists a gram twice: a doc twice in one posting list
            ngrams.insert(0, ngrams[0].clone());
        }
        let symbols = (0..rng.below(6))
            .map(|i| SymbolRec {
                name: rng.pick(&NAMES).to_string(),
                kind: SymbolKind::from_u8(rng.below(13) as u8),
                line: i as u32 * 3 + 1,
                col: 0,
                scope: rng.pick(&SCOPES).to_string(),
            })
            .collect();
        let calls = (0..rng.below(8))
            .map(|i| CallRec { callee: rng.pick(&NAMES).to_string(), caller: rng.pick(&SCOPES).to_string(), line: i as u32 + 2 })
            .collect();
        let lang = Lang::from_u16(rng.below(8) as u16);
        let meta = DocMeta { blob: BlobId::from_content(&content), repo_id, path, lang, raw_len: content.len() as u32 };
        let art = ExtractedArtifact { ngrams, symbols, calls, raw_len: content.len() as u32, lang };
        (meta, content, art)
    }

    /// `shards` shards (oldest first on disk) of about `docs` docs each over
    /// a pool of `paths` paths per repo, so the same (repo, path) turns up
    /// in several shards and in several repos; then tombstones on a share of
    /// every shard. The set is returned newest first.
    fn random_set(dir: &Path, seed: u64, shards: usize, docs: u64, paths: u64, words: u64) -> ShardSet {
        let mut rng = Lcg(seed);
        for k in 0..shards {
            // every shard has its own repo table: a rotation of the pool
            let repos: Vec<String> = (0..1 + rng.below(4)).map(|i| REPOS[(k + i as usize) % REPOS.len()].to_string()).collect();
            let mut w = ShardWriter::new(dir).unwrap();
            // order by age without sleeping a second per shard
            w.set_created_unix(1_700_000_000 + k as u64 * 10);
            let n = if docs == 0 { 0 } else { docs / 2 + rng.below(docs) };
            for _ in 0..n {
                let repo_id = rng.below(repos.len() as u64) as u32;
                let path = format!("src/{}/f{}.rs", rng.pick(&SCOPES), rng.below(paths));
                let len = if rng.below(40) == 0 { 0 } else { 1 + rng.below(words) };
                let (m, c, a) = doc(&mut rng, repo_id, path, len);
                w.add_doc(&m, &c, &a).unwrap();
            }
            let path = w.finish(&repos).unwrap();
            let mut shard = Shard::open(&path).unwrap();
            let dead: Vec<u32> = (0..shard.doc_count() as u32).filter(|_| rng.below(5) == 0).collect();
            shard.delete_docs(&dead).unwrap();
        }
        ShardSet::open_dir(dir).unwrap()
    }

    /// A compound shard and the directory it lives in (dropped after it).
    struct Opened {
        shard: Shard,
        _dir: tempfile::TempDir,
    }

    /// Both merges of `set.shards()[lo..hi]`; asserts the files are equal.
    fn assert_same_compound(set: &ShardSet, lo: usize, hi: usize) -> Opened {
        let run: Vec<&Shard> = set.shards()[lo..hi].iter().collect();
        let a_dir = tempfile::tempdir().unwrap();
        let b_dir = tempfile::tempdir().unwrap();
        let a = write_compound_in_memory(&run, a_dir.path()).unwrap();
        let b = write_compound(&run, b_dir.path()).unwrap();
        assert_eq!(a.origin, b.origin, "doc order");
        let (fa, fb) = (fs::read(&a.path).unwrap(), fs::read(&b.path).unwrap());
        if fa != fb {
            let at = fa.iter().zip(&fb).position(|(x, y)| x != y).unwrap_or(fa.len().min(fb.len()));
            panic!("compound shards differ: {} vs {} bytes, first difference at {at}", fa.len(), fb.len());
        }
        // nothing else is left behind (the spill and temp files are gone)
        let left: Vec<_> = fs::read_dir(b_dir.path()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left.len(), 1, "{left:?}");
        let shard = Shard::open(&b.path).unwrap();
        assert_eq!(shard.doc_count() as usize, b.origin.len());
        Opened { shard, _dir: b_dir }
    }

    #[test]
    fn the_streamed_compound_is_the_file_the_in_memory_merge_wrote() {
        for seed in 1..=6u64 {
            let dir = tempfile::tempdir().unwrap();
            // seeds differ in shape: few large shards, many small ones
            let (shards, docs) = [(3, 150), (6, 40), (9, 12), (2, 400), (5, 70), (12, 6)][seed as usize - 1];
            let set = random_set(dir.path(), seed, shards, docs, 60, 120);
            assert_eq!(set.len(), shards);
            // the whole set, the oldest and the youngest half, a single shard
            assert_same_compound(&set, 0, shards);
            assert_same_compound(&set, shards / 2, shards);
            assert_same_compound(&set, 0, shards / 2 + 1);
            assert_same_compound(&set, shards - 1, shards);
        }
    }

    #[test]
    fn the_compound_is_the_same_with_a_dictionary_and_long_docs() {
        // over 512 docs kept (the dictionary samples the first 512) and
        // docs past the 64 KiB a sample takes
        let dir = tempfile::tempdir().unwrap();
        let mut rng = Lcg(99);
        for k in 0..3u32 {
            let mut w = ShardWriter::new(dir.path()).unwrap();
            w.set_created_unix(1_700_000_000 + k as u64);
            for i in 0..260u32 {
                let words = if i % 97 == 0 { 20_000 } else { 30 + rng.below(200) };
                let (m, c, a) = doc(&mut rng, 0, format!("src/{k}/f{i}.rs"), words);
                w.add_doc(&m, &c, &a).unwrap();
            }
            w.finish(&["alpha".to_string()]).unwrap();
        }
        let set = ShardSet::open_dir(dir.path()).unwrap();
        let shard = assert_same_compound(&set, 0, 3).shard;
        assert_eq!(shard.doc_count(), 780);
        assert!(shard.meta().zstd_dict.is_some());
        assert!((0..780).any(|d| shard.doc(d).unwrap().raw_len > 64 * 1024));
    }

    #[test]
    fn the_compound_is_the_same_when_nothing_or_little_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let mut rng = Lcg(5);
        let mut paths = Vec::new();
        for k in 0..3u64 {
            let mut w = ShardWriter::new(dir.path()).unwrap();
            w.set_created_unix(1_700_000_000 + k);
            for i in 0..4 {
                let (m, c, a) = doc(&mut rng, 0, format!("f{i}.rs"), 20);
                w.add_doc(&m, &c, &a).unwrap();
            }
            paths.push(w.finish(&["alpha".to_string()]).unwrap());
        }
        // an empty shard in the run
        let mut w = ShardWriter::new(dir.path()).unwrap();
        w.set_created_unix(1_700_000_000 + 3);
        w.finish(&[]).unwrap();
        let set = ShardSet::open_dir(dir.path()).unwrap();
        // the same four paths in every shard: only the newest version of each is kept
        assert_eq!(assert_same_compound(&set, 0, 4).shard.doc_count(), 4);
        drop(set);
        // every doc tombstoned: an empty compound
        for p in &paths {
            Shard::open(p).unwrap().delete_docs(&[0, 1, 2, 3]).unwrap();
        }
        let set = ShardSet::open_dir(dir.path()).unwrap();
        assert_eq!(assert_same_compound(&set, 0, 4).shard.doc_count(), 0);
    }

    /// The merged set answers as the unmerged one did: same live docs with
    /// the same contents, same docs per gram, symbol and callee.
    #[test]
    fn a_merged_set_answers_like_the_unmerged_one() {
        fn view(set: &ShardSet) -> Vec<String> {
            let mut out = Vec::new();
            let name = |si: usize, docid: u32| {
                let dm = set.doc(si, docid).unwrap();
                format!("{}:{}", set.shard(si).unwrap().meta().repos[dm.repo_id as usize], dm.path)
            };
            for (si, docid, dm) in set.visible_iter() {
                let content = set.content(si, docid).unwrap();
                out.push(format!("doc {} {:?} {} {}", name(si, docid), dm.lang, dm.raw_len, BlobId::from_content(&content)));
            }
            let mut keys: BTreeSet<Vec<u8>> = BTreeSet::new();
            for shard in set.shards() {
                keys.extend(shard.fst_entries(kind::NGRAM_FST).into_iter().map(|(k, _)| k));
            }
            for gram in &keys {
                let mut docs: Vec<String> = set.posting_doc_ids(gram).into_iter().map(|(si, d)| name(si, d)).collect();
                docs.sort();
                if !docs.is_empty() {
                    out.push(format!("gram {gram:?} {docs:?}"));
                }
            }
            for n in NAMES {
                let mut syms: Vec<String> =
                    set.symbol_postings(n).into_iter().map(|(si, d, k, line, scope)| format!("{} {k} {line} {scope}", name(si, d))).collect();
                syms.sort();
                out.push(format!("sym {n} {syms:?}"));
                let mut calls: Vec<String> =
                    set.call_postings(n).into_iter().map(|(si, d, caller, line)| format!("{} {caller} {line}", name(si, d))).collect();
                calls.sort();
                out.push(format!("call {n} {calls:?}"));
            }
            out.sort();
            out
        }
        for (seed, max_shards) in [(11u64, 1usize), (12, 3), (13, 5)] {
            let dir = tempfile::tempdir().unwrap();
            let mut set = random_set(dir.path(), seed, 8, 30, 50, 80);
            let before = view(&set);
            assert!(before.len() > 200);
            set.merge(dir.path(), max_shards).unwrap();
            assert!(set.len() <= max_shards);
            assert_eq!(view(&set), before, "seed {seed}");
            assert_eq!(view(&ShardSet::open_dir(dir.path()).unwrap()), before, "seed {seed}, reopened");
        }
    }
}
