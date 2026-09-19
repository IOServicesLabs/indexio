//! ShardSet: newest-first collection of shards with fan-out reads and
//! compaction (merge of the oldest shards into one compound shard).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use indexio_types::codec::PostingCursor;
use indexio_types::{DocMeta, ExtractedArtifact, SymbolKind, SymbolRec, CallRec};
use tracing::info;

use crate::format::kind;
use crate::shard::{decode_call_payload, decode_sym_payload};
use crate::{invalid, Shard, ShardWriter};

/// Extension merged-but-still-mapped shards are parked under (see `merge`).
const STALE_EXT: &str = "stale";

pub struct ShardSet {
    /// Newest-first (by created_unix in the shard header).
    shards: Vec<Shard>,
    /// Live-doc view, computed on first use and dropped by every mutation
    /// (`delete_docs`, `merge`). Every fan-out lookup used to rescan all
    /// shards to build this; on a 40-repo set that was the dominant cost
    /// of a query (one full scan per n-gram).
    visible: OnceLock<VisibleCache>,
}

/// Cached result of the newest-wins / tombstone-aware doc scan.
struct VisibleCache {
    docs: Vec<(usize, u32, DocMeta)>,
    ids: HashSet<(usize, u32)>,
}

impl ShardSet {
    pub fn open_dir(dir: &Path) -> io::Result<Self> {
        let mut shards = Vec::new();
        if dir.exists() {
            for entry in fs::read_dir(dir)? {
                let path = entry?.path();
                match path.extension().and_then(|e| e.to_str()) {
                    Some("cidx") => shards.push(Shard::open(&path)?),
                    // A merged shard another process still had mapped when
                    // compaction ran (Windows refuses to unlink mapped
                    // files): parked by `merge`, reaped whenever possible.
                    Some(STALE_EXT) => {
                        let _ = fs::remove_file(&path);
                    }
                    _ => {}
                }
            }
        }
        shards.sort_by(|a, b| {
            b.created_unix()
                .cmp(&a.created_unix())
                .then_with(|| b.path().cmp(a.path()))
        });
        Ok(ShardSet { shards, visible: OnceLock::new() })
    }

    fn cache(&self) -> &VisibleCache {
        self.visible.get_or_init(|| {
            let docs = self.scan_visible();
            let ids = docs.iter().map(|(si, d, _)| (*si, *d)).collect();
            VisibleCache { docs, ids }
        })
    }

    fn invalidate(&mut self) {
        self.visible = OnceLock::new();
    }

    pub fn len(&self) -> usize {
        self.shards.len()
    }
    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }
    pub fn shards(&self) -> &[Shard] {
        &self.shards
    }
    pub fn shard(&self, idx: usize) -> Option<&Shard> {
        self.shards.get(idx)
    }

    /// Live doc count across all shards (tombstoned excluded).
    pub fn doc_count(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.doc_count() - s.tombstones().len())
            .sum()
    }

    pub fn doc(&self, shard_idx: usize, docid: u32) -> Option<DocMeta> {
        self.shards.get(shard_idx)?.doc(docid)
    }

    pub fn content(&self, shard_idx: usize, docid: u32) -> io::Result<Vec<u8>> {
        self.shards
            .get(shard_idx)
            .ok_or_else(|| invalid(format!("shard index {shard_idx} out of range")))?
            .content(docid)
    }

    pub fn is_tombstoned(&self, shard_idx: usize, docid: u32) -> bool {
        self.shards
            .get(shard_idx)
            .map_or(false, |s| s.tombstones().contains(docid))
    }

    pub fn delete_docs(&mut self, shard_idx: usize, docids: &[u32]) -> io::Result<()> {
        self.invalidate();
        self.shards
            .get_mut(shard_idx)
            .ok_or_else(|| invalid(format!("shard index {shard_idx} out of range")))?
            .delete_docs(docids)
    }

    /// Repo name for a doc (repo_id is shard-local).
    fn repo_name(shard: &Shard, repo_id: u32) -> String {
        shard
            .meta()
            .repos
            .get(repo_id as usize)
            .cloned()
            .unwrap_or_default()
    }

    /// All live docs, newest shard first (an owned copy of the cached
    /// view; prefer [`visible_slice`](Self::visible_slice) on hot paths).
    pub fn visible_docs(&self) -> Vec<(usize, u32, DocMeta)> {
        self.cache().docs.clone()
    }

    /// Borrowed live-doc view: newest shard first, newest-wins on
    /// (repo, path), tombstoned docs excluded.
    pub fn visible_slice(&self) -> &[(usize, u32, DocMeta)] {
        &self.cache().docs
    }

    /// Borrowed (shard_idx, docid) set of [`visible_slice`](Self::visible_slice).
    pub fn visible_set(&self) -> &HashSet<(usize, u32)> {
        &self.cache().ids
    }

    /// Duplicate suppression: same (repo, path) — the newest shard wins;
    /// tombstoned docs are skipped.
    fn scan_visible(&self) -> Vec<(usize, u32, DocMeta)> {
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let mut out = Vec::new();
        for (si, shard) in self.shards.iter().enumerate() {
            for docid in 0..shard.doc_count() as u32 {
                if shard.tombstones().contains(docid) {
                    continue;
                }
                let Some(dm) = shard.doc(docid) else { continue };
                let repo = Self::repo_name(shard, dm.repo_id);
                if seen.insert((repo, dm.path.clone())) {
                    out.push((si, docid, dm));
                }
            }
        }
        out
    }

    /// Fan-out gram postings: one cursor per shard that contains the gram.
    /// Cursors are raw — filter with `is_tombstoned` when consuming.
    pub fn postings(&self, gram: &[u8]) -> Vec<(usize, PostingCursor<'_>)> {
        self.shards
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.postings(gram).map(|c| (i, c)))
            .collect()
    }

    /// Live (docid, positions) pairs for a gram across all shards,
    /// tombstoned and duplicate-suppressed docs excluded.
    pub fn posting_docs(&self, gram: &[u8]) -> Vec<(usize, u32, Vec<u32>)> {
        let visible = self.visible_set();
        let mut out = Vec::new();
        for (si, mut cursor) in self.postings(gram) {
            while let Some((docid, positions)) = cursor.next_entry() {
                if visible.contains(&(si, docid)) {
                    out.push((si, docid, positions));
                }
            }
        }
        out
    }

    /// Summed posting payload bytes for `gram` over all shards (0 when no
    /// shard has it): a decode-free rarity estimate.
    pub fn posting_bytes(&self, gram: &[u8]) -> usize {
        self.shards.iter().map(|s| s.posting_bytes(gram)).sum()
    }

    /// Live doc refs for a gram across all shards, positions skipped (no
    /// per-posting allocation): what the query planner's candidate
    /// intersection needs. Sorted by (shard_idx, docid).
    pub fn posting_doc_ids(&self, gram: &[u8]) -> Vec<(usize, u32)> {
        let visible = self.visible_set();
        let mut out = Vec::new();
        for (si, mut cursor) in self.postings(gram) {
            while let Some(docid) = cursor.next_docid() {
                if visible.contains(&(si, docid)) {
                    out.push((si, docid));
                }
            }
        }
        out
    }

    /// The (shard_idx, docid) set `visible_docs` yields, without the metas
    /// (an owned copy; prefer [`visible_set`](Self::visible_set)).
    pub fn visible_ids(&self) -> HashSet<(usize, u32)> {
        self.cache().ids.clone()
    }

    /// Fan-out: (shard_idx, docid, kind, line, scope). Tombstoned and
    /// duplicate-suppressed docs excluded.
    pub fn symbol_postings(&self, name: &str) -> Vec<(usize, u32, u8, u32, String)> {
        self.symbol_postings_visible(name, self.visible_set())
    }

    /// `symbol_postings` against a precomputed visible set.
    pub fn symbol_postings_visible(
        &self,
        name: &str,
        visible: &HashSet<(usize, u32)>,
    ) -> Vec<(usize, u32, u8, u32, String)> {
        let mut out = Vec::new();
        for (si, shard) in self.shards.iter().enumerate() {
            for (docid, k, line, scope) in shard.symbol_postings(name) {
                if visible.contains(&(si, docid)) {
                    out.push((si, docid, k, line, scope));
                }
            }
        }
        out
    }

    /// Fan-out: (shard_idx, docid, caller, line).
    pub fn call_postings(&self, callee: &str) -> Vec<(usize, u32, String, u32)> {
        self.call_postings_visible(callee, self.visible_set())
    }

    /// `call_postings` against a precomputed visible set.
    pub fn call_postings_visible(
        &self,
        callee: &str,
        visible: &HashSet<(usize, u32)>,
    ) -> Vec<(usize, u32, String, u32)> {
        let mut out = Vec::new();
        for (si, shard) in self.shards.iter().enumerate() {
            for (docid, caller, line) in shard.call_postings(callee) {
                if visible.contains(&(si, docid)) {
                    out.push((si, docid, caller, line));
                }
            }
        }
        out
    }

    /// If shard_count > max_shards, merge the OLDEST shards (all but the
    /// newest `max_shards - 1`) into one compound shard. Applies tombstones
    /// (deleted docs are dropped) and newest-wins dedup within the merged
    /// group. Atomic swap: the compound shard is fully written (its own
    /// atomic rename) before the merged files are removed; the set is then
    /// updated in place.
    pub fn merge(&mut self, out_dir: &Path, max_shards: usize) -> io::Result<()> {
        if max_shards == 0 || self.shards.len() <= max_shards {
            return Ok(());
        }
        self.invalidate();
        let keep_n = max_shards - 1;
        let merged: Vec<usize> = (keep_n..self.shards.len()).collect();
        info!(merged = merged.len(), kept = keep_n, "merging shards");

        // Union repo string tables (oldest shard first), remap repo_ids.
        let mut repos: Vec<String> = Vec::new();
        let mut repo_ids: HashMap<String, u32> = HashMap::new();
        let mut repo_map: HashMap<(usize, u32), u32> = HashMap::new();
        for &si in merged.iter().rev() {
            for (rid, name) in self.shards[si].meta().repos.iter().enumerate() {
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
        for &si in merged.iter() {
            // merged indices are newest-first (ShardSet order)
            let shard = &self.shards[si];
            let mut kv = vec![false; shard.doc_count() as usize];
            for docid in 0..shard.doc_count() as u32 {
                if shard.tombstones().contains(docid) {
                    continue;
                }
                let Some(dm) = shard.doc(docid) else { continue };
                let repo = Self::repo_name(shard, dm.repo_id);
                if seen.insert((repo, dm.path)) {
                    kv[docid as usize] = true;
                }
            }
            keep.insert(si, kv);
        }

        let mut writer = ShardWriter::new(out_dir)?;
        // The compound shard must order as old as its newest input so that
        // kept (newer) shards still win duplicate suppression.
        let created = merged
            .iter()
            .map(|&si| self.shards[si].created_unix())
            .max()
            .unwrap_or(0);
        writer.set_created_unix(created);

        // Oldest shard first for a stable doc order in the compound shard.
        // `origin[new docid] = (shard idx, old docid)` lets tombstones that
        // land on an input while we merge be forwarded afterwards.
        let mut origin: Vec<(usize, u32)> = Vec::new();
        for &si in merged.iter().rev() {
            let shard = &self.shards[si];
            let kv = &keep[&si];
            let kept_ids: Vec<u32> = (0..shard.doc_count() as u32)
                .filter(|d| kv[*d as usize])
                .collect();
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

            // Invert NGRAM postings for kept docs (docids only; positions
            // an older shard may still carry are dropped here, so a
            // compaction also shrinks pre-P10 shards).
            for (gram, off) in shard.fst_entries(kind::NGRAM_FST) {
                let Some(payload) = shard.posting_payload(kind::NGRAM_POST, off) else {
                    continue;
                };
                let mut cursor = PostingCursor::new(payload);
                while let Some(docid) = cursor.next_docid() {
                    if let Some(&local) = local_of.get(&docid) {
                        arts[local].ngrams.push(gram.clone());
                    }
                }
            }
            // Symbol postings.
            for (name, off) in shard.fst_entries(kind::SYM_FST) {
                let Some(payload) = shard.posting_payload(kind::SYM_POST, off) else {
                    continue;
                };
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
            // Call postings.
            for (callee, off) in shard.fst_entries(kind::CALL_FST) {
                let Some(payload) = shard.posting_payload(kind::CALL_POST, off) else {
                    continue;
                };
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
                let mut dm = shard
                    .doc(docid)
                    .ok_or_else(|| invalid("merge: doc record missing"))?;
                dm.repo_id = *repo_map
                    .get(&(si, dm.repo_id))
                    .ok_or_else(|| invalid("merge: repo_id not mapped"))?;
                let content = shard.content(docid)?; // raw bytes, recompressed by writer
                let mut art = std::mem::take(&mut arts[local]);
                art.lang = dm.lang;
                art.raw_len = dm.raw_len;
                writer.add_doc(&dm, &content, &art)?;
            }
        }

        // finish() does its own atomic rename; only then remove old files.
        let compound_path: PathBuf = writer.finish(&repos)?;
        // Docs tombstoned in an input by another process while we were
        // merging would otherwise come back to life in the compound.
        let late: Vec<u32> = origin
            .iter()
            .enumerate()
            .filter(|(_, &(si, d))| {
                let now = self.shards[si].tombstones_on_disk();
                now.contains(d) && !self.shards[si].tombstones().contains(d)
            })
            .map(|(new_id, _)| new_id as u32)
            .collect();
        if !late.is_empty() {
            let mut compound = Shard::open(&compound_path)?;
            compound.delete_docs(&late)?;
            info!(forwarded = late.len(), "tombstones applied to the compound shard after the merge");
        }
        let merged_paths: Vec<PathBuf> = merged
            .iter()
            .map(|&si| self.shards[si].path().to_path_buf())
            .collect();
        // Drop merged shards (munmap) before unlinking their files. A file
        // some other process still has mapped cannot be unlinked on
        // Windows; park it under STALE_EXT (renaming a mapped file is
        // allowed), where `open_dir` ignores it and reaps it later.
        let kept: Vec<Shard> = self.shards.drain(..keep_n).collect();
        self.shards.clear();
        for p in &merged_paths {
            if let Err(e) = fs::remove_file(p) {
                let parked = p.with_extension(STALE_EXT);
                fs::rename(p, &parked).map_err(|e2| {
                    io::Error::new(
                        e.kind(),
                        format!("cannot remove merged shard {} ({e}) nor park it ({e2})", p.display()),
                    )
                })?;
            }
        }
        let mut shards = kept;
        shards.push(Shard::open(&compound_path)?);
        shards.sort_by(|a, b| {
            b.created_unix()
                .cmp(&a.created_unix())
                .then_with(|| b.path().cmp(a.path()))
        });
        self.shards = shards;
        Ok(())
    }
}
