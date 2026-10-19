//! ShardSet: newest-first collection of shards with fan-out reads and
//! compaction (merge of the oldest shards into one compound shard).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use indexio_types::codec::PostingCursor;
use indexio_types::{DocMeta, ExtractedArtifact, SymbolKind, SymbolRec, CallRec};
use tracing::info;

use crate::format::kind;
use crate::shard::{decode_call_payload, decode_sym_payload};
use crate::{invalid, Shard, ShardWriter};

/// Extension merged-but-still-mapped shards are parked under (see `merge`).
const STALE_EXT: &str = "stale";

/// A temp file untouched for an hour: its writer is gone.
pub fn is_abandoned(path: &Path) -> bool {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age.as_secs() > 3600)
}

pub struct ShardSet {
    /// Newest-first (by created_unix in the shard header); in a layered set
    /// ([`open_layers`](Self::open_layers)) newest-first within each layer,
    /// top layer first. Over a shared `base` these are the top layer only.
    shards: Vec<Shard>,
    /// A shared lower layer ([`open_over`](Self::open_over)): a team server
    /// keeps one base set and puts every user's overlay on top of it, so the
    /// base's mappings, live view and key map are built once for everyone.
    /// Its shards follow this set's own in the shard index space.
    base: Option<Arc<ShardSet>>,
    /// Live-doc view, computed on first use and dropped by every mutation
    /// (`delete_docs`, `merge`). Every fan-out lookup used to rescan all
    /// shards to build this; on a 40-repo set that was the dominant cost
    /// of a query (one full scan per n-gram). Over a `base`: the top layer's.
    visible: OnceLock<VisibleCache>,
    /// (repo, path) -> live doc of `visible`, built on first lookup.
    keys: OnceLock<Keys>,
    /// What the top layer takes out of the base (built on first use).
    over: OnceLock<Over>,
    /// The whole view materialized over a `base` (only for callers that
    /// need it as one slice or set; the hot paths stream instead).
    full: OnceLock<VisibleCache>,
    /// What this view hides on top of tombstones (team overlays, ACLs).
    view: Visibility,
    /// Opened from several directories: never compacted as one.
    layered: bool,
}

/// Docs a view hides beyond tombstones and newest-wins (team worktree
/// overlays): a `hidden` (repo, path) is a file the overlay deleted, so the
/// base doc under it must not surface; `allow` limits the view to the repos
/// a token may see. Every lookup goes through the visible set, so both
/// apply to search, symbols, calls, spans and the semantic legs alike.
#[derive(Clone, Default)]
pub struct Visibility {
    pub hidden: HashSet<(String, String)>,
    pub allow: Option<Arc<dyn Fn(&str) -> bool + Send + Sync>>,
}

/// Cached result of the newest-wins / tombstone-aware doc scan.
struct VisibleCache {
    docs: Vec<(usize, u32, DocMeta)>,
    ids: HashSet<(usize, u32)>,
    /// Summed `raw_len` of `docs`.
    raw: u64,
}

impl VisibleCache {
    fn new(docs: Vec<(usize, u32, DocMeta)>) -> Self {
        let ids = docs.iter().map(|(si, d, _)| (*si, *d)).collect();
        let raw = docs.iter().map(|(_, _, dm)| dm.raw_len as u64).sum();
        VisibleCache { docs, ids, raw }
    }
}

/// Live docs by repo, then path, with per-repo totals.
#[derive(Default)]
struct Keys {
    repos: HashMap<String, RepoKeys>,
}

#[derive(Default)]
struct RepoKeys {
    docs: HashMap<String, (usize, u32)>,
    raw: u64,
}

/// What a top layer takes out of its shared base.
struct Over {
    /// repo -> paths the top layer shadows (its own live docs) or hides.
    shadow: HashMap<String, HashSet<String>>,
    /// base shard -> shard-local repo id -> allowed by the view.
    allowed: Vec<Vec<bool>>,
    /// Neither applies: every live base doc is live here.
    trivial: bool,
    /// Live doc count and raw bytes of the whole view.
    totals: (usize, u64),
}

impl ShardSet {
    fn with_shards(shards: Vec<Shard>, base: Option<Arc<ShardSet>>, view: Visibility, layered: bool) -> Self {
        ShardSet {
            shards,
            base,
            visible: OnceLock::new(),
            keys: OnceLock::new(),
            over: OnceLock::new(),
            full: OnceLock::new(),
            view,
            layered,
        }
    }

    pub fn open_dir(dir: &Path) -> io::Result<Self> {
        Ok(Self::with_shards(Self::load_dir(dir)?, None, Visibility::default(), false))
    }

    /// One view over several shard directories, `dirs[0]` on top: a doc of
    /// an upper layer shadows a doc with the same (repo, path) below it, and
    /// `view` hides what the upper layers deleted. The lower layers are only
    /// read (their tombstones honoured, never written), so a shared base
    /// can sit under any number of per-user layers.
    pub fn open_layers(dirs: &[PathBuf], view: Visibility) -> io::Result<Self> {
        let mut shards = Vec::new();
        for d in dirs {
            shards.extend(Self::load_dir(d)?);
        }
        Ok(Self::with_shards(shards, None, view, true))
    }

    /// [`open_layers`](Self::open_layers) with an already opened base as the
    /// bottom layer: the same view, but only the `upper` shards are mapped
    /// and scanned; the base's live view and key map are shared, not copied.
    /// The base must not itself be layered over another set.
    pub fn open_over(upper: &[PathBuf], base: Arc<ShardSet>, view: Visibility) -> io::Result<Self> {
        if base.base.is_some() {
            return Err(invalid("a shared base cannot itself sit over another set"));
        }
        let mut shards = Vec::new();
        for d in upper {
            shards.extend(Self::load_dir(d)?);
        }
        Ok(Self::with_shards(shards, Some(base), view, true))
    }

    /// Every shard with its index: the top layer, then a shared base's.
    fn each_shard(&self) -> impl Iterator<Item = (usize, &Shard)> {
        let n = self.shards.len();
        self.shards
            .iter()
            .enumerate()
            .chain(self.base.iter().flat_map(move |b| b.shards.iter().enumerate().map(move |(i, s)| (i + n, s))))
    }

    /// Every shard, top layer first (a shared base's included).
    pub fn all_shards(&self) -> impl Iterator<Item = &Shard> {
        self.each_shard().map(|(_, s)| s)
    }

    /// The shards of one directory, newest first.
    fn load_dir(dir: &Path) -> io::Result<Vec<Shard>> {
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
                    // a writer that died between create and rename (seen:
                    // seven empty `.cidx.tmp` files weeks old); a live write
                    // finishes in seconds
                    Some("tmp") if is_abandoned(&path) => {
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
        Ok(shards)
    }

    /// The live docs of this set's own shards (the top layer over a base).
    fn cache(&self) -> &VisibleCache {
        self.visible.get_or_init(|| VisibleCache::new(self.scan_visible()))
    }

    fn keys(&self) -> &Keys {
        self.keys.get_or_init(|| {
            let mut k = Keys::default();
            for (si, docid, dm) in &self.cache().docs {
                let Some(r) = self.shards[*si].meta().repos.get(dm.repo_id as usize) else { continue };
                let repo = match k.repos.get_mut(r.as_str()) {
                    Some(p) => p,
                    None => k.repos.entry(r.clone()).or_default(),
                };
                repo.docs.insert(dm.path.clone(), (*si, *docid));
                repo.raw += dm.raw_len as u64;
            }
            k
        })
    }

    fn over(&self) -> &Over {
        self.over.get_or_init(|| {
            let base = self.base.as_ref().expect("only a set over a base has an overlay");
            let mut shadow: HashMap<String, HashSet<String>> = HashMap::new();
            for (repo, k) in &self.keys().repos {
                shadow.entry(repo.clone()).or_default().extend(k.docs.keys().cloned());
            }
            for (r, p) in &self.view.hidden {
                shadow.entry(r.clone()).or_default().insert(p.clone());
            }
            let allowed: Vec<Vec<bool>> = base
                .shards
                .iter()
                .map(|s| s.meta().repos.iter().map(|r| self.repo_allowed(r)).collect())
                .collect();
            let trivial = shadow.is_empty() && allowed.iter().all(|a| a.iter().all(|x| *x));
            // base totals, less what the view takes out of them
            let own = self.cache();
            let (mut n, mut raw) = (own.docs.len(), own.raw);
            let base_keys = base.keys();
            for (repo, k) in &base_keys.repos {
                if !self.repo_allowed(repo) {
                    continue;
                }
                n += k.docs.len();
                raw += k.raw;
                for p in shadow.get(repo).into_iter().flatten() {
                    if let Some(&(si, docid)) = k.docs.get(p) {
                        n -= 1;
                        raw -= base.doc(si, docid).map_or(0, |dm| dm.raw_len as u64);
                    }
                }
            }
            Over { shadow, allowed, trivial, totals: (n, raw) }
        })
    }

    fn repo_allowed(&self, repo: &str) -> bool {
        self.view.allow.as_ref().is_none_or(|f| f(repo))
    }

    /// Whether the live base doc at (`bsi`, repo id `rid`, `path`) survives
    /// the top layer (allowed, neither shadowed nor hidden).
    fn base_passes(&self, over: &Over, bsi: usize, rid: u32, path: &[u8]) -> bool {
        if over.trivial {
            return true;
        }
        if !over.allowed.get(bsi).and_then(|a| a.get(rid as usize)).copied().unwrap_or(false) {
            return false;
        }
        if over.shadow.is_empty() {
            return true;
        }
        let Some(repo) = self.base.as_ref().and_then(|b| b.shards.get(bsi)).and_then(|s| s.meta().repos.get(rid as usize)) else {
            return false;
        };
        match (over.shadow.get(repo.as_str()), std::str::from_utf8(path)) {
            (Some(paths), Ok(p)) => !paths.contains(p),
            _ => true,
        }
    }

    /// Whether (`shard_idx`, `docid`) is a live doc of this view: not
    /// tombstoned, newest of its (repo, path), allowed and not hidden.
    pub fn is_live(&self, shard_idx: usize, docid: u32) -> bool {
        let n = self.shards.len();
        match &self.base {
            None => self.cache().ids.contains(&(shard_idx, docid)),
            Some(_) if shard_idx < n => self.cache().ids.contains(&(shard_idx, docid)),
            Some(b) => {
                let bsi = shard_idx - n;
                if !b.is_live(bsi, docid) {
                    return false;
                }
                let over = self.over();
                if over.trivial {
                    return true;
                }
                match b.shards.get(bsi).and_then(|s| s.doc_key(docid)) {
                    Some((rid, path)) => self.base_passes(over, bsi, rid, path),
                    None => false,
                }
            }
        }
    }

    /// The live doc at (repo, path), through a map built on first use.
    pub fn locate(&self, repo: &str, path: &str) -> Option<(usize, u32)> {
        if let Some(r) = self.keys().repos.get(repo).and_then(|k| k.docs.get(path)) {
            return Some(*r);
        }
        let b = self.base.as_ref()?;
        let over = self.over();
        if !over.trivial
            && (!self.repo_allowed(repo) || over.shadow.get(repo).is_some_and(|p| p.contains(path)))
        {
            return None;
        }
        b.locate(repo, path).map(|(si, d)| (si + self.shards.len(), d))
    }

    /// Build the lazily computed views now (an engine prepared off the
    /// call path is warmed before use).
    pub fn warm(&self) {
        let _ = self.keys();
        if let Some(b) = &self.base {
            b.warm();
            let _ = self.over();
        }
    }

    /// Live docs of the whole view, borrowed, in [`visible_slice`]
    /// order (top layer first), without materializing a view over a base.
    ///
    /// [`visible_slice`]: Self::visible_slice
    pub fn visible_iter(&self) -> Box<dyn Iterator<Item = (usize, u32, &DocMeta)> + '_> {
        let own = self.cache().docs.iter().map(|(si, d, dm)| (*si, *d, dm));
        let Some(b) = &self.base else {
            return Box::new(own);
        };
        let n = self.shards.len();
        let over = self.over();
        Box::new(own.chain(b.visible_slice().iter().filter_map(move |(bsi, d, dm)| {
            self.base_passes(over, *bsi, dm.repo_id, dm.path.as_bytes()).then_some((bsi + n, *d, dm))
        })))
    }

    /// (live doc count, summed raw bytes) of the whole view.
    pub fn visible_totals(&self) -> (usize, u64) {
        match &self.base {
            None => {
                let c = self.cache();
                (c.docs.len(), c.raw)
            }
            Some(_) => self.over().totals,
        }
    }

    /// The whole view of a set over a base, materialized once.
    fn full(&self) -> &VisibleCache {
        match &self.base {
            None => self.cache(),
            Some(_) => self
                .full
                .get_or_init(|| VisibleCache::new(self.visible_iter().map(|(si, d, dm)| (si, d, dm.clone())).collect())),
        }
    }

    fn invalidate(&mut self) {
        self.visible = OnceLock::new();
        self.keys = OnceLock::new();
        self.over = OnceLock::new();
        self.full = OnceLock::new();
    }

    pub fn len(&self) -> usize {
        self.shards.len() + self.base.as_ref().map_or(0, |b| b.len())
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// This set's own shards: over a shared base, the top layer only
    /// ([`all_shards`](Self::all_shards) covers both).
    pub fn shards(&self) -> &[Shard] {
        &self.shards
    }
    pub fn shard(&self, idx: usize) -> Option<&Shard> {
        match self.shards.get(idx) {
            Some(s) => Some(s),
            None => self.base.as_ref()?.shard(idx - self.shards.len()),
        }
    }

    /// Live doc count across all shards (tombstoned excluded).
    pub fn doc_count(&self) -> u64 {
        self.all_shards().map(|s| s.doc_count() - s.tombstones().len()).sum()
    }

    pub fn doc(&self, shard_idx: usize, docid: u32) -> Option<DocMeta> {
        self.shard(shard_idx)?.doc(docid)
    }

    pub fn content(&self, shard_idx: usize, docid: u32) -> io::Result<Vec<u8>> {
        self.shard(shard_idx)
            .ok_or_else(|| invalid(format!("shard index {shard_idx} out of range")))?
            .content(docid)
    }

    pub fn is_tombstoned(&self, shard_idx: usize, docid: u32) -> bool {
        self.shard(shard_idx).is_some_and(|s| s.tombstones().contains(docid))
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
        self.visible_iter().map(|(si, d, dm)| (si, d, dm.clone())).collect()
    }

    /// The live docs of `repo` at `paths` only (what a delta over a few
    /// changed files compares against): the same newest-wins/tombstone
    /// rules as [`visible_docs`](Self::visible_docs) without building the
    /// whole view, and only the matches are decoded.
    pub fn visible_at(&self, repo: &str, paths: &HashSet<String>) -> Vec<(usize, u32, DocMeta)> {
        if self.base.is_some() {
            return self
                .visible_iter()
                .filter(|(si, _, dm)| paths.contains(&dm.path) && self.shard(*si).is_some_and(|s| Self::repo_name(s, dm.repo_id) == repo))
                .map(|(si, d, dm)| (si, d, dm.clone()))
                .collect();
        }
        let wanted: HashSet<&[u8]> = paths.iter().map(|p| p.as_bytes()).collect();
        let mut seen: HashSet<&[u8]> = HashSet::new();
        let mut out = Vec::new();
        for (si, shard) in self.shards.iter().enumerate() {
            let Some(rid) = shard.meta().repos.iter().position(|r| r == repo) else { continue };
            if self.view.allow.as_ref().is_some_and(|f| !f(repo)) {
                continue;
            }
            let tomb = shard.tombstones();
            for docid in 0..shard.doc_count() as u32 {
                if tomb.contains(docid) {
                    continue;
                }
                let Some((r, path)) = shard.doc_key(docid) else { continue };
                if r as usize != rid || !wanted.contains(path) || !seen.insert(path) {
                    continue;
                }
                if self.view.hidden.contains(&(repo.to_string(), String::from_utf8_lossy(path).into_owned())) {
                    continue;
                }
                if let Some(dm) = shard.doc(docid) {
                    out.push((si, docid, dm));
                }
            }
        }
        out
    }

    /// Borrowed live-doc view: newest shard first, newest-wins on
    /// (repo, path), tombstoned docs excluded.
    pub fn visible_slice(&self) -> &[(usize, u32, DocMeta)] {
        &self.full().docs
    }

    /// Borrowed (shard_idx, docid) set of [`visible_slice`](Self::visible_slice).
    pub fn visible_set(&self) -> &HashSet<(usize, u32)> {
        &self.full().ids
    }

    /// Duplicate suppression: same (repo, path) — the newest shard wins;
    /// tombstoned docs are skipped.
    fn scan_visible(&self) -> Vec<(usize, u32, DocMeta)> {
        // Dedupe on (set-wide repo id, path bytes borrowed from the maps):
        // superseded versions are skipped without decoding or allocating,
        // and only the docs that are returned are decoded (the old scan
        // built two Strings per doc of every shard).
        let mut repo_ids: HashMap<&str, u32> = HashMap::new();
        let mut hidden: HashSet<(u32, &[u8])> = HashSet::with_capacity(self.view.hidden.len());
        for (r, p) in &self.view.hidden {
            let n = repo_ids.len() as u32;
            let id = *repo_ids.entry(r.as_str()).or_insert(n);
            hidden.insert((id, p.as_bytes()));
        }
        let mut seen: HashSet<(u32, &[u8])> = HashSet::new();
        let mut out = Vec::new();
        for (si, shard) in self.shards.iter().enumerate() {
            // shard-local repo id -> (set-wide id, allowed)
            let local: Vec<(u32, bool)> = shard
                .meta()
                .repos
                .iter()
                .map(|r| {
                    let n = repo_ids.len() as u32;
                    let id = *repo_ids.entry(r.as_str()).or_insert(n);
                    (id, self.view.allow.as_ref().is_none_or(|f| f(r)))
                })
                .collect();
            let tomb = shard.tombstones();
            for docid in 0..shard.doc_count() as u32 {
                if tomb.contains(docid) {
                    continue;
                }
                let Some((rid, path)) = shard.doc_key(docid) else { continue };
                let Some(&(id, allowed)) = local.get(rid as usize) else { continue };
                if !allowed || hidden.contains(&(id, path)) || !seen.insert((id, path)) {
                    continue;
                }
                if let Some(dm) = shard.doc(docid) {
                    out.push((si, docid, dm));
                }
            }
        }
        out
    }

    /// Fan-out gram postings: one cursor per shard that contains the gram.
    /// Cursors are raw — filter with `is_tombstoned` when consuming.
    pub fn postings(&self, gram: &[u8]) -> Vec<(usize, PostingCursor<'_>)> {
        self.each_shard().filter_map(|(i, s)| s.postings(gram).map(|c| (i, c))).collect()
    }

    /// Live (docid, positions) pairs for a gram across all shards,
    /// tombstoned and duplicate-suppressed docs excluded.
    pub fn posting_docs(&self, gram: &[u8]) -> Vec<(usize, u32, Vec<u32>)> {
        let mut out = Vec::new();
        for (si, mut cursor) in self.postings(gram) {
            while let Some((docid, positions)) = cursor.next_entry() {
                if self.is_live(si, docid) {
                    out.push((si, docid, positions));
                }
            }
        }
        out
    }

    /// Summed posting payload bytes for `gram` over all shards (0 when no
    /// shard has it): a decode-free rarity estimate.
    pub fn posting_bytes(&self, gram: &[u8]) -> usize {
        self.all_shards().map(|s| s.posting_bytes(gram)).sum()
    }

    /// Live doc refs for a gram across all shards, positions skipped (no
    /// per-posting allocation): what the query planner's candidate
    /// intersection needs. Sorted by (shard_idx, docid).
    pub fn posting_doc_ids(&self, gram: &[u8]) -> Vec<(usize, u32)> {
        let mut out = Vec::new();
        for (si, mut cursor) in self.postings(gram) {
            while let Some(docid) = cursor.next_docid() {
                if self.is_live(si, docid) {
                    out.push((si, docid));
                }
            }
        }
        out
    }

    /// The (shard_idx, docid) set `visible_docs` yields, without the metas
    /// (an owned copy; prefer [`visible_set`](Self::visible_set)).
    pub fn visible_ids(&self) -> HashSet<(usize, u32)> {
        self.full().ids.clone()
    }

    /// Fan-out: (shard_idx, docid, kind, line, scope). Tombstoned and
    /// duplicate-suppressed docs excluded.
    pub fn symbol_postings(&self, name: &str) -> Vec<(usize, u32, u8, u32, String)> {
        let mut out = Vec::new();
        for (si, shard) in self.each_shard() {
            for (docid, k, line, scope) in shard.symbol_postings(name) {
                if self.is_live(si, docid) {
                    out.push((si, docid, k, line, scope));
                }
            }
        }
        out
    }

    /// Fan-out: (shard_idx, docid, caller, line), live docs only.
    pub fn call_postings(&self, callee: &str) -> Vec<(usize, u32, String, u32)> {
        let mut out = Vec::new();
        for (si, shard) in self.each_shard() {
            for (docid, caller, line) in shard.call_postings(callee) {
                if self.is_live(si, docid) {
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
        if self.layered {
            return Err(invalid("a layered shard set is compacted one layer at a time"));
        }
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
