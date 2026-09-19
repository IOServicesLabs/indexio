//! Random-indexing semantic model v2 (SPEC-P4 §2, upgraded per SPEC-P5 B1):
//! a self-contained, CPU-only embedder that learns corpus-native semantics
//! in-process. No model downloads, no GPU, no external services.
//!
//! Technique (Kanerva et al. 2000; Sahlgren's S-Space; Random Permutation
//! Model of Sahlgren et al. 2008): each token type gets a fixed sparse
//! ternary "label" vector derived deterministically from blake3; each token
//! also accumulates a "context" vector = weighted sum of the
//! (direction-sensitively *permuted*) label vectors of tokens it co-occurs
//! with. Chunk/query embedding = idf-weighted sum of its tokens' context
//! vectors plus a small label component, L2-normalized.
//!
//! SPEC-P5 B1 changes over v1:
//! - dim 1024 -> **2048**, 6 -> **8 nonzeros** per label (Sahlgren consensus).
//! - **True coordinate permutations** π/π' (xorshift64 Fisher-Yates seeded
//!   from blake3(b"ri-perm-v2")) replace the ±1 dim rotation: left context
//!   applies π, right context applies π' to the neighbor label dims.
//! - **Hub cut at embed time**: tokens with df > max(100, 40% of
//!   n_texts_seen) contribute neither label nor context.
//! - **Header field weighting ×2.5**: chunk text is `header + "\n" + body`;
//!   tokens before the first newline are weighted 2.5 in the pooling sum.
//! - model_id "rindex-v2"; `open()` loads only `sem/rindex-v2.rimodel` (v1
//!   files are ignored). Migration from v1: `indexio embed --rebuild-model`.
//! - SPEC-P5 B2: `nearest_tokens` / `expand_query` thesaurus expansion.
//!
//! Documented design choices (SPEC-P4 §2 invites documenting them):
//! - **min-df 2** is enforced as the *eviction priority* at the vocabulary
//!   cap (lowest-df types are dropped first, so singleton contexts are the
//!   first to go) rather than as an online gate on accumulation. Rationale:
//!   rare, corpus-specific identifiers are precisely what a code index must
//!   model, and gating accumulation at df >= 2 would leave df-1 bridge
//!   tokens (e.g. a term appearing in exactly one chunk) with no context at
//!   all, which defeats the cross-chunk semantic transfer the model exists
//!   for. Singletons with an empty context still embed via their own label
//!   (graceful cold start), exactly as the spec requires.
//! - **Vocab cap**: hard cap of 250_000 types; on overflow the lowest-df
//!   types (ties broken by term, deterministically) are evicted whole.
//! - **Unsplit identifier emission**: when sub-splitting an identifier
//!   produces exactly one part (no case/digit boundary), the token is
//!   emitted once at weight 1.0 (unsplit == its only part, so emitting
//!   "both" would just duplicate it and create degenerate self-bigrams).
//!   When a split does occur, the lowercase unsplit form is emitted at 0.5
//!   weight alongside its parts at 1.0, per spec.
//! - Bigram terms sit at the stream position of their first token, so
//!   unigram distances are measured in token steps (d = 1..=3); a unigram
//!   and its own outgoing bigram (same position, d = 0) do not accumulate
//!   into each other.
//! - **Context vectors are unit-normalized in `embed` before idf-weighting**
//!   (Sahlgren's normalization). Raw context magnitudes grow with token
//!   frequency, so without this, high-df tokens dominate every embedding
//!   they touch despite their low idf (frequency domination).
//!
//! Model drift note: `observe()` only ever *adds* contributions, so chunks
//! deleted since the model was built keep contributing until the model is
//! rebuilt (`indexio embed --rebuild-model`).

use std::collections::{HashMap, HashSet};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

use anyhow::Context;
use rayon::prelude::*;

use crate::embed::{l2_normalize, Embedder};
use crate::rimodel::{self, Comp, Snapshot};

/// Vector dimension (SPEC-P5 B1).
pub const DIM: usize = 2048;
/// Nonzero components per sparse ternary label vector (SPEC-P5 B1).
const LABEL_NONZEROS: usize = 8;
/// Context window radius: distances 1..=3.
const CTX_RADIUS: i64 = 3;
/// Context accumulators are truncated to the top-N components by |v| after
/// each observed text.
const CTX_TOP: usize = 128;
/// Weight of the label component in `embed`.
const LABEL_WEIGHT: f32 = 0.25;
/// Header field weight (SPEC-P5 B1): tokens before the first '\n' of a
/// chunk text count 2.5x in the pooling sum (BM25F/SIF evidence).
const HEADER_WEIGHT: f32 = 2.5;
/// Hub-cut absolute floor (SPEC-P5 B1): tokens with df strictly greater
/// than max(HUB_DF_FLOOR, 40% of n_texts_seen) are excluded at embed time.
const HUB_DF_FLOOR: u64 = 100;
/// Hard vocabulary cap (types), by frequency.
const VOCAB_CAP: usize = 250_000;

const MODEL_ID: &str = "rindex-v2";
const MODEL_FILE: &str = "rindex-v2.rimodel";

// ---------------------------------------------------------------------------
// Tokenizer (pub for tests, SPEC-P4 §2)
// ---------------------------------------------------------------------------

/// One token occurrence in the final token stream: the term, its weight,
/// and its stream position (bigrams sit at the position of their first
/// token).
#[derive(Clone, Debug, PartialEq)]
pub struct RiToken {
    pub term: String,
    pub weight: f32,
    pub pos: u32,
}

/// Keep tokens of len >= 2 that are not pure digits.
fn keep(term: &str) -> bool {
    term.chars().count() > 1 && !term.chars().all(|c| c.is_ascii_digit())
}

/// Identifier-aware sub-splitting: camelCase/PascalCase boundaries,
/// SCREAMING_CASE acronym boundaries (uppercase run followed by a
/// capitalized word: "URLParser" -> "url", "parser"), and digit/alpha
/// boundaries. Returns lowercase parts (never empty for non-empty input).
fn split_ident(ident: &str) -> Vec<String> {
    let chars: Vec<char> = ident.chars().collect();
    let mut parts: Vec<String> = Vec::new();
    let mut start = 0usize;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let cur = chars[i];
        let boundary = (prev.is_ascii_digit() != cur.is_ascii_digit())
            || (cur.is_uppercase() && prev.is_lowercase())
            || (cur.is_uppercase()
                && prev.is_uppercase()
                && chars.get(i + 1).is_some_and(|n| n.is_lowercase()));
        if boundary {
            parts.push(chars[start..i].iter().collect::<String>().to_lowercase());
            start = i;
        }
    }
    parts.push(chars[start..].iter().collect::<String>().to_lowercase());
    parts
}

/// Split chunk text (header + body) on non-alphanumerics, then identifier-
/// aware sub-splitting (camel/Pascal/snake/SCREAMING/digit boundaries).
/// Lowercases everything. Emits both the lowercase unsplit identifier (0.5
/// weight) and its parts (1.0) when a split occurred; otherwise the single
/// token at 1.0. Drops len-1 tokens and pure-digit tokens. Appends adjacent
/// bigrams of the final token stream (weight 1.0).
pub fn tokenize(text: &str) -> Vec<RiToken> {
    let mut stream: Vec<RiToken> = Vec::new();
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }
        let unsplit = raw.to_lowercase();
        let parts = split_ident(raw);
        if parts.len() == 1 {
            if keep(&unsplit) {
                let pos = stream.len() as u32;
                stream.push(RiToken {
                    term: unsplit,
                    weight: 1.0,
                    pos,
                });
            }
            continue;
        }
        // A real split: unsplit form at 0.5, then parts at 1.0.
        if keep(&unsplit) {
            let pos = stream.len() as u32;
            stream.push(RiToken {
                term: unsplit,
                weight: 0.5,
                pos,
            });
        }
        for p in parts {
            if keep(&p) {
                let pos = stream.len() as u32;
                stream.push(RiToken {
                    term: p,
                    weight: 1.0,
                    pos,
                });
            }
        }
    }
    // Adjacent bigrams of the final token stream, at the first token's pos.
    let n = stream.len();
    let mut out = stream;
    for i in 0..n.saturating_sub(1) {
        let (a, b) = (&out[i], &out[i + 1]);
        let term = format!("{} {}", a.term, b.term);
        out.push(RiToken {
            term,
            weight: 1.0,
            pos: a.pos,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Label vectors — deterministic sparse ternary from blake3
// ---------------------------------------------------------------------------

/// Sparse ternary label vector for a token type: LABEL_NONZEROS distinct
/// dims in [0, DIM) with signs, derived from blake3(b"ri-label" || token)
/// read as an XOF stream (retry past dim collisions). Returned sorted by
/// dim.
type Label = [(u32, f32); LABEL_NONZEROS];

/// Process-wide label cache. `label()` is pure but costs a blake3 XOF, a
/// 2 KiB scratch array and a sort per call, and `observe` visits every
/// token once per neighbor (6x) — this cache is the single largest win in
/// the embed pipeline (SPEC-P6 perf note). Bounded to keep memory sane on
/// pathological vocabularies; misses past the bound just recompute.
fn label_cached(term: &str) -> Label {
    const CACHE_MAX: usize = 2_000_000;
    static CACHE: OnceLock<RwLock<HashMap<String, Label>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(l) = cache.read().expect("label cache poisoned").get(term) {
        return *l;
    }
    let l = label(term);
    let mut w = cache.write().expect("label cache poisoned");
    if w.len() < CACHE_MAX {
        w.insert(term.to_string(), l);
    }
    l
}

fn label(term: &str) -> Label {
    let mut h = blake3::Hasher::new();
    h.update(b"ri-label");
    h.update(term.as_bytes());
    let mut xof = h.finalize_xof();
    let mut used = [false; DIM];
    let mut out = [(0u32, 0.0f32); LABEL_NONZEROS];
    let mut buf = [0u8; 4];
    let mut filled = 0;
    while filled < LABEL_NONZEROS {
        xof.fill(&mut buf);
        let v = u32::from_le_bytes(buf);
        let dim = ((v >> 1) as usize) % DIM;
        if used[dim] {
            continue;
        }
        used[dim] = true;
        let sign = if v & 1 == 0 { 1.0f32 } else { -1.0f32 };
        out[filled] = (dim as u32, sign);
        filled += 1;
    }
    out.sort_by_key(|&(d, _)| d);
    out
}

// ---------------------------------------------------------------------------
// Direction permutations (SPEC-P5 B1, Random Permutation Model)
// ---------------------------------------------------------------------------

/// Minimal xorshift64 PRNG (Marsaglia) — deterministic across platforms.
struct XorShift64(u64);

impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// A fixed permutation of 0..DIM built by Fisher-Yates with a xorshift64
/// stream seeded from `seed_bytes` (8 bytes of blake3(b"ri-perm-v2")).
fn build_permutation(seed_bytes: [u8; 8]) -> [u32; DIM] {
    let seed = u64::from_le_bytes(seed_bytes);
    // xorshift requires a nonzero state.
    let mut rng = XorShift64(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed });
    let mut perm: Vec<u32> = (0..DIM as u32).collect();
    for i in (1..DIM).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    perm.try_into().expect("permutation has exactly DIM entries")
}

/// The two direction permutations (SPEC-P5 B1): left context applies
/// PERM_LEFT (π), right context applies PERM_RIGHT (π') to the neighbor's
/// label dims. Both derive from blake3(b"ri-perm-v2"): π from digest bytes
/// 0..8, π' from bytes 8..16.
fn permutations() -> (&'static [u32; DIM], &'static [u32; DIM]) {
    static PERMS: OnceLock<([u32; DIM], [u32; DIM])> = OnceLock::new();
    let (l, r) = PERMS.get_or_init(|| {
        let digest = blake3::hash(b"ri-perm-v2");
        let b = digest.as_bytes();
        (
            build_permutation(b[0..8].try_into().expect("blake3 is 32 bytes")),
            build_permutation(b[8..16].try_into().expect("blake3 is 32 bytes")),
        )
    });
    (l, r)
}

/// Apply a direction permutation to a label: dims are remapped through
/// `perm`, signs kept. Output stays sorted is NOT guaranteed (permutation
/// scrambles order) — callers only iterate (dim, sign) pairs.
fn permute(lab: &[(u32, f32); LABEL_NONZEROS], perm: &[u32; DIM]) -> [(u32, f32); LABEL_NONZEROS] {
    let mut out = *lab;
    for (d, _) in out.iter_mut() {
        *d = perm[*d as usize];
    }
    out
}

// ---------------------------------------------------------------------------
// Per-text update (the parallel half of `observe`)
// ---------------------------------------------------------------------------

/// Everything `observe` needs from one text that does not touch the model,
/// fully reduced: the distinct terms (for df) and, per term, its context
/// contributions already summed per dim in stream-visit order and sorted by
/// dim. Pure, so batches are computed with rayon; `Model::apply_update`
/// then only merges into the model, sequentially in text order (per-text
/// CTX_TOP truncation makes order significant).
struct TextUpdate {
    /// Distinct terms of the text (first-seen order).
    terms: Vec<String>,
    /// Vocabulary shard of each term (see [`Vocab`]), computed here so the
    /// sequential half does no hashing.
    shards: Vec<u16>,
    /// Per term (index into `terms`): dim-sorted, pre-summed contributions.
    adds: Vec<Vec<(u32, f32)>>,
}

fn compute_update(text: &str) -> TextUpdate {
    let toks = tokenize(text);
    let labels: Vec<Label> = toks.iter().map(|t| label_cached(&t.term)).collect();
    // Distinct terms + token -> term index.
    let mut term_ix: HashMap<&str, u32> = HashMap::new();
    let mut terms: Vec<String> = Vec::new();
    let tok_term: Vec<u32> = toks
        .iter()
        .map(|t| {
            *term_ix.entry(t.term.as_str()).or_insert_with(|| {
                terms.push(t.term.clone());
                (terms.len() - 1) as u32
            })
        })
        .collect();
    // Token indices by stream position (positions are dense from 0).
    let npos = toks.iter().map(|t| t.pos as usize + 1).max().unwrap_or(0);
    let mut by_pos: Vec<Vec<usize>> = vec![Vec::new(); npos];
    for (i, t) in toks.iter().enumerate() {
        by_pos[t.pos as usize].push(i);
    }
    let (perm_left, perm_right) = permutations();
    // Raw contributions per term in stream-visit order: the outer loop is
    // the token stream, so appending to the token's term bucket preserves
    // exactly the visit order the original single-map implementation used.
    let mut raw: Vec<Vec<(u32, f32)>> = vec![Vec::new(); terms.len()];
    for (ti, t) in toks.iter().enumerate() {
        let bucket = &mut raw[tok_term[ti] as usize];
        for d in 1..=CTX_RADIUS {
            for side in [-1i64, 1i64] {
                let j = t.pos as i64 + side * d;
                if j < 0 || j as usize >= npos {
                    continue;
                }
                let w = 1.0f32 / (1.0 + d as f32);
                // Right context applies π', left applies π (SPEC-P5 B1).
                let perm = if side > 0 { perm_right } else { perm_left };
                for &ri in &by_pos[j as usize] {
                    let lab = permute(&labels[ri], perm);
                    let inc = w * toks[ri].weight;
                    for (dim, s) in lab {
                        bucket.push((dim, inc * s));
                    }
                }
            }
        }
    }
    // Reduce: stable sort by dim keeps visit order within a dim, so the
    // per-dim running sum is bit-for-bit the original's.
    let adds: Vec<Vec<(u32, f32)>> = raw
        .into_iter()
        .map(|mut b| {
            b.sort_by_key(|&(d, _)| d);
            let mut summed: Vec<(u32, f32)> = Vec::with_capacity(b.len());
            for (d, v) in b {
                match summed.last_mut() {
                    Some(last) if last.0 == d => last.1 += v,
                    _ => summed.push((d, v)),
                }
            }
            summed
        })
        .collect();
    let shards: Vec<u16> = terms.iter().map(|t| Vocab::shard_of(t) as u16).collect();
    TextUpdate { terms, shards, adds }
}

// ---------------------------------------------------------------------------
// Sharded vocabulary
// ---------------------------------------------------------------------------

/// Number of vocabulary shards (parallel apply in `observe`).
const VOCAB_SHARDS: usize = 32;

/// The vocabulary (SPEC-P10): a memory-mapped, immutable snapshot
/// ([`Snapshot`], the file on disk) plus an in-memory overlay of the entries
/// changed since it was written. A lookup checks the overlay first; a save
/// merges the two into a new snapshot and empties the overlay. The overlay
/// is sharded by term hash so `observe` can apply a batch of text updates
/// in parallel: a term's context evolution depends only on its own
/// contribution sequence, so applying each shard's items sequentially in
/// text order is exactly equivalent to a single serial pass. Every
/// iteration site sorts its output, so the shard layout is never observable.
#[derive(Default)]
struct Vocab {
    shards: Vec<HashMap<String, Entry>>,
    base: Option<Arc<Snapshot>>,
    /// Snapshot terms evicted by the cap (until the next save).
    removed: HashSet<String>,
}

/// A vocabulary entry borrowed from the overlay or the snapshot.
#[derive(Clone, Copy)]
struct EntryRef<'a> {
    df: u32,
    ctx: &'a [Comp],
}

impl Vocab {
    fn new() -> Self {
        Vocab {
            shards: (0..VOCAB_SHARDS).map(|_| HashMap::new()).collect(),
            base: None,
            removed: HashSet::new(),
        }
    }

    fn with_base(base: Arc<Snapshot>) -> Self {
        Vocab { base: Some(base), ..Self::new() }
    }

    /// FNV-1a over the term bytes: cheap, deterministic, no `Hasher` state.
    fn shard_of(term: &str) -> usize {
        (rimodel::fnv1a(term) % VOCAB_SHARDS as u64) as usize
    }

    fn get(&self, term: &str) -> Option<EntryRef<'_>> {
        if let Some(e) = self.shards.get(Self::shard_of(term))?.get(term) {
            return Some(EntryRef { df: e.df, ctx: &e.ctx });
        }
        if self.removed.contains(term) {
            return None;
        }
        let (df, ctx) = self.base.as_ref()?.get(term)?;
        Some(EntryRef { df, ctx })
    }

    #[cfg(test)]
    fn contains_key(&self, term: &str) -> bool {
        self.get(term).is_some()
    }

    fn insert(&mut self, term: String, e: Entry) {
        let i = Self::shard_of(&term);
        self.removed.remove(&term);
        self.shards[i].insert(term, e);
    }

    fn remove(&mut self, term: &str) {
        let i = Self::shard_of(term);
        self.shards[i].remove(term);
        if self.base.as_ref().map_or(false, |b| b.get(term).is_some()) {
            self.removed.insert(term.to_string());
        }
    }

    /// Whether `term` is in the snapshot and not evicted.
    fn in_base(&self, term: &str) -> bool {
        !self.removed.contains(term) && self.base.as_ref().map_or(false, |b| b.get(term).is_some())
    }

    fn len(&self) -> usize {
        let base = self.base.as_ref().map_or(0, |b| b.len()) - self.removed.len();
        let overlay_new = self.shards.iter().flat_map(|m| m.keys()).filter(|t| !self.in_base(t)).count();
        base + overlay_new
    }

    /// Every live entry, overlay first, then the snapshot's entries the
    /// overlay does not shadow.
    fn for_each(&self, mut f: impl FnMut(&str, EntryRef<'_>)) {
        for m in &self.shards {
            for (t, e) in m {
                f(t, EntryRef { df: e.df, ctx: &e.ctx });
            }
        }
        if let Some(b) = &self.base {
            for i in 0..b.len() {
                let (t, df, ctx) = b.entry(i);
                if self.removed.contains(t) || self.shards[Self::shard_of(t)].contains_key(t) {
                    continue;
                }
                f(t, EntryRef { df, ctx });
            }
        }
    }

    #[cfg(test)]
    fn values(&self) -> Vec<Entry> {
        let mut out = Vec::new();
        self.for_each(|_, e| out.push(Entry { df: e.df, ctx: e.ctx.to_vec() }));
        out
    }

    /// Sorted `(term, df, ctx)` of every live entry: what a snapshot holds.
    fn to_entries(&self) -> Vec<(String, u32, Vec<Comp>)> {
        let mut v: Vec<(String, u32, Vec<Comp>)> = Vec::with_capacity(self.len());
        self.for_each(|t, e| v.push((t.to_string(), e.df, e.ctx.to_vec())));
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }
}

/// Apply one term's pre-reduced contributions to its shard: df, merge-join
/// into the dim-sorted context, drop zeros, truncate to CTX_TOP by |v|.
/// Arithmetic is bit-for-bit the original single-map implementation's:
/// contributions to one (term, dim) were summed in stream-visit order
/// (done in `compute_update`), then added onto the existing value once.
fn apply_term(
    shard: &mut HashMap<String, Entry>,
    base: Option<(&Snapshot, &HashSet<String>)>,
    term: &str,
    summed: &[(u32, f32)],
) {
    if !shard.contains_key(term) {
        // first change since the snapshot: seed the overlay from it
        let seed = match base {
            Some((b, removed)) if !removed.contains(term) => match b.get(term) {
                Some((df, ctx)) => Entry { df, ctx: ctx.to_vec() },
                None => Entry::default(),
            },
            _ => Entry::default(),
        };
        shard.insert(term.to_string(), seed);
    }
    let e = shard.get_mut(term).expect("inserted above");
    e.df += 1;
    if summed.is_empty() {
        return;
    }
    let old = &e.ctx;
    let mut ctx: Vec<Comp> = Vec::with_capacity(old.len() + summed.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < old.len() || j < summed.len() {
        if j >= summed.len() || (i < old.len() && old[i].d < summed[j].0) {
            ctx.push(old[i]);
            i += 1;
        } else if i >= old.len() || summed[j].0 < old[i].d {
            ctx.push(Comp { d: summed[j].0, v: 0.0 + summed[j].1 });
            j += 1;
        } else {
            ctx.push(Comp { d: old[i].d, v: old[i].v + summed[j].1 });
            i += 1;
            j += 1;
        }
    }
    ctx.retain(|c| c.v != 0.0);
    if ctx.len() > CTX_TOP {
        // The (|v| desc, dim asc) order is total (dims are unique), so the
        // top-CTX_TOP set is unique: O(n) selection == full sort + truncate.
        let cmp = |a: &Comp, b: &Comp| b.v.abs().total_cmp(&a.v.abs()).then_with(|| a.d.cmp(&b.d));
        ctx.select_nth_unstable_by(CTX_TOP - 1, cmp);
        ctx.truncate(CTX_TOP);
        ctx.sort_by_key(|c| c.d);
    }
    e.ctx = ctx;
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Entry {
    df: u32,
    /// Sparse context accumulator, sorted by dim, truncated to top
    /// [`CTX_TOP`] components by |v| after each observed text.
    ctx: Vec<Comp>,
}

struct Model {
    n_texts_seen: u64,
    /// Sharded hash map (SPEC-P6 perf); every iteration site sorts
    /// (`to_disk`, `nearest`, `enforce_cap`) so results stay deterministic.
    vocab: Vocab,
}

impl Default for Model {
    fn default() -> Self {
        Model {
            n_texts_seen: 0,
            vocab: Vocab::new(),
        }
    }
}

/// The pre-P10 on-disk format (SPEC-P4 §2): bincode of { dim, n_texts_seen,
/// vocab }. Still read (and converted to a snapshot on the first save).
#[derive(serde::Serialize, serde::Deserialize)]
struct ModelDisk {
    dim: u32,
    n_texts_seen: u64,
    vocab: Vec<(String, u32, Vec<(u32, f32)>)>,
}

impl Model {
    fn from_disk(d: ModelDisk) -> Self {
        let mut vocab = Vocab::new();
        for (t, df, ctx) in d.vocab {
            vocab.insert(t, Entry { df, ctx: ctx.into_iter().map(|(d, v)| Comp { d, v }).collect() });
        }
        Model {
            n_texts_seen: d.n_texts_seen,
            vocab,
        }
    }

    fn from_snapshot(s: Arc<Snapshot>) -> Self {
        Model {
            n_texts_seen: s.n_texts_seen(),
            vocab: Vocab::with_base(s),
        }
    }

    /// Write the merged vocabulary as a snapshot at `path` and continue on
    /// top of it (empty overlay).
    ///
    /// Several servers share one model file (one per session, SPEC-P10
    /// §19): each saves its own overlay, so before merging, the base is
    /// rebased onto whatever snapshot is on disk now if another writer got
    /// there first — its terms survive, ours are applied on top (a term
    /// both changed keeps ours; `n_texts_seen` adds our share). Without
    /// this the last writer silently dropped the other's updates.
    fn save_snapshot(&mut self, path: &Path) -> anyhow::Result<()> {
        // (also when we started from nothing and another server has since
        // written the first snapshot)
        if path.is_file() && rimodel::is_snapshot(path) {
            if let Ok(fresh) = Snapshot::open(path) {
                let ours = self.vocab.base.as_ref().map_or(0, |b| b.n_texts_seen());
                if fresh.n_texts_seen() > ours && fresh.dim() as usize == DIM {
                    let observed_here = self.n_texts_seen.saturating_sub(ours);
                    self.n_texts_seen = fresh.n_texts_seen() + observed_here;
                    self.vocab.base = Some(Arc::new(fresh));
                    self.vocab.removed.clear();
                }
            }
        }
        let entries = self.vocab.to_entries();
        Snapshot::write(path, DIM as u32, self.n_texts_seen, &entries)?;
        drop(entries);
        let snap = Snapshot::open(path)?;
        self.vocab = Vocab::with_base(Arc::new(snap));
        Ok(())
    }

    /// Apply a batch of text updates: shards in parallel, each shard's
    /// items in text order (equivalent to a serial pass, see [`Vocab`]).
    fn apply_batch(&mut self, updates: &[TextUpdate]) {
        self.n_texts_seen += updates.len() as u64;
        let mut per_shard: Vec<Vec<(&str, &[(u32, f32)])>> =
            (0..VOCAB_SHARDS).map(|_| Vec::new()).collect();
        for u in updates {
            for ((term, &sh), adds) in u.terms.iter().zip(&u.shards).zip(&u.adds) {
                per_shard[sh as usize].push((term.as_str(), adds.as_slice()));
            }
        }
        let base = self.vocab.base.as_deref().map(|b| (b, &self.vocab.removed));
        self.vocab
            .shards
            .par_iter_mut()
            .zip(per_shard.into_par_iter())
            .for_each(|(shard, items)| {
                for (term, summed) in items {
                    apply_term(shard, base, term, summed);
                }
            });
    }

    /// Hub-cut threshold (SPEC-P5 B1): tokens with df strictly above this
    /// contribute nothing (label or context) at embed/expansion time.
    fn hub_cutoff(&self) -> u64 {
        (self.n_texts_seen * 2 / 5).max(HUB_DF_FLOOR)
    }

    /// idf-weighted sum of *unit-normalized* token context vectors +
    /// LABEL_WEIGHT * sum of token labels, densified to DIM and
    /// L2-normalized. Tokens with no context yet (or never observed)
    /// contribute only their label.
    ///
    /// Each context vector is L2-normalized BEFORE idf-weighting
    /// (Sahlgren's normalization): raw context magnitudes grow with token
    /// frequency (a token observed in thousands of contexts accumulates a
    /// proportionally large vector), so without this step high-df tokens
    /// dominate every embedding they appear in regardless of their low
    /// idf — generic central files outrank the distinctive ones. Label
    /// vectors are fixed-magnitude (LABEL_NONZEROS nonzeros) and stay as-is.
    ///
    /// SPEC-P5 B1 on top of the P4 formula:
    /// - Hub cut: tokens with df > max(100, 40% n_texts_seen) are skipped
    ///   entirely (label AND context excluded), evaluated at embed time.
    /// - Header field weighting: the text is split at the first '\n' into
    ///   header/body (chunk texts are `header + "\n" + body`); header
    ///   tokens weigh HEADER_WEIGHT (2.5) in the pooling sum, body 1.0.
    ///   A single-line text is all "header", a uniform scale that
    ///   L2-normalization cancels — so queries are unaffected.
    fn embed_text(&self, text: &str) -> Vec<f32> {
        let (header, body) = match text.find('\n') {
            Some(i) => (&text[..i], &text[i + 1..]),
            None => (text, ""),
        };
        let mut v = vec![0.0f32; DIM];
        let n = self.n_texts_seen as f64;
        let hub_cut = self.hub_cutoff();
        for (part, field_w) in [(header, HEADER_WEIGHT), (body, 1.0f32)] {
            if part.is_empty() {
                continue;
            }
            for t in tokenize(part) {
                // Hub cut: label AND context excluded.
                if let Some(e) = self.vocab.get(&t.term) {
                    if u64::from(e.df) > hub_cut {
                        continue;
                    }
                }
                let w = t.weight * field_w;
                for (d, s) in label_cached(&t.term) {
                    v[d as usize] += LABEL_WEIGHT * w * s;
                }
                if let Some(e) = self.vocab.get(&t.term) {
                    if !e.ctx.is_empty() && e.df > 0 {
                        let idf = (1.0 + n / f64::from(e.df)).ln() as f32;
                        // ctx is stored dim-sorted, so this sum order (and
                        // the resulting norm) is bit-for-bit deterministic.
                        let norm = e.ctx.iter().map(|c| c.v * c.v).sum::<f32>().sqrt();
                        if norm > 0.0 {
                            let w = w * idf / norm;
                            for c in e.ctx {
                                v[c.d as usize] += w * c.v;
                            }
                        }
                    }
                }
            }
        }
        l2_normalize(&mut v);
        v
    }

    /// Norm of a stored sparse context vector.
    fn ctx_norm(a: &[Comp]) -> f32 {
        a.iter().map(|c| c.v * c.v).sum::<f32>().sqrt()
    }

    /// Merge-join dot product of two dim-sorted sparse vectors.
    fn ctx_dot(a: &[Comp], b: &[Comp]) -> f32 {
        let mut dot = 0.0f32;
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].d.cmp(&b[j].d) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    dot += a[i].v * b[j].v;
                    i += 1;
                    j += 1;
                }
            }
        }
        dot
    }

    /// Nearest vocabulary tokens to `token` by context-vector cosine,
    /// descending (ties by term). Hub-cut tokens and `token` itself are
    /// excluded; only sims >= `min_sim` are returned. `token` without a
    /// (non-empty) context vector yields no neighbors.
    fn nearest(&self, token: &str, k: usize, min_sim: f32) -> Vec<(String, f32)> {
        self.nearest_many(&[token], k, min_sim).remove(0)
    }

    /// [`nearest`](Self::nearest) for several tokens in ONE pass over the
    /// vocabulary (SPEC-P9): the per-entry work — hub/df gate and the
    /// entry's own norm — is paid once for the whole query instead of once
    /// per term, and each query term's norm is computed once, not once per
    /// entry. Results are per input token, in order; unknown tokens or
    /// tokens without a context vector get an empty list.
    fn nearest_many(&self, tokens: &[&str], k: usize, min_sim: f32) -> Vec<Vec<(String, f32)>> {
        let hub_cut = self.hub_cutoff();
        let queries: Vec<Option<(&str, &[Comp], f32)>> = tokens
            .iter()
            .map(|t| {
                let e = self.vocab.get(t)?;
                if e.ctx.is_empty() {
                    return None;
                }
                let n = Self::ctx_norm(e.ctx);
                (n > 0.0).then_some((*t, e.ctx, n))
            })
            .collect();
        if queries.iter().all(Option::is_none) {
            return vec![Vec::new(); tokens.len()];
        }
        // Full-vocabulary scan (SPEC-P6 perf: this was ~700 ms per query on
        // a 250k-term vocabulary): the overlay shards and ranges of the
        // snapshot in parallel; the final sort makes the result independent
        // of shard/thread order.
        let score_one = |term: &str, e: EntryRef<'_>, out: &mut Vec<Vec<(f32, String)>>| {
            if e.ctx.is_empty() || u64::from(e.df) > hub_cut {
                return;
            }
            let nb = Self::ctx_norm(e.ctx);
            if nb <= 0.0 {
                return;
            }
            for (qi, q) in queries.iter().enumerate() {
                let Some((qt, qctx, qn)) = q else { continue };
                if term == *qt {
                    continue;
                }
                let sim = Self::ctx_dot(qctx, e.ctx) / (qn * nb);
                if sim >= min_sim {
                    out[qi].push((sim, term.to_string()));
                }
            }
        };
        let mut per_shard: Vec<Vec<Vec<(f32, String)>>> = self
            .vocab
            .shards
            .par_iter()
            .map(|shard| {
                let mut out: Vec<Vec<(f32, String)>> = vec![Vec::new(); queries.len()];
                for (term, e) in shard.iter() {
                    score_one(term, EntryRef { df: e.df, ctx: &e.ctx }, &mut out);
                }
                out
            })
            .collect();
        if let Some(b) = &self.vocab.base {
            const RANGE: usize = 8192;
            let n = b.len();
            let ranges: Vec<(usize, usize)> = (0..n).step_by(RANGE).map(|s| (s, (s + RANGE).min(n))).collect();
            let vocab = &self.vocab;
            // a read-only server never touches the overlay: skip the two
            // hash probes per entry then
            let shadowed = !vocab.removed.is_empty() || vocab.shards.iter().any(|m| !m.is_empty());
            per_shard.par_extend(ranges.into_par_iter().map(|(lo, hi)| {
                let mut out: Vec<Vec<(f32, String)>> = vec![Vec::new(); queries.len()];
                for i in lo..hi {
                    let (term, df, ctx) = b.entry(i);
                    if shadowed && (vocab.removed.contains(term) || vocab.shards[Vocab::shard_of(term)].contains_key(term)) {
                        continue;
                    }
                    score_one(term, EntryRef { df, ctx }, &mut out);
                }
                out
            }));
        }
        (0..queries.len())
            .map(|qi| {
                let mut scored: Vec<(f32, &str)> =
                    per_shard.iter().flat_map(|s| s[qi].iter().map(|(sim, t)| (*sim, t.as_str()))).collect();
                scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
                scored.truncate(k);
                scored.into_iter().map(|(s, t)| (t.to_string(), s)).collect()
            })
            .collect()
    }

    /// Hard vocabulary cap: evict lowest-df types (ties by term) until
    /// within `cap`. This is where min-df 2 bites: singletons are evicted
    /// before any df >= 2 type.
    fn enforce_cap(&mut self, cap: usize) {
        if self.vocab.len() <= cap {
            return;
        }
        let excess = self.vocab.len() - cap;
        let mut by_freq: Vec<(u32, String)> = Vec::with_capacity(cap + excess);
        self.vocab.for_each(|t, e| by_freq.push((e.df, t.to_string())));
        by_freq.sort();
        let drop: Vec<String> = by_freq.into_iter().take(excess).map(|(_, t)| t).collect();
        for t in drop {
            self.vocab.remove(&t);
        }
    }
}

// ---------------------------------------------------------------------------
// RandomIndexingEmbedder
// ---------------------------------------------------------------------------

/// Random-indexing semantic model + embedder (SPEC-P4 §2, SPEC-P5 B1/B2).
/// `model_id() == "rindex-v2"`, `dim() == 2048`.
/// Model file: `<data_dir>/sem/rindex-v2.rimodel` (bincode, tmp+rename
/// atomic; corrupt file -> start empty with a tracing::warn; dim mismatch
/// -> error, since a v2 file with dim != 2048 is a version mixup: rebuild
/// with `indexio embed --rebuild-model`). v1 `rindex-v1.rimodel` files are
/// ignored (clean break; migration = `indexio embed --rebuild-model`).
pub struct RandomIndexingEmbedder {
    model: RwLock<Model>,
    path: PathBuf,
    cap: usize,
    /// Thesaurus neighbours per query term (SPEC-P9): `nearest` is a full
    /// vocabulary scan (~10 ms per term on 250k types), and agents repeat
    /// topic words across a session. Cleared whenever the model changes.
    expand_cache: RwLock<HashMap<String, Vec<(String, f32)>>>,
    /// Lazy persistence (SPEC-P9): a long-lived server observes a few
    /// chunks per edit, and serialising the ~250 MB model each time would
    /// dominate the refresh. With `lazy_flush` set, `flush` only writes
    /// when the model changed and the last save is older than the
    /// interval; embeddings themselves are always persisted in the CAS,
    /// so an unsaved tail costs at most a little context-vector drift.
    lazy_flush: Option<std::time::Duration>,
    dirty: std::sync::atomic::AtomicBool,
    last_save: std::sync::Mutex<std::time::Instant>,
    /// Texts observed since the expansion cache was last cleared: a few
    /// chunks barely move any neighbourhood, so the cache survives the
    /// per-edit observes of the auto-refresh and is dropped only after
    /// [`EXPAND_CLEAR_AFTER`] texts.
    observed_since_clear: std::sync::atomic::AtomicUsize,
    /// The expansion cache gained entries since it was last written.
    expand_dirty: std::sync::atomic::AtomicBool,
}

/// Observed texts after which cached query expansions are considered stale.
const EXPAND_CLEAR_AFTER: usize = 2000;

/// Upper bound on cached query-term expansions.
const EXPAND_CACHE_CAP: usize = 4096;

impl RandomIndexingEmbedder {
    /// Load the model if `<data_dir>/sem/rindex-v2.rimodel` exists, else
    /// start empty.
    pub fn open(data_dir: &Path) -> anyhow::Result<Self> {
        Self::open_with_cap(data_dir, VOCAB_CAP)
    }

    /// `open` with a caller-chosen vocabulary cap (tests).
    pub(crate) fn open_with_cap(data_dir: &Path, cap: usize) -> anyhow::Result<Self> {
        let path = Self::model_path(data_dir);
        rimodel::reap_parked(&path);
        let model = if path.is_file() && rimodel::is_snapshot(&path) {
            // SPEC-P10: mapped, nothing read up front
            let snap = Snapshot::open(&path)?;
            if snap.dim() as usize != DIM {
                anyhow::bail!(
                    "rindex model {} has dim {} != expected {} \
                     (rebuild with `indexio embed --rebuild-model`)",
                    path.display(),
                    snap.dim(),
                    DIM,
                );
            }
            Model::from_snapshot(Arc::new(snap))
        } else {
            match std::fs::read(&path) {
                Ok(bytes) => match bincode::deserialize::<ModelDisk>(&bytes) {
                    Ok(disk) if disk.dim as usize == DIM => {
                        // pre-P10 bincode: load it once and convert to a
                        // snapshot, so the next open maps it
                        let mut m = Model::from_disk(disk);
                        match m.save_snapshot(&path) {
                            Ok(()) => tracing::info!(path = %path.display(), "rindex model converted to the mapped layout"),
                            Err(e) => tracing::warn!(error = %e, "rindex model: conversion failed; running from memory"),
                        }
                        m
                    }
                    Ok(disk) => {
                        // SPEC-P5 B1: persist+verify dim; a v2 file with the
                        // wrong dim is a version mixup, not corruption.
                        anyhow::bail!(
                            "rindex model {} has dim {} != expected {} \
                             (rebuild with `indexio embed --rebuild-model`)",
                            path.display(),
                            disk.dim,
                            DIM,
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            path = %path.display(),
                            error = %e,
                            "corrupt rindex model; starting empty"
                        );
                        Model::default()
                    }
                },
                Err(e) if e.kind() == ErrorKind::NotFound => Model::default(),
                Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
            }
        };
        // Persisted query expansions (SPEC-P9): the thesaurus scan costs
        // ~20 ms per new query word, and sessions repeat vocabulary.
        let expand_cache: HashMap<String, Vec<(String, f32)>> = std::fs::read(Self::expand_path(&path))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        Ok(RandomIndexingEmbedder {
            model: RwLock::new(model),
            path,
            cap,
            expand_cache: RwLock::new(expand_cache),
            lazy_flush: None,
            dirty: std::sync::atomic::AtomicBool::new(false),
            last_save: std::sync::Mutex::new(std::time::Instant::now()),
            observed_since_clear: std::sync::atomic::AtomicUsize::new(0),
            expand_dirty: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// `<model>.expand.json` next to the model file.
    fn expand_path(model_path: &Path) -> PathBuf {
        model_path.with_extension("expand.json")
    }

    /// Path of the on-disk model file for `data_dir`.
    pub fn model_path(data_dir: &Path) -> PathBuf {
        data_dir.join("sem").join(MODEL_FILE)
    }

    /// Number of token types in the vocabulary.
    pub fn vocab_len(&self) -> usize {
        self.model
            .read()
            .expect("rindex model lock poisoned")
            .vocab
            .len()
    }

    /// SPEC-P5 B2: nearest vocabulary tokens by context-vector cosine,
    /// sim >= `min_sim`, descending (ties by term). Hub-cut tokens and
    /// `token` itself are excluded.
    pub fn nearest_tokens(&self, token: &str, k: usize, min_sim: f32) -> Vec<(String, f32)> {
        self.model
            .read()
            .expect("rindex model lock poisoned")
            .nearest(token, k, min_sim)
    }

    /// SPEC-P5 B2: thesaurus query expansion. For each content token of
    /// the query (unigram, len >= 3, df >= 2, not hub-cut, not a
    /// stopword), append its top-3 nearest neighbors (min_sim 0.35) once
    /// each, after the original query text. Deterministic.
    pub fn expand_query(&self, query: &str) -> String {
        const EXPAND_K: usize = 3;
        const EXPAND_MIN_SIM: f32 = 0.35;
        let m = self.model.read().expect("rindex model lock poisoned");
        let hub_cut = m.hub_cutoff();
        let mut originals: Vec<String> = Vec::new();
        for t in tokenize(query) {
            // Unigrams only: bigram terms contain a space.
            if t.term.contains(' ') || originals.iter().any(|o| o == &t.term) {
                continue;
            }
            originals.push(t.term);
        }
        let mut out = query.to_string();
        // Content-token gate (SPEC-P5 B2), then one vocabulary scan for
        // every term the cache does not know yet (SPEC-P9).
        let eligible: Vec<&String> = originals
            .iter()
            .filter(|term| {
                if term.chars().count() < 3 || crate::embed::stopwords().contains(&term.as_str()) {
                    return false;
                }
                match m.vocab.get(term.as_str()) {
                    Some(e) => e.df >= 2 && u64::from(e.df) <= hub_cut,
                    None => false,
                }
            })
            .collect();
        let mut neighbors_of: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        let mut missing: Vec<&str> = Vec::new();
        {
            let c = self.expand_cache.read().expect("expand cache lock poisoned");
            for term in &eligible {
                match c.get(term.as_str()) {
                    Some(n) => {
                        neighbors_of.insert((*term).clone(), n.clone());
                    }
                    None => missing.push(term.as_str()),
                }
            }
        }
        if !missing.is_empty() {
            let found = m.nearest_many(&missing, EXPAND_K, EXPAND_MIN_SIM);
            let mut c = self.expand_cache.write().expect("expand cache lock poisoned");
            if c.len() + missing.len() >= EXPAND_CACHE_CAP {
                c.clear();
            }
            for (t, n) in missing.iter().zip(found) {
                c.insert((*t).to_string(), n.clone());
                neighbors_of.insert((*t).to_string(), n);
            }
            self.expand_dirty.store(true, std::sync::atomic::Ordering::Release);
        }
        for term in &eligible {
            let neighbors = neighbors_of.remove(term.as_str()).unwrap_or_default();
            for (neighbor, _) in neighbors {
                if !originals.iter().any(|o| o == &neighbor)
                    && !out.split(' ').any(|w| w == neighbor)
                {
                    out.push(' ');
                    out.push_str(&neighbor);
                }
            }
        }
        out
    }
}

impl RandomIndexingEmbedder {
    /// Save the model at most once per `interval` (see `lazy_flush`).
    pub fn with_lazy_flush(mut self, interval: std::time::Duration) -> Self {
        self.lazy_flush = Some(interval);
        self
    }

    /// Unconditional save (what `flush` does in eager mode): the overlay
    /// and the snapshot are merged into a new snapshot file, which becomes
    /// the base (SPEC-P10). Holds the write lock so no observe lands
    /// between the merge and the swap.
    pub fn save_now(&self) -> anyhow::Result<()> {
        let mut m = self.model.write().expect("rindex model lock poisoned");
        m.save_snapshot(&self.path)?;
        drop(m);
        self.dirty.store(false, std::sync::atomic::Ordering::Release);
        *self.last_save.lock().expect("last_save poisoned") = std::time::Instant::now();
        self.save_expand_cache();
        Ok(())
    }

    /// Persist the query-expansion cache when it changed (best effort;
    /// a few KB, so it runs on every flush).
    pub fn save_expand_cache(&self) {
        if !self.expand_dirty.swap(false, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        let path = Self::expand_path(&self.path);
        // several servers share the file: union with what is there
        let mut merged: HashMap<String, Vec<(String, f32)>> = std::fs::read(&path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        {
            let cache = self.expand_cache.read().expect("expand cache lock poisoned");
            if cache.is_empty() {
                return;
            }
            for (k, v) in cache.iter() {
                merged.insert(k.clone(), v.clone());
            }
        }
        if let Ok(bytes) = serde_json::to_vec(&merged) {
            let tmp = path.with_extension("expand.json.tmp");
            if std::fs::write(&tmp, &bytes).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    }
}

impl Embedder for RandomIndexingEmbedder {
    fn model_id(&self) -> &str {
        MODEL_ID
    }

    fn dim(&self) -> usize {
        DIM
    }

    /// In-process: a chunk embeds in ~0.4 ms, faster than an 8 KB read.
    fn recompute_is_cheap(&self) -> bool {
        true
    }

    fn texts_seen(&self) -> Option<u64> {
        Some(self.model.read().expect("rindex model lock poisoned").n_texts_seen)
    }

    /// Embedding is pure per text under a shared read lock: rayon across
    /// the batch. Output order = input order.
    fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let m = self.model.read().expect("rindex model lock poisoned");
        let model: &Model = &m;
        Ok(texts.par_iter().map(|t| model.embed_text(t)).collect())
    }

    /// Per-text updates are computed in parallel (pure), then applied
    /// sequentially in input order — identical results to a serial
    /// observe, including the streaming property (piecewise == batch).
    /// Batched so the buffered contributions stay bounded in memory.
    fn observe(&self, texts: &[String]) -> anyhow::Result<()> {
        const OBSERVE_BATCH: usize = 256;
        // Neighbourhoods shift as the model learns; drop cached expansions
        // once enough text has gone in to matter.
        let seen = self
            .observed_since_clear
            .fetch_add(texts.len(), std::sync::atomic::Ordering::Relaxed)
            + texts.len();
        if seen >= EXPAND_CLEAR_AFTER {
            self.expand_cache.write().expect("expand cache lock poisoned").clear();
            self.observed_since_clear.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        let mut m = self.model.write().expect("rindex model lock poisoned");
        let (mut compute_ms, mut apply_ms) = (0u128, 0u128);
        for batch in texts.chunks(OBSERVE_BATCH) {
            let t = std::time::Instant::now();
            let updates: Vec<TextUpdate> = batch.par_iter().map(|t| compute_update(t)).collect();
            compute_ms += t.elapsed().as_millis();
            let t = std::time::Instant::now();
            m.apply_batch(&updates);
            apply_ms += t.elapsed().as_millis();
        }
        let t = std::time::Instant::now();
        m.enforce_cap(self.cap);
        if !texts.is_empty() {
            self.dirty.store(true, std::sync::atomic::Ordering::Release);
        }
        tracing::debug!(
            texts = texts.len(),
            compute_ms = compute_ms as u64,
            apply_ms = apply_ms as u64,
            cap_ms = t.elapsed().as_millis() as u64,
            vocab = m.vocab.len(),
            "rindex observe"
        );
        Ok(())
    }

    fn expand_query(&self, q: &str) -> String {
        RandomIndexingEmbedder::expand_query(self, q)
    }

    /// Eager mode saves here (the CLI's `embed`/`sync`). Lazy mode (the
    /// server) never saves inline: the 250 MB snapshot merge used to land
    /// inside whichever tool call first embedded after the interval — a
    /// 2.7 s and a 5.7 s `read_span` in one afternoon (SPEC-P10 §30). The
    /// server asks [`save_due`](Self::save_due) and runs
    /// [`persist`](Self::persist) on its own thread.
    fn flush(&self) -> anyhow::Result<()> {
        self.save_expand_cache();
        if self.lazy_flush.is_some() {
            return Ok(());
        }
        self.save_now()
    }

    fn save_due(&self) -> bool {
        let Some(interval) = self.lazy_flush else { return false };
        self.dirty.load(std::sync::atomic::Ordering::Acquire)
            && self.last_save.lock().expect("last_save poisoned").elapsed() >= interval
    }

    fn persist(&self) -> anyhow::Result<()> {
        if !self.dirty.load(std::sync::atomic::Ordering::Acquire) {
            self.save_expand_cache();
            return Ok(());
        }
        self.save_now()
    }
}

// ---------------------------------------------------------------------------
// Tests (SPEC-P4 §2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::dot;

    fn terms(toks: &[RiToken]) -> Vec<(&str, f32)> {
        toks.iter().map(|t| (t.term.as_str(), t.weight)).collect()
    }

    /// SPEC-P10 §30: a lazily persisted model is never written inside
    /// `flush`; it reports when a save is due and `persist` writes it.
    #[test]
    fn lazy_flush_defers_the_save_to_persist() {
        let tmp = tempfile::tempdir().unwrap();
        let e = RandomIndexingEmbedder::open(tmp.path()).unwrap().with_lazy_flush(std::time::Duration::from_secs(3600));
        e.observe(&["lazyterm otherterm".to_string()]).unwrap();
        assert!(e.dirty.load(std::sync::atomic::Ordering::Acquire));
        e.flush().unwrap();
        assert!(e.dirty.load(std::sync::atomic::Ordering::Acquire), "flush must not save inline in lazy mode");
        assert!(!e.save_due(), "the interval has not passed");
        *e.last_save.lock().unwrap() = std::time::Instant::now() - std::time::Duration::from_secs(4000);
        assert!(e.save_due());
        e.persist().unwrap();
        assert!(!e.dirty.load(std::sync::atomic::Ordering::Acquire));
        assert!(!e.save_due());
        // the saved model reopens with the term
        let e2 = RandomIndexingEmbedder::open(tmp.path()).unwrap();
        assert!(e2.texts_seen().unwrap() >= 1);
    }

    #[test]
    fn tokenizer_identifier_splitting() {
        // camelCase: unsplit (0.5) + parts (1.0).
        let toks = tokenize("parseQuery");
        let ts = terms(&toks);
        assert!(ts.contains(&("parsequery", 0.5)), "{ts:?}");
        assert!(ts.contains(&("parse", 1.0)), "{ts:?}");
        assert!(ts.contains(&("query", 1.0)), "{ts:?}");
        // snake_case (underscores are non-alphanumeric separators).
        let toks = tokenize("parse_query");
        let ts = terms(&toks);
        assert!(ts.contains(&("parse", 1.0)), "{ts:?}");
        assert!(ts.contains(&("query", 1.0)), "{ts:?}");
        // PascalCase + SCREAMING acronym run + digit boundary.
        let toks = tokenize("URLParser2");
        let ts = terms(&toks);
        assert!(ts.contains(&("urlparser2", 0.5)), "{ts:?}");
        assert!(ts.contains(&("url", 1.0)), "{ts:?}");
        assert!(ts.contains(&("parser", 1.0)), "{ts:?}");
        // digit/alpha boundary; pure-digit part dropped.
        let toks = tokenize("utf8");
        let ts = terms(&toks);
        assert!(ts.contains(&("utf", 1.0)), "{ts:?}");
        assert!(!ts.iter().any(|(t, _)| *t == "8"), "{ts:?}");
        // len-1 tokens dropped, everything lowercased.
        let toks = tokenize("A B cd");
        let ts = terms(&toks);
        assert!(!ts.iter().any(|(t, _)| *t == "a" || *t == "b"), "{ts:?}");
        assert!(ts.contains(&("cd", 1.0)), "{ts:?}");
        // unsplit single word: emitted once at 1.0, not duplicated.
        let toks = tokenize("login");
        let ts = terms(&toks);
        assert_eq!(
            ts.iter().filter(|(t, _)| *t == "login").count(),
            1,
            "{ts:?}"
        );
        // adjacent bigrams of the final stream at weight 1.0.
        let toks = tokenize("login password session");
        let ts = terms(&toks);
        assert!(ts.contains(&("login password", 1.0)), "{ts:?}");
        assert!(ts.contains(&("password session", 1.0)), "{ts:?}");
        // bigram sits at its first token's position.
        let bg = toks.iter().find(|t| t.term == "login password").unwrap();
        let first = toks.iter().find(|t| t.term == "login").unwrap();
        assert_eq!(bg.pos, first.pos);
    }

    #[test]
    fn label_determinism_sparsity() {
        let a = label("authentication");
        let b = label("authentication");
        assert_eq!(a, b, "labels are deterministic");
        // LABEL_NONZEROS distinct dims, ternary signs.
        let dims: std::collections::BTreeSet<u32> = a.iter().map(|&(d, _)| d).collect();
        assert_eq!(dims.len(), LABEL_NONZEROS);
        assert!(a
            .iter()
            .all(|&(d, s)| d < DIM as u32 && (s == 1.0 || s == -1.0)));
        // Different tokens get different labels (overwhelmingly likely).
        assert_ne!(label("database"), label("authentication"));
    }

    /// SPEC-P5 B1: the direction permutations are deterministic, true
    /// permutations of 0..DIM, non-identity, and distinct from each other.
    #[test]
    fn permutations_deterministic_and_distinct() {
        let (l1, r1) = permutations();
        let (l2, r2) = permutations();
        assert_eq!(l1, l2, "π is deterministic");
        assert_eq!(r1, r2, "π' is deterministic");
        for (name, p) in [("pi", l1), ("pi'", r1)] {
            // True permutation: every dim is covered exactly once.
            let mut seen = vec![false; DIM];
            for &d in p.iter() {
                assert!((d as usize) < DIM, "{name} maps out of range");
                assert!(!seen[d as usize], "{name} collides at dim {d}");
                seen[d as usize] = true;
            }
            // Not the identity (with DIM=2048 a random permutation
            // fixing every point is impossible in practice).
            assert!(
                p.iter().enumerate().any(|(i, &d)| d as usize != i),
                "{name} must not be the identity"
            );
        }
        assert_ne!(l1, r1, "π != π'");
        // Permuting a label remaps dims through the permutation, keeps
        // signs, and the two directions land on different dims.
        let a = label("authentication");
        let left = permute(&a, l1);
        let right = permute(&a, r1);
        for i in 0..LABEL_NONZEROS {
            assert_eq!(left[i].0, l1[a[i].0 as usize]);
            assert_eq!(right[i].0, r1[a[i].0 as usize]);
            assert_eq!(left[i].1, a[i].1);
            assert_eq!(right[i].1, a[i].1);
        }
        assert_ne!(left, right, "direction matters");
    }

    fn fresh(dir: &Path) -> RandomIndexingEmbedder {
        RandomIndexingEmbedder::open(dir).unwrap()
    }

    const CORPUS: [&str; 3] = [
        "login password session",
        "database pool connection",
        "authentication login oauth",
    ];

    #[test]
    fn observe_embed_determinism_bitforbit() {
        let tmp = tempfile::tempdir().unwrap();
        let e1 = fresh(tmp.path());
        let e2 = fresh(tmp.path());
        let texts: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        e1.observe(&texts).unwrap();
        e2.observe(&texts).unwrap();
        let q = vec!["authentication".to_string(), "database".to_string()];
        assert_eq!(e1.embed(&q).unwrap(), e2.embed(&q).unwrap());
        // Piecewise observe == one batch observe (streaming property).
        let e3 = fresh(tmp.path());
        for t in &texts {
            e3.observe(std::slice::from_ref(t)).unwrap();
        }
        assert_eq!(e1.embed(&q).unwrap(), e3.embed(&q).unwrap());
    }

    #[test]
    fn dim_and_normalization() {
        let tmp = tempfile::tempdir().unwrap();
        let e = fresh(tmp.path());
        assert_eq!(e.dim(), DIM);
        assert_eq!(DIM, 2048, "SPEC-P5 B1: dim 2048");
        assert_eq!(LABEL_NONZEROS, 8, "SPEC-P5 B1: 8 nonzeros per label");
        assert_eq!(e.model_id(), "rindex-v2");
        let texts: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        e.observe(&texts).unwrap();
        for t in &texts {
            let v = e.embed(&[t.to_string()]).unwrap().remove(0);
            assert_eq!(v.len(), DIM);
            let norm = dot(&v, &v).sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
        }
        // Empty text -> zero vector, no NaN.
        let z = e.embed(&["   !!  ".to_string()]).unwrap().remove(0);
        assert!(z.iter().all(|&x| x == 0.0));
    }

    /// SPEC-P4 §2 critical test: semantic transfer across chunks with zero
    /// lexical overlap. After observing chunks A {login password session},
    /// B {database pool connection}, C {authentication login oauth}, the
    /// query "authentication" must rank A above B even though A shares no
    /// token with the query (login bridges A <-> C; their shared neighbor
    /// context aligns the sparse vectors).
    #[test]
    fn semantic_transfer() {
        let tmp = tempfile::tempdir().unwrap();
        let e = fresh(tmp.path());
        let texts: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        e.observe(&texts).unwrap();
        let q = e.embed(&["authentication".to_string()]).unwrap().remove(0);
        let va = e.embed(&[CORPUS[0].to_string()]).unwrap().remove(0);
        let vb = e.embed(&[CORPUS[1].to_string()]).unwrap().remove(0);
        let sa = dot(&q, &va);
        let sb = dot(&q, &vb);
        assert!(
            sa > sb,
            "query 'authentication' must rank login-chunk above database-chunk: {sa} vs {sb}"
        );
        assert!(sa > 0.0, "the transfer signal must be positive: {sa}");
    }

    /// Regression test for frequency domination (real-corpus pathology:
    /// NL query "search inside gzip bzip2 zstd archives" ranked generic
    /// central files above the decompression module). One hub token
    /// ("search") co-occurs with 50 generic texts; a pool of common
    /// code-ish filler tokens (fff/aaa/bbb/ccc) appears in 30 texts; the
    /// rare text mixes the hub token, the rare distinctive tokens and the
    /// common fillers. A query mixing the hub token with the rare tokens
    /// must rank the rare-token text #1. Verified against the raw
    /// (un-normalized) formula: it ranks a generic text #1 instead
    /// (0.55 vs 0.27), because ctx("search")'s raw magnitude outweighs
    /// every rare-token context ~50:1; unit normalization reverses it
    /// (0.92 vs 0.11).
    #[test]
    fn frequent_token_does_not_dominate_embed() {
        let tmp = tempfile::tempdir().unwrap();
        let e = fresh(tmp.path());
        let mut texts: Vec<String> = (0..50)
            .map(|i| format!("search engine index query parser topic{i}"))
            .collect();
        for i in 0..30 {
            texts.push(format!("fff aaa bbb ccc item{i}"));
        }
        let rare = "search gzip zstd fff aaa bbb ccc".to_string();
        texts.push(rare.clone());
        e.observe(&texts).unwrap();

        let q = e.embed(&["search gzip zstd".to_string()]).unwrap().remove(0);
        let score = |t: &String| dot(&q, &e.embed(&[t.clone()]).unwrap().remove(0));
        let rare_score = score(&rare);
        let best_other = texts[..texts.len() - 1]
            .iter()
            .map(score)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            rare_score > best_other,
            "rare-token text must rank #1: {rare_score} vs {best_other}"
        );
    }

    #[test]
    fn cold_start_empty_model() {
        let tmp = tempfile::tempdir().unwrap();
        let e = fresh(tmp.path());
        assert_eq!(e.vocab_len(), 0);
        // Embeds without panicking on an empty model.
        let a = e.embed(&["hello world".to_string()]).unwrap().remove(0);
        let b = e.embed(&["hello world".to_string()]).unwrap().remove(0);
        assert_eq!(a, b, "identical texts match on a cold model");
        assert!(dot(&a, &a) > 0.99);
        let c = e
            .embed(&["totally different content here".to_string()])
            .unwrap()
            .remove(0);
        assert!(dot(&a, &c) < 0.99);
        // Empty text is still a zero vector.
        let z = e.embed(&["".to_string()]).unwrap().remove(0);
        assert!(z.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn persistence_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let texts: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        let q = vec!["authentication".to_string(), "pool connection".to_string()];
        let before = {
            let e = fresh(tmp.path());
            e.observe(&texts).unwrap();
            let v = e.embed(&q).unwrap();
            e.flush().unwrap();
            assert!(RandomIndexingEmbedder::model_path(tmp.path()).is_file());
            v
        };
        // Reopen: identical vectors, vocab restored.
        let e2 = fresh(tmp.path());
        assert!(e2.vocab_len() > 0);
        let after = e2.embed(&q).unwrap();
        assert_eq!(
            before, after,
            "flush+reopen must reproduce vectors bit-for-bit"
        );
        // Continued observation keeps working after reload.
        let n = e2.vocab_len();
        e2.observe(&["zzqjv uncommonterms here".to_string()]).unwrap();
        assert!(e2.vocab_len() > n);
        e2.flush().unwrap();
        let e3 = fresh(tmp.path());
        assert_eq!(e3.vocab_len(), e2.vocab_len());
    }

    #[test]
    fn two_writers_keep_each_others_terms() {
        // SPEC-P10 §19: two servers on one data dir observe different
        // texts and save in turn; the second save must not drop the first
        let tmp = tempfile::tempdir().unwrap();
        let a = fresh(tmp.path());
        let b = fresh(tmp.path());
        a.observe(&["alphaterm gammaterm".to_string()]).unwrap();
        b.observe(&["betaterm gammaterm".to_string()]).unwrap();
        a.flush().unwrap();
        b.flush().unwrap();
        let c = fresh(tmp.path());
        let m = c.model.read().unwrap();
        assert!(m.vocab.contains_key("alphaterm"), "first writer's term survived");
        assert!(m.vocab.contains_key("betaterm"), "second writer's term saved");
        assert_eq!(m.n_texts_seen, 2, "both texts counted");
    }

    #[test]
    fn corrupt_model_file_starts_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let path = RandomIndexingEmbedder::model_path(tmp.path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not a bincode model at all").unwrap();
        let e = fresh(tmp.path());
        assert_eq!(e.vocab_len(), 0, "corrupt model -> empty start");
        let v = e.embed(&["hello world".to_string()]).unwrap().remove(0);
        assert!(dot(&v, &v) > 0.99);
    }

    #[test]
    fn vocab_cap_respected() {
        let tmp = tempfile::tempdir().unwrap();
        // Build 40 df-2 types first (83 types, under the cap), then push
        // the total (286) over the cap with 100 singleton texts so the
        // single eviction pass must choose singletons first.
        let e = RandomIndexingEmbedder::open_with_cap(tmp.path(), 250).unwrap();
        let commons: Vec<String> = (0..40).map(|i| format!("common{i} anchor")).collect();
        e.observe(&commons).unwrap();
        e.observe(&commons).unwrap();
        let mut texts = Vec::new();
        for i in 0..100 {
            texts.push(format!("unique{i} solo"));
        }
        e.observe(&texts).unwrap();
        assert!(e.vocab_len() <= 250, "cap enforced: {}", e.vocab_len());
        // min-df eviction priority: every df-2 unigram type survives while
        // singletons are evicted first.
        let m = e.model.read().unwrap();
        for i in 0..40 {
            assert!(
                m.vocab.contains_key(&format!("common{i}")),
                "df-2 type common{i} evicted"
            );
        }
        assert!(
            !m.vocab.contains_key("unique0"),
            "singletons are evicted before df>=2 types"
        );
        // Invariant: if any singleton survives, no df>=2 type was dropped.
        let singletons = m.vocab.values().iter().filter(|en| en.df < 2).count();
        let df2 = m.vocab.values().iter().filter(|en| en.df >= 2).count();
        assert!(singletons + df2 == m.vocab.len());
        assert!(df2 >= 40, "df>=2 types retained: {df2}");
    }

    /// Pipeline integration (SPEC-P4 §2): embed_repo on a 2-repo fixture
    /// with the rindex embedder writes the model file, CAS-dedups shared
    /// chunks, and builds the rindex-v2 vec index.
    #[test]
    fn pipeline_integration_two_repos() {
        use crate::pipeline::{embed_all, embed_repo};
        use indexio_index::{ShardSet, ShardWriter};
        use indexio_types::{BlobId, DocMeta, ExtractedArtifact, Lang};

        fn build_shardset(dir: &Path, repos: &[&str], docs: &[(u32, &str, &[u8])]) -> ShardSet {
            let shards_dir = dir.join("shards");
            std::fs::create_dir_all(&shards_dir).unwrap();
            let mut w = ShardWriter::new(&shards_dir).unwrap();
            for &(repo_id, path, content) in docs {
                let meta = DocMeta {
                    blob: BlobId::from_content(content),
                    repo_id,
                    path: path.to_string(),
                    lang: Lang::from_path(path),
                    raw_len: content.len() as u32,
                };
                let art = ExtractedArtifact {
                    lang: meta.lang,
                    raw_len: content.len() as u32,
                    ..Default::default()
                };
                w.add_doc(&meta, content, &art).unwrap();
            }
            let repos: Vec<String> = repos.iter().map(|s| s.to_string()).collect();
            w.finish(&repos).unwrap();
            ShardSet::open_dir(&shards_dir).unwrap()
        }

        let tmp = tempfile::tempdir().unwrap();
        let shared: &[u8] = b"pub fn login_session(token: &str) -> bool { check(token) }\n";
        let shards = build_shardset(
            tmp.path(),
            &["r1", "r2"],
            &[
                (
                    0,
                    "src/auth.rs",
                    b"fn authenticate_user(password: &str) -> Session { todo!() }\n",
                ),
                (0, "src/shared.rs", shared),
                (1, "src/shared.rs", shared),
                (
                    1,
                    "src/db.rs",
                    b"fn database_pool_connection() -> Pool { todo!() }\n",
                ),
            ],
        );
        let e = fresh(tmp.path());
        let rep1 = embed_repo(&shards, tmp.path(), "r1", &e).unwrap();
        assert_eq!(rep1.cas_hits, 0);
        assert!(rep1.embedded > 0);
        // flush() ran: the model file exists and reloads with vocab.
        assert!(RandomIndexingEmbedder::model_path(tmp.path()).is_file());
        assert!(fresh(tmp.path()).vocab_len() > 0);

        // rindex recomputes instead of caching vectors (SPEC-P10): the
        // shared chunks are misses the cache knows, so they are embedded
        // again but not observed again
        let rep2 = embed_repo(&shards, tmp.path(), "r2", &e).unwrap();
        assert_eq!(rep2.cas_hits, 0, "seen-only cache: {rep2:?}");
        assert!(rep2.cas_known > 0, "shared chunks are known: {rep2:?}");
        assert!(rep2.cas_known < rep2.chunks);

        // The rindex-v2 vec index is queryable and ranks the auth chunk
        // first for an authentication-flavored query.
        let idx = crate::index::VecSet::open(&tmp.path().join("vec"), "rindex-v2")
            .unwrap()
            .unwrap();
        assert!(idx.len() >= 3);
        let q = e
            .embed(&["user authentication login".to_string()])
            .unwrap()
            .remove(0);
        let hits = idx.search(&q, 3);
        assert!(!hits.is_empty());
        let top = idx.row_meta(hits[0].0);
        assert_eq!(top.repo, "r1");
        assert!(top.path.contains("auth.rs"), "top hit: {top:?}");

        // embed_all over both repos works: every chunk is known to the
        // seen-only cache, so nothing is observed twice.
        let reps = embed_all(&shards, tmp.path(), &e).unwrap();
        assert_eq!(reps.len(), 2);
        assert!(reps.iter().all(|r| r.cas_hits == 0 && r.cas_known + r.carried == r.chunks), "{reps:?}");
    }

    // ---------------------------------------------------------------------
    // SPEC-P5 B1/B2 tests
    // ---------------------------------------------------------------------

    /// Bridge corpus: "authentication" and "login" co-occur with shared
    /// neighbors (oauth/password/signin/identity) so their context vectors
    /// become mutual nearest neighbors; chunk "login password session"
    /// shares no token with an "authentication" query.
    const BRIDGE_CORPUS: [&str; 8] = [
        "login password session",
        "database pool connection",
        "authentication login oauth",
        "login authentication oauth",
        "authentication login password",
        "login authentication password",
        "authentication signin identity",
        "login signin identity",
    ];

    fn bridge_embedder(dir: &Path) -> RandomIndexingEmbedder {
        let e = fresh(dir);
        let texts: Vec<String> = BRIDGE_CORPUS.iter().map(|s| s.to_string()).collect();
        e.observe(&texts).unwrap();
        e
    }

    /// SPEC-P5 B1: a token with df > max(100, 40% n_texts_seen) is cut at
    /// embed time — its label AND context contribute nothing.
    #[test]
    fn hub_cut_excludes_token_at_embed_time() {
        let tmp = tempfile::tempdir().unwrap();
        let e = fresh(tmp.path());
        // 200 texts; "hubword" (and the "hubword hubword" bigram) appears
        // in 120 = 60% > max(100, 40% * 200 = 80) = 100.
        let texts: Vec<String> = (0..200)
            .map(|i| {
                if i < 120 {
                    format!("hubword hubword filler{}", i % 10)
                } else {
                    format!("rarex rarey topic{i}")
                }
            })
            .collect();
        e.observe(&texts).unwrap();
        // Probe vector with ONLY the hub token is the zero vector, exactly
        // as if the token were absent (presence/absence invariance). The
        // bigram probe stays zero because the bigram is hub-cut too.
        let v1 = e.embed(&["hubword".to_string()]).unwrap().remove(0);
        assert!(v1.iter().all(|&x| x == 0.0), "hub token must embed to zero");
        let v2 = e.embed(&["hubword hubword".to_string()]).unwrap().remove(0);
        assert!(
            v2.iter().all(|&x| x == 0.0),
            "hub-cut bigram must embed to zero"
        );
        // Non-hub tokens are unaffected: a rare text still embeds unit-norm.
        let vr = e.embed(&["rarex rarey".to_string()]).unwrap().remove(0);
        assert!(dot(&vr, &vr) > 0.99);

        // Control: the SAME construction with df = 90 (below the cutoff of
        // max(100, 0.4*200) = 100) is NOT cut — the token embeds nonzero.
        let tmp2 = tempfile::tempdir().unwrap();
        let e2 = fresh(tmp2.path());
        let texts2: Vec<String> = (0..200)
            .map(|i| {
                if i < 90 {
                    format!("hubword filler{}", i % 10)
                } else {
                    format!("rarex rarey topic{i}")
                }
            })
            .collect();
        e2.observe(&texts2).unwrap();
        let vc = e2.embed(&["hubword".to_string()]).unwrap().remove(0);
        assert!(
            vc.iter().any(|&x| x != 0.0),
            "df=90 < cutoff=100 must not be hub-cut"
        );
    }

    /// SPEC-P5 B1: header tokens (before the first '\n') weigh 2.5 in the
    /// pooling sum, so a probe term placed in the header aligns
    /// measurably better than the same term placed in the body.
    #[test]
    fn header_field_weighted_2_5() {
        let tmp = tempfile::tempdir().unwrap();
        let e = fresh(tmp.path());
        let texts: Vec<String> = [
            "zqxterm alpha beta",
            "gamma delta epsilon",
            "padding filler words here",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        e.observe(&texts).unwrap();
        let probe = e.embed(&["zqxterm".to_string()]).unwrap().remove(0);
        let v_head = e
            .embed(&["zqxterm\npadding padding padding".to_string()])
            .unwrap()
            .remove(0);
        let v_body = e
            .embed(&["padding padding padding\nzqxterm".to_string()])
            .unwrap()
            .remove(0);
        let s_head = dot(&probe, &v_head);
        let s_body = dot(&probe, &v_body);
        assert!(
            s_head > s_body,
            "term-in-header must beat term-in-body: {s_head} vs {s_body}"
        );
        // Single-line text is all "header": the uniform 2.5 scale cancels
        // under L2 normalization, so header-vs-body only matters when both
        // fields are present. Sanity: both vectors stay unit-norm.
        assert!((dot(&v_head, &v_head).sqrt() - 1.0).abs() < 1e-4);
        assert!((dot(&v_body, &v_body).sqrt() - 1.0).abs() < 1e-4);
    }

    /// SPEC-P5 B2: nearest_tokens finds corpus synonyms by context-vector
    /// cosine (login <-> authentication over the bridge fixture), filters
    /// by min_sim, and excludes the query token itself and hub-cut tokens.
    #[test]
    fn nearest_tokens_finds_synonyms() {
        let tmp = tempfile::tempdir().unwrap();
        let e = bridge_embedder(tmp.path());
        let nn = e.nearest_tokens("authentication", 8, 0.35);
        let terms: Vec<&str> = nn.iter().map(|(t, _)| t.as_str()).collect();
        assert!(
            terms.contains(&"login"),
            "login must be a neighbor of authentication: {nn:?}"
        );
        assert!(
            !terms.contains(&"authentication"),
            "query token excluded: {nn:?}"
        );
        // min_sim respected and results sorted by descending sim.
        assert!(nn.iter().all(|&(_, s)| s >= 0.35), "{nn:?}");
        assert!(
            nn.windows(2).all(|w| w[0].1 >= w[1].1),
            "sorted desc: {nn:?}"
        );
        // The reverse direction holds too (symmetric fixture).
        let nn = e.nearest_tokens("login", 8, 0.35);
        assert!(
            nn.iter().any(|(t, _)| t == "authentication"),
            "authentication must be a neighbor of login: {nn:?}"
        );
        // Unknown tokens and tokens with no context have no neighbors.
        assert!(e.nearest_tokens("zzqjv-never-seen", 5, 0.0).is_empty());
    }

    /// SPEC-P5 B2: expand_query appends top-3 nearest neighbors (min_sim
    /// 0.35) of each content token after the original query text, deduped;
    /// stopwords / df<2 / short tokens are not expanded.
    #[test]
    fn expand_query_appends_synonyms() {
        let tmp = tempfile::tempdir().unwrap();
        let e = bridge_embedder(tmp.path());
        let x = e.expand_query("authentication");
        assert!(
            x.starts_with("authentication"),
            "original tokens first: {x:?}"
        );
        // The bridging synonym rides in via the top-3 neighbor terms.
        assert!(
            x.split_whitespace().any(|w| w == "login"),
            "expansion must add the login bridge: {x:?}"
        );
        // Each appended neighbor term appears exactly once (dedup).
        let term_count = |term: &str| {
            x["authentication".len()..]
                .split_whitespace()
                .collect::<Vec<_>>()
                .windows(term.split_whitespace().count())
                .filter(|w| w.join(" ") == term)
                .count()
        };
        assert_eq!(term_count("authentication login"), 1, "{x:?}");
        // Stopword-only / content-free queries are returned unchanged.
        assert_eq!(e.expand_query("the of and"), "the of and");
        // df-1 tokens (never observed) are not expanded.
        assert_eq!(e.expand_query("zzqjv"), "zzqjv");
        // NL query: originals preserved verbatim and first.
        let x2 = e.expand_query("how does authentication work");
        assert!(x2.starts_with("how does authentication work"), "{x2:?}");
        assert!(x2.split_whitespace().any(|w| w == "login"), "{x2:?}");
    }

    /// SPEC-P5 B2 end-to-end: semantic search WITH expansion finds the
    /// zero-lexical-overlap chunk whose only path from the query token is
    /// a 2-hop bridge (query "authentication" -> expansion neighbor
    /// "login" -> chunk "login password session").
    #[test]
    fn expansion_bridges_two_hop_end_to_end() {
        use crate::index::{VecIndex, VecRowMeta};

        let tmp = tempfile::tempdir().unwrap();
        let e = bridge_embedder(tmp.path());
        // Indexed chunks: the bridge corpus trains the model, but only
        // these chunks are indexed — none contains the query token
        // "authentication" (zero lexical overlap with the query).
        let chunks = [
            ("a/login.rs", "login password session"),
            ("b/db.rs", "database pool connection"),
            ("c/greek.rs", "gamma delta epsilon"),
            ("d/letters.rs", "rho sigma tau"),
        ];
        let rows: Vec<(VecRowMeta, Vec<f32>)> = chunks
            .iter()
            .map(|(path, text)| {
                let v = e.embed(&[text.to_string()]).unwrap().remove(0);
                (
                    VecRowMeta {
                        chunk_hash: [0u8; 16],
                        repo: "r".to_string(),
                        path: path.to_string(),
                        start_line: 1,
                        end_line: 1,
                    },
                    v,
                )
            })
            .collect();
        let idx = VecIndex::create(&tmp.path().join("vec"), e.model_id(), e.dim(), rows).unwrap();

        // indexio-query semantic path behavior: expand, then embed, then search.
        let expanded = e.expand_query("authentication");
        assert_ne!(expanded, "authentication", "expansion must fire");
        let qv = e.embed(&[expanded]).unwrap().remove(0);
        let hits = idx.search(&qv, 4);
        assert_eq!(
            idx.row_meta(hits[0].0).path,
            "a/login.rs",
            "expanded query must surface the login chunk first: {:?}",
            hits.iter()
                .map(|&(r, s)| (idx.row_meta(r).path.clone(), s))
                .collect::<Vec<_>>()
        );

        // The 2-hop bridge is attributable to expansion: the unexpanded
        // query scores the target chunk strictly lower.
        let q_plain = e.embed(&["authentication".to_string()]).unwrap().remove(0);
        let target_v = e.embed(&[chunks[0].1.to_string()]).unwrap().remove(0);
        let s_plain = dot(&q_plain, &target_v);
        let s_expanded = dot(&qv, &target_v);
        assert!(
            s_expanded > s_plain,
            "expansion must strengthen the bridge: {s_expanded} vs {s_plain}"
        );
    }

    /// SPEC-P5 B2: expansion is bit-for-bit deterministic across fresh
    /// embedders and across flush/reload.
    #[test]
    fn expansion_determinism_bitforbit() {
        let tmp = tempfile::tempdir().unwrap();
        let e1 = bridge_embedder(tmp.path());
        // A second embedder built from scratch in its own data dir.
        let t2 = tempfile::tempdir().unwrap();
        let e2 = bridge_embedder(t2.path());
        for q in ["authentication", "login", "database", "how does authentication work"] {
            assert_eq!(
                e1.expand_query(q),
                e2.expand_query(q),
                "expansion must be deterministic for {q:?}"
            );
        }
        // flush + reopen reproduces expansion and embeddings bit-for-bit.
        let before = e1.expand_query("authentication");
        let v_before = e1.embed(&[before.clone()]).unwrap();
        e1.flush().unwrap();
        let e3 = fresh(tmp.path());
        let after = e3.expand_query("authentication");
        assert_eq!(before, after, "expansion survives persistence");
        assert_eq!(v_before, e3.embed(&[after]).unwrap());
    }

    /// SPEC-P5 B1: open() verifies the persisted dim (2048) and errors on
    /// a mismatch; v1 model files are ignored (clean break).
    #[test]
    fn dim_mismatch_errors_and_v1_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let path = RandomIndexingEmbedder::model_path(tmp.path());
        assert!(path.ends_with("sem/rindex-v2.rimodel"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // A v2-named file with the v1 dim must error, not silently load.
        let bogus = ModelDisk {
            dim: 1024,
            n_texts_seen: 0,
            vocab: Vec::new(),
        };
        std::fs::write(&path, bincode::serialize(&bogus).unwrap()).unwrap();
        let err = match RandomIndexingEmbedder::open(tmp.path()) {
            Ok(_) => panic!("dim-mismatched v2 model file must error"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("dim 1024"),
            "dim mismatch must error: {err}"
        );
        // A legacy v1 file at the v1 path is ignored entirely.
        std::fs::remove_file(&path).unwrap();
        let v1 = tmp.path().join("sem").join("rindex-v1.rimodel");
        std::fs::write(&v1, bincode::serialize(&bogus).unwrap()).unwrap();
        let e = RandomIndexingEmbedder::open(tmp.path()).unwrap();
        assert_eq!(e.vocab_len(), 0, "v1 model file ignored");
    }
}
