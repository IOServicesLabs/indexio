//! HNSW (Hierarchical Navigable Small World) ANN graph over the f32 vectors
//! already stored in `.civec` (SPEC-P3 §1).
//!
//! Pure safe Rust, no external dependencies. The graph is a rebuildable
//! sidecar — deleting `.cihnsw` only costs a rebuild, and `VecIndex` falls
//! back to the flat binary-quantized search path when it is absent.
//!
//! On-disk: `<data_dir>/vec/<model_id>.cihnsw`
//! ```text
//! magic "CIHNSW1" (8B) | u32 n_nodes | u32 m | u32 m0 | u32 entry_point
//! | per node: u8 level, then per level l=0..=level: u32 nbr_count + nbr_count x u32
//! ```
//! (little-endian; `entry_point` = `u32::MAX` when the graph is empty).
//!
//! Determinism: builds insert nodes in row order 0..n and draw each node's
//! level from a xorshift64 PRNG seeded from `(n, dim)` — no `rand` dep, so
//! identical input vectors always yield byte-identical graphs.
//!
//! Distance: `1 - dot` on L2-normalized vectors (cosine distance; smaller =
//! better). Neighbor pruning uses the paper's diversity heuristic (algorithm
//! 4, as in hnswlib's `getNeighborsByHeuristic2`, no backfill) — one of the
//! two strategies SPEC-P3 §1 allows; see [`select_neighbors`].

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{anyhow, ensure, Context};

use crate::embed::dot;

/// Magic is the 7 visible bytes of "CIHNSW1" NUL-padded to the spec'd 8B.
const MAGIC: &[u8; 8] = b"CIHNSW1\0";
/// Sentinel for "no entry point" (empty graph).
const NO_ENTRY: u32 = u32::MAX;
/// Hard cap on node levels (u8 on disk; generous headroom over ln-based draws).
const MAX_LEVEL: u8 = 64;

/// Build/search tuning for [`Hnsw`].
#[derive(Clone, Debug)]
pub struct HnswOptions {
    /// Max neighbors per node at levels >= 1.
    pub m: usize,
    /// Max neighbors per node at level 0.
    pub m0: usize,
    /// Beam width during graph construction.
    pub ef_construction: usize,
    /// Row count at/above which `VecIndex` builds a graph (used by
    /// `index::IndexOptions::default`; `INDEXIO_HNSW_THRESHOLD` overrides). The
    /// build is single-threaded and costs minutes per 100k rows, while the
    /// flat binary prescan is fast well past that size.
    pub threshold: usize,
}

impl Default for HnswOptions {
    fn default() -> Self {
        HnswOptions {
            m: 16,
            m0: 32,
            ef_construction: 200,
            threshold: 1_000_000,
        }
    }
}

/// Deterministic xorshift64 PRNG (same recurrence as the crate's test RNG).
struct XorShift64(u64);

impl XorShift64 {
    fn seeded(n: usize, dim: usize) -> Self {
        let mut s = 0x9E3779B97F4A7C15u64;
        s ^= (n as u64).wrapping_mul(0xA24BAED4963EE407);
        s ^= (dim as u64).wrapping_mul(0x9FB21C651E98DF25);
        if s == 0 {
            s = 0x2545F4914F6CDD1D;
        }
        XorShift64(s)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in (0, 1] (never 0, so `-ln(u)` is finite).
    fn next_unit_open1(&mut self) -> f64 {
        // 53-bit mantissa in [0,1); flip to (0,1].
        1.0 - (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// (distance, node) ordered by distance then id — total order via `total_cmp`
/// keeps heap behavior deterministic across platforms.
#[derive(Clone, Copy, PartialEq)]
struct Scored(f32, u32);

impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// Epoch-marked visited set: O(1) clear per search, no per-call allocation
/// when reused across build inserts.
struct Visited {
    marks: Vec<u32>,
    epoch: u32,
}

impl Visited {
    fn new(n: usize) -> Self {
        Visited {
            marks: vec![0; n],
            epoch: 0,
        }
    }

    fn reset(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // Wrapped (practically unreachable): re-zero to stay correct.
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    /// Returns true if already visited this epoch; marks it otherwise.
    fn check_and_mark(&mut self, id: u32) -> bool {
        let m = &mut self.marks[id as usize];
        if *m == self.epoch {
            true
        } else {
            *m = self.epoch;
            false
        }
    }
}

/// Per-node adjacency: `nbrs[l]` = neighbor ids at level `l` (0..=level).
#[derive(Clone, Debug, Default)]
struct NodeLevels {
    nbrs: Vec<Vec<u32>>,
}

impl NodeLevels {
    fn level(&self) -> u8 {
        (self.nbrs.len() as u8).saturating_sub(1)
    }
}

/// In-memory HNSW graph. Row `i` of the source vectors is node `i`.
pub struct Hnsw {
    m: u32,
    m0: u32,
    entry_point: u32,
    nodes: Vec<NodeLevels>,
}

/// Cosine distance on L2-normalized vectors: smaller = better.
#[inline]
fn dist(q: &[f32], v: &[f32]) -> f32 {
    1.0 - dot(q, v)
}

/// Neighbor selection per Malkov & Yashunin algorithm 4 (the diversity
/// heuristic, exactly as in hnswlib's `getNeighborsByHeuristic2`): walk
/// candidates in ascending distance to the query point and keep a candidate
/// only if it is closer to the query than to every already-selected neighbor
/// (a selected neighbor "covers" everything behind it). No backfill of
/// pruned candidates — measured on the SPEC-P3 §1 recall fixture (5,000
/// random 128-dim vectors), the pruned-connection backfill variant dropped
/// recall@10 from 0.955 to 0.940, and `extendCandidates` added build time
/// for zero recall gain, so both are omitted.
///
/// `cands` must be sorted by (distance to query, id) ascending.
fn select_neighbors<'a>(
    cands: &[(f32, u32)],
    mmax: usize,
    row: &dyn Fn(u32) -> &'a [f32],
) -> Vec<u32> {
    if cands.len() <= mmax {
        return cands.iter().map(|&(_, id)| id).collect();
    }
    let mut selected: Vec<u32> = Vec::with_capacity(mmax);
    for &(d_q, id) in cands {
        if selected.len() >= mmax {
            break;
        }
        let v = row(id);
        let diverse = selected.iter().all(|&r| dist(v, row(r)) >= d_q);
        if diverse {
            selected.push(id);
        }
    }
    selected
}

impl Hnsw {
    fn empty(m: usize, m0: usize) -> Self {
        Hnsw {
            m: m as u32,
            m0: m0 as u32,
            entry_point: NO_ENTRY,
            nodes: Vec::new(),
        }
    }

    /// Number of nodes (= rows) in the graph.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Builds a graph from row vectors (row `i` = node `i`). Deterministic:
    /// level assignment draws `level = floor(-ln(u) / ln(m))` with `u` from a
    /// xorshift64 PRNG seeded from `(n, dim)`.
    ///
    /// Vectors must be L2-normalized for distances to be cosine-meaningful.
    /// Defensive: empty or ragged (non-uniform dim) input yields an empty
    /// graph instead of panicking.
    pub fn build(vectors: &[Vec<f32>], opts: &HnswOptions) -> Self {
        if vectors.is_empty() {
            return Self::empty(opts.m, opts.m0);
        }
        let dim = vectors[0].len();
        if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
            return Self::empty(opts.m, opts.m0);
        }
        Self::build_inner(vectors.len(), dim, &|i| vectors[i as usize].as_slice(), opts)
    }

    /// Flat-layout builder for `VecIndex` (n_rows * dim row-major); avoids
    /// materializing per-row `Vec`s at index create time.
    pub(crate) fn build_flat(dim: usize, flat: &[f32], opts: &HnswOptions) -> Self {
        if dim == 0 || flat.is_empty() || flat.len() % dim != 0 {
            return Self::empty(opts.m, opts.m0);
        }
        let n = flat.len() / dim;
        Self::build_inner(n, dim, &|i| &flat[i as usize * dim..(i as usize + 1) * dim], opts)
    }

    fn build_inner<'a>(
        n: usize,
        dim: usize,
        row: &dyn Fn(u32) -> &'a [f32],
        opts: &HnswOptions,
    ) -> Self {
        let m = opts.m.max(1);
        let m0 = opts.m0.max(m);
        let ef_c = opts.ef_construction.max(1);
        let inv_ln_m = 1.0 / (m as f64).ln();

        let mut g = Hnsw {
            m: m as u32,
            m0: m0 as u32,
            entry_point: NO_ENTRY,
            nodes: Vec::with_capacity(n),
        };
        let mut rng = XorShift64::seeded(n, dim);
        let mut max_level = 0u8;
        let mut vis = Visited::new(n);

        for i in 0..n {
            let u = rng.next_unit_open1();
            let level = ((-u.ln() * inv_ln_m).floor() as u64).min(MAX_LEVEL as u64) as u8;
            g.nodes.push(NodeLevels {
                nbrs: vec![Vec::new(); level as usize + 1],
            });
            let id = i as u32;
            let qi = row(i as u32);

            if g.entry_point == NO_ENTRY {
                g.entry_point = id;
                max_level = level;
                continue;
            }

            // Greedy descent from the entry point down to level+1 (ef = 1).
            let mut w = g.entry_point;
            let mut lc = max_level;
            while lc > level {
                let best = g.search_layer(qi, row, w, 1, lc, &mut vis);
                w = best[0].1; // ef>=1 always yields the entry itself at worst
                lc -= 1;
            }

            // ef_construction beam at each level the new node joins.
            let top = max_level.min(level);
            for lc in (0..=top).rev() {
                let cands = g.search_layer(qi, row, w, ef_c, lc, &mut vis);
                let mmax = if lc == 0 { m0 } else { m };
                let sel = select_neighbors(&cands, mmax, row);
                g.nodes[i].nbrs[lc as usize] = sel.clone();
                for &nb in &sel {
                    let nbrs = &mut g.nodes[nb as usize].nbrs[lc as usize];
                    nbrs.push(id);
                    if nbrs.len() > mmax {
                        // Prune back to M with the same diversity heuristic,
                        // distances measured to the owner node `nb`.
                        let vnb = row(nb);
                        let mut d: Vec<(f32, u32)> = nbrs
                            .iter()
                            .map(|&x| (dist(vnb, row(x)), x))
                            .collect();
                        d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                        *nbrs = select_neighbors(&d, mmax, row);
                    }
                }
                // Entry point for the next lower level: nearest candidate.
                w = cands[0].1;
            }

            if level > max_level {
                max_level = level;
                g.entry_point = id;
            }
        }
        g
    }

    /// Beam search on one level. Returns up to `ef` candidates sorted by
    /// (distance, id) ascending; always non-empty (contains `ep` at worst).
    fn search_layer<'a>(
        &self,
        q: &[f32],
        row: &dyn Fn(u32) -> &'a [f32],
        ep: u32,
        ef: usize,
        lc: u8,
        vis: &mut Visited,
    ) -> Vec<(f32, u32)> {
        let ef = ef.max(1);
        vis.reset();
        vis.check_and_mark(ep);
        let d_ep = dist(q, row(ep));
        // candidates: min-heap (closest first); results: max-heap (worst on top).
        let mut cand: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        let mut res: BinaryHeap<Scored> = BinaryHeap::new();
        cand.push(Reverse(Scored(d_ep, ep)));
        res.push(Scored(d_ep, ep));

        while let Some(Reverse(c)) = cand.pop() {
            let worst = res.peek().map(|s| s.0).unwrap_or(f32::INFINITY);
            if res.len() >= ef && c.0 > worst {
                break;
            }
            for &nb in &self.nodes[c.1 as usize].nbrs[lc as usize] {
                if vis.check_and_mark(nb) {
                    continue;
                }
                let d = dist(q, row(nb));
                let worst = res.peek().map(|s| s.0).unwrap_or(f32::INFINITY);
                if res.len() < ef || d < worst {
                    cand.push(Reverse(Scored(d, nb)));
                    res.push(Scored(d, nb));
                    if res.len() > ef {
                        res.pop();
                    }
                }
            }
        }
        let mut out: Vec<(f32, u32)> = res.into_iter().map(|s| (s.0, s.1)).collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        out
    }

    /// Generic top-of-graph descent + level-0 beam over a row accessor.
    fn search_inner<'a>(
        &self,
        row: &dyn Fn(u32) -> &'a [f32],
        q: &[f32],
        ef: usize,
    ) -> Vec<u32> {
        if self.entry_point == NO_ENTRY || ef == 0 {
            return Vec::new();
        }
        let mut vis = Visited::new(self.nodes.len());
        let mut w = self.entry_point;
        let top = self.nodes[w as usize].level();
        // Greedy descent (ef = 1) from the top level down to level 1.
        for lc in (1..=top).rev() {
            let best = self.search_layer(q, row, w, 1, lc, &mut vis);
            w = best[0].1;
        }
        let res = self.search_layer(q, row, w, ef, 0, &mut vis);
        res.into_iter().map(|(_, id)| id).collect()
    }

    /// Approximate top-`ef` candidate row ids for query `q` (L2-normalized;
    /// dot = cosine). `vectors` must be the same rows the graph was built
    /// from (row i = node i). Returns ids sorted by ascending distance.
    /// Defensive: returns empty on empty input or node-count mismatch.
    pub fn search(&self, vectors: &[Vec<f32>], q: &[f32], ef: usize) -> Vec<u32> {
        if vectors.len() != self.nodes.len() {
            return Vec::new();
        }
        self.search_inner(&|i| vectors[i as usize].as_slice(), q, ef)
    }

    /// Flat-layout search for `VecIndex` (n_rows * dim row-major).
    pub(crate) fn search_flat(&self, dim: usize, flat: &[f32], q: &[f32], ef: usize) -> Vec<u32> {
        if dim == 0 || q.len() != dim || flat.len() != self.nodes.len() * dim {
            return Vec::new();
        }
        self.search_inner(&|i| &flat[i as usize * dim..(i as usize + 1) * dim], q, ef)
    }

    /// Serializes the graph via tmp+rename (atomic).
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.m.to_le_bytes());
        out.extend_from_slice(&self.m0.to_le_bytes());
        out.extend_from_slice(&self.entry_point.to_le_bytes());
        for node in &self.nodes {
            out.push(node.level());
            for nbrs in &node.nbrs {
                out.extend_from_slice(&(nbrs.len() as u32).to_le_bytes());
                for &nb in nbrs {
                    out.extend_from_slice(&nb.to_le_bytes());
                }
            }
        }
        let tmp = path.with_extension("cihnsw.tmp");
        {
            let mut f = fs::File::create(&tmp)
                .with_context(|| format!("create {}", tmp.display()))?;
            f.write_all(&out)?;
            f.sync_all().ok();
        }
        fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))?;
        Ok(())
    }

    /// Loads a graph written by [`Hnsw::save`].
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let buf = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let bad = || anyhow!("{}: corrupt .cihnsw file", path.display());
        let mut off = 0usize;
        let take = |off: &mut usize, n: usize| -> anyhow::Result<&[u8]> {
            let s = buf.get(*off..off.checked_add(n).ok_or_else(bad)?).ok_or_else(bad)?;
            *off += n;
            Ok(s)
        };
        let u32_at = |off: &mut usize| -> anyhow::Result<u32> {
            Ok(u32::from_le_bytes(take(off, 4)?.try_into().unwrap()))
        };

        ensure!(take(&mut off, 8)? == MAGIC, "bad .cihnsw magic");
        let n_nodes = u32_at(&mut off)? as usize;
        let m = u32_at(&mut off)?;
        let m0 = u32_at(&mut off)?;
        let entry_point = u32_at(&mut off)?;
        ensure!(
            n_nodes == 0 || entry_point < n_nodes as u32,
            "entry_point out of range"
        );

        let mut nodes = Vec::with_capacity(n_nodes);
        for _ in 0..n_nodes {
            let level = take(&mut off, 1)?[0];
            ensure!(level <= MAX_LEVEL, "level {level} exceeds cap");
            let mut nbrs = Vec::with_capacity(level as usize + 1);
            for _ in 0..=level {
                let cnt = u32_at(&mut off)? as usize;
                let raw = take(
                    &mut off,
                    cnt.checked_mul(4).ok_or_else(bad)?,
                )?;
                let mut list = Vec::with_capacity(cnt);
                for c in raw.chunks_exact(4) {
                    let id = u32::from_le_bytes(c.try_into().unwrap());
                    ensure!((id as usize) < n_nodes, "neighbor id out of range");
                    list.push(id);
                }
                nbrs.push(list);
            }
            nodes.push(NodeLevels { nbrs });
        }
        ensure!(off == buf.len(), "trailing bytes in .cihnsw");

        Ok(Hnsw {
            m,
            m0,
            entry_point: if n_nodes == 0 { NO_ENTRY } else { entry_point },
            nodes,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 data PRNG (no rand dep), same recurrence as
    /// the index tests.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            // uniform in [-1, 1)
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 1.0
        }
    }

    fn norm(mut v: Vec<f32>) -> Vec<f32> {
        let n = dot(&v, &v).sqrt();
        if n > 0.0 {
            for x in &mut v {
                *x /= n;
            }
        }
        v
    }

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Rng(seed);
        (0..n)
            .map(|_| norm((0..dim).map(|_| rng.next_f32()).collect()))
            .collect()
    }

    fn brute_force_topk(vectors: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(u32, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, dot(q, v)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// SPEC-P3 §1: recall@10 >= 0.95 vs exact brute force on 5,000 random
    /// 128-dim L2-normalized vectors, using the production ef = max(k*10, 100).
    #[test]
    fn recall_at_10_vs_brute_force_5000x128() {
        let (n, dim, k) = (5_000usize, 128usize, 10usize);
        let vectors = random_vectors(n, dim, 0x9E3779B97F4A7C15);
        let queries = random_vectors(20, dim, 0xDEADBEEFCAFEF00D);

        let h = Hnsw::build(&vectors, &HnswOptions::default());
        assert_eq!(h.n_nodes(), n);

        let ef = (k * 10).max(100);
        let mut total = 0f64;
        for q in &queries {
            let approx = h.search(&vectors, q, ef);
            assert_eq!(approx.len(), ef, "ef candidates expected on full graph");
            let top10: std::collections::BTreeSet<u32> =
                approx.iter().copied().take(k).collect();
            let truth: std::collections::BTreeSet<u32> =
                brute_force_topk(&vectors, q, k).into_iter().collect();
            let hit = top10.intersection(&truth).count();
            total += hit as f64 / k as f64;
        }
        let recall = total / queries.len() as f64;
        assert!(
            recall >= 0.95,
            "recall@10 {recall:.4} below 0.95 (queries={})",
            queries.len()
        );
    }

    /// SPEC-P3 §1: save/load round-trip yields identical search results.
    #[test]
    fn save_load_roundtrip_identical_results() {
        let tmp = tempfile::tempdir().unwrap();
        let (n, dim) = (1_000usize, 64usize);
        let vectors = random_vectors(n, dim, 0x1111222233334444);
        let queries = random_vectors(5, dim, 0x5555666677778888);

        let h = Hnsw::build(&vectors, &HnswOptions::default());
        let path = tmp.path().join("hash-v1.cihnsw");
        h.save(&path).unwrap();
        let h2 = Hnsw::load(&path).unwrap();
        assert_eq!(h2.n_nodes(), n);

        for (i, q) in queries.iter().enumerate() {
            for ef in [10usize, 100usize] {
                let a = h.search(&vectors, q, ef);
                let b = h2.search(&vectors, q, ef);
                assert_eq!(a, b, "query {i} ef {ef}: results must survive round-trip");
            }
        }

        // Corrupt files are errors, not panics.
        fs::write(&path, b"CIHNSW1\0garbage").unwrap();
        assert!(Hnsw::load(&path).is_err());
        fs::write(&path, b"NOMAGIC!").unwrap();
        assert!(Hnsw::load(&path).is_err());
    }

    /// Deterministic builds: identical input -> byte-identical graph file.
    #[test]
    fn build_is_deterministic() {
        let tmp = tempfile::tempdir().unwrap();
        let vectors = random_vectors(500, 32, 0xABCDEF0123456789);
        let opts = HnswOptions::default();
        let h1 = Hnsw::build(&vectors, &opts);
        let h2 = Hnsw::build(&vectors, &opts);
        let p1 = tmp.path().join("a.cihnsw");
        let p2 = tmp.path().join("b.cihnsw");
        h1.save(&p1).unwrap();
        h2.save(&p2).unwrap();
        assert_eq!(
            fs::read(&p1).unwrap(),
            fs::read(&p2).unwrap(),
            "same vectors + seed(n, dim) must serialize identically"
        );
    }

    /// Defensive API behavior on degenerate input (no panics).
    #[test]
    fn degenerate_inputs_are_safe() {
        let opts = HnswOptions::default();
        let empty = Hnsw::build(&[], &opts);
        assert_eq!(empty.n_nodes(), 0);
        assert!(empty.search(&[], &[1.0, 2.0], 10).is_empty());

        // Ragged rows -> empty graph, no panic.
        let ragged = Hnsw::build(&[vec![1.0, 0.0], vec![1.0]], &opts);
        assert_eq!(ragged.n_nodes(), 0);

        // Node-count mismatch -> empty result.
        let vectors = random_vectors(50, 8, 42);
        let h = Hnsw::build(&vectors, &opts);
        assert!(h.search(&vectors[..10], &[0.0; 8], 10).is_empty());
    }
}
