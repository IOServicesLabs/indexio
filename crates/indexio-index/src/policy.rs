//! Which shards a compaction merges.
//!
//! Shards are ordered by age and a doc of a newer shard shadows the same
//! (repo, path) in an older one, so only a run of ADJACENT ages can be
//! merged: the compound shard is dated by its newest input and then orders
//! exactly where the run did. Within that constraint the policy is tiered,
//! as in an LSM tree: the small young delta shards are merged among
//! themselves, and a large older shard is rewritten only once the shards
//! merged with it hold a fair share of its size.

/// Tombstoned share of all docs above which a merge rewrites the shard that
/// holds the most of them (dropping the dead docs) instead of the cheapest
/// run.
pub(crate) const MERGE_DEAD_SHARE: f64 = 0.3;

/// Size ratio of the tiers: a run being merged takes its older neighbour in
/// once the run holds at least `1 / MERGE_SIZE_RATIO` of the neighbour's
/// bytes. Every rewrite of a shard therefore adds at least half its size
/// to it, so a byte is rewritten O(log(index / delta)) times while the index
/// grows, and the shards a merge leaves behind grow at least threefold with
/// age (few shards). 1 rewrites a little less and leaves more shards; 3
/// the reverse (see the simulation in the tests).
pub(crate) const MERGE_SIZE_RATIO: u64 = 2;

/// The background compaction of a server starts once more than this many
/// shards exist. Every lookup probes each shard, and 0.1.14 servers sharing
/// the data dir start THEIR merge above 12 shards (the cheapest run that
/// leaves 6, which with a handful of large shards rewrites one of them for
/// a few small deltas): staying below that keeps them idle.
pub const COMPACT_ABOVE: usize = 10;

/// What the background compaction passes to [`ShardSet::merge`] as
/// `max_shards`. Close to [`COMPACT_ABOVE`] on purpose: the run a merge
/// must take is `shards - max_shards + 1` long, and with a lower bound than
/// the data dir has large shards that run cannot consist of deltas alone.
///
/// [`ShardSet::merge`]: crate::ShardSet::merge
pub const COMPACT_TO: usize = 8;

/// What the policy knows of one shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShardStat {
    /// File bytes less the fixed-size tombstone section: what holds docs.
    /// (A three-doc delta is 64 KiB of tombstone section and a few kB of
    /// everything else; with the section counted, a handful of them would
    /// look as large as a shard of hundreds of small docs.)
    pub bytes: u64,
    pub docs: u64,
    /// Tombstoned docs.
    pub dead: u64,
    /// Creation time (unix seconds) the set orders by.
    pub created: u64,
}

/// The run to merge, `lo..hi` of `stats` (newest first), for a set of more
/// than `max_shards` shards. It is at least `stats.len() - max_shards + 1`
/// long, so at most `max_shards` shards remain.
///
/// - The run starts as the cheapest (fewest bytes) one of that length, as
///   it did up to 0.1.14: delta shards from refreshes and hooks are the
///   small young ones, and the oldest run would rewrite the big base shards
///   with them.
/// - It then grows into older shards while it holds at least
///   `1 / MERGE_SIZE_RATIO` of the next one (the size-ratio rule): small
///   compounds are folded into larger ones as they catch up, and a large
///   shard is rewritten when the merge adds that share to it, not for a few
///   deltas. Newer neighbours are left alone; they are within `max_shards`.
/// - Once tombstoned docs are over [`MERGE_DEAD_SHARE`] of all docs, the
///   run is instead built around the shard with the most of them, with the
///   adjacent shards that are over the share themselves and the cheapest
///   neighbours up to the required length (up to 0.1.14: the oldest run).
/// - The run never starts between two shards created in the same second:
///   the compound is dated by its newest input and has the newest file
///   name, so it would order above a kept shard of that second.
pub(crate) fn merge_window(stats: &[ShardStat], max_shards: usize) -> (usize, usize) {
    let n = stats.len();
    debug_assert!(max_shards >= 1 && n > max_shards);
    let need = n - max_shards.max(1) + 1;
    let docs: u64 = stats.iter().map(|s| s.docs).sum();
    let dead: u64 = stats.iter().map(|s| s.dead).sum();
    let (mut lo, hi) = if over_dead_share(dead, docs) { dead_window(stats, need) } else { tier_window(stats, need) };
    while lo > 0 && stats[lo - 1].created == stats[lo].created {
        lo -= 1;
    }
    (lo, hi)
}

fn over_dead_share(dead: u64, docs: u64) -> bool {
    docs > 0 && dead as f64 / docs as f64 > MERGE_DEAD_SHARE
}

/// The cheapest run of `need` adjacent shards (the older one of two that
/// cost the same).
fn cheapest_run(stats: &[ShardStat], need: usize) -> (usize, usize) {
    let n = stats.len();
    (0..=n - need)
        .min_by_key(|&lo| (stats[lo..lo + need].iter().map(|s| s.bytes).sum::<u64>(), std::cmp::Reverse(lo)))
        .map_or((n - need, n), |lo| (lo, lo + need))
}

fn tier_window(stats: &[ShardStat], need: usize) -> (usize, usize) {
    let (lo, mut hi) = cheapest_run(stats, need);
    let mut sum: u64 = stats[lo..hi].iter().map(|s| s.bytes).sum();
    while hi < stats.len() && stats[hi].bytes <= MERGE_SIZE_RATIO.saturating_mul(sum) {
        sum = sum.saturating_add(stats[hi].bytes);
        hi += 1;
    }
    (lo, hi)
}

fn dead_window(stats: &[ShardStat], need: usize) -> (usize, usize) {
    let n = stats.len();
    let over = |i: usize| over_dead_share(stats[i].dead, stats[i].docs);
    // the total is over the share, so at least one shard is
    let seed = (0..n).filter(|&i| over(i)).max_by_key(|&i| (stats[i].dead, i)).unwrap_or(n - 1);
    let (mut lo, mut hi) = (seed, seed + 1);
    while lo > 0 && over(lo - 1) {
        lo -= 1;
    }
    while hi < n && over(hi) {
        hi += 1;
    }
    while hi - lo < need {
        match (lo.checked_sub(1).map(|i| stats[i].bytes), stats.get(hi).map(|s| s.bytes)) {
            (Some(newer), Some(older)) if newer < older => lo -= 1,
            (_, Some(_)) => hi += 1,
            (Some(_), None) => lo -= 1,
            (None, None) => break,
        }
    }
    (lo, hi)
}

/// The window 0.1.14 merges (kept for the simulation): the cheapest run of
/// exactly the required length, or the oldest one once tombstoned docs are
/// over the share.
#[cfg(test)]
pub(crate) fn merge_window_0_1_14(stats: &[ShardStat], max_shards: usize) -> (usize, usize) {
    let n = stats.len();
    let need = n - max_shards + 1;
    let docs: u64 = stats.iter().map(|s| s.docs).sum();
    let dead: u64 = stats.iter().map(|s| s.dead).sum();
    if over_dead_share(dead, docs) {
        return (n - need, n);
    }
    cheapest_run(stats, need)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1_000_000;

    fn shard(bytes: u64, created: u64) -> ShardStat {
        ShardStat { bytes, docs: (bytes / 20_000).max(1), dead: 0, created }
    }

    /// Shards newest first from sizes in MB, one second apart.
    fn layout(mb: &[f64]) -> Vec<ShardStat> {
        let n = mb.len() as u64;
        mb.iter().enumerate().map(|(i, m)| shard((m * MB as f64) as u64, 1_000 + n - i as u64)).collect()
    }

    /// The shards of the live data dir on 2026-11-20 when a 0.1.14 server
    /// was about to compact (newest first, MB).
    const LIVE: [f64; 13] = [0.25, 0.28, 0.26, 0.28, 0.26, 0.72, 0.33, 123.24, 73.61, 85.16, 1.27, 0.44, 210.53];

    #[test]
    fn deltas_merge_among_themselves_where_0_1_14_rewrote_the_large_compound() {
        let s = layout(&LIVE);
        // 0.1.14: above 12 shards, the cheapest run that leaves 6 = the
        // seven deltas and the 123 MB compound under them
        assert_eq!(merge_window_0_1_14(&s, 6), (0, 8));
        // now: the cheapest six, then the seventh delta (the run holds more
        // than half of it); the compound is fifty times the run
        assert_eq!(merge_window(&s, COMPACT_TO), (0, 7));
        // a caller that insists on 6 forces the compound into the run (six
        // shards are not deltas); the shards under it then follow by the
        // ratio rule, each no more than twice what is merged above it
        assert_eq!(merge_window(&s, 6), (0, 13));
    }

    #[test]
    fn a_large_shard_is_taken_once_the_run_holds_half_of_it() {
        // four deltas over a 5 MB compound over a 120 MB shard
        let mut s = layout(&[0.5, 0.5, 0.5, 0.5, 5.0, 120.0, 80.0]);
        // 2 MB of deltas: not yet half of the 5 MB compound
        assert_eq!(merge_window(&s, 4), (0, 4));
        // 3 MB of deltas: the compound joins (5 <= 2 * 3), the 120 MB shard
        // does not (120 > 2 * 8)
        s[0].bytes = 1_500_000;
        assert_eq!(merge_window(&s, 4), (0, 5));
        // a small tier of 51 MB under two deltas: the 120 MB shard stays
        let mut s = layout(&[0.5, 0.5, 50.0, 120.0, 80.0]);
        assert_eq!(merge_window(&s, 3), (0, 3));
        // at 71 MB it is taken (120 <= 2 * 71), and then the 80 MB shard
        // under it (80 <= 2 * 191)
        s[2].bytes = 70 * MB;
        assert_eq!(merge_window(&s, 3), (0, 5));
    }

    #[test]
    fn the_run_is_at_least_long_enough_to_leave_max_shards() {
        for n in 2..16usize {
            let mb: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 7) % 5) as f64 * 40.0).collect();
            let s = layout(&mb);
            for max in 1..n {
                let (lo, hi) = merge_window(&s, max);
                assert!(lo < hi && hi <= n, "n={n} max={max}: {lo}..{hi}");
                assert!(n - (hi - lo) < max, "n={n} max={max}: {lo}..{hi} leaves too many");
            }
        }
    }

    #[test]
    fn a_run_does_not_start_inside_a_second() {
        // the cheapest pair is the two oldest deltas, but the delta above
        // them was created in the same second as the newer of the two
        let mut s = layout(&[9.0, 2.0, 1.0, 1.0, 300.0]);
        assert_eq!(merge_window(&s, 4), (2, 4));
        s[1].created = s[2].created;
        assert_eq!(merge_window(&s, 4), (1, 4));
        // an older shard of the same second needs nothing: the compound
        // orders above it, as the run did
        let mut s = layout(&[9.0, 1.0, 1.0, 300.0]);
        s[3].created = s[2].created;
        assert_eq!(merge_window(&s, 3), (1, 3));
    }

    #[test]
    fn tombstones_over_the_share_rewrite_the_shards_that_hold_them() {
        let mut s = layout(&[0.5, 0.5, 0.5, 100.0, 70.0, 200.0]);
        for x in s.iter_mut() {
            x.docs = 1_000;
        }
        // under the share overall: the tiered window
        s[5].dead = 400;
        assert_eq!(merge_window(&s, 4), (0, 3));
        // over it: the shard holding the dead docs, with its cheapest
        // neighbours up to the required length; 0.1.14 took the oldest run,
        // which is the same here...
        s[5].dead = 900;
        s[3].dead = 950;
        s[4].dead = 10;
        assert_eq!(merge_window_0_1_14(&s, 4), (3, 6));
        // ...and now the 100 MB shard (most dead docs) with the two deltas
        // above it, not the clean 70 MB one below; the 200 MB shard is next
        // time's seed
        assert_eq!(merge_window(&s, 4), (1, 4));
        // a seed with nothing older extends toward the newer shards
        s[3].dead = 0;
        s[5].docs = 10_000;
        s[5].dead = 9_000;
        assert_eq!(merge_window(&s, 4), (3, 6));
        // adjacent shards that are over the share themselves come along
        let mut s = layout(&[0.5, 0.5, 50.0, 60.0, 70.0]);
        for x in s.iter_mut() {
            x.docs = 100;
            x.dead = 0;
        }
        s[2].dead = 60;
        s[3].dead = 80;
        s[4].dead = 70;
        assert_eq!(merge_window(&s, 4), (2, 5));
    }

    // ---- simulation -------------------------------------------------------

    /// A server's compaction rule: start above `above` shards, merge the
    /// run `window(stats, to)`.
    #[derive(Clone, Copy)]
    struct Rule {
        above: usize,
        to: usize,
        window: fn(&[ShardStat], usize) -> (usize, usize),
    }
    const OLD: Rule = Rule { above: 12, to: 6, window: merge_window_0_1_14 };
    const NEW: Rule = Rule { above: COMPACT_ABOVE, to: COMPACT_TO, window: merge_window };

    #[derive(Debug, Default)]
    struct Outcome {
        rewritten: u64,
        merges: u32,
        /// Merges of more than 50 MB.
        large: u32,
        largest: u64,
        max_shards: usize,
        shards: Vec<f64>,
    }

    struct Lcg(u64);
    impl Lcg {
        /// Uniform in [0, 1).
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as f64 / (1u64 << 31) as f64
        }
    }

    /// Replay `deltas` arrivals of a 0.1-1.0 MB delta shard on top of
    /// `base` (MB, newest first). After each arrival one server checks the
    /// shard count: a 0.1.14 one with probability `old_share`, else a
    /// current one; it merges while its rule says so. A compound is as large
    /// as its inputs together (nothing is superseded: the worst case for the
    /// bytes rewritten).
    fn simulate(base: &[f64], deltas: usize, old_share: f64) -> Outcome {
        let mut shards = layout(base);
        let mut clock = shards.iter().map(|s| s.created).max().unwrap_or(0);
        let mut rng = Lcg(7);
        let mut out = Outcome { max_shards: shards.len(), ..Default::default() };
        for _ in 0..deltas {
            clock += 180; // every few minutes
            let size = ((0.1 + 0.9 * rng.next()) * MB as f64) as u64;
            shards.insert(0, shard(size, clock));
            out.max_shards = out.max_shards.max(shards.len());
            let rule = if rng.next() < old_share { OLD } else { NEW };
            while shards.len() > rule.above {
                let (lo, hi) = (rule.window)(&shards, rule.to);
                let merged: u64 = shards[lo..hi].iter().map(|s| s.bytes).sum();
                out.rewritten += merged;
                out.merges += 1;
                out.large += (merged > 50 * MB) as u32;
                out.largest = out.largest.max(merged);
                let compound = shard(merged, shards[lo].created);
                shards.splice(lo..hi, [compound]);
            }
        }
        out.shards = shards.iter().map(|s| (s.bytes as f64 / MB as f64 * 10.0).round() / 10.0).collect();
        out
    }

    fn row(name: &str, o: &Outcome) -> String {
        format!(
            "{name:<34} rewritten {:>8.2} GB in {:>4} merges ({:>3} over 50 MB, largest {:>6.0} MB), max {:>2} shards, ends {:?}",
            o.rewritten as f64 / 1e9,
            o.merges,
            o.large,
            o.largest as f64 / 1e6,
            o.max_shards,
            o.shards
        )
    }

    /// The numbers behind the policy (`cargo test -p indexio-index
    /// policy -- --nocapture` prints them).
    #[test]
    fn simulation_of_deltas_on_top_of_the_large_shards() {
        // the four large shards of the live data dir, and the same with the
        // two small old shards that sit between them
        let four = [123.24, 73.61, 85.16, 210.53];
        let live = [123.24, 73.61, 85.16, 1.27, 0.44, 210.53];
        for (name, base, deltas) in [("four large shards", &four[..], 300), ("four large shards", &four[..], 2000), ("live layout", &live[..], 300), ("live layout", &live[..], 2000)] {
            let old = simulate(base, deltas, 1.0);
            let new = simulate(base, deltas, 0.0);
            let mixed = simulate(base, deltas, 0.5);
            println!("== {name}, {deltas} deltas of 0.1-1 MB");
            println!("{}", row("0.1.14 (above 12, to 6)", &old));
            println!("{}", row(&format!("tiered (above {COMPACT_ABOVE}, to {COMPACT_TO})"), &new));
            println!("{}", row("half the checks by 0.1.14 servers", &mixed));

            // 0.1.14 with six shards that are not deltas: a large shard is
            // rewritten in every merge
            if base.len() >= 6 {
                assert_eq!(old.large, old.merges, "{old:?}");
            }
            // the tiered policy rewrites a fraction of the bytes (one merge
            // of the whole index, when the deltas reach half of the newest
            // large shard, is most of what it rewrites in the short runs)...
            let factor = if deltas >= 2000 { 15 } else { 2 };
            assert!(new.rewritten * factor < old.rewritten, "{name}/{deltas}: {} vs {}", new.rewritten, old.rewritten);
            assert!(new.large * 4 < old.large, "{name}/{deltas}: {} vs {}", new.large, old.large);
            // ...within the same bound on the shard count (0.1.14: 13)
            assert_eq!(old.max_shards, 13);
            assert!(new.max_shards <= COMPACT_ABOVE + 1, "{}", new.max_shards);
            // a data dir shared with 0.1.14 servers: no worse than 0.1.14
            // alone, and the shard count stays within its bound
            assert!(mixed.rewritten < old.rewritten, "{name}/{deltas}: {} vs {}", mixed.rewritten, old.rewritten);
            assert!(mixed.max_shards <= 13, "{}", mixed.max_shards);
        }
    }

    /// Each merge takes shards out and puts one in, whoever runs it: the two
    /// policies cannot undo each other's work. What a 0.1.14 server can
    /// still do is run its expensive merge when it is the one to see more
    /// than 12 shards; current servers compact above 10, so that takes
    /// three deltas in a row seen by 0.1.14 servers only.
    #[test]
    fn simulation_with_0_1_14_servers_on_the_same_data_dir() {
        let live = [123.24, 73.61, 85.16, 1.27, 0.44, 210.53];
        let old = simulate(&live, 2000, 1.0);
        println!("{}", row("0.1.14 only", &old));
        let mut last = old.rewritten;
        for old_share in [0.75, 0.5, 0.25, 0.0] {
            let o = simulate(&live, 2000, old_share);
            println!("{}", row(&format!("{:.0}% of the checks by 0.1.14", old_share * 100.0), &o));
            assert!(o.rewritten < last, "fewer 0.1.14 servers, fewer bytes rewritten: {} vs {last}", o.rewritten);
            assert!(o.max_shards <= 13);
            last = o.rewritten;
        }
    }
}
