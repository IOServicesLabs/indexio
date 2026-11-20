//! Maps and sets a long-lived server can keep adding to.
//!
//! Two generations of at most `cap` entries each, like the engine's content
//! cache: entries go into `new`; when it is full it becomes `old` and the
//! previous `old` is dropped; touching an entry of `old` moves it back up.
//! Whatever was touched since the last turnover is kept, so a working set of
//! up to `cap` entries never falls out, and the structure never holds more
//! than twice that.

use std::collections::HashMap;
use std::hash::Hash;

pub(super) struct RecentMap<K, V> {
    new: HashMap<K, V>,
    old: HashMap<K, V>,
    cap: usize,
}

impl<K: Hash + Eq, V> RecentMap<K, V> {
    pub(super) fn new(cap: usize) -> Self {
        RecentMap { new: HashMap::new(), old: HashMap::new(), cap: cap.max(1) }
    }

    /// `HashMap::insert`: the value the key had, if it is still remembered.
    pub(super) fn insert(&mut self, key: K, value: V) -> Option<V> {
        let prev = match self.new.remove(&key) {
            Some(v) => Some(v),
            None => self.old.remove(&key),
        };
        if self.new.len() >= self.cap {
            self.old = std::mem::take(&mut self.new);
        }
        self.new.insert(key, value);
        prev
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.new.len() + self.old.len()
    }
}

pub(super) struct RecentSet<K>(RecentMap<K, ()>);

impl<K: Hash + Eq> RecentSet<K> {
    pub(super) fn new(cap: usize) -> Self {
        RecentSet(RecentMap::new(cap))
    }

    /// `HashSet::insert`: whether the key was not remembered.
    pub(super) fn insert(&mut self, key: K) -> bool {
        self.0.insert(key, ()).is_none()
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn behaves_like_a_map_below_the_cap() {
        let mut m = RecentMap::new(8);
        assert_eq!(m.insert("a", 1), None);
        assert_eq!(m.insert("b", 2), None);
        assert_eq!(m.insert("a", 3), Some(1));
        assert_eq!(m.insert("a", 3), Some(3));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn never_holds_more_than_two_generations() {
        let mut m = RecentMap::new(100);
        for i in 0..10_000u32 {
            m.insert(i, i);
            assert!(m.len() <= 200, "{} entries after {i} inserts", m.len());
        }
        // the most recent `cap` keys are all still there
        for i in 9_900..10_000u32 {
            assert_eq!(m.insert(i, 0), Some(i), "key {i} was forgotten");
        }
        // and the oldest are gone
        assert_eq!(m.insert(0, 0), None);
    }

    #[test]
    fn a_working_set_within_the_cap_is_never_forgotten() {
        // the coverage repair's pattern: every pass asks for the same docs
        // again, while other docs come and go
        let mut s = RecentSet::new(64);
        let stuck: Vec<u32> = (0..50).collect();
        for k in &stuck {
            assert!(s.insert(*k));
        }
        for pass in 0..200u32 {
            for k in &stuck {
                assert!(!s.insert(*k), "pass {pass}: doc {k} would be tried again");
            }
            // ten docs seen once
            for j in 0..10 {
                s.insert(1_000 + pass * 10 + j);
            }
            assert!(s.len() <= 128);
        }
    }
}
