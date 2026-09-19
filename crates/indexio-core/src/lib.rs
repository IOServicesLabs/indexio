//! indexio-core: n-gram extraction (sparse grams) and match verification.
//! Contract: docs/SPEC.md. CommonGrams is fully implemented by the orchestrator;
//! grams::extract / grams::grams_of / verify::* are implemented by the indexio-core agent.
#![forbid(unsafe_code)]

pub mod grams {
    use std::collections::HashMap;

    /// Maximum document size that gets n-gram extracted (SPEC: 4 MiB).
    pub const MAX_DOC_BYTES: usize = 4 * 1024 * 1024;
    /// Maximum gram length (sparse-gram extension cap).
    pub const MAX_GRAM_LEN: usize = 8;

    /// Sparse-gram table: for each *common* trigram, how many extra bytes
    /// (1..=5) a gram must include to be indexed. Trigrams not in the table
    /// are indexed as-is (ext_len == 0).
    #[derive(Clone, Debug, Default)]
    pub struct CommonGrams {
        ext: HashMap<[u8; 3], usize>,
    }

    impl CommonGrams {
        pub fn empty() -> Self {
            Self::default()
        }

        /// Build from gram statistics `(trigram, doc_count)` over a corpus of
        /// `total_docs` docs. Trigrams appearing in at least
        /// `threshold_doc_frac` of docs are marked common and require a
        /// 1-byte extension (longer extension is chosen adaptively at extract
        /// time when the extended gram is still common — see extract()).
        pub fn from_stats(
            counts: &[(Vec<u8>, u64)],
            total_docs: u64,
            threshold_doc_frac: f64,
        ) -> Self {
            let mut ext = HashMap::new();
            if total_docs == 0 {
                return Self { ext };
            }
            let cutoff = (total_docs as f64 * threshold_doc_frac).ceil() as u64;
            for (gram, n) in counts {
                if gram.len() == 3 && *n >= cutoff {
                    ext.insert([gram[0], gram[1], gram[2]], 1);
                }
            }
            Self { ext }
        }

        /// Required extension length (0 = plain trigram) for the trigram
        /// starting at `tri[0..3]`. `tri` may be longer than 3 bytes.
        pub fn ext_len(&self, tri: &[u8]) -> usize {
            if tri.len() < 3 {
                return 0;
            }
            self.ext.get(&[tri[0], tri[1], tri[2]]).copied().unwrap_or(0)
        }

        pub fn is_empty(&self) -> bool {
            self.ext.is_empty()
        }
    }

    /// The gram starting at byte offset `i` of `text` (requires `i + 3 <= text.len()`).
    ///
    /// Exact sparse-gram extension rule (deterministic, shared by `extract`
    /// and `grams_of` — this shared rule is the consistency invariant):
    ///
    /// 1. Let `tri = text[i..i + 3]` and `e = common.ext_len(tri)`.
    /// 2. If `e == 0`, the gram is exactly `tri` (3 bytes).
    /// 3. Otherwise start with `len = min(3 + e, MAX_GRAM_LEN, text.len() - i)`,
    ///    then extend adaptively: while `len < MAX_GRAM_LEN`, another byte
    ///    exists (`i + len < text.len()`), and the *trailing trigram* of the
    ///    current gram (`text[i + len - 3 .. i + len]`) is itself common
    ///    (`common.ext_len(..) > 0`), grow `len` by one.
    /// 4. **Boundary skip:** if the extension was cut short by the end of
    ///    `text` — i.e. `len < MAX_GRAM_LEN`, `i + len == text.len()`, and the
    ///    trailing trigram is still common (so more bytes *would* have
    ///    extended the gram) — no gram is emitted for this offset at all
    ///    (`None`). Such a gram's value depends on bytes that may not exist
    ///    in some other context, so emitting it would break the invariant
    ///    that every `grams_of(needle)` gram is an `extract(text)` gram key
    ///    for any `text` containing `needle`. Both doc side and query side
    ///    skip these positions symmetrically, so no recall is lost (the
    ///    remaining grams still filter; the verifier confirms).
    /// 5. The gram is `text[i..i + len]`.
    ///
    /// In words: common trigrams are indexed longer (4 bytes base, up to
    /// [`MAX_GRAM_LEN`] while the gram still ends in a common trigram), rare
    /// trigrams stay 3 bytes.
    fn gram_at<'a>(text: &'a [u8], i: usize, common: &CommonGrams) -> Option<&'a [u8]> {
        debug_assert!(i + 3 <= text.len());
        let e = common.ext_len(&text[i..i + 3]);
        if e == 0 {
            return Some(&text[i..i + 3]);
        }
        let mut len = (3 + e).min(MAX_GRAM_LEN).min(text.len() - i);
        while len < MAX_GRAM_LEN
            && i + len < text.len()
            && common.ext_len(&text[i + len - 3..i + len]) > 0
        {
            len += 1;
        }
        if len < MAX_GRAM_LEN
            && i + len == text.len()
            && common.ext_len(&text[i + len - 3..i + len]) > 0
        {
            // Extension truncated by the text boundary: skip (see rule 4).
            return None;
        }
        Some(&text[i..i + len])
    }

    /// Extract the sparse grams of `text`. See docs/SPEC.md (indexio-core).
    ///
    /// Returns the distinct grams sorted by gram bytes. Positions are not
    /// kept (SPEC-P10): the query planner intersects document sets and the
    /// verifier rescans the content, so per-occurrence offsets were never
    /// read — they only doubled the extraction work and made up 80 % of
    /// the shard and 90 % of the extraction cache. Documents larger than
    /// [`MAX_DOC_BYTES`] (or shorter than 3 bytes) yield no grams. See
    /// [`gram_at`] for the exact extension rule.
    pub fn extract(text: &[u8], common: &CommonGrams) -> Vec<Vec<u8>> {
        if text.len() > MAX_DOC_BYTES || text.len() < 3 {
            return Vec::new();
        }
        let mut set: std::collections::HashSet<&[u8]> = std::collections::HashSet::new();
        for i in 0..=(text.len() - 3) {
            if let Some(gram) = gram_at(text, i, common) {
                set.insert(gram);
            }
        }
        let mut out: Vec<Vec<u8>> = set.into_iter().map(<[u8]>::to_vec).collect();
        out.sort_unstable();
        out
    }

    /// All grams of `needle` as the query planner should look them up.
    ///
    /// Produces exactly the gram keys that [`extract`] would produce for a
    /// document consisting of `needle` alone (same extension rule, same
    /// [`MAX_DOC_BYTES`] cap), deduplicated, in first-occurrence order.
    pub fn grams_of(needle: &[u8], common: &CommonGrams) -> Vec<Vec<u8>> {
        if needle.len() > MAX_DOC_BYTES || needle.len() < 3 {
            return Vec::new();
        }
        let mut out: Vec<Vec<u8>> = Vec::new();
        let mut seen: std::collections::HashSet<&[u8]> = std::collections::HashSet::new();
        for i in 0..=(needle.len() - 3) {
            if let Some(gram) = gram_at(needle, i, common) {
                if seen.insert(gram) {
                    out.push(gram.to_vec());
                }
            }
        }
        out
    }
}

pub mod verify {
    use memchr::{memchr, memchr2, memchr_iter};

    /// Find literal matches; returns (1-based line, 0-based col) per hit.
    ///
    /// Case-sensitive matching uses a SIMD `memchr` scan. Case-insensitive
    /// matching is ASCII-only: it uses aho-corasick with
    /// `ascii_case_insensitive` when the needle is pure ASCII (which implies
    /// valid UTF-8); otherwise it falls back to a manual
    /// `memchr`-accelerated scan comparing bytes with
    /// `eq_ignore_ascii_case`. Overlapping occurrences are all reported,
    /// capped at `max_hits`. An empty needle yields no hits.
    pub fn find_literal(
        content: &[u8],
        needle: &[u8],
        case_insensitive: bool,
        max_hits: usize,
    ) -> Vec<(u32, u32)> {
        if needle.is_empty() || needle.len() > content.len() || max_hits == 0 {
            return Vec::new();
        }
        let mut offsets: Vec<usize> = Vec::new();
        if !case_insensitive {
            find_offsets_cs(content, needle, max_hits, &mut offsets);
        } else if needle.is_ascii() {
            // ASCII needle: aho-corasick with ASCII case folding.
            if let Ok(ac) = aho_corasick::AhoCorasick::builder()
                .ascii_case_insensitive(true)
                .build([needle])
            {
                for m in ac.find_overlapping_iter(content) {
                    offsets.push(m.start());
                    if offsets.len() >= max_hits {
                        break;
                    }
                }
            } else {
                find_offsets_ci(content, needle, max_hits, &mut offsets);
            }
        } else {
            // Non-ASCII / non-UTF-8 needle with ci requested: manual scan.
            find_offsets_ci(content, needle, max_hits, &mut offsets);
        }
        offsets_to_line_cols(content, offsets)
    }

    /// Case-sensitive occurrence offsets (ascending), via memchr on the
    /// first byte + slice compare. Reports overlapping occurrences.
    fn find_offsets_cs(content: &[u8], needle: &[u8], max_hits: usize, out: &mut Vec<usize>) {
        let n = needle.len();
        let first = needle[0];
        let mut start = 0usize;
        while start + n <= content.len() && out.len() < max_hits {
            let window_end = content.len() - n + 1;
            match memchr(first, &content[start..window_end]) {
                Some(rel) => {
                    let pos = start + rel;
                    if &content[pos..pos + n] == needle {
                        out.push(pos);
                    }
                    start = pos + 1;
                }
                None => break,
            }
        }
    }

    /// ASCII-case-insensitive occurrence offsets (ascending), manual
    /// memchr/memchr2-accelerated scan. Reports overlapping occurrences.
    fn find_offsets_ci(content: &[u8], needle: &[u8], max_hits: usize, out: &mut Vec<usize>) {
        let n = needle.len();
        let first = needle[0];
        let mut start = 0usize;
        while start + n <= content.len() && out.len() < max_hits {
            let window_end = content.len() - n + 1;
            let hay = &content[start..window_end];
            let found = if first.is_ascii_alphabetic() {
                memchr2(
                    first.to_ascii_lowercase(),
                    first.to_ascii_uppercase(),
                    hay,
                )
            } else {
                memchr(first, hay)
            };
            match found {
                Some(rel) => {
                    let pos = start + rel;
                    if content[pos..pos + n].eq_ignore_ascii_case(needle) {
                        out.push(pos);
                    }
                    start = pos + 1;
                }
                None => break,
            }
        }
    }

    /// Find regex matches; None = invalid pattern.
    ///
    /// Uses `regex::bytes::Regex`; returns the (1-based line, 0-based col) of
    /// each match start, capped at `max_hits`.
    pub fn find_regex(
        content: &[u8],
        pattern: &str,
        max_hits: usize,
    ) -> Option<Vec<(u32, u32)>> {
        let re = compile_regex(pattern).ok()?;
        Some(find_regex_with(content, &re, max_hits))
    }

    /// Compile a content regex the way `grep -n` reads one: `^` and `$`
    /// anchor at line boundaries (multi-line mode), `.` stops at the newline.
    /// Without this a `^## ` or `fn main\($` only matched at the very start
    /// or end of the file.
    pub fn compile_regex(pattern: &str) -> Result<regex::bytes::Regex, regex::Error> {
        regex::bytes::RegexBuilder::new(pattern).multi_line(true).build()
    }

    /// [`find_regex`] with a pre-compiled pattern: the query engine
    /// compiles once per query, not once per candidate document.
    pub fn find_regex_with(
        content: &[u8],
        re: &regex::bytes::Regex,
        max_hits: usize,
    ) -> Vec<(u32, u32)> {
        let mut offsets: Vec<usize> = Vec::new();
        for m in re.find_iter(content) {
            if offsets.len() >= max_hits {
                break;
            }
            offsets.push(m.start());
        }
        offsets_to_line_cols(content, offsets)
    }

    /// Convert ascending byte offsets to (1-based line, 0-based byte col),
    /// scanning the content for `\n` once.
    fn offsets_to_line_cols(content: &[u8], mut offsets: Vec<usize>) -> Vec<(u32, u32)> {
        offsets.sort_unstable();
        let mut out = Vec::with_capacity(offsets.len());
        let mut line: u32 = 1;
        let mut line_start = 0usize;
        let mut scanned = 0usize;
        for off in offsets {
            let off = off.min(content.len());
            let from = scanned.min(off);
            for nl in memchr_iter(b'\n', &content[from..off]) {
                line += 1;
                line_start = from + nl + 1;
            }
            scanned = off;
            out.push((line, (off - line_start) as u32));
        }
        out
    }

    /// RE2-style required literals from a regex; None if no usable literal.
    ///
    /// Strategy: parse the pattern with `regex-syntax` (bytes semantics) and
    /// compute, per alternation branch, the longest literal byte string that
    /// is guaranteed to *prefix* every match of that branch; concatenated
    /// literal runs are merged (e.g. `(foo|foobar)baz` yields `foobaz` and
    /// `foobarbaz`). Returns the best 1–4 literals, longest first.
    ///
    /// Returns `None` when there is no guaranteed literal: the pattern can
    /// match the empty string, starts with something non-literal such as
    /// `.*`, any alternation branch lacks a literal prefix, the pattern is
    /// invalid, or it uses `(?i)` case-insensitive matching (case-folded
    /// matches would not contain the literal bytes verbatim).
    pub fn required_literals(pattern: &str) -> Option<Vec<Vec<u8>>> {
        if pattern.contains("(?i") {
            return None;
        }
        let hir = regex_syntax::ParserBuilder::new()
            .utf8(false)
            .build()
            .parse(pattern)
            .ok()?;
        let mut lits = required_prefixes(&hir)?;
        lits.retain(|l| !l.is_empty());
        if lits.is_empty() {
            return None;
        }
        lits.sort();
        lits.dedup();
        // Longest first; keep the best 1-4.
        lits.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        lits.truncate(4);
        Some(lits)
    }

    /// Guaranteed literal prefixes of `hir`: `Some(list)` means every match
    /// of `hir` starts with at least one string in `list`; `None` means
    /// nothing is guaranteed. The list never contains empty strings (a
    /// zero-width guarantee is expressed as `Some(vec![b""])` only by
    /// [`required_prefixes_zw`]-style zero-width nodes, which `Concat`
    /// filters out).
    fn required_prefixes(hir: &regex_syntax::hir::Hir) -> Option<Vec<Vec<u8>>> {
        use regex_syntax::hir::HirKind;
        match hir.kind() {
            HirKind::Empty => Some(vec![Vec::new()]),
            HirKind::Literal(lit) => Some(vec![lit.0.to_vec()]),
            HirKind::Class(_) => None,
            HirKind::Look(_) => Some(vec![Vec::new()]), // zero-width, transparent
            HirKind::Capture(cap) => required_prefixes(&cap.sub),
            HirKind::Repetition(rep) => {
                if rep.min >= 1 {
                    // `X+` etc.: every match starts with a match of X.
                    required_prefixes(&rep.sub)
                } else {
                    // `X*`, `X?`: can match empty -> no guaranteed prefix.
                    None
                }
            }
            HirKind::Concat(subs) => {
                // `acc`: guaranteed prefixes so far; `None` = not started.
                let mut acc: Option<Vec<Vec<u8>>> = None;
                for sub in subs {
                    match sub.kind() {
                        HirKind::Look(_) | HirKind::Empty => continue,
                        HirKind::Literal(lit) => match &mut acc {
                            None => acc = Some(vec![lit.0.to_vec()]),
                            Some(ps) => {
                                for p in ps.iter_mut() {
                                    p.extend_from_slice(&lit.0);
                                }
                            }
                        },
                        _ => {
                            let ps = match required_prefixes(sub) {
                                Some(ps) => ps,
                                // No prefix guaranteed here: keep what we
                                // have (if anything) and stop.
                                None => {
                                    acc.as_ref()?;
                                    break;
                                }
                            };
                            let exact = is_exact(sub);
                            match &mut acc {
                                None => acc = Some(ps),
                                Some(cur) => {
                                    if cur.len() * ps.len() > 64 {
                                        // Avoid combinatorial blow-up; the
                                        // accumulated prefixes stay sound.
                                        break;
                                    }
                                    // Cross-append each branch prefix onto
                                    // each accumulated prefix (sound because
                                    // each accumulated prefix is immediately
                                    // followed by the sub-match it prefixes).
                                    let mut next: Vec<Vec<u8>> =
                                        Vec::with_capacity(cur.len() * ps.len());
                                    for a in cur.iter() {
                                        for b in &ps {
                                            let mut p = a.clone();
                                            p.extend_from_slice(b);
                                            if !next.contains(&p) {
                                                next.push(p);
                                            }
                                        }
                                    }
                                    *cur = next;
                                }
                            }
                            if !exact {
                                // After a variable-length element, following
                                // bytes are not part of the prefix.
                                break;
                            }
                        }
                    }
                }
                let ps = acc?;
                let ps: Vec<Vec<u8>> = ps.into_iter().filter(|p| !p.is_empty()).collect();
                if ps.is_empty() {
                    None
                } else {
                    Some(ps)
                }
            }
            HirKind::Alternation(alts) => {
                let mut out: Vec<Vec<u8>> = Vec::new();
                for alt in alts {
                    for p in required_prefixes(alt)? {
                        if p.is_empty() {
                            // A branch that guarantees nothing means the
                            // alternation as a whole guarantees nothing.
                            return None;
                        }
                        if !out.contains(&p) {
                            out.push(p);
                        }
                    }
                }
                if out.is_empty() {
                    None
                } else {
                    Some(out)
                }
            }
        }
    }

    /// True if every match of `hir` has exactly one fixed byte string per
    /// alternation branch — i.e. a following literal can soundly be appended
    /// to this node's guaranteed prefixes.
    fn is_exact(hir: &regex_syntax::hir::Hir) -> bool {
        use regex_syntax::hir::HirKind;
        match hir.kind() {
            HirKind::Empty | HirKind::Literal(_) | HirKind::Look(_) => true,
            HirKind::Class(_) => false,
            HirKind::Capture(cap) => is_exact(&cap.sub),
            HirKind::Concat(subs) => subs.iter().all(is_exact),
            HirKind::Alternation(alts) => alts.iter().all(is_exact),
            HirKind::Repetition(rep) => {
                rep.min >= 1 && rep.max == Some(rep.min) && is_exact(&rep.sub)
            }
        }
    }

    /// Byte offset -> (1-based line, 0-based col).
    ///
    /// Col is a byte offset within the line (a `\r` of a CRLF ending counts
    /// toward the preceding line's columns). Offsets past the end are
    /// clamped to `content.len()`.
    pub fn line_col(content: &[u8], offset: u32) -> (u32, u32) {
        let off = (offset as usize).min(content.len());
        let mut line: u32 = 1;
        let mut line_start = 0usize;
        for nl in memchr_iter(b'\n', &content[..off]) {
            line += 1;
            line_start = nl + 1;
        }
        (line, (off - line_start) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::grams::{extract, grams_of, CommonGrams, MAX_DOC_BYTES, MAX_GRAM_LEN};
    use super::verify::{find_literal, find_regex, line_col, required_literals};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn common_with(tris: &[&[u8]]) -> CommonGrams {
        let counts: Vec<(Vec<u8>, u64)> = tris.iter().map(|t| t.to_vec()).zip(std::iter::repeat(10u64)).collect();
        CommonGrams::from_stats(&counts, 10, 0.5)
    }

    fn keys(extracted: &[Vec<u8>]) -> std::collections::BTreeSet<Vec<u8>> {
        extracted.iter().cloned().collect()
    }

    // ---------- grams ----------

    #[test]
    fn extract_basic_trigrams() {
        let common = CommonGrams::empty();
        let out = extract(b"hello world", &common);
        let ks = keys(&out);
        for expect in [
            &b"hel"[..], b"ell", b"llo", b"lo ", b"o w", b" wo", b"wor", b"orl", b"rld",
        ] {
            assert!(ks.contains(expect), "missing gram {:?}", expect);
        }
        assert_eq!(out.len(), 9);
        // sorted by gram
        let mut sorted = out.clone();
        sorted.sort();
        assert_eq!(out, sorted);
    }

    #[test]
    fn extract_dedups_repeated_grams() {
        let common = CommonGrams::empty();
        let out = extract(b"ababab", &common);
        assert_eq!(out, vec![b"aba".to_vec(), b"bab".to_vec()]);
    }

    #[test]
    fn extract_skips_huge_docs() {
        let common = CommonGrams::empty();
        let big = vec![b'x'; MAX_DOC_BYTES + 1];
        assert!(extract(&big, &common).is_empty());
        let at_max = vec![b'y'; MAX_DOC_BYTES];
        assert!(!extract(&at_max, &common).is_empty());
    }

    #[test]
    fn extract_short_inputs() {
        let common = CommonGrams::empty();
        assert!(extract(b"", &common).is_empty());
        assert!(extract(b"ab", &common).is_empty());
        let out = extract(b"abc", &common);
        assert_eq!(out, vec![b"abc".to_vec()]);
        assert!(grams_of(b"ab", &common).is_empty());
    }

    #[test]
    fn common_gram_extension() {
        // "for" is common: grams starting at "for" extend to 4 bytes.
        let common = common_with(&[b"for"]);
        let out = extract(b"xx format yy", &common);
        let ks = keys(&out);
        assert!(ks.contains(b"form".as_slice()), "grams: {:?}", ks);
        assert!(!ks.contains(b"for".as_slice()));
        // non-common trigrams stay 3 bytes
        assert!(ks.contains(b"orm".as_slice()));
        // grams_of("format") matches extract of a doc consisting of "format"
        let q = grams_of(b"format", &common);
        assert_eq!(
            q,
            vec![b"form".to_vec(), b"orm".to_vec(), b"rma".to_vec(), b"mat".to_vec()]
        );
        let doc_keys = keys(&extract(b"format", &common));
        let q_set: std::collections::BTreeSet<_> = q.iter().cloned().collect();
        assert_eq!(q_set, doc_keys);
        // and every query gram is found in a larger doc containing "format"
        let big_keys = keys(&extract(b"xx format yy", &common));
        for g in &q {
            assert!(big_keys.contains(g), "missing {:?}", g);
        }
    }

    #[test]
    fn common_gram_adaptive_extension() {
        // "for" and "orm" common: at "format", gram extends adaptively
        // "form" -> trailing "orm" common -> "forma"; trailing "rma" not common -> stop.
        let common = common_with(&[b"for", b"orm"]);
        let q = grams_of(b"format", &common);
        assert_eq!(
            q,
            vec![b"forma".to_vec(), b"orma".to_vec(), b"rma".to_vec(), b"mat".to_vec()]
        );
        assert!(q[0].len() <= MAX_GRAM_LEN);
    }

    #[test]
    fn common_gram_extension_capped_at_max() {
        // All trigrams common: extension must cap at MAX_GRAM_LEN = 8.
        let tris: Vec<&[u8]> = b"aaaaaaaaaa"
            .windows(3)
            .collect::<Vec<_>>()
            .into_iter()
            .collect();
        let common = common_with(&tris);
        let q = grams_of(b"aaaaaaaaaa", &common);
        assert_eq!(q, vec![b"aaaaaaaa".to_vec()]);
        assert_eq!(q[0].len(), MAX_GRAM_LEN);
    }

    #[test]
    fn grams_of_empty_common_is_plain_trigrams() {
        let common = CommonGrams::empty();
        assert_eq!(
            grams_of(b"abcde", &common),
            vec![b"abc".to_vec(), b"bcd".to_vec(), b"cde".to_vec()]
        );
    }

    /// Deterministic pseudo-random byte texts of various flavours.
    fn sample_texts() -> Vec<Vec<u8>> {
        let mut rng = StdRng::seed_from_u64(0xC10E);
        let mut texts: Vec<Vec<u8>> = Vec::new();
        // source-code-like
        let code = r#"
fn main() {
    let common = CommonGrams::empty();
    for i in 0..100 {
        println!("{}: {:?}", i, extract(b"hello world", &common));
    }
    // TODO: handle CRLF\r\n
}
"#;
        texts.push(code.as_bytes().to_vec());
        texts.push(code.repeat(40).into_bytes()); // ~ bigger, many repeats
        // unicode
        texts.push("héllo wörld — ünïcödé — こんにちは世界 — 🦀🦀\n".repeat(30).into_bytes());
        // CRLF
        texts.push(b"line one\r\nline two\r\nfor format for\r\n".repeat(20));
        // long single line (>64KiB)
        let mut long_line = b"x".repeat(70 * 1024);
        long_line.extend_from_slice(b"needle-in-long-line");
        long_line.extend_from_slice(&b"y".repeat(1024));
        texts.push(long_line);
        // 1 MiB pseudo-random printable-ish bytes
        let mut big = Vec::with_capacity(1 << 20);
        while big.len() < (1 << 20) {
            let b = rng.gen_range(0u8..=255);
            // bias towards printable + newlines so it looks text-ish
            big.push(if b < 200 { rng.gen_range(b'a'..=b'z') } else { b });
        }
        texts.push(big);
        // pure random binary-ish (no NUL concern here; extract assumes text)
        let bin: Vec<u8> = (0..4096).map(|_| rng.gen::<u8>()).collect();
        texts.push(bin);
        texts
    }

    #[test]
    fn property_extract_grams_of_consistency() {
        let mut rng = StdRng::seed_from_u64(42);
        let commons = [
            CommonGrams::empty(),
            common_with(&[b"for"]),
            common_with(&[b"for", b"orm", b"hel", b"ell", b"lin", b"ine"]),
        ];
        for text in sample_texts() {
            assert!(text.len() < MAX_DOC_BYTES);
            for common in &commons {
                let extracted = extract(&text, common);
                let ks = keys(&extracted);
                // sample needles from the text
                for _ in 0..64 {
                    let max_len = text.len().min(64);
                    if max_len < 3 {
                        continue;
                    }
                    let start = rng.gen_range(0..=text.len() - 3);
                    let len = rng.gen_range(3..=max_len.min(text.len() - start));
                    let needle = &text[start..start + len];
                    let q = grams_of(needle, common);
                    // exact consistency with extract() of the needle as a doc
                    let doc_keys = keys(&extract(needle, common));
                    let q_set: std::collections::BTreeSet<_> = q.iter().cloned().collect();
                    assert_eq!(q_set, doc_keys, "grams_of != extract(doc=needle)");
                    // every query gram appears in the enclosing doc's grams
                    for g in &q {
                        assert!(
                            ks.contains(g),
                            "gram {:?} of needle missing from extract(text)",
                            g
                        );
                    }
                    // reverse sanity: extract of doc containing the needle is
                    // a superset of the needle's gram keys
                    let mut doc = b"pre ".to_vec();
                    doc.extend_from_slice(needle);
                    doc.extend_from_slice(b" post");
                    let doc_ks = keys(&extract(&doc, common));
                    for g in &q {
                        assert!(doc_ks.contains(g), "superset check failed for {:?}", g);
                    }
                }
            }
        }
    }

    #[test]
    fn property_gram_lengths() {
        let common = common_with(&[b"the", b"he ", b"e a"]);
        let text = b"the and the or thee a the a".as_slice();
        let out = extract(text, &common);
        let mut sorted = out.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(out, sorted, "grams must be sorted+unique");
        for g in &out {
            assert!((3..=MAX_GRAM_LEN).contains(&g.len()));
            // the gram actually occurs in the text
            assert!(text.windows(g.len()).any(|w| w == g.as_slice()), "{g:?} not in text");
        }
    }

    // ---------- verify ----------

    #[test]
    fn literal_basic() {
        let content = b"hello world\nhello again\n";
        assert_eq!(find_literal(content, b"hello", false, 10), vec![(1, 0), (2, 0)]);
        assert_eq!(find_literal(content, b"world", false, 10), vec![(1, 6)]);
        assert_eq!(find_literal(content, b"zzz", false, 10), Vec::<(u32, u32)>::new());
        // max_hits cap
        assert_eq!(find_literal(content, b"hello", false, 1), vec![(1, 0)]);
        assert_eq!(find_literal(content, b"hello", false, 0), Vec::<(u32, u32)>::new());
        // overlapping occurrences
        assert_eq!(find_literal(b"aaaa", b"aa", false, 10), vec![(1, 0), (1, 1), (1, 2)]);
        // empty needle / needle longer than content
        assert_eq!(find_literal(content, b"", false, 10), Vec::<(u32, u32)>::new());
        assert_eq!(find_literal(b"ab", b"abcdef", false, 10), Vec::<(u32, u32)>::new());
    }

    #[test]
    fn literal_unicode_and_crlf() {
        let content = "héllo wörld\r\nsecönd héllo\r\nthird".as_bytes();
        let hits = find_literal(content, "héllo".as_bytes(), false, 10);
        // "héllo wörld" = h(1) é(2) l l o space w ö(2) r l d = 13 bytes, then \r\n
        // "secönd " = s e c ö(2) n d space = 8 bytes -> line-2 "héllo" at col 8
        assert_eq!(hits, vec![(1, 0), (2, 8)]);
        // col counts bytes: 'ö' is 2 bytes
        let hits = find_literal(content, "wörld".as_bytes(), false, 10);
        assert_eq!(hits, vec![(1, 7)]);
        // CRLF: col of \r counts on its own line; "secönd héllo" = 14 bytes
        let hits = find_literal(content, b"\r\n", false, 10);
        assert_eq!(hits, vec![(1, 13), (2, 14)]);
    }

    #[test]
    fn literal_long_line() {
        let mut content = b"a".repeat(70 * 1024);
        content.extend_from_slice(b"TARGET");
        content.extend_from_slice(&b"b".repeat(100));
        let hits = find_literal(&content, b"TARGET", false, 10);
        assert_eq!(hits, vec![(1, 70 * 1024)]);
        let (line, col) = line_col(&content, 70 * 1024);
        assert_eq!((line, col), (1, 70 * 1024));
    }

    #[test]
    fn literal_case_insensitive() {
        let content = b"Foo BAR foo\nfOo\n";
        assert_eq!(
            find_literal(content, b"foo", true, 10),
            vec![(1, 0), (1, 8), (2, 0)]
        );
        assert_eq!(
            find_literal(content, b"FOO", true, 10),
            vec![(1, 0), (1, 8), (2, 0)]
        );
        assert_eq!(find_literal(content, b"bar", true, 10), vec![(1, 4)]);
        // non-ASCII needle with ci requested -> manual fallback; folding is
        // ASCII-only, so "SÉSÉ" does NOT match "sésé" (É != é), but the
        // exact "sésé" at byte 7 does.
        let c2 = "SÉSÉ sésé".as_bytes();
        assert_eq!(
            find_literal(c2, "sésé".as_bytes(), true, 10),
            vec![(1, 7)]
        );
        // ASCII-only folding still applies inside a mixed needle: "sAsÉ" ci
        // matches "SaSÉ".
        assert_eq!(
            find_literal("SaSÉ".as_bytes(), "sAsÉ".as_bytes(), true, 10),
            vec![(1, 0)]
        );
        // non-UTF-8 needle with ci requested -> fallback, no panic
        let needle = &[0x66u8, 0x6f, 0x80]; // "fo\x80"
        let content2 = b"xx FO\x80 yy fo\x80".as_slice();
        assert_eq!(find_literal(content2, needle, true, 10), vec![(1, 3), (1, 10)]);
        // case-sensitive must not match different ASCII case
        assert_eq!(find_literal(content, b"FOO", false, 10), Vec::<(u32, u32)>::new());
    }

    #[test]
    fn regex_basic_and_invalid() {
        let content = b"foo123 bar\nbaz foo456\n";
        let hits = find_regex(content, r"foo\d+", 10).unwrap();
        assert_eq!(hits, vec![(1, 0), (2, 4)]);
        assert_eq!(find_regex(content, r"foo\d+", 1).unwrap(), vec![(1, 0)]);
        assert!(find_regex(content, "(unclosed", 10).is_none());
        assert!(find_regex(content, "a{2,1}", 10).is_none());
        assert_eq!(find_regex(content, "zzz", 10).unwrap(), Vec::<(u32, u32)>::new());
    }

    #[test]
    fn regex_multiline_linecol() {
        let content = "ab\r\ncd ef\r\ngh".as_bytes();
        let hits = find_regex(content, "ef", 10).unwrap();
        assert_eq!(hits, vec![(2, 3)]);
    }

    #[test]
    fn regex_anchors_are_per_line() {
        // grep semantics: `^`/`$` anchor every line, `.` stops at the newline
        let content = b"intro
## 24. A real diff
fn main() {
}
";
        assert_eq!(find_regex(content, r"^## 2[0-9]\.", 10).unwrap(), vec![(2, 0)]);
        assert_eq!(find_regex(content, r"\{$", 10).unwrap(), vec![(3, 10)]);
        assert_eq!(find_regex(content, r"^fn .*\{$", 10).unwrap(), vec![(3, 0)]);
        assert_eq!(find_regex(content, "diff.fn", 10).unwrap(), Vec::<(u32, u32)>::new());
    }

    #[test]
    fn required_literals_basic() {
        assert_eq!(required_literals("foo"), Some(vec![b"foo".to_vec()]));
        assert_eq!(
            required_literals(r"foo\d+bar"),
            Some(vec![b"foo".to_vec()])
        );
        let lits = required_literals("(foo|foobar)baz").expect("some literals");
        assert!(
            lits.iter().any(|l| l.windows(3).any(|w| w == b"baz")),
            "expected a literal containing \"baz\", got {:?}",
            lits
        );
        assert_eq!(lits, vec![b"foobarbaz".to_vec(), b"foobaz".to_vec()]);
        assert_eq!(required_literals(".*x"), None);
        assert_eq!(required_literals("a*"), None);
        assert_eq!(required_literals(""), None);
        assert_eq!(required_literals("(unclosed"), None);
        assert_eq!(required_literals("(?i)foo"), None);
        // anchors are transparent
        assert_eq!(required_literals("^foo"), Some(vec![b"foo".to_vec()]));
        // alternation with one non-literal branch -> no guarantee
        assert_eq!(required_literals("(foo|.*)"), None);
        // repetition min>=1 still guarantees prefix
        assert_eq!(required_literals("ab+c"), Some(vec![b"ab".to_vec()]));
        // capped at 4, longest first
        let lits = required_literals("(aaaa|bbb|cc|d|eeeeee)(fff)").unwrap();
        assert!(lits.len() <= 4);
        assert!(lits.windows(2).all(|w| w[0].len() >= w[1].len()));
    }

    #[test]
    fn line_col_edges() {
        let content = b"ab\ncd\r\nef";
        assert_eq!(line_col(content, 0), (1, 0));
        assert_eq!(line_col(content, 2), (1, 2)); // at '\n'
        assert_eq!(line_col(content, 3), (2, 0)); // 'c'
        assert_eq!(line_col(content, 5), (2, 2)); // '\r'
        assert_eq!(line_col(content, 6), (2, 3)); // '\n' of CRLF
        assert_eq!(line_col(content, 7), (3, 0)); // 'e'
        assert_eq!(line_col(content, 100), (3, 2)); // clamped to end
    }
}
