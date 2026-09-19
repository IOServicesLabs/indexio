//! cAST-lite chunking (SPEC-P2 §1): split files at syntax boundaries using the
//! same tree-sitter grammars and def-node queries as `extract`.
//!
//! Strategy:
//! - files that fit in `max_chars` -> one chunk, header = path;
//! - otherwise walk top-level def nodes (from the per-language defs query):
//!   small defs become one chunk each (`path > scope chain > kind name`),
//!   large defs recurse into child defs, leaf defs that still overflow are
//!   split into line windows with ~10% overlap and `#partN` headers;
//! - runs of small adjacent non-def siblings (imports, comments, consts
//!   < 200 chars each) are merged into preamble chunks (header = path);
//! - Unknown language or parse failure -> sliding line-window fallback with
//!   `path > lines A-B` headers.
//!
//! Every chunk's `text` is `header + "\n" + source slice`. Untrusted input is
//! parsed under `catch_unwind`, mirroring `extract`.

use indexio_types::Lang;
use std::collections::{HashMap, HashSet};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Node, Parser};

use crate::{spec, text, LangSpec};

/// One semantic chunk of a file.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChunkRec {
    pub start_line: u32, // 1-based, inclusive
    pub end_line: u32,   // 1-based, inclusive
    pub header: String,  // e.g. "src/parser.rs > mod query > fn parse"
    pub text: String,    // header + "\n" + source slice
}

/// Chunk `content` into semantic pieces of at most `max_chars` source chars
/// each (header not counted). Files shorter than `max_chars` yield exactly one
/// chunk whose header is the path. Returns empty vec for empty/whitespace-only
/// files. Unknown/unsupported languages fall back to line-window chunking.
pub fn chunks(lang: Lang, path: &str, content: &[u8], max_chars: usize) -> Vec<ChunkRec> {
    if content.iter().all(|b| b.is_ascii_whitespace()) {
        return Vec::new();
    }
    let max_chars = max_chars.max(1);
    let lines = Lines::new(content);
    if content.len() <= max_chars {
        return vec![make_chunk(content, &lines, 0, content.len(), path.to_string())];
    }
    let Some(s) = spec(lang) else {
        return window_fallback(path, content, &lines, max_chars);
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        chunk_parsed(s, path, content, &lines, max_chars)
    }))
    .unwrap_or_else(|_| window_fallback(path, content, &lines, max_chars))
}

// ---------------------------------------------------------------------------
// Line index
// ---------------------------------------------------------------------------

/// Byte offsets of every line start, for byte-offset <-> 1-based-line mapping.
struct Lines {
    starts: Vec<usize>,
}

impl Lines {
    fn new(content: &[u8]) -> Self {
        let mut starts = vec![0];
        for (i, b) in content.iter().enumerate() {
            if *b == b'\n' {
                starts.push(i + 1);
            }
        }
        Lines { starts }
    }

    /// 1-based line containing byte offset `byte`.
    fn line_at(&self, byte: usize) -> u32 {
        let i = match self.starts.binary_search(&byte) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        i as u32 + 1
    }
}

// ---------------------------------------------------------------------------
// Def collection (reuses the per-language defs query from `extract`)
// ---------------------------------------------------------------------------

/// A definition node lifted to its outermost statement wrapper.
struct Def {
    start: usize,
    end: usize,
    name: String,
    kind_str: &'static str,
    is_const: bool,
    /// Index of the nearest enclosing def, if any.
    parent: Option<usize>,
}

/// Statement-level wrappers we climb through so the chunk range covers the
/// whole declaration (`const` keyword, decorators, `export`, ...).
const WRAPPERS: &[&str] = &[
    "expression_statement",
    "export_statement",
    "decorated_definition",
    "lexical_declaration",
    "variable_declaration",
    "const_declaration",
    "var_declaration",
    "field_declaration",
    "declaration",
    "type_declaration",
    "const_spec",
    "var_spec",
    "type_spec",
    "type_alias",
    "assignment",
];

fn kind_str(tag: &str) -> &'static str {
    match tag {
        "fn" => "fn",
        "method" => "method",
        "struct" => "struct",
        "class" => "class",
        "enum" => "enum",
        "trait" => "trait",
        "impl" => "impl",
        "mod" => "mod",
        "const" => "const",
        "interface" => "interface",
        _ => "type",
    }
}

/// Run the language's defs query and return all def nodes, lifted to their
/// outermost statement wrapper, sorted by byte range, with parent links
/// between nested defs.
fn collect_defs(s: &LangSpec, root: Node, content: &[u8]) -> Vec<Def> {
    let mut raw: Vec<(Node, String, &'static str, bool)> = Vec::new();
    let names = s.defs.capture_names();
    let Some(name_ix) = names.iter().position(|n| *n == "name") else {
        return Vec::new();
    };
    let mut cursor = tree_sitter::QueryCursor::new();
    let mut matches = cursor.matches(&s.defs, root, content);
    while let Some(m) = matches.next() {
        let mut name_node = None;
        let mut def_node = None;
        let mut tag = "";
        for cap in m.captures {
            if cap.index as usize == name_ix {
                name_node = Some(cap.node);
            } else if let Some(t) = names[cap.index as usize].strip_prefix("def.") {
                def_node = Some(cap.node);
                tag = t;
            }
        }
        let (Some(name_node), Some(def_node)) = (name_node, def_node) else {
            continue;
        };
        let name = text(content, name_node);
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        // Lift to the outermost statement wrapper.
        let mut node = def_node;
        while let Some(p) = node.parent() {
            if WRAPPERS.contains(&p.kind()) {
                node = p;
            } else {
                break;
            }
        }
        raw.push((node, name.to_string(), kind_str(tag), tag == "const"));
    }
    raw.sort_by_key(|(n, _, _, _)| (n.start_byte(), n.end_byte()));
    // Dedup on the lifted node (e.g. Go const groups share one declaration).
    let mut seen: HashSet<usize> = HashSet::new();
    raw.retain(|(n, _, _, _)| seen.insert(n.id()));

    let by_id: HashMap<usize, usize> = raw
        .iter()
        .enumerate()
        .map(|(i, (n, _, _, _))| (n.id(), i))
        .collect();
    raw.iter()
        .map(|(node, name, ks, is_const)| {
            // Nearest ancestor that is itself a def -> parent link.
            let mut parent = None;
            let mut cur = node.parent();
            while let Some(anc) = cur {
                if let Some(&pi) = by_id.get(&anc.id()) {
                    parent = Some(pi);
                    break;
                }
                cur = anc.parent();
            }
            Def {
                start: node.start_byte(),
                end: node.end_byte(),
                name: name.clone(),
                kind_str: ks,
                is_const: *is_const,
                parent,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Parsed-language chunking
// ---------------------------------------------------------------------------

/// One top-level item: either a top-level def or a non-def byte range.
#[derive(Clone, Copy)]
struct Item {
    start: usize,
    end: usize,
    def: Option<usize>,
}

fn chunk_parsed(
    s: &LangSpec,
    path: &str,
    content: &[u8],
    lines: &Lines,
    max_chars: usize,
) -> Vec<ChunkRec> {
    let mut parser = Parser::new();
    if parser.set_language(&s.language).is_err() {
        return window_fallback(path, content, lines, max_chars);
    }
    let Some(tree) = parser.parse(content, None) else {
        return window_fallback(path, content, lines, max_chars);
    };
    let root = tree.root_node();
    let defs = collect_defs(s, root, content);
    let mut top: Vec<usize> = defs
        .iter()
        .enumerate()
        .filter(|(_, d)| d.parent.is_none())
        .map(|(i, _)| i)
        .collect();
    if top.is_empty() {
        return window_fallback(path, content, lines, max_chars);
    }
    top.sort_by_key(|&i| defs[i].start);

    // Children-of-def index, each list sorted by byte range.
    let mut kids: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, d) in defs.iter().enumerate() {
        if let Some(p) = d.parent {
            kids.entry(p).or_default().push(i);
        }
    }
    for v in kids.values_mut() {
        v.sort_by_key(|&i| (defs[i].start, defs[i].end));
    }

    // Flatten the top level into items: defs plus non-def siblings.
    let top_by_id: HashMap<usize, usize> = top
        .iter()
        .copied()
        .map(|i| (defs_id(root, &defs[i]), i))
        .collect();
    let top_ranges: Vec<(usize, usize)> = top.iter().map(|&i| (defs[i].start, defs[i].end)).collect();
    let mut items: Vec<Item> = Vec::new();
    classify(root, &top_by_id, &top_ranges, &mut items);
    items.sort_by_key(|it| (it.start, it.end));
    // Fill uncovered byte gaps (whitespace, wrapper prefixes, ...) as non-def
    // items so small interstitial comments join preamble runs.
    let mut filled: Vec<Item> = Vec::new();
    let mut pos = 0usize;
    for it in items {
        if it.start > pos {
            filled.push(Item { start: pos, end: it.start, def: None });
        }
        filled.push(it);
        pos = pos.max(it.end);
    }
    if pos < content.len() {
        filled.push(Item { start: pos, end: content.len(), def: None });
    }

    let mut ctx = Ctx {
        path,
        content,
        lines,
        max_chars,
        defs: &defs,
        kids: &kids,
        out: Vec::new(),
    };

    // Walk items, merging runs of preamble-eligible siblings.
    let mut i = 0;
    while i < filled.len() {
        let it = filled[i];
        if !ctx.eligible(it) {
            match it.def {
                Some(di) => ctx.emit_def(di, &[]),
                None => ctx.emit_oversized_nondef(it.start, it.end),
            }
            i += 1;
            continue;
        }
        // Preamble run: merge adjacent eligible siblings up to max_chars.
        let run_start = it.start;
        let mut run_end = it.end;
        let mut j = i + 1;
        while j < filled.len()
            && ctx.eligible(filled[j])
            && filled[j].end - run_start <= max_chars
        {
            run_end = filled[j].end;
            j += 1;
        }
        if run_end - run_start > max_chars {
            // The first item alone overflows max_chars.
            match it.def {
                Some(di) => ctx.emit_def(di, &[]),
                None => ctx.emit_oversized_nondef(it.start, it.end),
            }
            i += 1;
        } else {
            ctx.push_trimmed(run_start, run_end, path.to_string());
            i = j;
        }
    }
    ctx.out
}

/// Node id for a def's byte range (ids came from the same tree, but the Def
/// struct stores only ranges; re-resolve cheaply via cursor walk).
fn defs_id(root: Node, d: &Def) -> usize {
    root.descendant_for_byte_range(d.start, d.end)
        .filter(|n| n.start_byte() == d.start && n.end_byte() == d.end)
        .map(|n| n.id())
        .unwrap_or(usize::MAX)
}

/// Walk `node`'s named children: top-level defs become def items, nodes
/// containing defs are traversed transparently, everything else is non-def.
fn classify(
    node: Node,
    top_by_id: &HashMap<usize, usize>,
    top_ranges: &[(usize, usize)],
    out: &mut Vec<Item>,
) {
    let mut walk = node.walk();
    for child in node.named_children(&mut walk) {
        if let Some(&di) = top_by_id.get(&child.id()) {
            out.push(Item {
                start: child.start_byte(),
                end: child.end_byte(),
                def: Some(di),
            });
        } else if contains_top_def(child, top_ranges) {
            classify(child, top_by_id, top_ranges, out);
        } else {
            out.push(Item {
                start: child.start_byte(),
                end: child.end_byte(),
                def: None,
            });
        }
    }
}

fn contains_top_def(node: Node, top_ranges: &[(usize, usize)]) -> bool {
    let idx = top_ranges.partition_point(|&(s, _)| s < node.start_byte());
    top_ranges
        .get(idx)
        .is_some_and(|&(_, e)| e <= node.end_byte())
}

// ---------------------------------------------------------------------------
// Emission context
// ---------------------------------------------------------------------------

struct Ctx<'a> {
    path: &'a str,
    content: &'a [u8],
    lines: &'a Lines,
    max_chars: usize,
    defs: &'a [Def],
    kids: &'a HashMap<usize, Vec<usize>>,
    out: Vec<ChunkRec>,
}

impl Ctx<'_> {
    /// Preamble-eligible: non-def siblings, or const defs shorter than 200.
    fn eligible(&self, it: Item) -> bool {
        match it.def {
            None => true,
            Some(di) => {
                let d = &self.defs[di];
                d.is_const && d.end - d.start < 200
            }
        }
    }

    /// Emit one def node; recurse or window-split when it overflows.
    fn emit_def(&mut self, di: usize, chain: &[String]) {
        let defs = self.defs;
        let d = &defs[di];
        let mut base = self.path.to_string();
        for c in chain {
            base.push_str(" > ");
            base.push_str(c);
        }
        base.push_str(" > ");
        base.push_str(d.kind_str);
        base.push(' ');
        base.push_str(&d.name);

        if d.end - d.start <= self.max_chars {
            self.push(d.start, d.end, base);
            return;
        }
        let children = self.kids.get(&di).cloned().unwrap_or_default();
        if children.is_empty() {
            // Leaf def still too big: line windows with #partN.
            self.emit_split(d.start, d.end, &base);
            return;
        }
        // Recurse into child defs; cover interstitial gaps with the node's
        // own header so no source is lost.
        let mut chain2 = chain.to_vec();
        chain2.push(format!("{} {}", d.kind_str, d.name));
        let mut prev = d.start;
        for ki in children {
            let (ks, ke) = (defs[ki].start, defs[ki].end);
            self.emit_gap(prev, ks, &base);
            self.emit_def(ki, &chain2);
            prev = prev.max(ke);
        }
        self.emit_gap(prev, d.end, &base);
    }

    fn emit_gap(&mut self, start: usize, end: usize, base: &str) {
        let Some((s, e)) = trim(self.content, start, end) else {
            return;
        };
        if e - s <= self.max_chars {
            self.push(s, e, base.to_string());
        } else {
            self.emit_split(s, e, base);
        }
    }

    /// Oversized non-def content: window-split with `path > lines A-B`.
    fn emit_oversized_nondef(&mut self, start: usize, end: usize) {
        let Some((s, e)) = trim(self.content, start, end) else {
            return;
        };
        if e - s <= self.max_chars {
            self.push(s, e, self.path.to_string());
            return;
        }
        for (ws, we) in window_split(self.content, self.lines, s, e, self.max_chars) {
            let a = self.lines.line_at(ws);
            let b = self.lines.line_at(we - 1);
            self.push(ws, we, format!("{} > lines {}-{}", self.path, a, b));
        }
    }

    /// Split [start, end) into line windows, headers `<base>#partN`.
    fn emit_split(&mut self, start: usize, end: usize, base: &str) {
        let parts = window_split(self.content, self.lines, start, end, self.max_chars);
        if parts.len() == 1 {
            let (s, e) = parts[0];
            self.push(s, e, base.to_string());
            return;
        }
        for (n, (s, e)) in parts.into_iter().enumerate() {
            self.push(s, e, format!("{base}#part{}", n + 1));
        }
    }

    fn push_trimmed(&mut self, start: usize, end: usize, header: String) {
        if let Some((s, e)) = trim(self.content, start, end) {
            self.push(s, e, header);
        }
    }

    fn push(&mut self, start: usize, end: usize, header: String) {
        self.out
            .push(make_chunk(self.content, self.lines, start, end, header));
    }
}

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

/// Trim ASCII whitespace off both ends; None if nothing remains.
fn trim(content: &[u8], start: usize, end: usize) -> Option<(usize, usize)> {
    let mut s = start.min(end);
    let mut e = end.min(content.len());
    while s < e && content[s].is_ascii_whitespace() {
        s += 1;
    }
    while e > s && content[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    if s < e {
        Some((s, e))
    } else {
        None
    }
}

fn make_chunk(content: &[u8], lines: &Lines, start: usize, end: usize, header: String) -> ChunkRec {
    let src = String::from_utf8_lossy(&content[start..end]);
    ChunkRec {
        start_line: lines.line_at(start),
        end_line: lines.line_at(end - 1),
        text: format!("{header}\n{src}"),
        header,
    }
}

/// Sliding line-window fallback: chunks of <= max_chars, 10% line overlap,
/// header = `path > lines A-B`.
fn window_fallback(path: &str, content: &[u8], lines: &Lines, max_chars: usize) -> Vec<ChunkRec> {
    window_split(content, lines, 0, content.len(), max_chars)
        .into_iter()
        .map(|(s, e)| {
            let a = lines.line_at(s);
            let b = lines.line_at(e - 1);
            make_chunk(content, lines, s, e, format!("{path} > lines {a}-{b}"))
        })
        .collect()
}

/// Split [start, end) into windows of whole lines totaling <= max_chars,
/// with ~10% line overlap between consecutive windows. Lines longer than
/// max_chars are hard-split at UTF-8 boundaries.
fn window_split(
    content: &[u8],
    lines: &Lines,
    start: usize,
    end: usize,
    max_chars: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if start >= end {
        return out;
    }
    // Line indices (0-based) intersecting [start, end).
    let li_start = lines.line_at(start) - 1;
    let li_end = lines.line_at(end - 1) - 1;
    let line_start = |li: u32| lines.starts[li as usize];
    let line_end = |li: u32| {
        if (li as usize + 1) < lines.starts.len() {
            lines.starts[li as usize + 1]
        } else {
            content.len()
        }
    };
    let mut i = li_start;
    while i <= li_end {
        let w_start = start.max(line_start(i));
        let mut j = i;
        while j < li_end && line_end(j + 1).min(end) - w_start <= max_chars {
            j += 1;
        }
        let w_end = end.min(line_end(j));
        if w_end - w_start > max_chars {
            // Single line longer than max_chars: hard-split at char boundaries.
            let mut p = w_start;
            while p < w_end {
                let mut e = (p + max_chars).min(w_end);
                while e > p + 1 && e < w_end && (content[e] & 0xC0) == 0x80 {
                    e -= 1;
                }
                out.push((p, e));
                p = e;
            }
        } else {
            out.push((w_start, w_end));
        }
        if j >= li_end {
            break; // the range is covered; an overlap-only tail would duplicate it
        }
        let count = j - i + 1;
        // ~10% line overlap, at least one line, never the whole window.
        let overlap = count.div_ceil(10).min(count - 1);
        i = (j + 1 - overlap).max(i + 1);
    }
    // SPEC-P9: a tail window that adds nothing but closing brackets and
    // blank lines beyond the overlap (a split that landed just before a
    // function's final `}`) is folded into the previous window instead of
    // becoming a chunk of its own — such chunks embed as their path header
    // alone and surface as `}` hits.
    if out.len() >= 2 {
        let (ts, te) = out[out.len() - 1];
        let prev_end = out[out.len() - 2].1;
        let fresh = &content[prev_end.max(ts).min(te)..te];
        let trivial = fresh
            .iter()
            .all(|b| b.is_ascii_whitespace() || matches!(b, b'{' | b'}' | b'(' | b')' | b'[' | b']' | b';' | b','));
        if trivial {
            out.pop();
            let n = out.len();
            out[n - 1].1 = te;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Total source-slice length of a chunk (text minus header and newline).
    fn src_len(c: &ChunkRec) -> usize {
        c.text.len() - c.header.len() - 1
    }

    const RUST_SRC: &str = concat!(
        "use std::io;\n",
        "// a comment\n",
        "const MAX: usize = 10;\n",
        "\n",
        "fn alpha(x: i32) -> i32 {\n",
        "    x + 1\n",
        "}\n",
        "\n",
        "fn beta(y: i32) -> i32 {\n",
        "    y * 2\n",
        "}\n",
        "\n",
        "fn gamma(z: i32) -> i32 {\n",
        "    z - 3\n",
        "}\n",
    );

    #[test]
    fn rust_fns_each_get_own_chunk() {
        let chunks = chunks(Lang::Rust, "src/lib.rs", RUST_SRC.as_bytes(), 60);
        let find = |name: &str| chunks.iter().find(|c| c.header.contains(name)).cloned();
        let alpha = find("fn alpha").expect("alpha chunk");
        assert_eq!(alpha.header, "src/lib.rs > fn alpha");
        assert!(alpha.text.contains("fn alpha(x: i32) -> i32 {"));
        assert_eq!(alpha.start_line, 5);
        assert_eq!(alpha.end_line, 7);
        assert!(find("fn beta").is_some());
        assert!(find("fn gamma").is_some());
        // Each header carries the path.
        assert!(chunks.iter().all(|c| c.header.starts_with("src/lib.rs")));
        // text = header + "\n" + source slice.
        for c in &chunks {
            assert!(c.text.starts_with(&c.header));
            assert_eq!(c.text.as_bytes()[c.header.len()], b'\n');
        }
    }

    #[test]
    fn huge_fn_splits_with_part_headers_and_overlap() {
        let mut src = String::from("fn huge() {\n");
        for i in 0..60 {
            src.push_str(&format!("    let v{i:02} = compute_something_quite_long({i});\n"));
        }
        src.push_str("}\n");
        let chunks = chunks(Lang::Rust, "src/huge.rs", src.as_bytes(), 400);
        let parts: Vec<&ChunkRec> = chunks.iter().filter(|c| c.header.contains("#part")).collect();
        assert!(parts.len() >= 2, "expected split parts: {chunks:?}");
        assert_eq!(parts[0].header, "src/huge.rs > fn huge#part1");
        assert!(parts[1].header.contains("#part2"));
        // ~10% overlap: consecutive part windows share at least one line.
        assert!(parts[0].end_line >= parts[1].start_line, "{parts:?}");
        for p in &parts {
            assert!(src_len(p) <= 400, "part too big: {} chars", src_len(p));
        }
    }

    #[test]
    fn python_class_methods_carry_class_scope() {
        let src = concat!(
            "class Animal:\n",
            "    def speak(self):\n",
            "        return self.voice()\n",
            "\n",
            "    def voice(self):\n",
            "        return \"...\"\n",
        );
        // max_chars small enough that the class itself must be split.
        let chunks = chunks(Lang::Python, "pkg/animals.py", src.as_bytes(), 60);
        let speak = chunks
            .iter()
            .find(|c| c.header.contains("speak"))
            .expect("speak chunk");
        assert_eq!(speak.header, "pkg/animals.py > class Animal > fn speak");
        let voice = chunks
            .iter()
            .find(|c| c.header.contains("voice"))
            .expect("voice chunk");
        assert_eq!(voice.header, "pkg/animals.py > class Animal > fn voice");
    }

    #[test]
    fn tiny_file_single_chunk_with_path_header() {
        let src = b"fn main() {}\n";
        let tiny = chunks(Lang::Rust, "src/main.rs", src, 1200);
        assert_eq!(tiny.len(), 1);
        assert_eq!(tiny[0].header, "src/main.rs");
        assert_eq!(tiny[0].start_line, 1);
        assert_eq!(tiny[0].end_line, 1);
        assert_eq!(tiny[0].text, "src/main.rs\nfn main() {}\n");
        // Same for unknown languages.
        let unknown = chunks(Lang::Unknown, "README.txt", b"hello\n", 1200);
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].header, "README.txt");
    }

    #[test]
    fn unknown_language_window_fallback() {
        let mut src = String::new();
        for i in 0..100 {
            src.push_str(&format!("log line number {i:03} with some padding text\n"));
        }
        let chunks = chunks(Lang::Unknown, "logs/app.log", src.as_bytes(), 300);
        assert!(chunks.len() >= 3, "{chunks:?}");
        for c in &chunks {
            assert!(src_len(c) <= 300, "window too big: {}", src_len(c));
            // Header format: "path > lines A-B" matches the chunk line range.
            let hdr = format!("logs/app.log > lines {}-{}", c.start_line, c.end_line);
            assert_eq!(c.header, hdr);
        }
        // 10% line overlap between consecutive windows.
        assert!(chunks[0].end_line >= chunks[1].start_line);
    }

    #[test]
    fn empty_and_whitespace_files_yield_no_chunks() {
        assert!(chunks(Lang::Rust, "a.rs", b"", 100).is_empty());
        assert!(chunks(Lang::Rust, "a.rs", b"  \n\t \n", 100).is_empty());
        assert!(chunks(Lang::Unknown, "a.txt", b"", 100).is_empty());
        assert!(chunks(Lang::Python, "a.py", b"\n\n", 100).is_empty());
    }

    #[test]
    fn max_chars_respected_across_languages() {
        let max = 100;
        let cases: [(Lang, &str, String); 3] = [
            (Lang::Rust, "t/a.rs", RUST_SRC.to_string()),
            (
                Lang::Python,
                "t/a.py",
                concat!(
                    "import os\n",
                    "MAX_RETRIES = 3\n",
                    "class Animal:\n",
                    "    def speak(self):\n",
                    "        return self.voice()\n",
                    "    def voice(self):\n",
                    "        return \"...\"\n",
                    "def make_animal(name):\n",
                    "    a = Animal()\n",
                    "    return a.speak()\n",
                )
                .to_string(),
            ),
            (
                Lang::Go,
                "t/a.go",
                concat!(
                    "package main\n",
                    "func NewServer(port int) *Server { return &Server{port: port} }\n",
                    "func (s *Server) Start() error { return nil }\n",
                    "func main() { NewServer(8080) }\n",
                )
                .to_string(),
            ),
        ];
        for (lang, path, src) in cases {
            assert!(src.len() > max, "{path}: fixture should exercise splitting");
            for c in chunks(lang, path, src.as_bytes(), max) {
                assert!(
                    src_len(&c) <= max + max / 10,
                    "{path}: chunk {:?} is {} chars (max {max})",
                    c.header,
                    src_len(&c)
                );
            }
        }
    }

    #[test]
    fn preamble_run_merges_imports_comments_consts() {
        let chunks = chunks(Lang::Rust, "src/lib.rs", RUST_SRC.as_bytes(), 60);
        let preamble = &chunks[0];
        assert_eq!(preamble.header, "src/lib.rs");
        assert!(preamble.text.contains("use std::io;"));
        assert!(preamble.text.contains("// a comment"));
        assert!(preamble.text.contains("const MAX: usize = 10;"));
        assert_eq!(preamble.start_line, 1);
        assert_eq!(preamble.end_line, 3);
    }

    #[test]
    fn rust_mod_scope_chain_in_header() {
        let src = concat!(
            "mod outer {\n",
            "    fn inner_fn() {\n",
            "        do_work();\n",
            "    }\n",
            "}\n",
            "fn top_level() {}\n",
        );
        let chunks = chunks(Lang::Rust, "src/m.rs", src.as_bytes(), 50);
        let inner = chunks
            .iter()
            .find(|c| c.header.contains("inner_fn"))
            .expect("inner_fn chunk");
        assert_eq!(inner.header, "src/m.rs > mod outer > fn inner_fn");
        let top = chunks
            .iter()
            .find(|c| c.header.contains("top_level"))
            .expect("top_level chunk");
        assert_eq!(top.header, "src/m.rs > fn top_level");
    }
}
