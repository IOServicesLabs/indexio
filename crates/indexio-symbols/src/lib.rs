//! indexio-symbols: tree-sitter symbol + call-site extraction.
//! Contract: docs/SPEC.md "indexio-symbols".
//!
//! Strategy: one lazily-compiled `Parser`-less table per language (grammar +
//! two tree-sitter queries). Definitions are captured as `@name` plus a
//! kind-tagged `@def.<kind>` capture; call sites as `@call` plus the callee
//! expression `@callee` (reduced to its last identifier segment). Everything
//! is wrapped in `catch_unwind`: untrusted input must never panic.
#![forbid(unsafe_code)]

use indexio_types::{CallRec, Lang, SymbolKind, SymbolRec};
use std::collections::HashSet;
use std::sync::OnceLock;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Node, Parser, Query, QueryCursor};

/// Per-language extraction tables (compiled once, shared).
struct LangSpec {
    language: Language,
    defs: Query,
    calls: Query,
    /// Node kinds that act as a named caller scope for call edges.
    caller_kinds: &'static [&'static str],
    /// Node kinds that contribute a path component to a symbol's scope.
    container_kinds: &'static [&'static str],
}

// ---------------------------------------------------------------------------
// Query sources
// ---------------------------------------------------------------------------

const RUST_DEFS: &str = r#"
(function_item name: (identifier) @name) @def.fn
(struct_item name: (type_identifier) @name) @def.struct
(union_item name: (type_identifier) @name) @def.struct
(enum_item name: (type_identifier) @name) @def.enum
(trait_item name: (type_identifier) @name) @def.trait
(impl_item type: (type_identifier) @name) @def.impl
(impl_item type: (generic_type type: (type_identifier) @name)) @def.impl
(mod_item name: (identifier) @name) @def.mod
(const_item name: (identifier) @name) @def.const
(static_item name: (identifier) @name) @def.const
(type_item name: (type_identifier) @name) @def.type
"#;

const RUST_CALLS: &str = r#"
(call_expression function: (_) @callee) @call
(macro_invocation macro: (_) @callee) @call
"#;

const PYTHON_DEFS: &str = r#"
(function_definition name: (identifier) @name) @def.fn
(class_definition name: (identifier) @name) @def.class
(module (expression_statement (assignment left: (identifier) @name) @def.const))
"#;

const PYTHON_CALLS: &str = r#"
(call function: (_) @callee) @call
"#;

const GO_DEFS: &str = r#"
(function_declaration name: (identifier) @name) @def.fn
(method_declaration name: (field_identifier) @name) @def.method
(type_declaration (type_spec name: (type_identifier) @name type: (struct_type)) @def.struct)
(type_declaration (type_spec name: (type_identifier) @name type: (interface_type)) @def.interface)
(type_declaration (type_spec name: (type_identifier) @name) @def.type)
(type_declaration (type_alias name: (type_identifier) @name) @def.type)
(const_declaration (const_spec name: (identifier) @name @def.const))
"#;

const GO_CALLS: &str = r#"
(call_expression function: (_) @callee) @call
"#;

const TS_DEFS: &str = r#"
(function_declaration name: (identifier) @name) @def.fn
(generator_function_declaration name: (identifier) @name) @def.fn
(method_definition name: (property_identifier) @name) @def.method
(class_declaration name: (type_identifier) @name) @def.class
(interface_declaration name: (type_identifier) @name) @def.interface
(enum_declaration name: (identifier) @name) @def.enum
(program (lexical_declaration (variable_declarator name: (identifier) @name) @def.const))
(program (export_statement declaration: (lexical_declaration (variable_declarator name: (identifier) @name) @def.const)))
"#;

const TS_CALLS: &str = r#"
(call_expression function: (_) @callee) @call
(new_expression constructor: (_) @callee) @call
"#;

const JAVA_DEFS: &str = r#"
(method_declaration name: (identifier) @name) @def.method
(constructor_declaration name: (identifier) @name) @def.method
(class_declaration name: (identifier) @name) @def.class
(interface_declaration name: (identifier) @name) @def.interface
(enum_declaration name: (identifier) @name) @def.enum
(field_declaration declarator: (variable_declarator name: (identifier) @name) @def.const)
"#;

const JAVA_CALLS: &str = r#"
(method_invocation name: (identifier) @callee) @call
(object_creation_expression type: (_) @callee) @call
"#;

const CPP_DEFS: &str = r#"
(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def.fn
(function_definition declarator: (function_declarator declarator: (field_identifier) @name)) @def.fn
(function_definition declarator: (function_declarator declarator: (qualified_identifier name: (identifier) @name)) @def.fn)
(function_definition declarator: (pointer_declarator declarator: (function_declarator declarator: (identifier) @name))) @def.fn
(function_definition declarator: (pointer_declarator declarator: (function_declarator declarator: (qualified_identifier name: (identifier) @name)))) @def.fn
(struct_specifier name: (type_identifier) @name body: (field_declaration_list)) @def.struct
(class_specifier name: (type_identifier) @name body: (field_declaration_list)) @def.class
(enum_specifier name: (type_identifier) @name body: (enumerator_list)) @def.enum
(namespace_definition name: (namespace_identifier) @name) @def.mod
"#;

const CPP_CALLS: &str = r#"
(call_expression function: (_) @callee) @call
"#;

// ---------------------------------------------------------------------------
// Lazy per-language tables
// ---------------------------------------------------------------------------

fn build_spec(
    language: Language,
    defs_src: &str,
    calls_src: &str,
    caller_kinds: &'static [&'static str],
    container_kinds: &'static [&'static str],
) -> Option<LangSpec> {
    let defs = Query::new(&language, defs_src).ok()?;
    let calls = Query::new(&language, calls_src).ok()?;
    Some(LangSpec {
        language,
        defs,
        calls,
        caller_kinds,
        container_kinds,
    })
}

fn spec(lang: Lang) -> Option<&'static LangSpec> {
    static RUST: OnceLock<Option<LangSpec>> = OnceLock::new();
    static PYTHON: OnceLock<Option<LangSpec>> = OnceLock::new();
    static GO: OnceLock<Option<LangSpec>> = OnceLock::new();
    static TS: OnceLock<Option<LangSpec>> = OnceLock::new();
    static JAVA: OnceLock<Option<LangSpec>> = OnceLock::new();
    static CPP: OnceLock<Option<LangSpec>> = OnceLock::new();
    let cell = match lang {
        Lang::Rust => &RUST,
        Lang::Python => &PYTHON,
        Lang::Go => &GO,
        Lang::TsJs => &TS,
        Lang::Java => &JAVA,
        Lang::Cpp => &CPP,
        Lang::Unknown | Lang::Text => return None,
    };
    cell.get_or_init(|| match lang {
        Lang::Rust => build_spec(
            tree_sitter_rust::LANGUAGE.into(),
            RUST_DEFS,
            RUST_CALLS,
            &["function_item"],
            &[
                "function_item",
                "struct_item",
                "union_item",
                "enum_item",
                "trait_item",
                "impl_item",
                "mod_item",
            ],
        ),
        Lang::Python => build_spec(
            tree_sitter_python::LANGUAGE.into(),
            PYTHON_DEFS,
            PYTHON_CALLS,
            &["function_definition"],
            &["function_definition", "class_definition"],
        ),
        Lang::Go => build_spec(
            tree_sitter_go::LANGUAGE.into(),
            GO_DEFS,
            GO_CALLS,
            &["function_declaration", "method_declaration"],
            &["function_declaration", "method_declaration"],
        ),
        // TSX grammar covers TS + JS + TSX + JSX.
        Lang::TsJs => build_spec(
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            TS_DEFS,
            TS_CALLS,
            &[
                "function_declaration",
                "generator_function_declaration",
                "method_definition",
            ],
            &[
                "function_declaration",
                "generator_function_declaration",
                "method_definition",
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
            ],
        ),
        Lang::Java => build_spec(
            tree_sitter_java::LANGUAGE.into(),
            JAVA_DEFS,
            JAVA_CALLS,
            &["method_declaration", "constructor_declaration"],
            &[
                "method_declaration",
                "constructor_declaration",
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
            ],
        ),
        // C++ grammar; covers C reasonably via error recovery.
        Lang::Cpp => build_spec(
            tree_sitter_cpp::LANGUAGE.into(),
            CPP_DEFS,
            CPP_CALLS,
            &["function_definition"],
            &[
                "function_definition",
                "namespace_definition",
                "struct_specifier",
                "class_specifier",
                "enum_specifier",
            ],
        ),
        Lang::Unknown | Lang::Text => None,
    })
    .as_ref()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Whether a tree-sitter grammar is available for `lang`.
pub fn supported(lang: Lang) -> bool {
    spec(lang).is_some()
}

/// Extract (symbols, calls) from source content. Never panics on
/// unparseable input; returns empty vecs for unsupported languages.
pub fn extract(lang: Lang, content: &[u8]) -> (Vec<SymbolRec>, Vec<CallRec>) {
    let Some(s) = spec(lang) else {
        return (Vec::new(), Vec::new());
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| extract_inner(s, lang, content)))
        .unwrap_or_default()
}

fn extract_inner(spec: &LangSpec, lang: Lang, content: &[u8]) -> (Vec<SymbolRec>, Vec<CallRec>) {
    let mut parser = Parser::new();
    if parser.set_language(&spec.language).is_err() {
        return (Vec::new(), Vec::new());
    }
    let Some(tree) = parser.parse(content, None) else {
        return (Vec::new(), Vec::new());
    };
    let root = tree.root_node();
    let ranged = dedup_symbols(collect_symbols(spec, lang, root, content));
    let mut symbols: Vec<SymbolRec> = ranged.into_iter().map(|(s, _)| s).collect();
    let mut calls = collect_calls(spec, root, content);
    symbols.sort_by_key(|s| s.line);
    calls.sort_by_key(|c| c.line);
    (symbols, calls)
}

/// Shared dedup for `extract` and `outline` (SPEC-P6 §1: same symbols, same
/// lines): drop Go's generic `Type` twin of a Struct/Interface at the same
/// (name, line), then keep the first (name, kind, line) occurrence.
fn dedup_symbols(mut symbols: Vec<(SymbolRec, u32)>) -> Vec<(SymbolRec, u32)> {
    let specific: HashSet<(String, u32)> = symbols
        .iter()
        .filter(|(s, _)| s.kind != SymbolKind::Type)
        .map(|(s, _)| (s.name.clone(), s.line))
        .collect();
    symbols.retain(|(s, _)| {
        s.kind != SymbolKind::Type || !specific.contains(&(s.name.clone(), s.line))
    });
    let mut seen: HashSet<(String, u8, u32)> = HashSet::new();
    symbols.retain(|(s, _)| seen.insert((s.name.clone(), s.kind.as_u8(), s.line)));
    symbols
}

// ---------------------------------------------------------------------------
// Outline (SPEC-P6 §1): the same definitions as `extract`, with node ranges
// ---------------------------------------------------------------------------

/// One definition with its full source range (query-time view, never
/// persisted — SPEC-P6 §1).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OutlineItem {
    pub name: String,
    pub kind: SymbolKind,
    /// Enclosing definition path, as `SymbolRec::scope`.
    pub scope: String,
    /// 1-based inclusive line of the definition node's first line.
    pub start_line: u32,
    /// 1-based inclusive line of the definition node's last line.
    pub end_line: u32,
}

/// The outline of a markdown document (SPEC-P10 §38): one item per ATX
/// heading (`#` to `######`, outside fenced code), named by its text, kind
/// `Section`, scoped by its ancestor headings (`Install::Windows`), and
/// spanning to the line before the next heading of the same or a higher
/// level. `read_span` with no `end` on a heading line then returns the
/// whole section, as it returns a whole function in code.
pub fn markdown_outline(content: &[u8]) -> Vec<OutlineItem> {
    let text = String::from_utf8_lossy(content);
    let mut heads: Vec<(u32, usize, String)> = Vec::new(); // (line, level, title)
    let mut in_fence = false;
    let mut total = 0u32;
    for (i, raw) in text.lines().enumerate() {
        total = i as u32 + 1;
        let line = raw.trim_end();
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence || !line.starts_with('#') {
            continue;
        }
        let level = line.chars().take_while(|&c| c == '#').count();
        if level > 6 || !line[level..].starts_with(' ') {
            continue;
        }
        let title = line[level..].trim().trim_end_matches('#').trim();
        if title.is_empty() {
            continue;
        }
        heads.push((i as u32 + 1, level, title.to_string()));
    }
    let mut out = Vec::with_capacity(heads.len());
    for (idx, (line, level, title)) in heads.iter().enumerate() {
        let end = heads[idx + 1..]
            .iter()
            .find(|(_, l, _)| l <= level)
            .map(|(next, _, _)| next - 1)
            .unwrap_or(total)
            .max(*line);
        // ancestors: the nearest preceding heading of each higher level
        let mut scope: Vec<&str> = Vec::new();
        let mut want = level - 1;
        for (_, l, t) in heads[..idx].iter().rev() {
            if want == 0 {
                break;
            }
            if *l == want {
                scope.push(t);
                want -= 1;
            } else if *l < want {
                want = *l;
                scope.push(t);
                want -= 1;
            }
        }
        scope.reverse();
        out.push(OutlineItem {
            name: title.clone(),
            kind: SymbolKind::Section,
            scope: scope.join("::"),
            start_line: *line,
            end_line: end,
        });
    }
    out
}

/// The definitions `extract` reports, each with the full range of its
/// definition node. Sorted by (start_line asc, end_line desc) so containers
/// precede their members. Empty for unsupported languages; never panics.
pub fn outline(lang: Lang, content: &[u8]) -> Vec<OutlineItem> {
    let Some(s) = spec(lang) else {
        return Vec::new();
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| outline_inner(s, lang, content)))
        .unwrap_or_default()
}

fn outline_inner(spec: &LangSpec, lang: Lang, content: &[u8]) -> Vec<OutlineItem> {
    let mut parser = Parser::new();
    if parser.set_language(&spec.language).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(content, None) else {
        return Vec::new();
    };
    let root = tree.root_node();
    let mut items: Vec<OutlineItem> = dedup_symbols(collect_symbols(spec, lang, root, content))
        .into_iter()
        .map(|(s, end_line)| OutlineItem {
            name: s.name,
            kind: s.kind,
            scope: s.scope,
            start_line: s.line,
            end_line: end_line.max(s.line),
        })
        .collect();
    items.sort_by(|a, b| {
        a.start_line
            .cmp(&b.start_line)
            .then_with(|| b.end_line.cmp(&a.end_line))
    });
    items
}

/// Innermost outline item whose range covers 1-based `line` (smallest span
/// wins; ties resolve to the later-starting item).
pub fn enclosing(items: &[OutlineItem], line: u32) -> Option<&OutlineItem> {
    items
        .iter()
        .filter(|it| it.start_line <= line && line <= it.end_line)
        .min_by_key(|it| (it.end_line - it.start_line, u32::MAX - it.start_line))
}

// ---------------------------------------------------------------------------
// Symbol collection
// ---------------------------------------------------------------------------

/// Definitions with the 1-based last line of their definition node
/// (SPEC-P6 §1: `extract` drops it, `outline` keeps it).
fn collect_symbols(
    spec: &LangSpec,
    lang: Lang,
    root: Node,
    content: &[u8],
) -> Vec<(SymbolRec, u32)> {
    let mut out = Vec::new();
    let names = spec.defs.capture_names();
    let Some(name_ix) = names.iter().position(|n| *n == "name") else {
        return out;
    };
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&spec.defs, root, content);
    while let Some(m) = matches.next() {
        let mut name_node = None;
        let mut def_node = None;
        let mut kind: Option<SymbolKind> = None;
        for cap in m.captures {
            if cap.index as usize == name_ix {
                name_node = Some(cap.node);
            } else if let Some(tag) = names[cap.index as usize].strip_prefix("def.") {
                def_node = Some(cap.node);
                kind = Some(match tag {
                    "fn" => SymbolKind::Fn,
                    "method" => SymbolKind::Method,
                    "struct" => SymbolKind::Struct,
                    "class" => SymbolKind::Class,
                    "enum" => SymbolKind::Enum,
                    "trait" => SymbolKind::Trait,
                    "impl" => SymbolKind::Impl,
                    "mod" => SymbolKind::Mod,
                    "const" => SymbolKind::Const,
                    "interface" => SymbolKind::Interface,
                    _ => SymbolKind::Type,
                });
            }
        }
        let (Some(name_node), Some(def_node), Some(mut kind)) = (name_node, def_node, kind) else {
            continue;
        };
        let name = text(content, name_node);
        if name.is_empty() {
            continue;
        }

        // Language-specific filters and kind adjustments.
        match lang {
            Lang::Rust
                if kind == SymbolKind::Fn && has_ancestor(def_node, &["impl_item", "trait_item"]) => {
                    kind = SymbolKind::Method;
                }
            Lang::Python => {
                if kind == SymbolKind::Fn && has_ancestor(def_node, &["class_definition"]) {
                    kind = SymbolKind::Method;
                }
                if kind == SymbolKind::Const && !is_screaming_const(&name) {
                    continue;
                }
            }
            Lang::TsJs
                if kind == SymbolKind::Const && !is_const_decl(content, def_node) => {
                    continue;
                }
            Lang::Java
                if kind == SymbolKind::Const && !is_static_final(content, def_node) => {
                    continue;
                }
            _ => {}
        }

        // Scope: nearest enclosing definition names, outermost first.
        let scope = if lang == Lang::Go && kind == SymbolKind::Method {
            go_receiver_type(content, def_node).unwrap_or_default()
        } else {
            let (s, impl_contributed) = scope_of(content, def_node, spec.container_kinds);
            // Rust methods in impl blocks: "Type::method" form (best-effort).
            if lang == Lang::Rust && kind == SymbolKind::Method && impl_contributed {
                if s.is_empty() { name.clone() } else { format!("{s}::{name}") }
            } else {
                s
            }
        };

        let pos = name_node.start_position();
        let end_line = def_node.end_position().row as u32 + 1;
        out.push((
            SymbolRec {
                name,
                kind,
                line: pos.row as u32 + 1,
                col: pos.column as u32,
                scope,
            },
            end_line,
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Call collection
// ---------------------------------------------------------------------------

fn collect_calls(spec: &LangSpec, root: Node, content: &[u8]) -> Vec<CallRec> {
    let mut out = Vec::new();
    let names = spec.calls.capture_names();
    let (Some(call_ix), Some(callee_ix)) = (
        names.iter().position(|n| *n == "call"),
        names.iter().position(|n| *n == "callee"),
    ) else {
        return out;
    };
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&spec.calls, root, content);
    while let Some(m) = matches.next() {
        let mut call_node = None;
        let mut callee_node = None;
        for cap in m.captures {
            if cap.index as usize == call_ix {
                call_node = Some(cap.node);
            } else if cap.index as usize == callee_ix {
                callee_node = Some(cap.node);
            }
        }
        let (Some(call_node), Some(callee_node)) = (call_node, callee_node) else {
            continue;
        };
        let callee = callee_name(content, callee_node);
        if callee.is_empty() {
            continue;
        }
        let caller = caller_of(spec, content, call_node);
        out.push(CallRec {
            callee,
            caller,
            line: call_node.start_position().row as u32 + 1,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Node helpers
// ---------------------------------------------------------------------------

fn text(content: &[u8], node: Node) -> String {
    String::from_utf8_lossy(&content[node.start_byte()..node.end_byte()]).into_owned()
}

/// Reduce a callee expression to its last identifier segment:
/// `a.b.c()` -> "c", `foo::bar::baz()` -> "baz", `foo()` -> "foo".
fn callee_name(content: &[u8], node: Node) -> String {
    const FIELDS: [&str; 7] = [
        "field", "property", "attribute", "name", "function", "macro", "type",
    ];
    let mut cur = node;
    'outer: loop {
        for f in FIELDS {
            if let Some(c) = cur.child_by_field_name(f) {
                cur = c;
                continue 'outer;
            }
        }
        break;
    }
    text(content, cur).trim().to_string()
}

/// Name of a definition-like node, handling nodes whose name is not in a
/// plain `name` field (Rust impl type, C++ declarator chains).
fn node_name(content: &[u8], node: Node) -> Option<String> {
    if node.kind() == "impl_item" {
        return impl_type_name(content, node);
    }
    if let Some(n) = node.child_by_field_name("name") {
        let s = text(content, n);
        if !s.is_empty() {
            return Some(s);
        }
    }
    if node.kind() == "function_definition" {
        return cpp_fn_name(content, node);
    }
    None
}

fn impl_type_name(content: &[u8], node: Node) -> Option<String> {
    let t = node.child_by_field_name("type")?;
    resolve_type_name(content, t)
}

fn resolve_type_name(content: &[u8], node: Node) -> Option<String> {
    match node.kind() {
        "type_identifier" | "identifier" => Some(text(content, node)),
        "generic_type" => node
            .child_by_field_name("type")
            .and_then(|t| resolve_type_name(content, t)),
        "scoped_type_identifier" => node.child_by_field_name("name").map(|n| text(content, n)),
        _ => None,
    }
}

/// Walk a C++ declarator chain down to the declared name.
fn cpp_fn_name(content: &[u8], node: Node) -> Option<String> {
    let mut cur = node;
    while let Some(d) = cur.child_by_field_name("declarator") {
        cur = d;
    }
    if cur.kind() == "qualified_identifier" {
        if let Some(n) = cur.child_by_field_name("name") {
            cur = n;
        }
    }
    if cur.kind() == "template_function" || cur.kind() == "template_method" {
        if let Some(n) = cur.child_by_field_name("name") {
            cur = n;
        }
    }
    let s = text(content, cur);
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Nearest enclosing function/method name for a call site; "" at top level.
fn caller_of(spec: &LangSpec, content: &[u8], node: Node) -> String {
    let mut cur = node.parent();
    while let Some(n) = cur {
        if spec.caller_kinds.contains(&n.kind()) {
            if let Some(name) = node_name(content, n) {
                return name;
            }
        }
        cur = n.parent();
    }
    String::new()
}

/// Enclosing scope path for a definition: container names, outermost first,
/// joined with "::". Also reports whether a Rust `impl_item` contributed.
fn scope_of(content: &[u8], def_node: Node, container_kinds: &[&str]) -> (String, bool) {
    let mut comps: Vec<String> = Vec::new();
    let mut impl_contributed = false;
    let mut cur = def_node.parent();
    while let Some(anc) = cur {
        if container_kinds.contains(&anc.kind()) {
            if let Some(n) = node_name(content, anc) {
                if anc.kind() == "impl_item" {
                    impl_contributed = true;
                }
                comps.push(n);
            }
        }
        cur = anc.parent();
    }
    comps.reverse();
    (comps.join("::"), impl_contributed)
}

fn has_ancestor(node: Node, kinds: &[&str]) -> bool {
    let mut cur = node.parent();
    while let Some(n) = cur {
        if kinds.contains(&n.kind()) {
            return true;
        }
        cur = n.parent();
    }
    false
}

/// `MAX_RETRIES`-style name: uppercase/digits/underscore, >=1 uppercase letter.
fn is_screaming_const(name: &str) -> bool {
    name.chars().any(|c| c.is_ascii_uppercase())
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// TS/JS: is this `variable_declarator` part of a `const` (not `let`) decl?
fn is_const_decl(content: &[u8], declarator: Node) -> bool {
    let Some(parent) = declarator.parent() else {
        return false;
    };
    if parent.kind() != "lexical_declaration" {
        return false;
    }
    parent
        .child_by_field_name("kind")
        .map(|k| text(content, k) == "const")
        .unwrap_or(false)
}

/// Java: is this `variable_declarator` a `static final` field?
fn is_static_final(content: &[u8], declarator: Node) -> bool {
    let Some(parent) = declarator.parent() else {
        return false;
    };
    if parent.kind() != "field_declaration" {
        return false;
    }
    let mut walk = parent.walk();
    for child in parent.children(&mut walk) {
        if child.kind() == "modifiers" {
            let m = text(content, child);
            let has = |kw: &str| m.split_whitespace().any(|w| w == kw);
            return has("static") && has("final");
        }
    }
    false
}

/// Go: receiver type name of a method_declaration (best-effort).
fn go_receiver_type(content: &[u8], node: Node) -> Option<String> {
    fn first_type_ident(content: &[u8], node: Node) -> Option<String> {
        if node.kind() == "type_identifier" {
            return Some(text(content, node));
        }
        let mut walk = node.walk();
        for child in node.children(&mut walk) {
            if let Some(s) = first_type_ident(content, child) {
                return Some(s);
            }
        }
        None
    }
    let recv = node.child_by_field_name("receiver")?;
    first_type_ident(content, recv)
}

// ---------------------------------------------------------------------------
// Chunking (SPEC-P2 §1) — cAST-lite semantic chunker, see src/chunking.rs.
// ---------------------------------------------------------------------------

pub mod chunking;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_headings_become_sections_with_ranges_and_scopes() {
        let md = b"# Title\n\nintro\n\n## Install\n\ntext\n\n### Windows\n\n```sh\n# not a heading\n```\n\n### Linux\n\nx\n\n## Use\n\ny\n";
        let o = markdown_outline(md);
        let names: Vec<&str> = o.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, ["Title", "Install", "Windows", "Linux", "Use"]);
        assert!(o.iter().all(|i| i.kind == SymbolKind::Section));
        let by = |n: &str| o.iter().find(|i| i.name == n).unwrap();
        assert_eq!((by("Title").start_line, by("Title").end_line), (1, 21));
        assert_eq!((by("Install").start_line, by("Install").end_line), (5, 18));
        assert_eq!((by("Windows").start_line, by("Windows").end_line), (9, 14));
        assert_eq!(by("Windows").scope, "Title::Install");
        assert_eq!(by("Use").scope, "Title");
        assert!(markdown_outline(b"no headings here\n").is_empty());
    }

    fn sym<'a>(symbols: &'a [SymbolRec], name: &str, kind: SymbolKind) -> Option<&'a SymbolRec> {
        symbols.iter().find(|s| s.name == name && s.kind == kind)
    }

    fn has_call(calls: &[CallRec], callee: &str, caller: &str) -> bool {
        calls.iter().any(|c| c.callee == callee && c.caller == caller)
    }

    #[test]
    fn supported_languages() {
        assert!(supported(Lang::Rust));
        assert!(supported(Lang::Python));
        assert!(supported(Lang::Go));
        assert!(supported(Lang::TsJs));
        assert!(supported(Lang::Java));
        assert!(supported(Lang::Cpp));
        assert!(!supported(Lang::Unknown));
        let (syms, calls) = extract(Lang::Unknown, b"fn main() {}");
        assert!(syms.is_empty() && calls.is_empty());
    }

    #[test]
    fn rust_symbols_and_calls() {
        let src = concat!(
            "pub mod util {\n",                                  // 1
            "    pub const MAX: usize = 10;\n",                  // 2
            "    pub static NAME: &str = \"x\";\n",              // 3
            "    pub struct Point { pub x: i32 }\n",             // 4
            "    pub enum Color { Red }\n",                      // 5
            "    pub trait Shape { fn area(&self) -> f64; }\n",  // 6
            "    pub type Id = u64;\n",                          // 7
            "    pub fn helper() -> i32 { 1 }\n",                // 8
            "}\n",                                               // 9
            "pub struct Foo;\n",                                 // 10
            "impl Foo {\n",                                      // 11
            "    pub fn new() -> Self { Foo }\n",                // 12
            "    pub fn run(&self) { util::helper(); println!(\"hi\"); }\n", // 13
            "}\n",                                               // 14
            "fn main() {\n",                                     // 15
            "    let _x = foo::bar::baz();\n",                   // 16
            "}\n",                                               // 17
        );
        let (syms, calls) = extract(Lang::Rust, src.as_bytes());

        let m = sym(&syms, "util", SymbolKind::Mod).expect("mod util");
        assert_eq!((m.line, m.col, m.scope.as_str()), (1, 8, ""));
        assert_eq!(sym(&syms, "MAX", SymbolKind::Const).unwrap().line, 2);
        assert_eq!(sym(&syms, "MAX", SymbolKind::Const).unwrap().scope, "util");
        assert_eq!(sym(&syms, "NAME", SymbolKind::Const).unwrap().line, 3);
        assert_eq!(sym(&syms, "Point", SymbolKind::Struct).unwrap().line, 4);
        assert_eq!(sym(&syms, "Point", SymbolKind::Struct).unwrap().scope, "util");
        assert_eq!(sym(&syms, "Color", SymbolKind::Enum).unwrap().line, 5);
        assert_eq!(sym(&syms, "Shape", SymbolKind::Trait).unwrap().line, 6);
        assert_eq!(sym(&syms, "Id", SymbolKind::Type).unwrap().line, 7);
        assert_eq!(sym(&syms, "helper", SymbolKind::Fn).unwrap().line, 8);
        assert_eq!(sym(&syms, "Foo", SymbolKind::Struct).unwrap().line, 10);
        assert_eq!(sym(&syms, "Foo", SymbolKind::Impl).unwrap().line, 11);

        let new = sym(&syms, "new", SymbolKind::Method).expect("method new");
        assert_eq!((new.line, new.scope.as_str()), (12, "Foo::new"));
        let run = sym(&syms, "run", SymbolKind::Method).expect("method run");
        assert_eq!((run.line, run.scope.as_str()), (13, "Foo::run"));
        assert_eq!(sym(&syms, "main", SymbolKind::Fn).unwrap().line, 15);

        // Key call edges.
        assert!(has_call(&calls, "baz", "main"), "baz <- main: {calls:?}");
        assert!(has_call(&calls, "helper", "run"), "helper <- run: {calls:?}");
        assert!(has_call(&calls, "println", "run"), "println <- run: {calls:?}");
        // Sorted by line.
        assert!(syms.windows(2).all(|w| w[0].line <= w[1].line));
        assert!(calls.windows(2).all(|w| w[0].line <= w[1].line));
    }

    #[test]
    fn python_symbols_and_calls() {
        let src = concat!(
            "import os\n",                      // 1
            "\n",                               // 2
            "MAX_RETRIES = 3\n",                // 3
            "not_a_const = 1\n",                // 4
            "\n",                               // 5
            "class Animal:\n",                  // 6
            "    def speak(self):\n",           // 7
            "        return self.voice()\n",    // 8
            "\n",                               // 9
            "    def voice(self):\n",           // 10
            "        return \"...\"\n",         // 11
            "\n",                               // 12
            "def make_animal(name):\n",         // 13
            "    a = Animal()\n",               // 14
            "    return a.speak()\n",           // 15
            "\n",                               // 16
            "result = make_animal(\"cat\")\n",  // 17
        );
        let (syms, calls) = extract(Lang::Python, src.as_bytes());

        assert_eq!(sym(&syms, "MAX_RETRIES", SymbolKind::Const).unwrap().line, 3);
        assert!(sym(&syms, "not_a_const", SymbolKind::Const).is_none());
        assert_eq!(sym(&syms, "Animal", SymbolKind::Class).unwrap().line, 6);
        let speak = sym(&syms, "speak", SymbolKind::Method).expect("method speak");
        assert_eq!((speak.line, speak.scope.as_str()), (7, "Animal"));
        assert_eq!(sym(&syms, "voice", SymbolKind::Method).unwrap().line, 10);
        let make = sym(&syms, "make_animal", SymbolKind::Fn).expect("fn make_animal");
        assert_eq!((make.line, make.scope.as_str()), (13, ""));

        assert!(has_call(&calls, "voice", "speak"), "{calls:?}");
        assert!(has_call(&calls, "Animal", "make_animal"), "{calls:?}");
        assert!(has_call(&calls, "speak", "make_animal"), "{calls:?}");
        assert!(has_call(&calls, "make_animal", ""), "top-level call: {calls:?}");
    }

    #[test]
    fn go_symbols_and_calls() {
        let src = concat!(
            "package main\n",                                       // 1
            "\n",                                                   // 2
            "const Version = \"1.0\"\n",                            // 3
            "\n",                                                   // 4
            "type Server struct{ port int }\n",                     // 5
            "type Greeter interface{ Greet() string }\n",           // 6
            "type Port = int\n",                                    // 7
            "\n",                                                   // 8
            "func NewServer(port int) *Server { return &Server{port: port} }\n", // 9
            "\n",                                                   // 10
            "func (s *Server) Start() error {\n",                   // 11
            "\ts.listen()\n",                                       // 12
            "\treturn nil\n",                                       // 13
            "}\n",                                                  // 14
            "\n",                                                   // 15
            "func (s *Server) listen() {}\n",                       // 16
            "\n",                                                   // 17
            "func main() {\n",                                      // 18
            "\ts := NewServer(8080)\n",                             // 19
            "\ts.Start()\n",                                        // 20
            "}\n",                                                  // 21
        );
        let (syms, calls) = extract(Lang::Go, src.as_bytes());

        assert_eq!(sym(&syms, "Version", SymbolKind::Const).unwrap().line, 3);
        assert_eq!(sym(&syms, "Server", SymbolKind::Struct).unwrap().line, 5);
        // struct/interface must not also show up as Type.
        assert!(sym(&syms, "Server", SymbolKind::Type).is_none());
        assert_eq!(sym(&syms, "Greeter", SymbolKind::Interface).unwrap().line, 6);
        assert_eq!(sym(&syms, "Port", SymbolKind::Type).unwrap().line, 7);
        assert_eq!(sym(&syms, "NewServer", SymbolKind::Fn).unwrap().line, 9);
        let start = sym(&syms, "Start", SymbolKind::Method).expect("method Start");
        assert_eq!((start.line, start.scope.as_str()), (11, "Server"));
        assert_eq!(sym(&syms, "listen", SymbolKind::Method).unwrap().line, 16);
        assert_eq!(sym(&syms, "main", SymbolKind::Fn).unwrap().line, 18);

        assert!(has_call(&calls, "NewServer", "main"), "{calls:?}");
        assert!(has_call(&calls, "Start", "main"), "{calls:?}");
        assert!(has_call(&calls, "listen", "Start"), "{calls:?}");
    }

    #[test]
    fn tsjs_symbols_and_calls() {
        let src = concat!(
            "const LIMIT = 100;\n",                                 // 1
            "export const NAME = \"svc\";\n",                       // 2
            "let counter = 0;\n",                                   // 3
            "\n",                                                   // 4
            "enum Level { Info, Warn }\n",                          // 5
            "\n",                                                   // 6
            "interface Logger { log(msg: string): void; }\n",       // 7
            "\n",                                                   // 8
            "class ConsoleLogger implements Logger {\n",            // 9
            "  log(msg: string): void { console.log(msg); }\n",     // 10
            "}\n",                                                  // 11
            "\n",                                                   // 12
            "function makeLogger(): Logger {\n",                    // 13
            "  return new ConsoleLogger();\n",                      // 14
            "}\n",                                                  // 15
            "\n",                                                   // 16
            "const l = makeLogger();\n",                            // 17
            "l.log(\"hello\");\n",                                  // 18
        );
        let (syms, calls) = extract(Lang::TsJs, src.as_bytes());

        assert_eq!(sym(&syms, "LIMIT", SymbolKind::Const).unwrap().line, 1);
        assert_eq!(sym(&syms, "NAME", SymbolKind::Const).unwrap().line, 2);
        assert!(sym(&syms, "counter", SymbolKind::Const).is_none(), "let is not const");
        assert_eq!(sym(&syms, "Level", SymbolKind::Enum).unwrap().line, 5);
        assert_eq!(sym(&syms, "Logger", SymbolKind::Interface).unwrap().line, 7);
        assert_eq!(sym(&syms, "ConsoleLogger", SymbolKind::Class).unwrap().line, 9);
        let log = sym(&syms, "log", SymbolKind::Method).expect("method log");
        assert_eq!((log.line, log.scope.as_str()), (10, "ConsoleLogger"));
        assert_eq!(sym(&syms, "makeLogger", SymbolKind::Fn).unwrap().line, 13);
        // module-level `const l` is also a Const per contract.
        assert_eq!(sym(&syms, "l", SymbolKind::Const).unwrap().line, 17);

        assert!(has_call(&calls, "log", "log"), "console.log in log(): {calls:?}");
        assert!(has_call(&calls, "ConsoleLogger", "makeLogger"), "new: {calls:?}");
        assert!(has_call(&calls, "makeLogger", ""), "{calls:?}");
        assert!(has_call(&calls, "log", ""), "{calls:?}");
    }

    #[test]
    fn java_symbols_and_calls() {
        let src = concat!(
            "public class App {\n",                                 // 1
            "    private static final int MAX_CONN = 10;\n",        // 2
            "    private int count;\n",                             // 3
            "\n",                                                   // 4
            "    public App() { this.count = 0; }\n",               // 5
            "\n",                                                   // 6
            "    public static void main(String[] args) {\n",       // 7
            "        App app = new App();\n",                       // 8
            "        app.run();\n",                                 // 9
            "    }\n",                                              // 10
            "\n",                                                   // 11
            "    public void run() {\n",                            // 12
            "        Helper.assist(count);\n",                      // 13
            "    }\n",                                              // 14
            "}\n",                                                  // 15
            "\n",                                                   // 16
            "interface Service { void serve(); }\n",                // 17
            "\n",                                                   // 18
            "enum State { ON, OFF }\n",                             // 19
            "\n",                                                   // 20
            "class Helper {\n",                                     // 21
            "    static void assist(int n) { System.out.println(n); }\n", // 22
            "}\n",                                                  // 23
        );
        let (syms, calls) = extract(Lang::Java, src.as_bytes());

        assert_eq!(sym(&syms, "App", SymbolKind::Class).unwrap().line, 1);
        assert_eq!(sym(&syms, "MAX_CONN", SymbolKind::Const).unwrap().line, 2);
        assert!(sym(&syms, "count", SymbolKind::Const).is_none(), "not static final");
        let ctor = sym(&syms, "App", SymbolKind::Method).expect("constructor");
        assert_eq!((ctor.line, ctor.scope.as_str()), (5, "App"));
        let main = sym(&syms, "main", SymbolKind::Method).expect("main");
        assert_eq!((main.line, main.scope.as_str()), (7, "App"));
        assert_eq!(sym(&syms, "run", SymbolKind::Method).unwrap().line, 12);
        assert_eq!(sym(&syms, "Service", SymbolKind::Interface).unwrap().line, 17);
        assert_eq!(sym(&syms, "State", SymbolKind::Enum).unwrap().line, 19);
        assert_eq!(sym(&syms, "Helper", SymbolKind::Class).unwrap().line, 21);
        assert_eq!(sym(&syms, "assist", SymbolKind::Method).unwrap().line, 22);

        assert!(has_call(&calls, "App", "main"), "new App(): {calls:?}");
        assert!(has_call(&calls, "run", "main"), "{calls:?}");
        assert!(has_call(&calls, "assist", "run"), "{calls:?}");
        assert!(has_call(&calls, "println", "assist"), "{calls:?}");
    }

    #[test]
    fn cpp_symbols_and_calls() {
        let src = concat!(
            "#include <cstdio>\n",                          // 1
            "\n",                                           // 2
            "namespace util {\n",                           // 3
            "\n",                                           // 4
            "struct Point {\n",                             // 5
            "    int x;\n",                                 // 6
            "    int y;\n",                                 // 7
            "};\n",                                         // 8
            "\n",                                           // 9
            "enum class Color { Red, Green };\n",           // 10
            "\n",                                           // 11
            "int add(int a, int b) { return a + b; }\n",    // 12
            "\n",                                           // 13
            "} // namespace util\n",                        // 14
            "\n",                                           // 15
            "class Greeter {\n",                            // 16
            "public:\n",                                    // 17
            "    void greet() { printf(\"hi\"); }\n",       // 18
            "};\n",                                         // 19
            "\n",                                           // 20
            "static int global_helper() { return util::add(1, 2); }\n", // 21
            "\n",                                           // 22
            "int main() {\n",                               // 23
            "    Greeter g;\n",                             // 24
            "    g.greet();\n",                             // 25
            "    int v = global_helper();\n",               // 26
            "    return v;\n",                              // 27
            "}\n",                                          // 28
        );
        let (syms, calls) = extract(Lang::Cpp, src.as_bytes());

        assert_eq!(sym(&syms, "util", SymbolKind::Mod).unwrap().line, 3);
        let point = sym(&syms, "Point", SymbolKind::Struct).expect("struct Point");
        assert_eq!((point.line, point.scope.as_str()), (5, "util"));
        assert_eq!(sym(&syms, "Color", SymbolKind::Enum).unwrap().line, 10);
        let add = sym(&syms, "add", SymbolKind::Fn).expect("fn add");
        assert_eq!((add.line, add.scope.as_str()), (12, "util"));
        assert_eq!(sym(&syms, "Greeter", SymbolKind::Class).unwrap().line, 16);
        assert!(sym(&syms, "greet", SymbolKind::Fn).is_some(), "inline method: {syms:?}");
        assert_eq!(sym(&syms, "global_helper", SymbolKind::Fn).unwrap().line, 21);
        assert_eq!(sym(&syms, "main", SymbolKind::Fn).unwrap().line, 23);

        assert!(has_call(&calls, "printf", "greet"), "{calls:?}");
        assert!(has_call(&calls, "add", "global_helper"), "{calls:?}");
        assert!(has_call(&calls, "greet", "main"), "{calls:?}");
        assert!(has_call(&calls, "global_helper", "main"), "{calls:?}");
    }

    #[test]
    fn empty_files_no_panic() {
        for lang in [Lang::Rust, Lang::Python, Lang::Go, Lang::TsJs, Lang::Java, Lang::Cpp] {
            let (syms, calls) = extract(lang, b"");
            assert!(syms.is_empty(), "{lang:?} empty file");
            assert!(calls.is_empty(), "{lang:?} empty file");
        }
    }

    #[test]
    fn invalid_syntax_no_panic() {
        let cases: [(Lang, &[u8]); 6] = [
            (Lang::Rust, b"fn ((( broken {{{"),
            (Lang::Python, b"def f(:\nclass:\n  ***"),
            (Lang::Go, b"package\nfunc 123("),
            (Lang::TsJs, b"class { function ( ) enum }"),
            (Lang::Java, b"public class { void ((( ;"),
            (Lang::Cpp, b"int main( {{{ ::: ~"),
        ];
        for (lang, src) in cases {
            // Must not panic; partial or empty results are both fine.
            let _ = extract(lang, src);
        }
        // Truncated UTF-8 must not panic either.
        let _ = extract(Lang::Rust, b"fn caf\xc3");
        let _ = extract(Lang::Python, b"\xff\xfe\x00binary-ish");
    }

    #[test]
    fn large_file_no_pathological_slowdown() {
        // ~1 MiB of generated Rust with a call in each function.
        let mut src = String::with_capacity(1 << 20);
        let mut i = 0;
        while src.len() < (1 << 20) {
            src.push_str(&format!(
                "pub fn generated_fn_{i}(x: i32) -> i32 {{ helper_shared(x + {i}) }}\n"
            ));
            i += 1;
        }
        assert!(src.len() >= (1 << 20));
        let start = std::time::Instant::now();
        let (syms, calls) = extract(Lang::Rust, src.as_bytes());
        let elapsed = start.elapsed();
        assert!(syms.len() >= i, "expected >= {i} fns, got {}", syms.len());
        assert_eq!(calls.len(), i);
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "extract took {elapsed:?} for {} bytes",
            src.len()
        );
    }
}
