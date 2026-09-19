//! indexio-types: shared types and the posting-list byte codec.
//! This crate is the contract layer — see docs/SPEC.md. Do not modify
//! public items without updating SPEC.md.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

pub mod codec;


/// Global content key: first 16 bytes of blake3(content).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlobId(pub [u8; 16]);

impl BlobId {
    pub fn from_content(content: &[u8]) -> Self {
        let h = blake3::hash(content);
        let mut id = [0u8; 16];
        id.copy_from_slice(&h.as_bytes()[..16]);
        BlobId(id)
    }
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

impl std::fmt::Debug for BlobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlobId({})", &self.hex()[..8])
    }
}

impl std::fmt::Display for BlobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hex())
    }
}

/// Language identifier (stored as u16 in the doc table).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[repr(u16)]
pub enum Lang {
    Unknown = 0,
    Rust = 1,
    Python = 2,
    Go = 3,
    TsJs = 4,
    Java = 5,
    Cpp = 6,
    /// Any other text a session may search — docs, configs, scripts,
    /// SQL, markup, code in languages without a grammar here (SPEC-P9).
    /// Lexical index + window chunks; no symbols, calls or outline.
    Text = 7,
}

impl Default for Lang {
    fn default() -> Self {
        Lang::Unknown
    }
}

impl Lang {
    pub fn from_u16(v: u16) -> Lang {
        match v {
            1 => Lang::Rust,
            2 => Lang::Python,
            3 => Lang::Go,
            4 => Lang::TsJs,
            5 => Lang::Java,
            6 => Lang::Cpp,
            7 => Lang::Text,
            _ => Lang::Unknown,
        }
    }
    pub fn as_u16(self) -> u16 {
        self as u16
    }
    pub fn name(self) -> &'static str {
        match self {
            Lang::Unknown => "unknown",
            Lang::Rust => "rust",
            Lang::Python => "python",
            Lang::Go => "go",
            Lang::TsJs => "typescript",
            Lang::Java => "java",
            Lang::Cpp => "c++",
            Lang::Text => "text",
        }
    }
    pub fn from_name(s: &str) -> Lang {
        match s.to_ascii_lowercase().as_str() {
            "rust" | "rs" => Lang::Rust,
            "python" | "py" => Lang::Python,
            "go" | "golang" => Lang::Go,
            "typescript" | "ts" | "javascript" | "js" | "tsx" | "jsx" => Lang::TsJs,
            "java" => Lang::Java,
            "c" | "cpp" | "c++" | "cc" | "cxx" => Lang::Cpp,
            "text" | "txt" | "md" | "markdown" | "json" | "yaml" | "toml" | "sql" | "shell" | "sh" => Lang::Text,
            _ => Lang::Unknown,
        }
    }
    pub fn from_path(path: &str) -> Lang {
        let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
        let lower = name.to_ascii_lowercase();
        // generated / lock / minified files: never worth a lookup
        if lower.ends_with(".min.js")
            || lower.ends_with(".min.css")
            || lower.ends_with(".map")
            || lower.ends_with(".lock")
            || lower.ends_with("-lock.json")
            || lower.ends_with("-lock.yaml")
            || lower == "cargo.lock"
            || lower == "package-lock.json"
            || lower == "yarn.lock"
            || lower == "pnpm-lock.yaml"
        {
            return Lang::Unknown;
        }
        let ext = match lower.rfind('.') {
            Some(i) if i > 0 => &lower[i + 1..],
            _ => "",
        };
        match ext {
            "rs" => Lang::Rust,
            "py" | "pyi" => Lang::Python,
            "go" => Lang::Go,
            "ts" | "tsx" | "js" | "jsx" | "mts" | "cts" | "mjs" | "cjs" => Lang::TsJs,
            "java" => Lang::Java,
            "c" | "h" | "cc" | "cpp" | "cxx" | "hpp" | "hh" | "c++" => Lang::Cpp,
            // docs, configs, data, scripts, markup, styles
            "md" | "markdown" | "mdx" | "rst" | "txt" | "adoc" | "org"
            | "json" | "jsonc" | "json5" | "yaml" | "yml" | "toml" | "ini" | "cfg" | "conf" | "env" | "properties"
            | "xml" | "html" | "htm" | "css" | "scss" | "sass" | "less" | "svg" | "vue" | "svelte" | "astro"
            | "sh" | "bash" | "zsh" | "fish" | "ps1" | "psm1" | "bat" | "cmd"
            | "sql" | "graphql" | "gql" | "proto" | "thrift"
            | "tf" | "hcl" | "nix" | "cmake" | "gradle" | "sbt" | "make" | "mk" | "dockerfile"
            // code without a grammar here
            | "rb" | "php" | "kt" | "kts" | "swift" | "cs" | "fs" | "scala" | "dart" | "lua" | "r" | "jl"
            | "ex" | "exs" | "erl" | "hs" | "ml" | "mli" | "clj" | "cljs" | "edn" | "zig" | "sol" | "move"
            | "m" | "mm" | "pl" | "pm" | "vb" | "groovy" | "elm" | "nim" | "d" | "v" | "vhd" | "vhdl"
            // firmware, GPU, assembly, older languages, templates, tabular data
            | "ino" | "pde" | "cu" | "cuh" | "cl" | "glsl" | "hlsl" | "wgsl" | "metal"
            | "asm" | "s" | "f" | "f90" | "f95" | "f03" | "for" | "pas" | "tcl" | "awk"
            | "el" | "lisp" | "scm" | "rkt" | "coffee" | "pug" | "hbs" | "mustache" | "jinja" | "j2" | "twig"
            | "erb" | "haml" | "liquid" | "njk" | "ejs" | "tsv" | "csv" | "log" => Lang::Text,
            "" => match lower.as_str() {
                "dockerfile" | "makefile" | "justfile" | "rakefile" | "gemfile" | "procfile"
                | "readme" | "license" | "changelog" | "authors" | "contributing" | "notice" => Lang::Text,
                _ => Lang::Unknown,
            },
            _ => Lang::Unknown,
        }
    }
    /// Files with symbols, calls and an outline (a tree-sitter grammar).
    pub fn is_code(self) -> bool {
        !matches!(self, Lang::Unknown | Lang::Text)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[repr(u8)]
pub enum SymbolKind {
    Fn = 0,
    Struct = 1,
    Class = 2,
    Enum = 3,
    Trait = 4,
    Impl = 5,
    Mod = 6,
    Const = 7,
    Type = 8,
    Method = 9,
    Var = 10,
    Interface = 11,
}

impl SymbolKind {
    pub fn from_u8(v: u8) -> SymbolKind {
        match v {
            0 => SymbolKind::Fn,
            1 => SymbolKind::Struct,
            2 => SymbolKind::Class,
            3 => SymbolKind::Enum,
            4 => SymbolKind::Trait,
            5 => SymbolKind::Impl,
            6 => SymbolKind::Mod,
            7 => SymbolKind::Const,
            8 => SymbolKind::Type,
            9 => SymbolKind::Method,
            10 => SymbolKind::Var,
            _ => SymbolKind::Interface,
        }
    }
    pub fn as_u8(self) -> u8 {
        self as u8
    }
    pub fn name(self) -> &'static str {
        match self {
            SymbolKind::Fn => "fn",
            SymbolKind::Struct => "struct",
            SymbolKind::Class => "class",
            SymbolKind::Enum => "enum",
            SymbolKind::Trait => "trait",
            SymbolKind::Impl => "impl",
            SymbolKind::Mod => "mod",
            SymbolKind::Const => "const",
            SymbolKind::Type => "type",
            SymbolKind::Method => "method",
            SymbolKind::Var => "var",
            SymbolKind::Interface => "interface",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolRec {
    pub name: String,
    pub kind: SymbolKind,
    pub line: u32,
    pub col: u32,
    /// Enclosing definition path (e.g. "MyStruct::new"), "" at top level.
    pub scope: String,
}

/// Best-effort, name-based call edge. `caller` is "" for top-level calls.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CallRec {
    pub callee: String,
    pub caller: String,
    pub line: u32,
}

/// The unit cached in the CAS and consumed by the index writer.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExtractedArtifact {
    /// Sorted unique grams (3..=8 bytes). No positions (SPEC-P10): the
    /// planner works on document sets and the verifier rescans content.
    pub ngrams: Vec<Vec<u8>>,
    pub symbols: Vec<SymbolRec>,
    pub calls: Vec<CallRec>,
    pub raw_len: u32,
    pub lang: Lang,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocMeta {
    pub blob: BlobId,
    pub repo_id: u32,
    pub path: String,
    pub lang: Lang,
    pub raw_len: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchHit {
    pub repo: String,
    pub path: String,
    pub line: u32,
    pub col: u32,
    pub snippet: String,
    pub score: f32,
    pub lang: Lang,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EngineStats {
    pub repos: Vec<String>,
    pub doc_count: u64,
    pub total_raw_bytes: u64,
    pub shard_count: u64,
    pub cas_entries: u64,
    pub cas_bytes: u64,
    pub index_bytes: u64,
}
