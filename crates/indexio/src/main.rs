//! indexio: binary crate — CLI (clap), HTTP server (axum) and MCP server
//! (stdio JSON-RPC) for the indexio engine.
//!
//! Contract: docs/SPEC.md, section "indexio — binary".

#![allow(clippy::type_complexity)]

mod impact_cli;
mod mcp;
mod hook;
mod mcp_text;
mod runs;
mod sessions;
mod usage;
mod watch;
mod serve;

use std::io::{BufRead as _, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::Context;
use clap::{ArgGroup, Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use indexio_embed::embed::{Embedder, HashEmbedder, HttpEmbedder};
use indexio_embed::rindex::RandomIndexingEmbedder;
use indexio_embed::pipeline::EmbedReport;
use indexio_embed::rerank::{HttpReranker, NoopReranker, Reranker};
use indexio_embed::store::EmbedCas;
use indexio_index::ShardSet;
use indexio_ingest::org_sync::{OrgSyncOptions, OrgSyncReport};
use indexio_ingest::sources::{self, SyncOptions, SyncReport};
use indexio_ingest::{Cas, IndexReport};
use indexio_query::{Engine, FusionAlgo, SearchMode};
use indexio_types::{EngineStats, SearchHit};

#[derive(Parser)]
#[command(name = "indexio", version, about = "indexio: local code search engine")]
struct Cli {
    /// Data directory (shards/, cas/, repos/).
    /// Overrides $INDEXIO_DATA_DIR; default ~/.indexio.
    #[arg(long, global = true, value_name = "DIR")]
    data_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a source and index it now (SPEC-P7). A source is a local folder
    /// (every git repo under it, any depth; a folder with no repos is
    /// indexed as-is), `github:ORG`, `github:OWNER/REPO`,
    /// `azdo:ORG/PROJECT`, `azdo:ORG/PROJECT/REPO`, or any clone URL.
    /// Remembered in <data-dir>/sources.json so `indexio sync` keeps it fresh.
    Add {
        /// Folder path, github:…, azdo:…, or a git URL.
        source: String,
        /// Where remote repos are cloned (default <data-dir>/remotes/…).
        #[arg(long)]
        dest: Option<PathBuf>,
        /// Include forked repos (GitHub).
        #[arg(long)]
        include_forks: bool,
        /// Include archived (GitHub) / disabled (Azure DevOps) repos.
        #[arg(long)]
        include_archived: bool,
        /// Full clone instead of --depth 1.
        #[arg(long)]
        full_clone: bool,
        /// Cap the number of repos taken from a remote source (trial run).
        #[arg(long)]
        limit: Option<usize>,
        /// Remember the source but do not sync it now.
        #[arg(long)]
        no_sync: bool,
        /// Skip the embed step after indexing.
        #[arg(long)]
        no_embed: bool,
    },
    /// Sync every remembered source (clone/pull, discover, delta re-index)
    /// plus any repo registered outside a source, then embed. Cron this.
    Sync {
        /// Skip the embed step.
        #[arg(long)]
        no_embed: bool,
        /// Cap repos per remote source (trial run).
        #[arg(long)]
        limit: Option<usize>,
    },
    /// List remembered sources and registered repos.
    Sources,
    /// Forget a source (indexed repos stay searchable until compacted away).
    Remove {
        /// The source as typed for `indexio add`.
        source: String,
    },
    /// Index a local git repo at HEAD (or a plain folder without git).
    Index {
        /// Path to the git repository.
        repo_path: PathBuf,
        /// Repo name (default: directory basename).
        #[arg(long)]
        name: Option<String>,
    },
    /// Delta re-index of registered repo(s).
    #[command(group = ArgGroup::new("target").required(true).multiple(false).args(["repo", "all"]))]
    Reindex {
        /// Re-index this registered repo.
        #[arg(long)]
        repo: Option<String>,
        /// Re-index all registered repos.
        #[arg(long)]
        all: bool,
        /// Index the WORKING TREE (uncommitted edits, untracked files not
        /// ignored by git) instead of HEAD; delta by content hash.
        #[arg(long)]
        worktree: bool,
    },
    /// Search the index.
    Search {
        /// Query string (literals, "phrases", /regex/, repo:/lang:/path:/case: filters).
        query: String,
        /// Maximum number of hits.
        #[arg(long, default_value_t = 50)]
        limit: usize,
        /// Print the SearchResult as JSON.
        #[arg(long)]
        json: bool,
        /// Search mode: lexical (exact identifiers/regex), semantic
        /// (natural-language concepts), hybrid (RRF fusion of both).
        #[arg(long, default_value = "lexical")]
        mode: String,
        /// Rerank the hybrid results (SPEC-P3 §2). Only valid with
        /// --mode hybrid; reranker = INDEXIO_RERANK_BASE HttpReranker when set,
        /// else the offline OverlapReranker.
        #[arg(long)]
        rerank: bool,
        /// Fusion algorithm for --mode hybrid (SPEC-P5 §A3): rrf
        /// (reciprocal-rank, k=60; default) or combmnz (min-max normalized
        /// scores, sum x number of legs).
        #[arg(long, default_value = "rrf", value_name = "rrf|combmnz")]
        fusion: String,
        /// Embedder for semantic/hybrid modes (SPEC-P4 §2): "hash" selects
        /// the legacy HashEmbedder (A/B escape hatch), "rindex" the default
        /// self-contained random-indexing model. INDEXIO_EMBED_BASE (HTTP)
        /// overrides this flag.
        #[arg(long, value_name = "hash|rindex")]
        embedder: Option<String>,
    },
    /// Embed repos into the semantic vector sidecar (chunk -> CAS-deduped
    /// embed -> quantized vec index).
    #[command(group = ArgGroup::new("embtarget").multiple(false).args(["repo", "all"]))]
    Embed {
        /// Embed only this repo.
        #[arg(long)]
        repo: Option<String>,
        /// Embed all repos (default).
        #[arg(long)]
        all: bool,
        /// Chunk size cap in source chars.
        #[arg(long, default_value_t = indexio_embed::pipeline::MAX_CHARS)]
        max_chars: usize,
        /// Embedder: "hash" selects the legacy HashEmbedder (A/B escape
        /// hatch), "rindex" the default self-contained random-indexing
        /// model (SPEC-P4 §2). INDEXIO_EMBED_BASE (HTTP) overrides this flag.
        #[arg(long, value_name = "hash|rindex")]
        embedder: Option<String>,
        /// Rebuild the semantic model from scratch (SPEC-P4 §2): delete
        /// the on-disk model and this model's embcas namespace, then
        /// re-observe ALL chunks (every chunk becomes a CAS miss). Needed
        /// because the RI model state still includes contributions of
        /// documents deleted since it was built (observe only ever adds).
        #[arg(long)]
        rebuild_model: bool,
    },
    /// Sync and index every repo of a GitHub organization.
    OrgSync {
        /// GitHub organization name.
        #[arg(long)]
        org: String,
        /// Env var holding the GitHub PAT (unauthenticated when unset/empty).
        #[arg(long, default_value = "GITHUB_TOKEN")]
        token_env: String,
        /// Clone destination (default: <data-dir>/org-repos/<org>).
        #[arg(long)]
        dest: Option<PathBuf>,
        /// Cap the number of repos (for trials).
        #[arg(long)]
        limit: Option<usize>,
        /// Include forked repos.
        #[arg(long)]
        include_forks: bool,
        /// Include archived repos.
        #[arg(long)]
        include_archived: bool,
        /// Full clone instead of --depth 1.
        #[arg(long)]
        full_clone: bool,
    },
    /// Embedding CAS statistics (semantic plane).
    EmbcasStats {
        /// Print stats as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Find a symbol by name (exact, substring fallback).
    Symbol { name: String },
    /// Find callers of a function/method.
    Calls { name: String },
    /// Claude Code hooks (SPEC-P9). `hook bash` is a PreToolUse hook for the
    /// Bash tool: it reads the hook JSON on stdin and denies plain shell
    /// reads/searches (cat, sed -n, head, grep, rg, find) of files the index
    /// holds, naming the indexio call to make instead. `hook install` writes
    /// it (and the PreCompact `sessions --quiet` import) into
    /// `~/.claude/settings.json` with a path every shell accepts.
    Hook {
        /// bash or read (act as that PreToolUse hook), or install.
        which: String,
        /// install: Claude config dir (default: $CLAUDE_CONFIG_DIR or ~/.claude).
        #[arg(long)]
        claude_dir: Option<PathBuf>,
        /// install: show the changes without writing settings.json.
        #[arg(long)]
        print_only: bool,
    },
    /// Import Claude Code session transcripts into the `sessions` source
    /// (SPEC-P9 recall): `<claude dir>/projects/*/*.jsonl` become compact
    /// Markdown under `<data-dir>/sessions`, registered as a plain-folder
    /// source and re-indexed. Idempotent; only changed transcripts are
    /// re-rendered. Meant for a PreCompact hook and the MCP server.
    Sessions {
        /// Claude config dir (default: $CLAUDE_CONFIG_DIR or ~/.claude).
        #[arg(long)]
        claude_dir: Option<PathBuf>,
        /// Only this project folder (a Claude project slug).
        #[arg(long)]
        project: Option<String>,
        /// Render only; do not re-index.
        #[arg(long)]
        no_index: bool,
        /// Print nothing unless something failed.
        #[arg(long)]
        quiet: bool,
    },
    /// Run a command and keep its output out of the model's context
    /// (SPEC-P10 §31): short output is printed as it is; long output is
    /// stored under `<data-dir>/runs/<repo>/` (indexed, searchable, rolled
    /// off after $INDEXIO_RETAIN_DAYS, default 30) and a digest with the
    /// error lines and a `read_span` pointer is printed instead. The Bash
    /// hook routes scripts and builds through this.
    Run {
        /// Working directory (default: the current one).
        #[arg(long)]
        cwd: Option<PathBuf>,
        /// The command, after `--`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
        command: Vec<String>,
    },
    /// Aggregate the MCP usage log (`<data-dir>/usage/*.jsonl`): calls,
    /// result size and latency per tool and per repo, and what the
    /// harness's own tools would have returned for the same calls.
    Usage {
        /// Only the last N days (default 7).
        #[arg(long, default_value_t = 7)]
        days: u64,
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
    },
    /// Compact shards (merge oldest down to --max-shards).
    Compact {
        /// Target maximum shard count.
        #[arg(long, default_value_t = 4)]
        max_shards: usize,
        /// Also fold every vector segment into one (re-encodes pre-P10 f32
        /// segments as int8 rows).
        #[arg(long)]
        vectors: bool,
    },
    /// Engine + CAS statistics.
    Stats {
        /// Print stats as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Content-addressed store statistics.
    CasStats,
    /// Run the HTTP server (loopback unless --bind says otherwise).
    Serve {
        /// Port to bind.
        #[arg(long, default_value_t = 7717)]
        port: u16,
        /// Address to bind. The default is loopback; anything else (a
        /// container's 0.0.0.0, a LAN address) requires --auth-token or
        /// --acl-file so the index is never on a network unauthenticated.
        /// Falls back to $INDEXIO_BIND when unset.
        #[arg(long, value_name = "ADDR")]
        bind: Option<std::net::IpAddr>,
        /// Bearer token required on all routes except GET /health
        /// (SPEC-P3 §3). Falls back to $INDEXIO_AUTH_TOKEN when unset.
        #[arg(long, value_name = "TOKEN")]
        auth_token: Option<String>,
        /// JSON ACL file: {"tokens":{"tok":{"allow":["core","team-*"]}}}.
        /// Supersedes --auth-token; adds per-token repo allowlists and
        /// admin-route gating (allow must contain exactly "*").
        #[arg(long, value_name = "PATH")]
        acl_file: Option<PathBuf>,
    },
    /// Run the MCP server over stdio (newline-delimited JSON-RPC 2.0).
    Mcp {
        /// The registered repo this session works in: its hits are listed
        /// first. Default: $INDEXIO_REPO, else the repo whose folder
        /// contains the current working directory.
        #[arg(long)]
        repo: Option<String>,
    },
    /// Register indexio with an agent harness (SPEC-P8). `setup claude`
    /// runs `claude mcp add` for this binary + data dir and prints the
    /// guidance block to put in CLAUDE.md (or appends it with --claude-md).
    Setup {
        /// Harness: "claude" (Claude Code).
        harness: String,
        /// `claude mcp add` scope: user (all projects) or project (.mcp.json).
        #[arg(long, default_value = "user")]
        scope: String,
        /// Append the guidance block to this CLAUDE.md (created if missing).
        #[arg(long, value_name = "PATH")]
        claude_md: Option<PathBuf>,
        /// Only print what would be done.
        #[arg(long)]
        print_only: bool,
    },
    /// Impact analysis (SPEC-P6): what else does a change touch? Exactly
    /// one of --symbol / --diff / --diff-file / --file selects the target.
    Impact {
        /// Root symbol name to walk (repeatable).
        #[arg(long = "symbol", value_name = "NAME")]
        symbols: Vec<String>,
        /// Registered repo: analyse `git diff -U0 <base>` of its working tree.
        #[arg(long, value_name = "REPO")]
        diff: Option<String>,
        /// Git base ref for --diff (default HEAD = all uncommitted changes).
        #[arg(long, default_value = "HEAD")]
        base: String,
        /// Read a unified diff from this file ("-" = stdin); requires --repo.
        #[arg(long, value_name = "PATCH")]
        diff_file: Option<PathBuf>,
        /// Registered repo the --diff-file patch applies to.
        #[arg(long)]
        repo: Option<String>,
        /// REPO:PATH — importers + impact of every definition in the file.
        #[arg(long, value_name = "REPO:PATH")]
        file: Option<String>,
        /// Reverse-call hops to follow (1 = direct callers).
        #[arg(long, default_value_t = 2)]
        depth: u32,
        /// Stop after this many call sites (sets truncated).
        #[arg(long, default_value_t = 500)]
        max_sites: usize,
        /// Call sites consumed per symbol name (hub protection).
        #[arg(long, default_value_t = 200)]
        max_fanout: usize,
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Definitions with line ranges for one indexed file (REPO:PATH).
    Outline {
        target: String,
        #[arg(long)]
        json: bool,
    },
    /// Print lines of an indexed file: REPO:PATH START[:END] (max 400 lines).
    Span { target: String, range: String },
}

/// Exactly one impact target must be selected (SPEC-P6 §3).
pub(crate) fn validate_impact_target(
    symbols: &[String],
    diff: Option<&str>,
    diff_file: Option<&Path>,
    repo: Option<&str>,
    file: Option<&str>,
) -> anyhow::Result<()> {
    let n = usize::from(!symbols.is_empty())
        + usize::from(diff.is_some())
        + usize::from(diff_file.is_some())
        + usize::from(file.is_some());
    if n != 1 {
        anyhow::bail!("select exactly one of --symbol, --diff, --diff-file, --file");
    }
    if diff_file.is_some() && repo.is_none() {
        anyhow::bail!("--diff-file requires --repo");
    }
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        // Never write logs to stdout: `indexio mcp` speaks JSON-RPC there.
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    let data_dir = resolve_data_dir(&cli);
    match cli.command {
        Commands::Add {
            source,
            dest,
            include_forks,
            include_archived,
            full_clone,
            limit,
            no_sync,
            no_embed,
        } => {
            let mut src = sources::parse_source(&source)?;
            src.dest = dest;
            src.include_forks = include_forks;
            src.include_archived = include_archived;
            src.shallow = !full_clone;
            let added = sources::add_source(&data_dir, src.clone())?;
            println!(
                "{} source {}",
                if added { "added" } else { "updated" },
                src.label()
            );
            if !no_sync {
                let cas = Cas::open(&data_dir.join("cas"))?;
                let opts = SyncOptions {
                    limit,
                    ..SyncOptions::default()
                };
                let report = sources::sync_source(&src, &data_dir, &cas, &opts)?;
                print_sync_report(&report);
                if !no_embed {
                    embed_after_sync(&data_dir, &report.indexed)?;
                }
                print_next_steps(&data_dir);
            }
        }
        Commands::Sync { no_embed, limit } => {
            let cas = Cas::open(&data_dir.join("cas"))?;
            let opts = SyncOptions {
                limit,
                ..SyncOptions::default()
            };
            let reports = sources::sync_all(&data_dir, &cas, &opts)?;
            if reports.is_empty() {
                println!("nothing to sync: add a source first, e.g. `indexio add ~/code` or `indexio add github:my-org`");
            }
            for r in &reports {
                print_sync_report(r);
            }
            if !no_embed && !reports.is_empty() {
                let indexed: Vec<IndexReport> = reports.iter().flat_map(|r| r.indexed.iter().cloned()).collect();
                embed_after_sync(&data_dir, &indexed)?;
            }
        }
        Commands::Sources => {
            let srcs = sources::load_sources(&data_dir)?;
            println!("sources ({}):", srcs.len());
            for s in &srcs {
                let mut flags = Vec::new();
                if s.include_forks {
                    flags.push("forks");
                }
                if s.include_archived {
                    flags.push("archived");
                }
                if !s.shallow {
                    flags.push("full-clone");
                }
                if let Some(d) = &s.dest {
                    flags.push(Box::leak(format!("dest={}", d.display()).into_boxed_str()));
                }
                println!(
                    "  {:<12} {}{}",
                    format!("{:?}", s.kind).to_lowercase(),
                    s.label(),
                    if flags.is_empty() { String::new() } else { format!("  [{}]", flags.join(", ")) }
                );
            }
            let names = sources::registered_repo_names(&data_dir)?;
            println!("registered repos ({}):", names.len());
            for n in &names {
                match indexio_ingest::repo_state(&data_dir, n) {
                    Ok(st) => println!(
                        "  {:<28} {}{}  (indexed {})",
                        n,
                        st.path.display(),
                        if st.plain { "  [plain folder]" } else { "" },
                        st.indexed_at
                    ),
                    Err(_) => println!("  {n}"),
                }
            }
        }
        Commands::Remove { source } => {
            if sources::remove_source(&data_dir, &source)? {
                println!("removed source {source}");
            } else {
                anyhow::bail!("no such source: {source} (see `indexio sources`)");
            }
        }
        Commands::Index { repo_path, name } => {
            let name = match name {
                Some(n) => n,
                None => repo_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
                    .context("cannot derive repo name from path; pass --name")?,
            };
            let cas = Cas::open(&data_dir.join("cas"))?;
            let report = indexio_ingest::index_repo(&repo_path, &name, &data_dir, &cas)?;
            print_report(&report);
        }
        Commands::Reindex { repo, all, worktree } => {
            let cas = Cas::open(&data_dir.join("cas"))?;
            let names: Vec<String> = if all {
                registered_repos(&data_dir)?
            } else {
                vec![repo.expect("clap group guarantees --repo")]
            };
            if names.is_empty() {
                println!("no registered repos");
            }
            for name in names {
                let report = if worktree {
                    indexio_ingest::reindex_worktree(&name, &data_dir, &cas)?
                } else {
                    indexio_ingest::reindex_repo(&name, &data_dir, &cas)?
                };
                print_report(&report);
            }
        }
        Commands::Search {
            query,
            limit,
            json,
            mode,
            rerank,
            fusion,
            embedder: embedder_flag,
        } => {
            if mode == "bm25" {
                // evaluation aid (SPEC-P10 §14): the chunk-BM25 leg alone
                let engine = Engine::open(&data_dir)
                    .with_context(|| format!("opening index at {}", data_dir.display()))?;
                let embedder = select_embedder(embedder_flag.as_deref(), &data_dir)?;
                let scope = engine.repo_scope(&query);
                let hits = engine.search_bm25_in(&query, limit, embedder.model_id(), scope.as_deref())?;
                if json {
                    let arr: Vec<serde_json::Value> =
                        hits.iter().map(|h| serde_json::to_value(h).expect("SearchHit is Serialize")).collect();
                    println!("{}", serde_json::json!({ "hits": arr }));
                } else {
                    print_hits(&hits);
                }
                return Ok(());
            }
            let mode = SearchMode::parse(&mode).map_err(anyhow::Error::msg)?;
            validate_rerank_flag(mode, rerank)?;
            let fusion = FusionAlgo::parse(&fusion).map_err(anyhow::Error::msg)?;
            if fusion != FusionAlgo::Rrf && mode != SearchMode::Hybrid {
                anyhow::bail!("--fusion requires --mode hybrid");
            }
            let engine = Engine::open(&data_dir)
                .with_context(|| format!("opening index at {}", data_dir.display()))?;
            match mode {
                SearchMode::Lexical => {
                    let q = indexio_query::parse(&query).map_err(|e| anyhow::anyhow!("{e}"))?;
                    let res = engine.search(&q, limit);
                    if json {
                        println!("{}", serve::search_result_to_json(&res));
                    } else {
                        print_hits(&res.hits);
                        if res.truncated {
                            println!("(results truncated)");
                        }
                    }
                }
                SearchMode::Semantic => {
                    let embedder = select_embedder(embedder_flag.as_deref(), &data_dir)?;
                    let scope = engine.repo_scope(&query);
                    let hits = engine.search_semantic_in(&query, limit, embedder.as_ref(), scope.as_deref())?;
                    if json {
                        let arr: Vec<serde_json::Value> = hits
                            .iter()
                            .map(|h| serde_json::to_value(h).expect("SearchHit is Serialize"))
                            .collect();
                        println!("{}", serde_json::json!({ "hits": arr }));
                    } else {
                        print_hits(&hits);
                    }
                }
                SearchMode::Hybrid => {
                    let embedder = select_embedder(embedder_flag.as_deref(), &data_dir)?;
                    let fused = if rerank {
                        let reranker = select_reranker()?;
                        engine.search_hybrid_fused_reranked(
                            &query,
                            limit,
                            embedder.as_ref(),
                            reranker.as_ref(),
                            fusion,
                        )?
                    } else {
                        let scope = engine.repo_scope(&query);
                        engine.search_hybrid_in(&query, limit, embedder.as_ref(), fusion, scope.as_deref())?
                    };
                    if json {
                        println!(
                            "{}",
                            serde_json::json!({
                                "hits": fused.iter().map(serve::hybrid_hit_to_json).collect::<Vec<_>>(),
                            })
                        );
                    } else {
                        print_hybrid_hits(&fused);
                    }
                }
            }
        }
        Commands::Embed {
            repo,
            all: _,
            max_chars,
            embedder: embedder_flag,
            rebuild_model,
        } => {
            let embedder = select_embedder(embedder_flag.as_deref(), &data_dir)?;
            let embedder = if rebuild_model {
                // SPEC-P4 §2: wipe the on-disk semantic model and this
                // model's embcas namespace, then re-observe ALL chunks
                // (with the CAS gone, every chunk is a miss, so the
                // pipeline observes all of them). Drift note: incremental
                // observe() only ever ADDS contributions, so docs deleted
                // since the model was built keep contributing until this
                // rebuild.
                let model_file = data_dir
                    .join("sem")
                    .join(format!("{}.rimodel", embedder.model_id()));
                if model_file.exists() {
                    std::fs::remove_file(&model_file).with_context(|| {
                        format!("removing {}", model_file.display())
                    })?;
                }
                let cas_dir = data_dir.join("embcas").join(embedder.model_id());
                if cas_dir.exists() {
                    std::fs::remove_dir_all(&cas_dir)
                        .with_context(|| format!("removing {}", cas_dir.display()))?;
                }
                // SPEC-P9: the incremental embed carries rows of unchanged
                // files, which would keep the OLD model's vectors.
                indexio_embed::pipeline::clear_segments(&data_dir, embedder.model_id());
                // Re-open so no stale in-memory model state survives.
                select_embedder(embedder_flag.as_deref(), &data_dir)?
            } else {
                embedder
            };
            let shards_dir = data_dir.join("shards");
            let set = ShardSet::open_dir(&shards_dir)
                .with_context(|| format!("opening shards at {}", shards_dir.display()))?;
            let reports = match repo {
                Some(r) => vec![indexio_embed::pipeline::embed_repo_with(
                    &set,
                    &data_dir,
                    &r,
                    embedder.as_ref(),
                    max_chars,
                )?],
                None => indexio_embed::pipeline::embed_all_with(
                    &set,
                    &data_dir,
                    embedder.as_ref(),
                    max_chars,
                )?,
            };
            print_embed_reports(&reports);
        }
        Commands::OrgSync {
            org,
            token_env,
            dest,
            limit,
            include_forks,
            include_archived,
            full_clone,
        } => {
            let token = std::env::var(&token_env).ok().filter(|t| !t.is_empty());
            let dest_dir = dest.unwrap_or_else(|| data_dir.join("org-repos").join(&org));
            let opts = OrgSyncOptions {
                org: org.clone(),
                token,
                dest_dir,
                include_forks,
                include_archived,
                limit,
                shallow: !full_clone,
            };
            let cas = Cas::open(&data_dir.join("cas"))?;
            let report = indexio_ingest::org_sync::sync_org(&opts, &data_dir, &cas)?;
            print_org_sync_report(&report);
        }
        Commands::EmbcasStats { json } => {
            let embedder = select_embedder(None, &data_dir)?;
            let cas = EmbedCas::open_for(&data_dir, embedder.as_ref());
            let (entries, bytes) = cas.stats();
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "model_id": embedder.model_id(),
                        "entries": entries,
                        "bytes": bytes,
                    })
                );
            } else {
                println!("embcas model:   {}", embedder.model_id());
                println!("embcas entries: {entries}");
                println!("embcas bytes:   {bytes}");
            }
        }
        Commands::Symbol { name } => {
            let engine = Engine::open(&data_dir)
                .with_context(|| format!("opening index at {}", data_dir.display()))?;
            print_hits(&engine.find_symbol(&name, 50));
        }
        Commands::Calls { name } => {
            let engine = Engine::open(&data_dir)
                .with_context(|| format!("opening index at {}", data_dir.display()))?;
            print_hits(&engine.who_calls(&name, 50));
        }
        Commands::Usage { days, json } => {
            print_usage(&data_dir, days, json)?;
        }
        Commands::Hook { which, claude_dir, print_only } => match which.as_str() {
            "bash" => {
                // never fail the tool call: any error means "allow"
                if let Err(e) = hook::run_bash_hook(&data_dir) {
                    tracing::debug!(error = %e, "hook bash: allowing");
                }
            }
            "read" => {
                if let Err(e) = hook::run_read_hook(&data_dir) {
                    tracing::debug!(error = %e, "hook read: allowing");
                }
            }
            "install" => {
                let claude = match claude_dir.or_else(sessions::default_claude_dir) {
                    Some(d) => d,
                    None => anyhow::bail!("cannot locate the Claude config dir; pass --claude-dir"),
                };
                let exe = std::env::current_exe().context("locating the indexio binary")?;
                hook::install(&claude, &exe, print_only)?;
            }
            other => anyhow::bail!("unknown hook '{other}' (expected: bash, read or install)"),
        },
        Commands::Run { cwd, command } => {
            let cwd = match cwd {
                Some(c) => c,
                None => std::env::current_dir()?,
            };
            let code = runs::run(&data_dir, &cwd, &runs::shell_join(&command))?;
            if code != 0 {
                // the command's own status, without an "error:" line of ours
                std::process::exit(code.clamp(1, 255));
            }
        }
        Commands::Sessions { claude_dir, project, no_index, quiet } => {
            let claude = match claude_dir.or_else(sessions::default_claude_dir) {
                Some(d) => d,
                None => anyhow::bail!("cannot locate the Claude config dir; pass --claude-dir"),
            };
            let out = data_dir.join("sessions");
            let report = sessions::import(&claude, &out, project.as_deref())?;
            let rolled = runs::retain(&data_dir, runs::retain_days());
            if rolled > 0 && !quiet {
                println!("runs: {rolled} log(s) older than {} days removed", runs::retain_days());
            }
            if !quiet {
                println!(
                    "sessions: {} seen, {} imported ({} parts, {} KB)",
                    report.sessions_seen,
                    report.sessions_imported,
                    report.parts_written,
                    report.bytes_written / 1024
                );
            }
            if !no_index && report.sessions_imported > 0 {
                let cas = Cas::open(&data_dir.join("cas"))?;
                let r = sessions::index_sessions(&data_dir, &cas, None)?;
                if !quiet {
                    println!("indexed: +{} -{} ={} docs", r.docs_added, r.docs_deleted, r.docs_unchanged);
                }
            }
        }
        Commands::Compact { max_shards, vectors } => {
            let shards_dir = data_dir.join("shards");
            let mut set = ShardSet::open_dir(&shards_dir)?;
            let before = set.len();
            set.merge(&shards_dir, max_shards)?;
            let after = set.len();
            if after < before {
                println!("compacted: {before} -> {after} shards");
            } else {
                println!("nothing to compact ({before} shards, max {max_shards})");
            }
            // semantic plane (SPEC-P9): fold delta segments / purge tombstones
            let model = select_embedder(None, &data_dir).map(|e| e.model_id().to_string());
            if let Ok(model) = model {
                let _lock = indexio_embed::pipeline::CompactLock::try_acquire(&data_dir);
                if _lock.is_none() {
                    println!("vector segments: another process is compacting; skipped");
                } else {
                    let (n, rows, dead) = indexio_embed::index::segment_stats(&data_dir.join("vec"), &model);
                    match indexio_embed::pipeline::compact_vectors_opts(&data_dir, &model, vectors) {
                        Ok(true) => {
                            let (m, _, _) = indexio_embed::index::segment_stats(&data_dir.join("vec"), &model);
                            println!("vector segments: {n} -> {m} ({rows} rows, {dead} were tombstoned)");
                        }
                        Ok(false) => println!("vector segments: nothing to compact ({n} segments, {dead}/{rows} tombstoned)"),
                        Err(e) => println!("vector segments: compaction failed: {e:#}"),
                    }
                }
            }
        }
        Commands::Stats { json } => {
            let stats = merged_stats(&data_dir)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                print_stats(&stats);
            }
        }
        Commands::CasStats => {
            let (entries, bytes) = cas_stats(&data_dir);
            println!("cas entries: {entries}");
            println!("cas bytes:   {bytes}");
        }
        Commands::Serve {
            port,
            bind,
            auth_token,
            acl_file,
        } => {
            let auth = resolve_auth_config(auth_token, acl_file.as_deref())?;
            let bind = bind
                .or_else(|| std::env::var("INDEXIO_BIND").ok().and_then(|s| s.trim().parse().ok()))
                .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
            if !bind.is_loopback() && matches!(auth, serve::AuthConfig::Open) {
                anyhow::bail!(
                    "--bind {bind} is not a loopback address: pass --auth-token (or $INDEXIO_AUTH_TOKEN) or --acl-file before exposing the index beyond this host"
                );
            }
            serve::run_server(data_dir, bind, port, auth).await?;
        }
        Commands::Mcp { repo } => {
            let engine = Engine::open(&data_dir)
                .with_context(|| format!("opening index at {}", data_dir.display()))?;
            // A server observes a few chunks per edit: persist the model
            // at most every few minutes instead of on every refresh.
            let embedder: Arc<dyn Embedder> = if std::env::var_os("INDEXIO_EMBED_BASE").is_none() {
                Arc::new(
                    RandomIndexingEmbedder::open(&data_dir)?
                        .with_lazy_flush(std::time::Duration::from_secs(300)),
                )
            } else {
                select_embedder(None, &data_dir)?
            };
            let reranker = select_reranker()?;
            let current = mcp::resolve_current_repo(&data_dir, repo.as_deref());
            let server = mcp::McpServer::new(engine, embedder, reranker).with_current_repo(current);
            server.prewarm_sidecars();
            run_mcp_stdio(server)?;
        }
        Commands::Setup {
            harness,
            scope,
            claude_md,
            print_only,
        } => {
            if harness != "claude" {
                anyhow::bail!("unknown harness '{harness}' (supported: claude)");
            }
            if !matches!(scope.as_str(), "user" | "project" | "local") {
                anyhow::bail!("--scope must be user, project or local");
            }
            let exe = std::env::current_exe().context("locating the indexio binary")?;
            let cmd = format!(
                "claude mcp add --scope {scope} indexio -- \"{}\" mcp --data-dir \"{}\"",
                exe.display(),
                data_dir.display()
            );
            if print_only {
                println!("{cmd}");
            } else {
                println!("running: {cmd}");
                let status = std::process::Command::new("claude")
                    .args(["mcp", "add", "--scope", &scope, "indexio", "--"])
                    .arg(&exe)
                    .arg("mcp")
                    .arg("--data-dir")
                    .arg(&data_dir)
                    .status()
                    .context("running `claude` (is Claude Code installed and on PATH?)")?;
                anyhow::ensure!(status.success(), "claude mcp add failed ({status})");
            }
            if let Some(path) = claude_md {
                if !print_only {
                    let existing = std::fs::read_to_string(&path).unwrap_or_default();
                    if existing.contains("## indexio (code index MCP server)") {
                        println!("{} already has the indexio guidance block", path.display());
                    } else {
                        let mut s = existing;
                        if !s.is_empty() && !s.ends_with('\n') {
                            s.push('\n');
                        }
                        s.push('\n');
                        s.push_str(mcp::HARNESS_GUIDANCE);
                        std::fs::write(&path, s)
                            .with_context(|| format!("writing {}", path.display()))?;
                        println!("appended the indexio guidance block to {}", path.display());
                    }
                }
            } else {
                println!();
                println!("Add this to your CLAUDE.md (project or ~/.claude/CLAUDE.md) so the agent reaches for the index first:");
                println!();
                print!("{}", mcp::HARNESS_GUIDANCE);
                println!();
                println!("(or re-run with --claude-md <path> to append it)");
            }
        }
        Commands::Impact {
            symbols,
            diff,
            base,
            diff_file,
            repo,
            file,
            depth,
            max_sites,
            max_fanout,
            json,
        } => {
            validate_impact_target(
                &symbols,
                diff.as_deref(),
                diff_file.as_deref(),
                repo.as_deref(),
                file.as_deref(),
            )?;
            let opts = impact_cli::options(depth, max_sites, max_fanout);
            let engine = Engine::open(&data_dir)
                .with_context(|| format!("opening index at {}", data_dir.display()))?;
            let (changed, report) = if !symbols.is_empty() {
                (None, engine.impact_symbols(&symbols, &opts))
            } else if let Some(target) = file {
                let (r, p) = impact_cli::parse_repo_path(&target)?;
                (None, engine.impact_file(&r, &p, &opts))
            } else {
                // --diff REPO (run git) or --diff-file PATCH --repo REPO.
                let repo_name = diff.clone().or(repo).expect("validated above");
                let repo_path = impact_cli::resolve_repo_path(&data_dir, &repo_name)?;
                let patch = match diff_file {
                    Some(p) if p.as_os_str() == "-" => {
                        let mut s = String::new();
                        std::io::stdin().lock().read_to_string(&mut s)?;
                        s
                    }
                    Some(p) => std::fs::read_to_string(&p)
                        .with_context(|| format!("reading {}", p.display()))?,
                    None => impact_cli::git_diff(&repo_path, &base)?,
                };
                let provider = impact_cli::working_tree_provider(Some(repo_path));
                let (c, r) = engine.impact_diff(&repo_name, &patch, &provider, &opts);
                (Some(c), r)
            };
            if json {
                println!("{}", impact_cli::impact_to_json(changed.as_deref(), &report));
            } else {
                impact_cli::print_impact(changed.as_deref(), &report);
            }
        }
        Commands::Outline { target, json } => {
            let (r, p) = impact_cli::parse_repo_path(&target)?;
            let engine = Engine::open(&data_dir)
                .with_context(|| format!("opening index at {}", data_dir.display()))?;
            let items = engine
                .outline(&r, &p)
                .with_context(|| format!("{r}:{p} is not indexed"))?;
            if json {
                println!("{}", impact_cli::outline_to_json(&r, &p, &items));
            } else {
                impact_cli::print_outline(&r, &p, &items);
            }
        }
        Commands::Span { target, range } => {
            let (r, p) = impact_cli::parse_repo_path(&target)?;
            let (start, end) = impact_cli::parse_line_range(&range)?;
            let engine = Engine::open(&data_dir)
                .with_context(|| format!("opening index at {}", data_dir.display()))?;
            let (text, last) = engine
                .read_span(&r, &p, start, end)
                .with_context(|| format!("{r}:{p} is not indexed"))?;
            print!("{text}");
            if last < end {
                eprintln!("(stopped at line {last})");
            }
        }
    }
    Ok(())
}

/// Embedder selection (SPEC-P2 §4 + SPEC-P4 §2), in precedence order:
/// 1. INDEXIO_EMBED_BASE set -> `HttpEmbedder::from_env()` (production:
///    OpenAI-compatible /v1/embeddings); overrides `--embedder`.
/// 2. `--embedder hash` -> deterministic offline `HashEmbedder::new(512)`
///    (A/B escape hatch).
/// 3. default (`--embedder rindex` or no flag) ->
///    `RandomIndexingEmbedder::open(data_dir)`, the self-contained
///    CPU-only random-indexing model (created on first `indexio embed`).
pub(crate) fn select_embedder(
    embedder: Option<&str>,
    data_dir: &Path,
) -> anyhow::Result<Arc<dyn Embedder>> {
    if std::env::var_os("INDEXIO_EMBED_BASE").is_some() {
        return Ok(Arc::new(HttpEmbedder::from_env()?));
    }
    match embedder {
        Some("hash") => Ok(Arc::new(HashEmbedder::new(512))),
        Some("rindex") | None => Ok(Arc::new(RandomIndexingEmbedder::open(data_dir)?)),
        Some(other) => Err(anyhow::anyhow!(
            "unknown --embedder '{other}' (expected 'hash' or 'rindex')"
        )),
    }
}

/// Embedder selection that never fails (HTTP server startup): falls back
/// to the HashEmbedder with a warning when the configured embedder cannot
/// be initialized (e.g. INDEXIO_EMBED_BASE set but incomplete, or an
/// unreadable rindex model directory).
pub(crate) fn select_embedder_or_fallback(data_dir: &Path) -> Arc<dyn Embedder> {
    match select_embedder(None, data_dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("embedder init failed ({e}); using HashEmbedder");
            Arc::new(HashEmbedder::new(512))
        }
    }
}

/// Reranker selection (SPEC-P3 §2): `HttpReranker::from_env()` when
/// INDEXIO_RERANK_BASE is set (production: TEI/vLLM/Jina-style /rerank), else
/// a no-op. The offline `OverlapReranker` used to be the fallback; on 80
/// model-written questions over the registered repos it halved hybrid
/// recall@5 (67.5 → 35 %, SPEC-P10 §15), so `rerank` without a configured
/// reranker now leaves the fused order alone.
pub(crate) fn select_reranker() -> anyhow::Result<Arc<dyn Reranker>> {
    if let Some(r) = HttpReranker::from_env()? {
        Ok(Arc::new(r))
    } else {
        Ok(Arc::new(NoopReranker))
    }
}

/// Reranker selection that never fails (HTTP server startup): falls back
/// to the no-op with a warning when the HttpReranker env configuration is
/// incomplete.
pub(crate) fn select_reranker_or_fallback() -> Arc<dyn Reranker> {
    match select_reranker() {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("INDEXIO_RERANK_BASE set but HttpReranker init failed ({e}); rerank is a no-op");
            Arc::new(NoopReranker)
        }
    }
}

/// `--rerank` is only valid with `--mode hybrid` (SPEC-P3 §2). Clap cannot
/// express "flag valid only for one value of another arg" declaratively,
/// so this check runs right after parsing, before any index is opened.
pub(crate) fn validate_rerank_flag(mode: SearchMode, rerank: bool) -> anyhow::Result<()> {
    if rerank && mode != SearchMode::Hybrid {
        anyhow::bail!("--rerank requires --mode hybrid");
    }
    Ok(())
}

/// Serve auth selection (SPEC-P3 §3): --acl-file supersedes --auth-token;
/// --auth-token falls back to $INDEXIO_AUTH_TOKEN; neither -> open.
pub(crate) fn resolve_auth_config(
    auth_token: Option<String>,
    acl_file: Option<&Path>,
) -> anyhow::Result<serve::AuthConfig> {
    if let Some(path) = acl_file {
        let map = serve::load_acl_file(path)?;
        return Ok(serve::AuthConfig::Acl(Arc::new(map)));
    }
    let token = auth_token.or_else(|| {
        std::env::var("INDEXIO_AUTH_TOKEN")
            .ok()
            .filter(|t| !t.is_empty())
    });
    Ok(match token {
        Some(t) => serve::AuthConfig::Token(t),
        None => serve::AuthConfig::Open,
    })
}

/// `indexio usage`: per-tool and per-repo aggregates of the MCP usage log,
/// including the built-in-tool comparison (SPEC-P10 §22).
fn print_usage(data_dir: &Path, days: u64, as_json: bool) -> anyhow::Result<()> {
    let r = usage::aggregate(data_dir, days);
    if as_json {
        println!("{}", usage::to_json(&r));
        return Ok(());
    }
    println!("{}", usage::render(&r));
    Ok(())
}

/// MCP stdio loop: newline-delimited JSON-RPC 2.0 messages (one per line;
/// Content-Length framing is NOT supported). Notifications produce no line.
fn run_mcp_stdio(server: mcp::McpServer) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if let Some(resp) = server.handle(&line) {
            writeln!(out, "{resp}")?;
            out.flush()?;
        }
    }
    server.finish_background_work();
    Ok(())
}

/// Expand a leading "~/" to $HOME.
/// Home directory: `HOME`, else `USERPROFILE` (Windows processes launched
/// outside a Unix-style shell usually have no `HOME`).
fn home_dir() -> Option<PathBuf> {
    ["HOME", "USERPROFILE"]
        .iter()
        .filter_map(std::env::var_os)
        .find(|h| !h.is_empty())
        .map(PathBuf::from)
}

fn expand_tilde(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix("~/").or_else(|| s.strip_prefix("~\\")) {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    p.to_path_buf()
}

/// Data dir resolution: --data-dir flag > $INDEXIO_DATA_DIR > ~/.indexio.
fn resolve_data_dir(cli: &Cli) -> PathBuf {
    if let Some(d) = &cli.data_dir {
        return expand_tilde(d);
    }
    if let Some(d) = std::env::var_os("INDEXIO_DATA_DIR") {
        if !d.is_empty() {
            return expand_tilde(Path::new(&d));
        }
    }
    home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".indexio")
}

/// Names of registered repos (<data_dir>/repos/*.json), sorted.
fn registered_repos(data_dir: &Path) -> anyhow::Result<Vec<String>> {
    let dir = data_dir.join("repos");
    let mut out = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    out.push(stem.to_string());
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

/// CAS (entries, bytes); zeros when the CAS dir does not exist yet.
fn cas_stats(data_dir: &Path) -> (u64, u64) {
    let cas_dir = data_dir.join("cas");
    if !cas_dir.is_dir() {
        return (0, 0);
    }
    match Cas::open(&cas_dir) {
        Ok(cas) => cas.stats(),
        Err(_) => (0, 0),
    }
}

/// Engine::stats (which already sums shard file sizes into index_bytes)
/// merged with CAS stats.
fn merged_stats(data_dir: &Path) -> anyhow::Result<EngineStats> {
    let engine = Engine::open(data_dir)
        .with_context(|| format!("opening index at {}", data_dir.display()))?;
    let mut stats = engine.stats();
    let (entries, bytes) = cas_stats(data_dir);
    stats.cas_entries = entries;
    stats.cas_bytes = bytes;
    Ok(stats)
}

/// `repo:path:line:col` locator for one hit.
fn hit_location(h: &SearchHit) -> String {
    format!("{}:{}:{}:{}", h.repo, h.path, h.line, h.col)
}

/// Human search output: `repo:path:line:col  score  snippet`, aligned.
fn print_hits(hits: &[SearchHit]) {
    let locs: Vec<String> = hits.iter().map(hit_location).collect();
    let w = locs.iter().map(|l| l.len()).max().unwrap_or(0);
    for (h, loc) in hits.iter().zip(locs.iter()) {
        println!("{loc:<w$}  {:.2}  {}", h.score, h.snippet);
    }
}

/// Human hybrid output: locator + fused RRF score + ranks (+ rerank score
/// when a reranker ran) + snippet.
fn print_hybrid_hits(hits: &[indexio_query::HybridHit]) {
    let locs: Vec<String> = hits.iter().map(|h| hit_location(&h.hit)).collect();
    let w = locs.iter().map(|l| l.len()).max().unwrap_or(0);
    for (h, loc) in hits.iter().zip(locs.iter()) {
        let rank = |r: Option<usize>| r.map(|n| n.to_string()).unwrap_or_else(|| "-".into());
        let rr = h
            .rerank_score
            .map(|s| format!("  rerank {s:.4}"))
            .unwrap_or_default();
        println!(
            "{loc:<w$}  rrf {:.4}  lex {}  sem {}{}  {}",
            h.rrf,
            rank(h.lex_rank),
            rank(h.sem_rank),
            rr,
            h.hit.snippet.replace('\n', " / ")
        );
    }
}

/// Embed reports as a small table: repo, chunks, cas_hits, cas_misses,
/// embedded, ms.
fn print_embed_reports(reports: &[EmbedReport]) {
    println!(
        "{:<24} {:>7} {:>8} {:>10} {:>8} {:>6}",
        "repo", "chunks", "cas_hits", "cas_misses", "embedded", "ms"
    );
    for r in reports {
        println!(
            "{:<24} {:>7} {:>8} {:>10} {:>8} {:>6}",
            r.repo, r.chunks, r.cas_hits, r.cas_misses, r.embedded, r.elapsed_ms
        );
    }
    if let Some(r) = reports.first() {
        println!(
            "index build (vec + bm25, once for all repos above): {} ms",
            r.index_build_ms
        );
    }
}

/// `indexio add` / `indexio sync`: embed what the sync changed (SPEC-P10
/// §28) — the repos whose docs were added or deleted, incrementally into a
/// delta segment — and rebuild the whole plane only when there is none yet.
/// (Before: every `add` of one folder rebuilt every repo's vectors, 25 s
/// and a 500 MB segment that every running server had to reopen.)
fn embed_after_sync(data_dir: &Path, indexed: &[IndexReport]) -> anyhow::Result<()> {
    let embedder = select_embedder(None, data_dir)?;
    let shards_dir = data_dir.join("shards");
    if !shards_dir.is_dir() {
        return Ok(());
    }
    let set = ShardSet::open_dir(&shards_dir)?;
    let changed: Vec<String> = indexed
        .iter()
        .filter(|r| r.docs_added > 0 || r.docs_deleted > 0)
        .map(|r| r.repo.clone())
        .collect();
    let reports = if !indexio_embed::pipeline::plane_exists(data_dir, embedder.model_id()) {
        indexio_embed::pipeline::embed_all(&set, data_dir, embedder.as_ref())?
    } else if changed.is_empty() {
        Vec::new()
    } else {
        indexio_embed::pipeline::embed_repos_with(
            &set,
            data_dir,
            &changed,
            embedder.as_ref(),
            indexio_embed::pipeline::MAX_CHARS,
        )?
    };
    if !reports.is_empty() {
        println!("semantic plane:");
        print_embed_reports(&reports);
    }
    Ok(())
}

fn print_sync_report(r: &SyncReport) {
    let added: u64 = r.indexed.iter().map(|i| i.docs_added).sum();
    let deleted: u64 = r.indexed.iter().map(|i| i.docs_deleted).sum();
    let unchanged: u64 = r.indexed.iter().map(|i| i.docs_unchanged).sum();
    println!(
        "{}: {} repos ({} skipped), {} files added, {} deleted, {} unchanged{}",
        r.source,
        r.indexed.len(),
        r.skipped,
        added,
        deleted,
        unchanged,
        if r.failed.is_empty() { String::new() } else { format!(", {} FAILED", r.failed.len()) }
    );
    for ir in &r.indexed {
        println!(
            "  {:<28} +{} -{} ={}  {} ms",
            ir.repo, ir.docs_added, ir.docs_deleted, ir.docs_unchanged, ir.elapsed_ms
        );
    }
    for (name, err) in &r.failed {
        println!("  FAILED {name}: {err}");
    }
}

fn print_next_steps(data_dir: &Path) {
    println!();
    println!("ready. try:");
    println!("  indexio search \"<identifier or words>\" --mode hybrid --data-dir {}", data_dir.display());
    println!("  indexio mcp --data-dir {}      # for Claude Code / any MCP harness", data_dir.display());
    println!("  indexio sync --data-dir {}     # keep it fresh (cron every 15 min)", data_dir.display());
}

fn print_org_sync_report(r: &OrgSyncReport) {
    println!("org sync: {} listed, {} cloned/updated, {} skipped", r.repos_listed, r.repos_cloned, r.repos_skipped);
    for (name, err) in &r.repos_failed {
        println!("  FAILED {name}: {err}");
    }
    for ir in &r.indexed {
        print_report(ir);
    }
}

fn print_report(r: &IndexReport) {
    let short_commit: String = r.commit.chars().take(12).collect();
    println!("repo '{}' @ {short_commit}", r.repo);
    println!(
        "  docs:    {} added, {} deleted, {} unchanged",
        r.docs_added, r.docs_deleted, r.docs_unchanged
    );
    println!("  cas:     {} hits, {} misses", r.cas_hits, r.cas_misses);
    println!("  bytes:   {} indexed", r.bytes_indexed);
    println!("  elapsed: {} ms", r.elapsed_ms);
}

fn print_stats(s: &EngineStats) {
    println!("repos:           {}", s.repos.len());
    for r in &s.repos {
        println!("  - {r}");
    }
    println!("docs:            {}", s.doc_count);
    println!("total raw bytes: {}", s.total_raw_bytes);
    println!("shards:          {}", s.shard_count);
    println!("index bytes:     {}", s.index_bytes);
    println!("cas entries:     {}", s.cas_entries);
    println!("cas bytes:       {}", s.cas_bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parse_search_flags() {
        let cli = Cli::try_parse_from(["indexio", "search", "foo", "--limit", "5", "--json"]).unwrap();
        match cli.command {
            Commands::Search {
                query,
                limit,
                json,
                mode,
                rerank,
                fusion,
                embedder,
            } => {
                assert_eq!(query, "foo");
                assert_eq!(limit, 5);
                assert!(json);
                assert_eq!(mode, "lexical");
                assert!(!rerank);
                assert_eq!(fusion, "rrf"); // SPEC-P5 §A3 default
                assert!(embedder.is_none());
            }
            _ => panic!("expected search"),
        }
        // --embedder flag parses on search (SPEC-P4 §2).
        let cli =
            Cli::try_parse_from(["indexio", "search", "foo", "--mode", "semantic", "--embedder", "hash"])
                .unwrap();
        match cli.command {
            Commands::Search { embedder, .. } => {
                assert_eq!(embedder.as_deref(), Some("hash"))
            }
            _ => panic!("expected search"),
        }
        // defaults
        let cli = Cli::try_parse_from(["indexio", "search", "foo"]).unwrap();
        match cli.command {
            Commands::Search { limit, json, mode, rerank, .. } => {
                assert_eq!(limit, 50);
                assert!(!json);
                assert_eq!(mode, "lexical");
                assert!(!rerank);
            }
            _ => panic!("expected search"),
        }
        let cli =
            Cli::try_parse_from(["indexio", "search", "foo", "--mode", "hybrid"]).unwrap();
        match cli.command {
            Commands::Search { mode, .. } => assert_eq!(mode, "hybrid"),
            _ => panic!("expected search"),
        }
        // --rerank flag parses (validation is post-parse, see below)
        let cli =
            Cli::try_parse_from(["indexio", "search", "foo", "--mode", "hybrid", "--rerank"]).unwrap();
        match cli.command {
            Commands::Search { rerank, mode, .. } => {
                assert!(rerank);
                assert_eq!(mode, "hybrid");
            }
            _ => panic!("expected search"),
        }
        // SPEC-P5 §A3: --fusion parses; value validation is post-parse.
        let cli = Cli::try_parse_from([
            "indexio", "search", "foo", "--mode", "hybrid", "--fusion", "combmnz",
        ])
        .unwrap();
        match cli.command {
            Commands::Search { fusion, mode, .. } => {
                assert_eq!(fusion, "combmnz");
                assert_eq!(mode, "hybrid");
                assert_eq!(
                    FusionAlgo::parse(&fusion).unwrap(),
                    FusionAlgo::CombMnz
                );
            }
            _ => panic!("expected search"),
        }
    }

    #[test]
    fn rerank_flag_requires_hybrid_mode() {
        // SPEC-P3 §2: --rerank is only valid with --mode hybrid.
        assert!(validate_rerank_flag(SearchMode::Hybrid, true).is_ok());
        assert!(validate_rerank_flag(SearchMode::Hybrid, false).is_ok());
        assert!(validate_rerank_flag(SearchMode::Lexical, false).is_ok());
        let err = validate_rerank_flag(SearchMode::Lexical, true).unwrap_err();
        assert!(format!("{err}").contains("--mode hybrid"), "{err}");
        let err = validate_rerank_flag(SearchMode::Semantic, true).unwrap_err();
        assert!(format!("{err}").contains("--mode hybrid"), "{err}");
        // End-to-end at the clap layer: the invalid combination parses
        // (clap cannot express the cross-arg value rule) but validation
        // rejects it before any engine work.
        let cli =
            Cli::try_parse_from(["indexio", "search", "foo", "--mode", "lexical", "--rerank"])
                .unwrap();
        match cli.command {
            Commands::Search { mode, rerank, .. } => {
                let mode = SearchMode::parse(&mode).unwrap();
                assert!(validate_rerank_flag(mode, rerank).is_err());
            }
            _ => panic!("expected search"),
        }
    }

    #[test]
    fn parse_serve_auth_flags() {
        // defaults: no auth flags -> open behavior
        let cli = Cli::try_parse_from(["indexio", "serve"]).unwrap();
        match cli.command {
            Commands::Serve {
                port,
                bind,
                auth_token,
                acl_file,
            } => {
                assert_eq!(port, 7717);
                assert!(bind.is_none(), "loopback unless asked");
                assert!(auth_token.is_none());
                assert!(acl_file.is_none());
            }
            _ => panic!("expected serve"),
        }
        let cli = Cli::try_parse_from(["indexio", "serve", "--bind", "0.0.0.0", "--port", "8080"]).unwrap();
        match cli.command {
            Commands::Serve { port, bind, .. } => {
                assert_eq!(port, 8080);
                assert!(!bind.unwrap().is_loopback());
            }
            _ => panic!("expected serve"),
        }
        let cli = Cli::try_parse_from([
            "indexio", "serve", "--port", "9000", "--auth-token", "tok",
        ])
        .unwrap();
        match cli.command {
            Commands::Serve {
                port,
                bind: _,
                auth_token,
                acl_file,
            } => {
                assert_eq!(port, 9000);
                assert_eq!(auth_token.as_deref(), Some("tok"));
                assert!(acl_file.is_none());
            }
            _ => panic!("expected serve"),
        }
        let cli = Cli::try_parse_from(["indexio", "serve", "--acl-file", "/tmp/acl.json"]).unwrap();
        match cli.command {
            Commands::Serve { acl_file, .. } => {
                assert_eq!(acl_file.as_deref(), Some(Path::new("/tmp/acl.json")));
            }
            _ => panic!("expected serve"),
        }
    }

    #[test]
    fn resolve_auth_config_precedence() {
        // acl-file supersedes --auth-token (SPEC-P3 §3)
        let tmp = tempfile::tempdir().unwrap();
        let acl = tmp.path().join("acl.json");
        std::fs::write(&acl, r#"{"tokens":{"t":{"allow":["*"]}}}"#).unwrap();
        let cfg = resolve_auth_config(Some("tok".to_string()), Some(&acl)).unwrap();
        assert!(matches!(cfg, serve::AuthConfig::Acl(_)));
        // token flag without acl-file
        let cfg = resolve_auth_config(Some("tok".to_string()), None).unwrap();
        assert!(matches!(cfg, serve::AuthConfig::Token(ref t) if t == "tok"));
        // env fallback when the flag is absent
        std::env::set_var("INDEXIO_AUTH_TOKEN", "env-tok");
        let cfg = resolve_auth_config(None, None).unwrap();
        assert!(matches!(cfg, serve::AuthConfig::Token(ref t) if t == "env-tok"));
        // neither flag nor env -> open
        std::env::remove_var("INDEXIO_AUTH_TOKEN");
        let cfg = resolve_auth_config(None, None).unwrap();
        assert!(matches!(cfg, serve::AuthConfig::Open));
        // missing acl file -> error, not panic
        assert!(resolve_auth_config(None, Some(Path::new("/definitely/missing.json"))).is_err());
    }

    #[test]
    fn parse_embed_and_org_sync_and_embcas() {
        // embed defaults: no target flag = all repos, MAX_CHARS default.
        let cli = Cli::try_parse_from(["indexio", "embed"]).unwrap();
        match cli.command {
            Commands::Embed {
                repo,
                all,
                max_chars,
                embedder,
                rebuild_model,
            } => {
                assert!(repo.is_none());
                assert!(!all);
                assert_eq!(max_chars, indexio_embed::pipeline::MAX_CHARS);
                assert!(embedder.is_none());
                assert!(!rebuild_model);
            }
            _ => panic!("expected embed"),
        }
        // --embedder + --rebuild-model parse on embed (SPEC-P4 §2).
        let cli =
            Cli::try_parse_from(["indexio", "embed", "--embedder", "rindex", "--rebuild-model"])
                .unwrap();
        match cli.command {
            Commands::Embed {
                embedder,
                rebuild_model,
                ..
            } => {
                assert_eq!(embedder.as_deref(), Some("rindex"));
                assert!(rebuild_model);
            }
            _ => panic!("expected embed"),
        }
        let cli = Cli::try_parse_from(["indexio", "embed", "--repo", "x", "--max-chars", "800"]).unwrap();
        match cli.command {
            Commands::Embed { repo, max_chars, .. } => {
                assert_eq!(repo.as_deref(), Some("x"));
                assert_eq!(max_chars, 800);
            }
            _ => panic!("expected embed"),
        }
        // --repo and --all conflict
        assert!(Cli::try_parse_from(["indexio", "embed", "--repo", "x", "--all"]).is_err());

        // org-sync smoke: flags parse; --org is required
        assert!(Cli::try_parse_from(["indexio", "org-sync"]).is_err());
        let cli = Cli::try_parse_from([
            "indexio", "org-sync", "--org", "acme", "--limit", "5",
            "--include-forks", "--include-archived", "--full-clone",
        ])
        .unwrap();
        match cli.command {
            Commands::OrgSync {
                org,
                token_env,
                dest,
                limit,
                include_forks,
                include_archived,
                full_clone,
            } => {
                assert_eq!(org, "acme");
                assert_eq!(token_env, "GITHUB_TOKEN");
                assert!(dest.is_none());
                assert_eq!(limit, Some(5));
                assert!(include_forks && include_archived && full_clone);
            }
            _ => panic!("expected org-sync"),
        }

        assert!(Cli::try_parse_from(["indexio", "embcas-stats"]).is_ok());
        assert!(Cli::try_parse_from(["indexio", "embcas-stats", "--json"]).is_ok());
    }

    #[test]
    fn mode_parse_invalid_is_error() {
        assert!(SearchMode::parse("lexical").is_ok());
        let err = SearchMode::parse("bogus").unwrap_err();
        assert!(err.contains("bogus"), "{err}");
    }

    #[test]
    fn parse_index_and_defaults() {
        let cli = Cli::try_parse_from(["indexio", "index", "/tmp/repo"]).unwrap();
        match cli.command {
            Commands::Index { repo_path, name } => {
                assert_eq!(repo_path, PathBuf::from("/tmp/repo"));
                assert!(name.is_none());
            }
            _ => panic!("expected index"),
        }
        let cli = Cli::try_parse_from(["indexio", "index", "/tmp/repo", "--name", "mine"]).unwrap();
        match cli.command {
            Commands::Index { name, .. } => assert_eq!(name.as_deref(), Some("mine")),
            _ => panic!("expected index"),
        }
    }

    #[test]
    fn parse_reindex_group() {
        // exactly one of --repo / --all is required
        assert!(Cli::try_parse_from(["indexio", "reindex"]).is_err());
        assert!(Cli::try_parse_from(["indexio", "reindex", "--all"]).is_ok());
        assert!(Cli::try_parse_from(["indexio", "reindex", "--repo", "x"]).is_ok());
        assert!(Cli::try_parse_from(["indexio", "reindex", "--repo", "x", "--all"]).is_err());
    }

    #[test]
    fn parse_global_data_dir_after_subcommand() {
        let cli = Cli::try_parse_from(["indexio", "stats", "--data-dir", "/tmp/x"]).unwrap();
        assert_eq!(cli.data_dir, Some(PathBuf::from("/tmp/x")));
    }

    #[test]
    fn parse_serve_and_compact_defaults() {
        let cli = Cli::try_parse_from(["indexio", "serve"]).unwrap();
        match cli.command {
            Commands::Serve { port, .. } => assert_eq!(port, 7717),
            _ => panic!("expected serve"),
        }
        let cli = Cli::try_parse_from(["indexio", "compact"]).unwrap();
        match cli.command {
            Commands::Compact { max_shards, vectors } => assert_eq!((max_shards, vectors), (4, false)),
            _ => panic!("expected compact"),
        }
    }

    #[test]
    fn data_dir_flag_wins_over_env() {
        std::env::set_var("INDEXIO_DATA_DIR", "/tmp/from-env");
        let cli = Cli::try_parse_from(["indexio", "stats", "--data-dir", "/tmp/from-flag"]).unwrap();
        assert_eq!(resolve_data_dir(&cli), PathBuf::from("/tmp/from-flag"));
        let cli = Cli::try_parse_from(["indexio", "stats"]).unwrap();
        assert_eq!(resolve_data_dir(&cli), PathBuf::from("/tmp/from-env"));
        std::env::remove_var("INDEXIO_DATA_DIR");
    }

    #[test]
    fn tilde_expansion() {
        let home = std::env::var_os("HOME").unwrap();
        assert_eq!(
            expand_tilde(Path::new("~/x/y")),
            PathBuf::from(home).join("x/y")
        );
        assert_eq!(expand_tilde(Path::new("/abs")), PathBuf::from("/abs"));
    }

    /// SPEC-P4 §2: embedder selection precedence + the random-indexing
    /// embedder wired end-to-end through the pipeline and Engine.
    #[test]
    fn embedder_selection_and_rindex_end_to_end() {
        use indexio_index::{ShardSet, ShardWriter};
        use indexio_types::{BlobId, DocMeta, ExtractedArtifact, Lang};

        // INDEXIO_EMBED_BASE would take precedence in selection; ensure unset.
        let saved_base = std::env::var_os("INDEXIO_EMBED_BASE");
        std::env::remove_var("INDEXIO_EMBED_BASE");

        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();

        // Selection precedence: default -> rindex-v2, flag -> hash-v1.
        let e = select_embedder(None, data_dir).unwrap();
        assert_eq!(e.model_id(), "rindex-v2");
        assert_eq!(e.dim(), indexio_embed::rindex::DIM);
        let e = select_embedder(Some("rindex"), data_dir).unwrap();
        assert_eq!(e.model_id(), "rindex-v2");
        let e = select_embedder(Some("hash"), data_dir).unwrap();
        assert_eq!(e.model_id(), "hash-v1");
        assert!(select_embedder(Some("bogus"), data_dir).is_err());

        // Build a tiny 1-repo shard set under data_dir/shards.
        let shards_dir = data_dir.join("shards");
        std::fs::create_dir_all(&shards_dir).unwrap();
        let mut w = ShardWriter::new(&shards_dir).unwrap();
        let docs: [(u32, &str, &[u8]); 3] = [
            (
                0,
                "src/auth.rs",
                b"fn authenticate_user(password: &str) -> Session { login(password) }\n",
            ),
            (
                0,
                "src/db.rs",
                b"fn database_pool_connection() -> Pool { Pool::open() }\n",
            ),
            (
                0,
                "src/main.rs",
                b"fn main() { let s = authenticate_user(\"hunter2\"); }\n",
            ),
        ];
        for (repo_id, path, content) in &docs {
            let meta = DocMeta {
                blob: BlobId::from_content(content),
                repo_id: *repo_id,
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
        w.finish(&["r1".to_string()]).unwrap();

        // Default-selected embedder drives the pipeline; model file lands
        // under <data_dir>/sem, vec index under rindex-v2.
        let embedder = select_embedder(None, data_dir).unwrap();
        let set = ShardSet::open_dir(&shards_dir).unwrap();
        let reports =
            indexio_embed::pipeline::embed_all(&set, data_dir, embedder.as_ref()).unwrap();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].embedded > 0);
        assert!(RandomIndexingEmbedder::model_path(data_dir).is_file());

        // Engine::search_semantic works end-to-end on the rindex vectors.
        let engine = Engine::open(data_dir).unwrap();
        let hits = engine
            .search_semantic("user authentication login", 5, embedder.as_ref())
            .unwrap();
        assert!(!hits.is_empty(), "semantic search returned no hits");
        assert!(
            hits[0].path.contains("auth.rs"),
            "top hit should be the auth chunk: {:?}",
            hits.iter().map(|h| &h.path).collect::<Vec<_>>()
        );

        if let Some(v) = saved_base {
            std::env::set_var("INDEXIO_EMBED_BASE", v);
        }
    }

    /// SPEC-P6 §3: impact/outline/span parse; exactly one impact target.
    #[test]
    fn parse_impact_outline_span() {
        let cli = Cli::try_parse_from([
            "indexio", "impact", "--symbol", "a", "--symbol", "b", "--depth", "3", "--json",
        ])
        .unwrap();
        match cli.command {
            Commands::Impact {
                symbols,
                diff,
                base,
                diff_file,
                repo,
                file,
                depth,
                max_sites,
                max_fanout,
                json,
            } => {
                assert_eq!(symbols, vec!["a".to_string(), "b".to_string()]);
                assert!(diff.is_none() && diff_file.is_none() && repo.is_none() && file.is_none());
                assert_eq!(base, "HEAD");
                assert_eq!((depth, max_sites, max_fanout, json), (3, 500, 200, true));
                validate_impact_target(&symbols, None, None, None, None).unwrap();
            }
            _ => panic!("expected impact"),
        }
        let cli = Cli::try_parse_from(["indexio", "impact", "--diff", "core", "--base", "main"]).unwrap();
        match cli.command {
            Commands::Impact { diff, base, .. } => {
                assert_eq!(diff.as_deref(), Some("core"));
                assert_eq!(base, "main");
            }
            _ => panic!("expected impact"),
        }
        // target validation (post-parse)
        assert!(validate_impact_target(&[], None, None, None, None).is_err());
        assert!(validate_impact_target(&["a".into()], Some("r"), None, None, None).is_err());
        assert!(validate_impact_target(&[], None, Some(Path::new("p.diff")), None, None).is_err());
        assert!(validate_impact_target(&[], None, Some(Path::new("p.diff")), Some("r"), None).is_ok());
        assert!(validate_impact_target(&[], None, None, None, Some("r:p")).is_ok());

        let cli = Cli::try_parse_from(["indexio", "outline", "core:src/a.rs", "--json"]).unwrap();
        match cli.command {
            Commands::Outline { target, json } => {
                assert_eq!(target, "core:src/a.rs");
                assert!(json);
            }
            _ => panic!("expected outline"),
        }
        let cli = Cli::try_parse_from(["indexio", "span", "core:src/a.rs", "10:20"]).unwrap();
        match cli.command {
            Commands::Span { target, range } => {
                assert_eq!(target, "core:src/a.rs");
                assert_eq!(range, "10:20");
            }
            _ => panic!("expected span"),
        }
        assert!(Cli::try_parse_from(["indexio", "span", "core:src/a.rs"]).is_err());
    }

    /// SPEC-P7: add / sync / sources / remove parse.
    #[test]
    fn parse_add_sync_sources_remove() {
        let cli = Cli::try_parse_from([
            "indexio", "add", "github:acme", "--include-forks", "--full-clone", "--limit", "5", "--no-embed",
        ])
        .unwrap();
        match cli.command {
            Commands::Add { source, include_forks, include_archived, full_clone, limit, no_sync, no_embed, dest } => {
                assert_eq!(source, "github:acme");
                assert!(include_forks && full_clone && no_embed);
                assert!(!include_archived && !no_sync);
                assert_eq!(limit, Some(5));
                assert!(dest.is_none());
            }
            _ => panic!("expected add"),
        }
        assert!(Cli::try_parse_from(["indexio", "add"]).is_err());
        assert!(matches!(Cli::try_parse_from(["indexio", "sync"]).unwrap().command, Commands::Sync { no_embed: false, limit: None }));
        assert!(matches!(Cli::try_parse_from(["indexio", "sync", "--no-embed"]).unwrap().command, Commands::Sync { no_embed: true, .. }));
        assert!(matches!(Cli::try_parse_from(["indexio", "sources"]).unwrap().command, Commands::Sources));
        match Cli::try_parse_from(["indexio", "remove", "azdo:acme/p"]).unwrap().command {
            Commands::Remove { source } => assert_eq!(source, "azdo:acme/p"),
            _ => panic!("expected remove"),
        }
    }

    #[test]
    fn parse_setup() {
        let cli = Cli::try_parse_from(["indexio", "setup", "claude", "--scope", "project", "--print-only"]).unwrap();
        match cli.command {
            Commands::Setup { harness, scope, claude_md, print_only } => {
                assert_eq!(harness, "claude");
                assert_eq!(scope, "project");
                assert!(claude_md.is_none() && print_only);
            }
            _ => panic!("expected setup"),
        }
        assert!(Cli::try_parse_from(["indexio", "setup"]).is_err());
    }

    #[test]
    fn hit_location_format() {
        let h = SearchHit {
            repo: "r".into(),
            path: "src/a.rs".into(),
            line: 12,
            col: 3,
            snippet: "x".into(),
            score: 1.0,
            lang: indexio_types::Lang::Rust,
        };
        assert_eq!(hit_location(&h), "r:src/a.rs:12:3");
    }
}
