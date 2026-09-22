# indexio-cli

[![PyPI](https://img.shields.io/pypi/v/indexio-cli?label=pypi)](https://pypi.org/project/indexio-cli/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/IOServicesLabs/indexio/blob/main/LICENSE)

Code search, impact analysis and MCP server for AI coding agents, as one binary.
It indexes your repositories once and answers a query in milliseconds, over the Model
Context Protocol (MCP) and an HTTP API. No database and no external service.

```bash
pip install indexio-cli
indexio add ~/code            # index every git repository under a folder
indexio search parse_config   # search it
indexio setup claude          # register the MCP server with Claude Code
indexio hook install          # route reads and greps through the index
```

Agents get `code_search`, `code_grep`, `find_symbol`, `who_calls`, `file_outline`,
`read_span`, `impact_of_symbol`, `impact_of_diff` and `recall` as tools. On real traffic
that cuts what the tools return by 50 to 90 percent against the built-in Read, Grep and
Glob, and a lookup takes single-digit milliseconds instead of tens.

The wheel bundles the `indexio` binary for your platform, so nothing is downloaded at
install time and no Rust toolchain is needed. Wheels are published for Linux and macOS
(x86_64, arm64) and Windows (x86_64). `git` must be on the PATH. The package is called
`indexio-cli` because `indexio` on PyPI belongs to an unrelated project; the command it
installs is `indexio`.

Documentation and source: https://github.com/IOServicesLabs/indexio
Licence: Apache-2.0
