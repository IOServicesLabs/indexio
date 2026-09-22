# indexio

[![npm](https://img.shields.io/npm/v/indexio?label=npm)](https://www.npmjs.com/package/indexio)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/IOServicesLabs/indexio/blob/main/LICENSE)

Code search, impact analysis and MCP server for AI coding agents, as one binary.
It indexes your repositories once and answers a query in milliseconds, over the Model
Context Protocol (MCP) and an HTTP API. No database and no external service.

```bash
npm install -g indexio        # or: npx indexio --help
indexio add ~/code            # index every git repository under a folder
indexio search parse_config   # search it
indexio setup claude          # register the MCP server with Claude Code
indexio hook install          # route reads and greps through the index
```

Agents get `code_search`, `code_grep`, `find_symbol`, `who_calls`, `file_outline`,
`read_span`, `impact_of_symbol`, `impact_of_diff` and `recall` as tools. On real traffic
that cuts what the tools return by 50 to 90 percent against the built-in Read, Grep and
Glob, and a lookup takes single-digit milliseconds instead of tens.

This package downloads the prebuilt binary for your platform from the matching GitHub
release at install time and verifies its SHA-256. Nothing is sent anywhere: the index is a
set of files in `~/.indexio`. Supported: Linux and macOS (x86_64, arm64), Windows (x86_64).
`git` must be on the PATH.

If the download is blocked, install the binary another way and the shim will use it:
https://github.com/IOServicesLabs/indexio#install

Documentation and source: https://github.com/IOServicesLabs/indexio
Licence: Apache-2.0
