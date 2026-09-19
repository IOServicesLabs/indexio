# indexio

Code search, impact analysis and MCP server for AI coding agents, as one binary.

```bash
npm install -g indexio        # or: npx indexio --help
indexio add ~/code            # index every git repository under a folder
indexio setup claude          # register the MCP server with Claude Code
indexio hook install          # route reads and greps through the index
```

This package downloads the prebuilt binary for your platform from the GitHub
release at install time and checks its SHA-256. Documentation, source and other
install options: https://github.com/IOServicesLabs/indexio
