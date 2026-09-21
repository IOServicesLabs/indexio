# Security

indexio runs as the local user. The MCP server speaks over stdio and has no
authentication; do not expose it to a network. The HTTP API refuses to bind a
non-loopback address without a bearer token or an ACL file. Credentials for
remote sources are read from the environment only and never written to disk.

## Report a vulnerability

Use GitHub's private vulnerability reporting on this repository (Security tab,
"Report a vulnerability"). Do not open a public issue for a security problem.
Include the version (`indexio --version`), the platform, and steps to reproduce.
You will get an acknowledgement within a few days.
