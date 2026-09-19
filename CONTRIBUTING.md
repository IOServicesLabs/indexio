# Contributing

Thank you for your interest in indexio.

## Report a problem

Open an issue with the command you ran, the output you got, the output you expected, and
`indexio --version`. For a search result that looks wrong, add `indexio stats` and the
query. Do not include tokens or private code.

## Change the code

1. Fork the repository and create a branch.
2. Build and test: `cargo build --release` and `cargo test --workspace`.
3. Keep a change small and give it one purpose. Add a test for a behaviour change.
4. Write the commit message in the imperative: what the change does and why.
5. Open a pull request. CI checks the manifests, the scripts and the workflows.

`docs/DEVELOPMENT.md` explains the crates, the benchmarks and the data layout. The design
of each subsystem is in `docs/SPEC*.md`.

## License

By contributing you agree that your contribution is licensed under the Apache License 2.0,
the license of this project.
