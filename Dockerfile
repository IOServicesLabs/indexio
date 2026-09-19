# indexio — code search, impact analysis and MCP server for AI coding agents.
#
#   docker build -t indexio .
#   docker run --rm -v indexio-data:/data -v ~/code:/repos:ro indexio add /repos
#   docker run -d -p 7717:7717 -v indexio-data:/data -e INDEXIO_AUTH_TOKEN=... indexio
#
# The data dir is /data (a volume); repos to index are mounted wherever you
# like and added by path. The default command serves the HTTP API on all
# container interfaces, which the CLI only permits with an auth token or an
# ACL file — set INDEXIO_AUTH_TOKEN. See README "Deploying with Docker".

FROM rust:1-bookworm AS build
WORKDIR /src
# dependency layer: build once per Cargo.lock change
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release -p indexio --locked \
    && strip target/release/indexio

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 indexio \
    && mkdir -p /data && chown indexio:indexio /data
COPY --from=build /src/target/release/indexio /usr/local/bin/indexio
USER indexio
ENV INDEXIO_DATA_DIR=/data \
    INDEXIO_BIND=0.0.0.0
VOLUME ["/data"]
EXPOSE 7717
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD ["/bin/bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/7717 && printf 'GET /health HTTP/1.0\\r\\n\\r\\n' >&3 && head -c 12 <&3 | grep -q 200"]
ENTRYPOINT ["/usr/bin/tini", "--", "indexio"]
CMD ["serve", "--port", "7717"]
