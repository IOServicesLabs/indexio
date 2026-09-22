#!/usr/bin/env node
// postinstall: download the indexio binary of this package's version from
// the GitHub release, check its SHA-256, and place it in vendor/.
// Nothing here runs at `indexio` time; bin/indexio.js only execs the binary.
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");
const https = require("https");
const crypto = require("crypto");
const { execFileSync } = require("child_process");

const REPO = "IOServicesLabs/indexio";
const version = require("./package.json").version;
const platform = { linux: "linux", darwin: "macos", win32: "windows" }[process.platform];
const arch = { x64: "x86_64", arm64: "aarch64" }[process.arch];
const vendor = path.join(__dirname, "vendor");
const exe = path.join(vendor, process.platform === "win32" ? "indexio.exe" : "indexio");

function fetch(url, redirects = 5) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { "User-Agent": "indexio-npm" } }, (res) => {
      if ([301, 302, 307, 308].includes(res.statusCode) && res.headers.location && redirects > 0) {
        res.resume();
        return fetch(res.headers.location, redirects - 1).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(`GET ${url}: HTTP ${res.statusCode}`));
      }
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => resolve(Buffer.concat(chunks)));
      res.on("error", reject);
    }).on("error", reject);
  });
}

/// Unpack `file` into `dir`. On Windows the `tar` first on PATH is often
/// Git's GNU tar, which reads `C:\...` as a remote host ("Cannot connect
/// to C"), so the system bsdtar is called by its full path and PowerShell's
/// Expand-Archive is the fallback. Elsewhere plain `tar` handles both.
function extract(file, dir) {
  const attempts = [];
  if (process.platform === "win32") {
    const sysTar = path.join(process.env.SystemRoot || "C:\Windows", "System32", "tar.exe");
    if (fs.existsSync(sysTar)) {
      attempts.push(() => execFileSync(sysTar, ["-xf", file, "-C", dir], { stdio: "inherit" }));
    }
    attempts.push(() =>
      execFileSync(
        "powershell",
        ["-NoProfile", "-NonInteractive", "-Command", `Expand-Archive -LiteralPath '${file}' -DestinationPath '${dir}' -Force`],
        { stdio: "inherit" }
      )
    );
  }
  attempts.push(() => execFileSync("tar", ["-xf", file, "-C", dir], { stdio: "inherit" }));
  let last;
  for (const run of attempts) {
    try {
      run();
      return;
    } catch (e) {
      last = e;
    }
  }
  throw last;
}

async function main() {
  if (process.env.INDEXIO_SKIP_DOWNLOAD) return;
  if (!platform || !arch) {
    console.error(`indexio: no prebuilt binary for ${process.platform}/${process.arch}; build from source (cargo install --git https://github.com/${REPO} indexio)`);
    return;
  }
  const name = `indexio-${version}-${platform}-${arch}`;
  const ext = platform === "windows" ? "zip" : "tar.gz";
  const base = `https://github.com/${REPO}/releases/download/v${version}`;
  const archive = await fetch(`${base}/${name}.${ext}`);
  const want = (await fetch(`${base}/${name}.${ext}.sha256`)).toString("utf8").trim().split(/\s+/)[0].toLowerCase();
  const have = crypto.createHash("sha256").update(archive).digest("hex");
  if (want !== have) throw new Error("checksum mismatch");

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "indexio-"));
  const file = path.join(tmp, `${name}.${ext}`);
  fs.writeFileSync(file, archive);
  extract(file, tmp);
  fs.mkdirSync(vendor, { recursive: true });
  fs.copyFileSync(path.join(tmp, name, path.basename(exe)), exe);
  fs.chmodSync(exe, 0o755);
  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(`indexio ${version} installed (${platform}-${arch})`);
}

main().catch((e) => {
  console.error(`indexio: download failed: ${e.message}`);
  console.error(`install it another way: https://github.com/${REPO}#install`);
  // do not fail the whole npm install; the shim explains what to do
});
