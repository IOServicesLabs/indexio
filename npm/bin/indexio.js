#!/usr/bin/env node
"use strict";
const path = require("path");
const fs = require("fs");
const { spawnSync } = require("child_process");

const exe = path.join(__dirname, "..", "vendor", process.platform === "win32" ? "indexio.exe" : "indexio");
if (!fs.existsSync(exe)) {
  console.error("indexio: the binary was not downloaded at install time.");
  console.error("run `npm rebuild indexio` with network access, or install from https://github.com/IOServicesLabs/indexio#install");
  process.exit(1);
}
const r = spawnSync(exe, process.argv.slice(2), { stdio: "inherit", windowsHide: true });
if (r.error) {
  console.error(`indexio: ${r.error.message}`);
  process.exit(1);
}
process.exit(r.status === null ? 1 : r.status);
