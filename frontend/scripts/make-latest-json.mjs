#!/usr/bin/env node
// Writes the updater manifest (latest.json) for a local macOS release build.
// Usage: node scripts/make-latest-json.mjs [notes]
// Run from frontend/ after `pnpm run tauri:build` with the signing key set.
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const REPO = 'kiriyk/meeting-minutes-russian';
const conf = JSON.parse(readFileSync('src-tauri/tauri.conf.json', 'utf8'));
const version = conf.version;

const bundleDir = ['../target/release/bundle/macos', 'src-tauri/target/release/bundle/macos'].find(existsSync);
if (!bundleDir) throw new Error('No macOS bundle found; run `pnpm run tauri:build` first');

const archive = readdirSync(bundleDir).find(
  (name) => name.endsWith('.app.tar.gz') && existsSync(join(bundleDir, `${name}.sig`)),
);
if (!archive) {
  throw new Error(`No signed .app.tar.gz in ${bundleDir}; set TAURI_SIGNING_PRIVATE_KEY before building`);
}

const manifest = {
  version,
  notes: process.argv[2] ?? `Meetily RU v${version}`,
  pub_date: new Date().toISOString(),
  platforms: {
    'darwin-aarch64': {
      signature: readFileSync(join(bundleDir, `${archive}.sig`), 'utf8').trim(),
      url: `https://github.com/${REPO}/releases/download/v${version}/${encodeURIComponent(archive)}`,
    },
  },
};

const out = join(bundleDir, 'latest.json');
writeFileSync(out, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`Wrote ${out}`);
