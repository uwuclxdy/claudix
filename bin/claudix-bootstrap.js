#!/usr/bin/env node
// claudix bootstrap: resolve, cache, verify, and download the native binary, then
// exec it with the caller's args. Replaces bin/claudix (bash shim) +
// scripts/ensure-binary.sh + the scripts/*.sh hook wrappers, so the plugin needs no
// bash at runtime (node is present wherever Claude Code runs).
//
// Invoked as: node claudix-bootstrap.js <mcp|hook <Event>|--install|--check-only> [forward args...]
//
// INVARIANTS (breaking these breaks the plugin silently):
// - Hooks (argv[0]==='hook') ALWAYS exit 0. A missing binary, a failed download, or a
//   panicking native handler must never break the agent's session (fail-open).
// - mcp / --install / --check-only exit non-zero on failure. mcp cannot fail open: CC
//   must see it fail so search surfaces as unavailable, not as a silent no-op.
// - MCP speaks JSON-RPC over stdio. The native binary is spawned with stdio:'inherit'
//   so it owns the real FDs. The bootstrap must NEVER write to stdout in a code path
//   that reaches spawnNative (all logging -> stderr via log()), and must NEVER touch
//   process.stdin (no read/pause/resume). process.stdout.write only appears in the
//   --check-only / --install exit branches, which never spawn.
'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const https = require('https');
const http = require('http');
const { spawn } = require('child_process');

const PLUGIN_NAME = 'claudix';
const GITHUB_REPO = process.env.CLAUDIX_GITHUB_REPO || 'uwuclxdy/claudix';
const PREBUILT = new Set(['linux-x86_64', 'darwin-aarch64', 'windows-x86_64']);
const GC_KEEP = 2;
const MAX_REDIRECTS = 5;
// windows loader failure STATUS_DLL_NOT_FOUND: the MSVC-built binary can't find a
// required system DLL, almost always the VC++ 2015-2022 runtime a stock windows box
// lacks. node surfaces the code as this unsigned value or its signed int32 form.
const STATUS_DLL_NOT_FOUND = 0xc0000135;
// a freshly tagged release's binary asset isn't uploaded until release.yml finishes
// building; retry transient failures so an install in that window doesn't hard-fail.
const RELEASE_WAIT_MS = parseInt(process.env.CLAUDIX_RELEASE_WAIT_MS || '180000', 10);
const RELEASE_RETRY_MS = parseInt(process.env.CLAUDIX_RELEASE_RETRY_MS || '5000', 10);

const argv = process.argv.slice(2);
const isHook = argv[0] === 'hook';
const isWindows = process.platform === 'win32';

// --- paths (mirror ensure-binary.sh resolution exactly) ---
function resolvePluginRoot() {
  const env = process.env.CLAUDE_PLUGIN_ROOT;
  if (env && fs.existsSync(path.join(env, 'bin', 'claudix-bootstrap.js'))) return env;
  return path.resolve(__dirname, '..');
}
function resolveCacheDir() {
  return (
    process.env.CLAUDE_PLUGIN_DATA ||
    process.env.CLAUDIX_HOME ||
    path.join(process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share'), PLUGIN_NAME)
  );
}
const CACHE_DIR = resolveCacheDir();
const BIN_DIR = path.join(CACHE_DIR, 'bin');
const LOCK_DIR = path.join(CACHE_DIR, 'install.lock');
const PID_FILE = LOCK_DIR + '.pid';
const LOCAL_BIN = process.env.CLAUDIX_LOCAL_BIN || path.join(os.homedir(), '.local', 'bin');
const INSTALL_LOG = path.join(CACHE_DIR, 'install.log');

function cargoBinPath() {
  const cargoHome = process.env.CARGO_HOME || path.join(os.homedir(), '.cargo');
  return path.join(cargoHome, 'bin', PLUGIN_NAME + (isWindows ? '.exe' : ''));
}

// --- logging: stderr only; download-path lines also tee to install.log (AU-1) ---
function log(msg) {
  process.stderr.write('claudix: ' + msg + '\n');
}
async function logAppend(line) {
  try {
    await fs.promises.appendFile(INSTALL_LOG, line.endsWith('\n') ? line : line + '\n');
  } catch {
    /* install.log is best-effort; never fail the install on a log write */
  }
}

// --- version + platform ---
async function readWantedVersion(pluginRoot) {
  const manifestPath = path.join(pluginRoot, '.claude-plugin', 'plugin.json');
  let manifest;
  try {
    manifest = JSON.parse(await fs.promises.readFile(manifestPath, 'utf8'));
  } catch {
    throw new Error('plugin manifest missing or unreadable: ' + manifestPath);
  }
  if (!manifest.version) throw new Error('could not read version from ' + manifestPath);
  return manifest.version;
}
function detectPlatform() {
  if (process.env.CLAUDIX_PLATFORM_OVERRIDE) return process.env.CLAUDIX_PLATFORM_OVERRIDE;
  let osName;
  switch (process.platform) {
    case 'linux': osName = 'linux'; break;
    case 'darwin': osName = 'darwin'; break;
    case 'win32': osName = 'windows'; break;
    default: throw new Error('unsupported os: ' + process.platform);
  }
  let arch;
  switch (process.arch) {
    case 'x64': arch = 'x86_64'; break;
    case 'arm64': arch = 'aarch64'; break;
    default: throw new Error('unsupported arch: ' + process.arch);
  }
  return osName + '-' + arch;
}
function assetName(platform) {
  return PLUGIN_NAME + '-' + platform + (platform.startsWith('windows') ? '.exe' : '');
}
// derive the cached path from the asset name so the .exe suffix is always consistent
// (bash omitted .exe on the windows cached name; that never ran natively on windows so
// fixing it here is safe and correct).
function binaryPathFor(version, platform) {
  const cached = assetName(platform).replace(/^claudix-/, 'claudix-v' + version + '-');
  return path.join(BIN_DIR, cached);
}

async function canExec(p) {
  try {
    const s = await fs.promises.stat(p);
    if (!s.isFile()) return false;
    if (isWindows) return true;
    return (s.mode & 0o111) !== 0; // any execute bit
  } catch {
    return false;
  }
}
function cargoVersion(cb) {
  return new Promise((resolve) => {
    const c = spawn(cb, ['-V'], { stdio: ['ignore', 'pipe', 'ignore'] });
    let out = '';
    c.stdout.on('data', (d) => (out += d));
    c.on('error', () => resolve(null));
    c.on('exit', (code) => {
      if (code !== 0) return resolve(null);
      const parts = out.trim().split(/\s+/);
      resolve(parts.length >= 2 ? parts[1] : null);
    });
  });
}
async function installedPath(version, platform) {
  const binary = binaryPathFor(version, platform);
  if (await canExec(binary)) return binary;
  const cb = cargoBinPath();
  if (await canExec(cb)) {
    const v = await cargoVersion(cb);
    if (v === version) return cb;
  }
  return null;
}

// --- development_mode (dev-only escape hatch: use a cargo-built binary, never download) ---
function configDevValue(file) {
  let content;
  try {
    content = fs.readFileSync(file, 'utf8');
  } catch {
    return null;
  }
  const lines = content.split('\n').filter((l) => /^\s*development_mode\s*=\s*(true|false)/.test(l));
  if (lines.length === 0) return null;
  const m = lines[lines.length - 1].match(/=\s*(true|false)/);
  return m ? m[1] : null;
}
function developmentModeOn() {
  const project = process.env.CLAUDE_PROJECT_DIR || process.env.PWD || process.cwd();
  let val = configDevValue(path.join(project, '.claude', 'claudix.toml'));
  if (val === null) val = configDevValue(path.join(os.homedir(), '.claude', 'claudix.toml'));
  return val === 'true';
}
async function resolveDevBinary() {
  const cb = cargoBinPath();
  if (await canExec(cb)) return cb;
  log(
    "development_mode is on but no cargo binary at " +
      cb +
      "; run 'cargo install --path .' from the claudix repo"
  );
  return null;
}

// --- download + verify ---
async function sha256File(p) {
  const h = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(p)) h.update(chunk);
  return h.digest('hex');
}
function fetchFile(url, dest) {
  return new Promise((resolve, reject) => {
    if (url.startsWith('file://')) {
      fs.promises.copyFile(url.slice('file://'.length), dest).then(resolve, (e) => {
        fs.promises.unlink(dest).catch(() => {});
        reject(e);
      });
      return;
    }
    const proto = url.startsWith('https') ? https : http;
    let hops = 0;
    const doGet = (u) => {
      const req = proto.get(u, { timeout: 270000 }, (res) => {
        if (
          res.statusCode >= 300 &&
          res.statusCode < 400 &&
          res.headers.location &&
          hops < MAX_REDIRECTS
        ) {
          hops += 1;
          res.destroy();
          let next = res.headers.location;
          if (next.startsWith('/')) next = new URL(u).origin + next;
          doGet(next);
          return;
        }
        if (res.statusCode !== 200) {
          res.destroy();
          reject(new Error('HTTP ' + res.statusCode + ' for ' + u));
          return;
        }
        const stream = fs.createWriteStream(dest);
        res.pipe(stream);
        stream.on('finish', () => stream.close(resolve));
        stream.on('error', (e) => {
          fs.promises.unlink(dest).catch(() => {});
          reject(e);
        });
      });
      req.on('error', (e) => {
        fs.promises.unlink(dest).catch(() => {});
        reject(e);
      });
      req.on('timeout', () => req.destroy(new Error('download timed out for ' + u)));
    };
    doGet(url);
  });
}

// transient = the asset may land shortly (release still publishing, rate-limit,
// server hiccup). ENOENT from a file:// fixture or a 4xx auth error won't resolve.
function isTransientNetError(err) {
  const m = String((err && err.message) || err);
  if (/^HTTP (404|408|429|5\d\d)\b/.test(m)) return true;
  return /timed out|ECONNRESET|ENOTFOUND|ECONNREFUSED|EAI_AGAIN|ECONNABORTED|socket hang up/i.test(
    m
  );
}
async function fetchFileWithRetry(label, url, dest) {
  if (!/^https?:/i.test(url)) return fetchFile(url, dest);
  const deadline = Date.now() + RELEASE_WAIT_MS;
  for (;;) {
    try {
      await fetchFile(url, dest);
      return;
    } catch (err) {
      if (!isTransientNetError(err) || Date.now() >= deadline) throw err;
      log(label + ' not ready (' + (err.message || err) + '); retrying...');
      await new Promise((r) => setTimeout(r, RELEASE_RETRY_MS));
    }
  }
}

async function gcOldVersions(platform) {
  let entries;
  try {
    entries = await fs.promises.readdir(BIN_DIR);
  } catch {
    return;
  }
  const suffix = '-' + platform + (platform.startsWith('windows') ? '.exe' : '');
  const ours = entries.filter((f) => f.startsWith(PLUGIN_NAME + '-v') && f.endsWith(suffix));
  if (ours.length <= GC_KEEP) return;
  const statted = await Promise.all(
    ours.map(async (f) => {
      try {
        const s = await fs.promises.stat(path.join(BIN_DIR, f));
        return { f, mtime: s.mtimeMs };
      } catch {
        return null;
      }
    })
  );
  statted
    .filter(Boolean)
    .sort((a, b) => b.mtime - a.mtime)
    .slice(GC_KEEP)
    .forEach(({ f }) => fs.promises.unlink(path.join(BIN_DIR, f)).catch(() => {}));
}

// a stray cargo-installed claudix can shadow the symlink on PATH. warn but do not delete
// it: it may be an intentional dev / standalone install (the release symlink still wins
// when LOCAL_BIN precedes ~/.cargo/bin on PATH).
async function warnCargoDuplicate(releaseBinary) {
  const cb = cargoBinPath();
  if (!(await canExec(cb))) return;
  let cbReal;
  let relReal;
  try {
    cbReal = await fs.promises.realpath(cb);
    relReal = await fs.promises.realpath(releaseBinary);
  } catch {
    return;
  }
  if (cbReal === relReal) return;
  const ver = await cargoVersion(cb);
  log(
    'found cargo-installed ' +
      PLUGIN_NAME +
      ' at ' +
      cb +
      ' (v' +
      (ver || 'unknown') +
      "); left in place, remove with 'cargo uninstall " +
      PLUGIN_NAME +
      "' if undesired"
  );
}

async function exposeCommand(binary, platform) {
  if (platform.startsWith('windows')) return; // windows symlinks need privileges; PATH differs
  await fs.promises.mkdir(LOCAL_BIN, { recursive: true }).catch(() => {});
  const link = path.join(LOCAL_BIN, PLUGIN_NAME);
  try {
    await fs.promises.symlink(binary, link);
  } catch (e) {
    if (e.code !== 'EEXIST') return;
    await fs.promises.unlink(link).catch(() => {}); // ln -sf semantics: replace the target
    await fs.promises.symlink(binary, link).catch(() => {});
  }
  log('linked ' + link + ' -> ' + path.basename(binary));
  await warnCargoDuplicate(binary);
  if ((':' + (process.env.PATH || '') + ':').indexOf(':' + LOCAL_BIN + ':') === -1) {
    log("add " + LOCAL_BIN + " to PATH to run '" + PLUGIN_NAME + "' from your shell");
  }
}

async function installFromRelease(version, platform) {
  const asset = assetName(platform);
  const binaryDest = binaryPathFor(version, platform);
  const baseUrl =
    process.env.CLAUDIX_RELEASE_BASE_URL ||
    'https://github.com/' + GITHUB_REPO + '/releases/download/v' + version;

  await fs.promises.mkdir(BIN_DIR, { recursive: true });
  const tmpDir = await fs.promises.mkdtemp(path.join(CACHE_DIR, 'download.'));
  const assetPath = path.join(tmpDir, asset);
  const sumsPath = path.join(tmpDir, 'SHA256SUMS');
  try {
    log('downloading ' + asset + ' v' + version);
    await logAppend('downloading ' + asset + ' v' + version);
    await fetchFileWithRetry(asset, baseUrl + '/' + asset, assetPath);
    await fetchFileWithRetry('SHA256SUMS', baseUrl + '/SHA256SUMS', sumsPath);
    const sums = await fs.promises.readFile(sumsPath, 'utf8');
    let expected = null;
    for (const line of sums.split('\n')) {
      const parts = line.trim().split(/\s+/);
      if (parts.length === 2 && parts[1] === asset) {
        expected = parts[0];
        break;
      }
    }
    if (!expected) throw new Error('SHA256SUMS does not contain ' + asset);
    const actual = await sha256File(assetPath);
    if (actual !== expected) throw new Error('checksum mismatch for ' + asset);
    if (!isWindows) await fs.promises.chmod(assetPath, 0o755);
    await fs.promises.rename(assetPath, binaryDest); // atomic on same fs (tmpDir + BIN_DIR under CACHE_DIR)
    log('installed claudix v' + version + ' for ' + platform);
    await gcOldVersions(platform);
    await exposeCommand(binaryDest, platform);
    return binaryDest;
  } finally {
    await fs.promises.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  }
}

function runOk(cmd, args, shell) {
  return new Promise((resolve) => {
    const c = spawn(cmd, args, { stdio: 'ignore', shell: !!shell });
    c.on('error', () => resolve(false));
    c.on('exit', (code) => resolve(code === 0));
  });
}
async function tryCargo(version) {
  if (!(await runOk('cargo', ['--version'], isWindows))) return null;
  log('no prebuilt for this target; falling back to cargo install');
  await logAppend('falling back to cargo install claudix@' + version);
  if (!(await runOk('cargo', ['install', PLUGIN_NAME + '@' + version, '--quiet'], isWindows))) {
    return null;
  }
  const cb = cargoBinPath();
  if (await canExec(cb)) return cb;
  return null;
}

// --- lock (mkdir-as-lock + pidfile; node's trap EXIT doesn't cover signals) ---
function pidAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return e.code !== 'ESRCH'; // ESRCH = no such process; EPERM = exists, not ours -> treat alive
  }
}
async function acquireLock() {
  await fs.promises.mkdir(CACHE_DIR, { recursive: true });
  let attempts = 0;
  for (;;) {
    try {
      await fs.promises.mkdir(LOCK_DIR);
      break;
    } catch (e) {
      if (e.code !== 'EEXIST') throw e;
      let pid = null;
      try {
        pid = parseInt(await fs.promises.readFile(PID_FILE, 'utf8'), 10);
      } catch {
        /* no pid file */
      }
      if (pid && !pidAlive(pid)) {
        await fs.promises.rm(LOCK_DIR, { recursive: true, force: true }).catch(() => {});
        await fs.promises.rm(PID_FILE, { force: true }).catch(() => {});
        continue;
      }
      if (++attempts > 120) throw new Error('install lock stuck; remove ' + LOCK_DIR + ' and retry');
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
  await fs.promises.writeFile(PID_FILE, String(process.pid));
}
function releaseLock() {
  return Promise.all([
    fs.promises.rm(PID_FILE, { force: true }).catch(() => {}),
    fs.promises.rmdir(LOCK_DIR).catch(() => {}),
  ]);
}

async function doInstall(version, platform) {
  await acquireLock();
  try {
    const hit = await installedPath(version, platform);
    if (hit) {
      await exposeCommand(hit, platform); // relink even on cache-hit (fix 12)
      return hit;
    }
    if (PREBUILT.has(platform)) return await installFromRelease(version, platform);
    const cargoBin = await tryCargo(version);
    if (cargoBin) return cargoBin;
    throw new Error(
      'no prebuilt for ' + platform + ' and cargo install failed; install rust from https://rustup.rs and retry'
    );
  } catch (e) {
    // AU-1: record every install/download failure (network, 404, checksum, cargo) so
    // /doctor can surface it instead of leaving MCP silently dead across sessions.
    await logAppend('error: ' + ((e && e.message) || String(e)));
    throw e;
  } finally {
    await releaseLock();
  }
}

// unconditional cleanup of orphan temp dirs + dead locks (runs once per invocation,
// including --check-only, mirroring ensure-binary.sh's prologue)
async function cleanupStale() {
  let entries;
  try {
    entries = await fs.promises.readdir(CACHE_DIR);
  } catch {
    return;
  }
  for (const e of entries) {
    if (!e.startsWith('download.')) continue;
    const p = path.join(CACHE_DIR, e);
    try {
      const s = await fs.promises.stat(p);
      if (s.isDirectory()) await fs.promises.rm(p, { recursive: true, force: true });
    } catch {
      /* best-effort */
    }
  }
  if (fs.existsSync(LOCK_DIR) && fs.existsSync(PID_FILE)) {
    let pid = null;
    try {
      pid = parseInt(await fs.promises.readFile(PID_FILE, 'utf8'), 10);
    } catch {
      /* no pid */
    }
    if (pid && !pidAlive(pid)) {
      await fs.promises.rm(LOCK_DIR, { recursive: true, force: true }).catch(() => {});
      await fs.promises.rm(PID_FILE, { force: true }).catch(() => {});
    }
  }
}

// a windows loader DLL-not-found exit dies before main(), so the bootstrap is the
// only layer that can diagnose it. map it to an actionable redistributable hint.
function dllNotFoundHint(code) {
  if (code !== STATUS_DLL_NOT_FOUND && code !== STATUS_DLL_NOT_FOUND - 0x100000000) return null;
  return (
    'windows binary failed to load (0xC0000135, missing system DLL); install the ' +
    'Visual C++ 2015-2022 x64 redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe'
  );
}

// --- spawn the resolved native binary with the caller's args ---
function spawnNative(binary, forwardArgs, hook) {
  const child = spawn(binary, forwardArgs, {
    stdio: 'inherit', // MCP JSON-RPC owns the real FDs; never pipe/transform here
    env: process.env,
  });
  ['SIGINT', 'SIGTERM', 'SIGHUP'].forEach((sig) => {
    process.on(sig, () => {
      try {
        child.kill(sig);
      } catch {
        /* child may already be gone */
      }
    });
  });
  child.on('error', () => process.exit(hook ? 0 : 1));
  child.on('exit', (code) => {
    const hint = dllNotFoundHint(code);
    if (hint) {
      log(hint);
      try {
        fs.appendFileSync(INSTALL_LOG, 'error: ' + hint + '\n');
      } catch {
        /* install.log is best-effort */
      }
    }
    // hooks ALWAYS succeed (fail-open); mcp propagates the native exit code
    process.exit(hook ? 0 : code == null ? 1 : code);
  });
}

// cache miss during a session: spawn the install detached so the session keeps running,
// with stdout+stderr tee'd to install.log (replaces bash's >/dev/null so failures are
// auditable). ready next session.
function spawnInstallBackground() {
  try {
    const out = fs.openSync(INSTALL_LOG, 'a');
    const child = spawn(
      process.execPath,
      [path.join(__dirname, 'claudix-bootstrap.js'), '--install'],
      {
        detached: true,
        stdio: ['ignore', out, out],
        env: { ...process.env, CLAUDE_PLUGIN_ROOT: resolvePluginRoot() },
      }
    );
    child.unref();
  } catch (e) {
    logAppend('error: failed to spawn background install: ' + e.message).catch(() => {});
  }
}

async function main() {
  await cleanupStale();
  const pluginRoot = resolvePluginRoot();

  // development_mode bypasses release resolution entirely: cargo binary or a clear
  // failure (exit 3 for check-only/install, fail-closed for mcp, fail-open for hook).
  if (developmentModeOn()) {
    const cb = await resolveDevBinary();
    if (cb) {
      if (argv[0] === '--check-only' || argv[0] === '--install') {
        process.stdout.write(cb + '\n');
        process.exit(0);
      }
      spawnNative(cb, argv, isHook);
      return;
    }
    if (isHook) process.exit(0);
    if (argv[0] === 'mcp') process.exit(1);
    process.exit(3);
  }

  const version = await readWantedVersion(pluginRoot);
  const platform = detectPlatform();

  switch (argv[0]) {
    case '--check-only': {
      const hit = await installedPath(version, platform);
      if (hit) {
        process.stdout.write(hit + '\n');
        process.exit(0);
      }
      process.exit(1);
    }
    case '--install': {
      const hit = await installedPath(version, platform);
      if (hit) {
        await exposeCommand(hit, platform);
        process.stdout.write(hit + '\n');
        process.exit(0);
      }
      const installed = await doInstall(version, platform);
      process.stdout.write(installed + '\n');
      process.exit(0);
    }
    case 'mcp':
    case 'hook':
    default: {
      const hit = await installedPath(version, platform);
      if (hit) {
        spawnNative(hit, argv, isHook);
        return;
      }
      spawnInstallBackground();
      if (argv[0] === 'mcp') {
        process.stderr.write(
          'claudix: native binary not yet cached — downloading in background, restart Claude Code once it completes\n'
        );
        process.exit(1); // mcp fails closed
      }
      process.stderr.write(
        'claudix: binary installing in background; hooks activate on next session start\n'
      );
      process.exit(0); // hooks fail open
    }
  }
}

// top-level fail-open: any escaped error exits 0 for hooks, 1 otherwise.
function failTop(err) {
  try {
    process.stderr.write('claudix: ' + ((err && err.stack) || err) + '\n');
  } catch {
    /* ignore */
  }
  process.exit(isHook ? 0 : 1);
}
// requiring this file (tests) must not run main(); expose the pure helper instead.
if (require.main === module) {
  process.on('uncaughtException', failTop);
  process.on('unhandledRejection', failTop);
  main().catch(failTop);
} else {
  module.exports = { dllNotFoundHint };
}
