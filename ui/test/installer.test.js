'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { Installer, lineReader, resolveComponents, loadManifest, platformKey } = require('../src/installer');

const REPO = path.resolve(__dirname, '..', '..');
const RES = path.join(REPO, 'packaging', 'env');

function fakeUv(dir, { failOn = null, slow = false } = {}) {
  // un "uv" falso: registra los argumentos y crea el python del entorno cuando se le pide `venv`
  const log = path.join(dir, 'calls.log');
  const file = path.join(dir, process.platform === 'win32' ? 'uv.cmd' : 'uv');
  fs.writeFileSync(file, `#!/bin/sh
echo "$@" >> "${log}"
echo "salida de uv: $1 $2"
echo "aviso en stderr" >&2
${failOn ? `[ "$1 $2" = "${failOn}" ] && { echo "boom" >&2; exit 3; }` : ''}
if [ "$1" = "venv" ]; then
  for last; do :; done
  mkdir -p "$last/bin"; printf '#!/bin/sh\\n' > "$last/bin/python"; chmod +x "$last/bin/python"
fi
${slow ? 'sleep 30' : ''}
exit 0
`, { mode: 0o755 });
  return { file, log };
}
const sh = { skip: process.platform === 'win32' };

test('platformKey and manifest: every component has a lock for each platform it declares', () => {
  assert.strictEqual(platformKey('win32'), 'windows');
  assert.strictEqual(platformKey('linux'), 'linux');
  const m = loadManifest(RES);
  assert.strictEqual(m.components[0].id, 'base');
  for (const c of m.components) {
    for (const p of c.platforms) assert.ok(fs.existsSync(path.join(RES, 'locks', `${c.id}-${p}.txt`)), `${c.id}-${p}`);
  }
  // bilingual labels everywhere
  assert.ok(m.components.every((c) => c.label.en && c.label.es));
});

test('locks are hash-locked and pin every package', () => {
  for (const f of fs.readdirSync(path.join(RES, 'locks'))) {
    const text = fs.readFileSync(path.join(RES, 'locks', f), 'utf8');
    const reqs = text.split('\n').filter((l) => /^[A-Za-z0-9_.\-]+==/.test(l));
    assert.ok(reqs.length > 0, f);
    assert.ok((text.match(/--hash=sha256:/g) || []).length >= reqs.length, `${f}: falta algún hash`);
  }
});

test('resolveComponents: base first, whitelist, platform availability', () => {
  const m = loadManifest(RES);
  assert.deepStrictEqual(resolveComponents(m, ['keras'], 'linux'), ['base', 'keras']);
  assert.deepStrictEqual(resolveComponents(m, [], 'linux'), ['base']);
  assert.deepStrictEqual(resolveComponents(m, ['base', 'tui', 'tui'], 'win32'), ['base', 'tui']);
  assert.throws(() => resolveComponents(m, ['numpy==1; rm -rf /'], 'linux'), /desconocido/);
  assert.throws(() => resolveComponents(m, ['tflm'], 'win32'), /no disponible/);
  assert.throws(() => resolveComponents(m, [{ id: 'keras' }], 'linux'), /desconocido/);
});

test('lineReader splits chunks into lines (\\n and \\r\\n) and ignores blanks', () => {
  const lines = [];
  const feed = lineReader((l) => lines.push(l));
  feed(Buffer.from('uno\ndo'));
  feed(Buffer.from('s\r\n\r\ntres\n'));
  assert.deepStrictEqual(lines, ['uno', 'dos', 'tres']);
});

test('plan: venv first, locked installs with hashes and wheels only, then tinyml-studio', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tms-inst-'));
  const inst = new Installer({ root: path.join(dir, 'root'), uv: '/x/uv', resourcesDir: RES, repoRoot: REPO, platform: 'linux' });
  const steps = inst.plan(['keras']);
  assert.deepStrictEqual(steps.map((s) => s.label), ['Python', 'base', 'keras', 'tinyml-studio (dev)']);
  assert.deepStrictEqual(steps[0].args.slice(0, 5), ['venv', '--python', '3.12', '--python-preference', 'only-managed']);
  for (const s of steps.slice(1, 3)) {
    assert.ok(s.args.includes('--require-hashes') && s.args.includes('--only-binary'));
    assert.match(s.args[s.args.indexOf('-r') + 1], /locks[\\/](base|keras)-linux\.txt$/);
  }
  assert.throws(() => new Installer({ root: dir, uv: '/x/uv', resourcesDir: RES, platform: 'linux' }).plan([]), /rueda/);
});

test('install runs the steps, streams logs, writes state and status reflects it', sh, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tms-inst-'));
  const { file, log } = fakeUv(dir);
  const inst = new Installer({ root: path.join(dir, 'root'), uv: file, resourcesDir: RES, repoRoot: REPO, platform: 'linux' });
  assert.strictEqual(inst.status().installed, false);
  const lines = [];
  const states = [];
  inst.on('log', (l) => lines.push(l));
  inst.on('state', (s) => states.push(s.state));
  await inst.install(['tui']);
  const calls = fs.readFileSync(log, 'utf8').trim().split('\n');
  assert.strictEqual(calls.length, 4); // venv, base, tui, tinyml-studio
  assert.match(calls[0], /^venv --python 3\.12/);
  assert.ok(lines.some((l) => l.stream === 'err' && /aviso en stderr/.test(l.line)));
  assert.deepStrictEqual(states, ['running', 'done']);
  const st = inst.status();
  assert.strictEqual(st.installed, true);
  assert.deepStrictEqual(st.components.filter((c) => c.installed).map((c) => c.id), ['base', 'tui']);
  assert.strictEqual(st.components.find((c) => c.id === 'keras').installed, false);
  // reinstalling skips `venv` (the env exists) and a failure is reported with the tail of uv's output
  fs.writeFileSync(log, '');
  const bad = fakeUv(dir, { failOn: 'pip install' });
  const inst2 = new Installer({ root: path.join(dir, 'root'), uv: bad.file, resourcesDir: RES, repoRoot: REPO, platform: 'linux' });
  await assert.rejects(inst2.install(['keras']), /uv pip install falló \(código 3\).*boom/);
  assert.ok(!/^venv/m.test(fs.readFileSync(bad.log, 'utf8')));
  // reset removes everything under root only
  inst.reset();
  assert.strictEqual(inst.status().installed, false);
  assert.ok(fs.existsSync(dir));
});

test('cancel stops the running step', sh, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tms-inst-'));
  const { file } = fakeUv(dir, { slow: true });
  const inst = new Installer({ root: path.join(dir, 'root'), uv: file, resourcesDir: RES, repoRoot: REPO, platform: 'linux' });
  const p = inst.install([]);
  p.catch(() => {});
  await new Promise((r) => setTimeout(r, 400));
  assert.strictEqual(inst.cancel(), true);
  await assert.rejects(p, /cancelado/);
  assert.strictEqual(inst.status().busy, false);
});

test('install refuses a missing uv and concurrent runs', sh, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tms-inst-'));
  const none = new Installer({ root: dir, uv: path.join(dir, 'no-uv'), resourcesDir: RES, repoRoot: REPO, platform: 'linux' });
  await assert.rejects(none.install([]), /no se encuentra uv/);
  const { file } = fakeUv(dir, { slow: true });
  const inst = new Installer({ root: path.join(dir, 'r2'), uv: file, resourcesDir: RES, repoRoot: REPO, platform: 'linux' });
  const p = inst.install([]);
  p.catch(() => {});
  await new Promise((r) => setTimeout(r, 300));
  await assert.rejects(inst.install([]), /en curso/);
  inst.cancel();
  await p.catch(() => {});
});

test('planExternal: wheel without deps, then only the missing requirements; rejects anything else', () => {
  const inst = new Installer({ root: '/tmp/x', uv: '/tmp/uv', resourcesDir: RES, repoRoot: REPO, platform: 'linux' });
  const py = '/home/u/env/bin/python';
  const steps = inst.planExternal({ python: py, wheel: '/app/env/wheels/tinyml_studio-0.0.1-py3-none-any.whl', requirements: ['scipy>=1.10,<2', 'fastapi>=0.100', 'scipy>=1.10,<2'] });
  assert.strictEqual(steps.length, 2);
  assert.deepStrictEqual(steps[0].args, ['pip', 'install', '--python', py, '--no-deps', '/app/env/wheels/tinyml_studio-0.0.1-py3-none-any.whl']);
  assert.deepStrictEqual(steps[1].args, ['pip', 'install', '--python', py, 'scipy>=1.10,<2', 'fastapi>=0.100']);
  assert.deepStrictEqual(inst.planExternal({ python: py }), []);
  for (const bad of ['--index-url=http://evil', 'git+https://x/y.git', 'a b', '../x', 'pkg @ http://x', '-r reqs.txt', '']) {
    assert.throws(() => inst.planExternal({ python: py, requirements: [bad] }), /requisito no válido/, bad);
  }
  assert.throws(() => inst.planExternal({ python: 'relative/python' }), /ruta de Python/);
  assert.throws(() => inst.planExternal({ python: py, wheel: '/tmp/evil.whl' }), /rueda no válida/);
});

test('installExternal runs uv against the user python, tags events as "own" and writes no state', sh, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tms-ext-'));
  const { file, log } = fakeUv(dir);
  const inst = new Installer({ root: path.join(dir, 'root'), uv: file, resourcesDir: RES, repoRoot: REPO, platform: 'linux' });
  const events = [];
  for (const k of ['log', 'step', 'state']) inst.on(k, (e) => events.push({ k, ...e }));
  await inst.installExternal({ python: '/home/u/env/bin/python', wheel: '/app/tinyml_studio-0.0.1-py3-none-any.whl', requirements: ['scipy>=1.10,<2'] });
  const calls = fs.readFileSync(log, 'utf8').trim().split('\n');
  assert.strictEqual(calls.length, 2);
  assert.match(calls[0], /^pip install --python \/home\/u\/env\/bin\/python --no-deps /);
  assert.ok(events.length > 0 && events.every((e) => e.scope === 'own'));
  assert.deepStrictEqual(events.filter((e) => e.k === 'state').map((e) => e.state), ['running', 'done']);
  assert.strictEqual(inst.scope, 'managed');
  assert.strictEqual(fs.existsSync(inst.stateFile), false);
});

test('refreshWheel: reinstalls only the bundled wheel when its hash changed after an app update', async () => {
  const { EventEmitter } = require('node:events');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tms-wheel-'));
  const res = path.join(dir, 'res');
  fs.mkdirSync(path.join(res, 'wheels'), { recursive: true });
  fs.writeFileSync(path.join(res, 'manifest.json'), JSON.stringify({ python: '3.12', components: [] }));
  const whl = path.join(res, 'wheels', 'tinyml_studio-0.0.1-py3-none-any.whl');
  fs.writeFileSync(whl, 'version A');
  const uv = path.join(dir, 'uv');
  fs.writeFileSync(uv, '');
  const calls = [];
  const spawnFn = (cmd, args) => {
    calls.push(args);
    const p = new EventEmitter();
    p.stdout = new EventEmitter();
    p.stderr = new EventEmitter();
    setImmediate(() => p.emit('close', 0));
    return p;
  };
  const inst = new Installer({ root: path.join(dir, 'root'), uv, resourcesDir: res, spawnFn });
  fs.mkdirSync(path.dirname(inst.python), { recursive: true });
  fs.writeFileSync(inst.python, '');
  fs.writeFileSync(inst.stateFile, JSON.stringify({ components: { base: { time: 'x' } } })); // instalado sin hash: entorno antiguo
  assert.deepStrictEqual(await inst.refreshWheel(), { refreshed: true });
  assert.deepStrictEqual(calls[0].slice(0, 2), ['pip', 'install']);
  assert.ok(calls[0].includes('--no-deps') && calls[0].includes('--reinstall') && calls[0].includes(whl));
  assert.deepStrictEqual(await inst.refreshWheel(), { refreshed: false, reason: 'current' }); // mismo hash: nada
  assert.strictEqual(calls.length, 1);
  fs.writeFileSync(whl, 'version B'); // la app se actualizó con una rueda nueva
  assert.deepStrictEqual(await inst.refreshWheel(), { refreshed: true });
  assert.strictEqual(calls.length, 2);
  // sin entorno gestionado no hace nada
  fs.rmSync(inst.stateFile);
  assert.strictEqual((await inst.refreshWheel()).refreshed, false);
});
