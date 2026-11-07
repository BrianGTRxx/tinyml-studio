'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { resolveCore } = require('../src/core');
const E = require('../src/envsettings');

const base = { isPackaged: true, resourcesPath: '/r', repoRoot: '/repo', platform: 'linux', env: {} };

test('resolveCore: external interpreter wins over the packaged core, TINYML_CORE_EXE wins over both', () => {
  const ext = resolveCore({ ...base, userPython: '/home/u/venv/bin/python' });
  assert.strictEqual(ext.kind, 'external');
  assert.strictEqual(ext.command, '/home/u/venv/bin/python');
  assert.deepStrictEqual(ext.args, ['-m', 'tinyml_studio.api']);
  assert.strictEqual(resolveCore(base).kind, 'frozen');
  assert.strictEqual(resolveCore({ ...base, env: { TINYML_CORE_EXE: '/x/core' }, userPython: '/p' }).kind, 'frozen');
});

test('settings: only an absolute path is accepted; corrupt or missing files give defaults', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tms-'));
  const f = path.join(dir, 'sub', 'settings.json');
  assert.deepStrictEqual(E.loadSettings(f), { python: null });
  const abs = path.resolve('/opt/py/bin/python');
  E.saveSettings(f, { python: abs });
  assert.deepStrictEqual(E.loadSettings(f), { python: abs });
  fs.writeFileSync(f, JSON.stringify({ python: 'relative/python' }));
  assert.deepStrictEqual(E.loadSettings(f), { python: null });
  fs.writeFileSync(f, '{no json');
  assert.deepStrictEqual(E.loadSettings(f), { python: null });
});

test('doctorCommand: own interpreter runs the standalone script on itself; builtin uses cli/-m; whitelists', () => {
  const own = E.doctorCommand({ kind: 'none', command: '' }, '/py/bin/python', ['base', 'keras', 'rm -rf'], 'en', '/res/env/doctor_standalone.py');
  assert.strictEqual(own.command, '/py/bin/python');
  assert.deepStrictEqual(own.args, ['-I', '/res/env/doctor_standalone.py', '--lang', 'en', '--json', '--components', 'base,keras']);
  assert.throws(() => E.doctorCommand({ kind: 'none', command: '' }, '/py', ['base'], 'en'), /doctor_standalone/);
  const frozen = E.doctorCommand({ kind: 'frozen', command: '/c/tinyml-core' }, null, ['base'], 'es');
  assert.deepStrictEqual(frozen.args, ['cli', '--lang', 'es', 'doctor', '--json', '--components', 'base']);
  const dev = E.doctorCommand({ kind: 'dev', command: 'python3' }, null, undefined, 'xx');
  assert.deepStrictEqual(dev.args, ['-m', 'tinyml_studio.cli', 'doctor', '--json', '--components', 'base']);
});

test('runDoctor: JSON on exit code 1 is a valid report; no JSON is an error', async () => {
  const ok = (cb) => cb(Object.assign(new Error('exit 1'), { code: 1 }), '{"ok":false,"components":{}}', '');
  const rep = await E.runDoctor({ kind: 'dev', command: 'py' }, null, ['base'], 'en', { execFileFn: (c, a, o, cb) => ok(cb) });
  assert.strictEqual(rep.ok, false);
  await assert.rejects(E.runDoctor({ kind: 'none', command: '' }, '/nope', ['base'], 'en'), /doctor_standalone/);
  await assert.rejects(
    E.runDoctor({ kind: 'dev', command: 'py' }, '/nope', ['base'], 'en', { standalone: '/s.py', execFileFn: (c, a, o, cb) => cb(new Error('x'), '', 'error: no existe el intérprete') }),
    /no existe el intérprete/);
});

test('the renderer cannot set an interpreter path: no channel takes a path argument', () => {
  const pre = fs.readFileSync(path.join(__dirname, '..', 'src', 'preload.js'), 'utf8');
  assert.match(pre, /envPick: \(lang\)/);
  assert.match(pre, /envApply: \(\) =>/);
  assert.match(pre, /envReset: \(\) =>/);
  // la lista detectada se elige por ID (la ruta solo la conoce el proceso principal), nunca por ruta
  assert.match(pre, /envUseScanned: \(id, lang\)/);
  assert.doesNotMatch(pre, /env(Apply|Set|Use(?!Scanned))\w*: \(\w+/);
});

test('resolveCore: managed env beats the packaged core; raw packaged app (no core) is "none"', () => {
  const m = resolveCore({ ...base, managedPython: '/ud/python-env/env/bin/python' });
  assert.strictEqual(m.kind, 'managed');
  assert.deepStrictEqual(m.args, ['-m', 'tinyml_studio.api']);
  assert.strictEqual(resolveCore({ ...base, hasFrozen: false }).kind, 'none');
  assert.strictEqual(resolveCore({ ...base, hasFrozen: true }).kind, 'frozen');
  // user's own interpreter still wins over the managed one; dev mode never becomes "none"
  assert.strictEqual(resolveCore({ ...base, userPython: '/u/py', managedPython: '/m/py' }).kind, 'external');
  assert.strictEqual(resolveCore({ ...base, isPackaged: false, hasFrozen: false }).kind, 'dev');
});

test('findUv: env override, bundled uv when packaged, PATH in development', () => {
  const { findUv } = require('../src/installer');
  assert.strictEqual(findUv({ isPackaged: true, resourcesPath: '/r', repoRoot: '/repo', env: { TINYML_UV: '/x/uv' }, platform: 'linux' }), '/x/uv');
  assert.strictEqual(findUv({ isPackaged: true, resourcesPath: '/r', repoRoot: '/repo', env: {}, platform: 'linux' }), path.join('/r', 'uv', 'uv'));
  assert.ok(findUv({ isPackaged: true, resourcesPath: 'C:\\r', repoRoot: 'C:\\repo', env: {}, platform: 'win32' }).endsWith('uv.exe'));
  // el separador de PATH y de rutas es el del SO donde corre la prueba: la rama de PATH solo se prueba en POSIX
  if (process.platform !== 'win32') assert.strictEqual(findUv({ isPackaged: false, resourcesPath: '', repoRoot: '/repo', env: { PATH: '/a:/b' }, platform: 'linux', existsSync: (f) => f === '/b/uv' }), '/b/uv');
});
