'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { resolveCore, coreExeName } = require('../src/core');
const { decideSandbox } = require('../src/linux-sandbox');

const REPO = '/repo';

test('desarrollo: python -m tinyml_studio.api desde la raíz del repo', () => {
  const c = resolveCore({ isPackaged: false, repoRoot: REPO, platform: 'linux', env: {} });
  assert.deepStrictEqual(c, { kind: 'dev', command: 'python3', args: ['-m', 'tinyml_studio.api'], cwd: REPO });
  const w = resolveCore({ isPackaged: false, repoRoot: REPO, platform: 'win32', env: {} });
  assert.strictEqual(w.command, 'python');
});

test('desarrollo: TINYML_PYTHON se respeta', () => {
  const c = resolveCore({ isPackaged: false, repoRoot: REPO, platform: 'linux', env: { TINYML_PYTHON: '/v/bin/python' } });
  assert.strictEqual(c.command, '/v/bin/python');
  assert.deepStrictEqual(c.args, ['-m', 'tinyml_studio.api']);
});

test('empaquetado Linux: resources/core/tinyml-core sin argumentos, cwd = su carpeta', () => {
  const c = resolveCore({ isPackaged: true, resourcesPath: '/opt/TinyML Studio/resources', repoRoot: REPO, platform: 'linux', env: { TINYML_PYTHON: '/ignorado' } });
  assert.deepStrictEqual(c, {
    kind: 'frozen',
    command: '/opt/TinyML Studio/resources/core/tinyml-core',
    args: [],
    cwd: '/opt/TinyML Studio/resources/core',
  });
});

test('empaquetado Windows: sufijo .exe y separadores de Windows', () => {
  const c = resolveCore({ isPackaged: true, resourcesPath: 'C:\\Users\\a\\AppData\\Local\\Programs\\TinyML Studio\\resources', repoRoot: REPO, platform: 'win32', env: {} });
  assert.strictEqual(c.command, 'C:\\Users\\a\\AppData\\Local\\Programs\\TinyML Studio\\resources\\core\\tinyml-core.exe');
  assert.strictEqual(c.cwd, 'C:\\Users\\a\\AppData\\Local\\Programs\\TinyML Studio\\resources\\core');
  assert.deepStrictEqual(c.args, []);
  assert.strictEqual(coreExeName('win32'), 'tinyml-core.exe');
  assert.strictEqual(coreExeName('linux'), 'tinyml-core');
});

test('TINYML_CORE_EXE manda sobre todo, también sin empaquetar', () => {
  const c = resolveCore({ isPackaged: false, repoRoot: REPO, platform: 'linux', env: { TINYML_CORE_EXE: '/x/y/tinyml-core', TINYML_PYTHON: 'python' } });
  assert.deepStrictEqual(c, { kind: 'frozen', command: '/x/y/tinyml-core', args: [], cwd: '/x/y' });
  const w = resolveCore({ isPackaged: false, repoRoot: REPO, platform: 'win32', env: { TINYML_CORE_EXE: 'D:\\core\\tinyml-core.exe' } });
  assert.strictEqual(w.cwd, 'D:\\core');
});

test('empaquetado sin resourcesPath es un error claro', () => {
  assert.throws(() => resolveCore({ isPackaged: true, repoRoot: REPO, platform: 'linux', env: {} }), /resourcesPath/);
});

// ---- política de sandbox en Linux empaquetado -------------------------------------------------
const fakeFs = ({ files = {}, stat = null } = {}) => ({
  readFileSync: (f) => {
    if (f in files) return files[f];
    throw new Error('ENOENT');
  },
  statSync: () => {
    if (!stat) throw new Error('ENOENT');
    return stat;
  },
});
const base = { platform: 'linux', isPackaged: true, execPath: '/opt/app/tinyml-studio', env: {}, argv: [] };

test('sandbox: nunca se desactiva en Windows/macOS ni en desarrollo', () => {
  const blocked = fakeFs({ files: { '/proc/sys/user/max_user_namespaces': '0' } });
  assert.strictEqual(decideSandbox({ ...base, platform: 'win32', fsMod: blocked }).disable, false);
  assert.strictEqual(decideSandbox({ ...base, platform: 'darwin', fsMod: blocked }).disable, false);
  assert.strictEqual(decideSandbox({ ...base, isPackaged: false, fsMod: blocked }).disable, false);
});

test('sandbox: Linux empaquetado sin SUID ni userns -> respaldo --no-sandbox', () => {
  for (const [file, val] of [
    ['/proc/sys/user/max_user_namespaces', '0'],
    ['/proc/sys/kernel/unprivileged_userns_clone', '0'],
    ['/proc/sys/kernel/apparmor_restrict_unprivileged_userns', '1'],
  ]) {
    const d = decideSandbox({ ...base, fsMod: fakeFs({ files: { [file]: val } }) });
    assert.strictEqual(d.disable, true, file);
  }
});

test('sandbox: se conserva si hay userns, o chrome-sandbox con setuid root', () => {
  assert.strictEqual(decideSandbox({ ...base, fsMod: fakeFs({ files: { '/proc/sys/user/max_user_namespaces': '63000' } }) }).disable, false);
  const suid = fakeFs({ files: { '/proc/sys/user/max_user_namespaces': '0' }, stat: { uid: 0, mode: 0o104755 } });
  assert.strictEqual(decideSandbox({ ...base, fsMod: suid }).disable, false);
  const notRoot = fakeFs({ files: { '/proc/sys/user/max_user_namespaces': '0' }, stat: { uid: 1000, mode: 0o104755 } });
  assert.strictEqual(decideSandbox({ ...base, fsMod: notRoot }).disable, true);
});

test('sandbox: TINYML_FORCE_SANDBOX=1 impide el respaldo', () => {
  const blocked = fakeFs({ files: { '/proc/sys/user/max_user_namespaces': '0' } });
  assert.strictEqual(decideSandbox({ ...base, env: { TINYML_FORCE_SANDBOX: '1' }, fsMod: blocked }).disable, false);
});
