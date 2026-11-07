'use strict';
const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const { scanPythons, verdictFor, candidatePaths } = require('../src/pyscan');

test('veredicto por versión', () => {
  assert.strictEqual(verdictFor([3, 9, 18]), 'too_old');
  assert.strictEqual(verdictFor([3, 10, 4]), 'ok');
  assert.strictEqual(verdictFor([3, 11, 6]), 'recommended');
  assert.strictEqual(verdictFor([3, 12, 1]), 'recommended');
  assert.strictEqual(verdictFor([3, 14, 0]), 'newer');
  assert.strictEqual(verdictFor([2, 7, 18]), 'too_old');
});

function fakeFs(files) {
  const set = new Set(files.map((f) => path.normalize(f)));
  return {
    statSync: (p) => { if (!set.has(path.normalize(p))) throw new Error('ENOENT'); return { isFile: () => true }; },
    readdirSync: () => { throw new Error('ENOENT'); },
    realpathSync: (p) => p,
  };
}

test('encuentra candidatos en el PATH y no repite', () => {
  const a = path.join('/x', 'bin', 'python3');
  const c = candidatePaths({ platform: 'linux', env: { PATH: `${path.join('/x', 'bin')}:${path.join('/x', 'bin')}` }, home: '/nohome', fsMod: fakeFs([a]) });
  assert.deepStrictEqual(c, [a]);
});

test('descarta lo que no es un Python real y ordena mejor primero', async () => {
  const p312 = path.join('/py312', 'bin', 'python3');
  const p39 = path.join('/py39', 'bin', 'python3');
  const stub = path.join('/stub', 'bin', 'python3');
  const reply = {
    [p312]: { v: [3, 12, 1], exe: p312, venv: false, conda: false },
    [p39]: { v: [3, 9, 7], exe: p39, venv: false, conda: false },
  };
  const execFileFn = (exe, _args, _o, cb) => (reply[exe] ? cb(null, `${JSON.stringify(reply[exe])}\n`) : cb(new Error('Microsoft Store')));
  const list = await scanPythons({
    platform: 'linux',
    env: { PATH: [p39, p312, stub].map((p) => path.dirname(p)).join(':') },
    home: '/nohome',
    fsMod: fakeFs([p312, p39, stub]),
    execFileFn,
  });
  assert.deepStrictEqual(list.map((x) => [x.version, x.verdict]), [['3.12.1', 'recommended'], ['3.9.7', 'too_old']]);
  assert.ok(list.every((x) => !('_v' in x)));
});

test('un intérprete con salida corrupta se ignora', async () => {
  const p = path.join('/bad', 'python3');
  const list = await scanPythons({ platform: 'linux', env: { PATH: path.dirname(p) }, home: '/h', fsMod: fakeFs([p]), execFileFn: (_e, _a, _o, cb) => cb(null, 'no json') });
  assert.deepStrictEqual(list, []);
});
