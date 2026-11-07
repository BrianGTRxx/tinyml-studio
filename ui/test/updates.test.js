'use strict';
const test = require('node:test');
const assert = require('node:assert');
const U = require('../src/updates');

const REPO = 'BrianGTRxx/tinyml-studio';
const rel = (tag, extra = {}) => ({ tag_name: tag, html_url: `https://github.com/${REPO}/releases/tag/${tag}`, body: 'notes', draft: false, prerelease: false, ...extra });
const fakeFetch = (status, body) => async () => ({ status, ok: status >= 200 && status < 300, json: async () => body });

test('parseRepo y compareVersions', () => {
  assert.strictEqual(U.parseRepo('https://github.com/BrianGTRxx/tinyml-studio'), REPO);
  assert.strictEqual(U.parseRepo('https://example.com/a/b'), null);
  assert.strictEqual(U.compareVersions('0.1.0', '0.0.9'), 1);
  assert.strictEqual(U.compareVersions('v0.0.1', '0.0.1'), 0);
  assert.strictEqual(U.compareVersions('0.0.1', '0.10.0'), -1);
  assert.strictEqual(U.compareVersions('1.2', '1.2.0'), 0);
});

test('avisa de una versión más nueva y la elige entre varias, ignorando borradores', async () => {
  const r = await U.checkForUpdate({ repo: REPO, current: '0.0.1', fetchFn: fakeFetch(200, [rel('v0.0.2'), rel('v0.1.0', { prerelease: true }), rel('v9.9.9', { draft: true })]) });
  assert.strictEqual(r.status, 'newer');
  assert.strictEqual(r.latest.version, '0.1.0');
  assert.strictEqual(r.latest.prerelease, true);
  assert.ok(r.latest.url.startsWith(`https://github.com/${REPO}/releases`));
});

test('misma versión, sin releases, error de red y limitación de GitHub', async () => {
  assert.strictEqual((await U.checkForUpdate({ repo: REPO, current: '0.0.1', fetchFn: fakeFetch(200, [rel('v0.0.1')]) })).status, 'current');
  assert.strictEqual((await U.checkForUpdate({ repo: REPO, current: '0.0.1', fetchFn: fakeFetch(404, {}) })).status, 'none');
  assert.strictEqual((await U.checkForUpdate({ repo: REPO, current: '0.0.1', fetchFn: fakeFetch(200, []) })).status, 'none');
  assert.strictEqual((await U.checkForUpdate({ repo: REPO, current: '0.0.1', fetchFn: fakeFetch(403, {}) })).status, 'error');
  const down = await U.checkForUpdate({ repo: REPO, current: '0.0.1', fetchFn: async () => { throw new Error('offline'); } });
  assert.strictEqual(down.status, 'error');
  assert.match(down.error, /offline/);
});

test('una dirección que no es de los releases del repositorio no se usa', async () => {
  const r = await U.checkForUpdate({ repo: REPO, current: '0.0.1', fetchFn: fakeFetch(200, [rel('v0.0.2', { html_url: 'https://evil.example/x' })]) });
  assert.strictEqual(r.latest.url, `https://github.com/${REPO}/releases`);
  assert.strictEqual(U.validReleaseUrl('https://github.com/other/repo/releases', REPO), false);
});

test('preferencias: valores por defecto, guardado y cuándo toca comprobar', () => {
  const mem = {};
  const fsMod = { readFileSync: (f) => { if (!(f in mem)) throw new Error('ENOENT'); return mem[f]; }, mkdirSync() {}, writeFileSync: (f, d) => { mem[f] = d; } };
  assert.deepStrictEqual(U.loadPrefs('p.json', fsMod), { auto: null, lastCheck: 0, skipped: '' });
  U.savePrefs('p.json', { auto: true, lastCheck: 1000, skipped: '0.2.0' }, fsMod);
  assert.deepStrictEqual(U.loadPrefs('p.json', fsMod), { auto: true, lastCheck: 1000, skipped: '0.2.0' });
  assert.strictEqual(U.dueForAutoCheck({ auto: null, lastCheck: 0 }, 1e12), false); // sin permiso no se comprueba
  assert.strictEqual(U.dueForAutoCheck({ auto: false, lastCheck: 0 }, 1e12), false);
  assert.strictEqual(U.dueForAutoCheck({ auto: true, lastCheck: 1000 }, 1000 + U.DAY_MS - 1), false);
  assert.strictEqual(U.dueForAutoCheck({ auto: true, lastCheck: 1000 }, 1000 + U.DAY_MS), true);
});
