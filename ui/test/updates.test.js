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

// ---- descarga verificada del instalador ----------------------------------------------------------------------
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const TAG = 'v0.0.9';
const EXE = 'TinyML-Studio-raw-Setup-0.0.9.exe';
const dl = (name) => `https://github.com/${REPO}/releases/download/${TAG}/${name}`;
const assets = (size = 10) => [
  { name: EXE, size, browser_download_url: dl(EXE) },
  { name: 'SHA256SUMS.txt', size: 100, browser_download_url: dl('SHA256SUMS.txt') },
  { name: 'TinyML-Studio-raw-0.0.9-linux-x86_64.AppImage', size: 5, browser_download_url: dl('TinyML-Studio-raw-0.0.9-linux-x86_64.AppImage') },
];

test('pickInstaller solo acepta el .exe y las huellas del mismo repositorio y etiqueta', () => {
  const ok = U.pickInstaller(assets(), REPO, TAG);
  assert.strictEqual(ok.name, EXE);
  assert.strictEqual(ok.url, dl(EXE));
  assert.strictEqual(ok.sumsUrl, dl('SHA256SUMS.txt'));
  assert.strictEqual(ok.size, 10);
  // dirección de otro sitio, de otro repositorio o de otra etiqueta: se ignora
  const evil = assets();
  evil[0].browser_download_url = `https://evil.example/${EXE}`;
  assert.strictEqual(U.pickInstaller(evil, REPO, TAG), null);
  const other = assets();
  other[1].browser_download_url = `https://github.com/other/repo/releases/download/${TAG}/SHA256SUMS.txt`;
  assert.strictEqual(U.pickInstaller(other, REPO, TAG), null);
  assert.strictEqual(U.pickInstaller(assets(), REPO, 'v1.0.0'), null);
  // sin huellas, sin .exe, tamaño absurdo o datos que no son una lista
  assert.strictEqual(U.pickInstaller(assets().slice(0, 1), REPO, TAG), null);
  assert.strictEqual(U.pickInstaller(assets().slice(1), REPO, TAG), null);
  assert.strictEqual(U.pickInstaller(assets(U.MAX_INSTALLER_BYTES + 1), REPO, TAG), null);
  assert.strictEqual(U.pickInstaller(null, REPO, TAG), null);
  assert.strictEqual(U.pickInstaller([{ name: 'x.exe', browser_download_url: 5 }], REPO, TAG), null);
});

test('checkForUpdate incluye el instalador solo si el release lo trae completo', async () => {
  const r = await U.checkForUpdate({ repo: REPO, current: '0.0.1', fetchFn: fakeFetch(200, [rel(TAG, { assets: assets() })]) });
  assert.strictEqual(r.latest.installer.name, EXE);
  const none = await U.checkForUpdate({ repo: REPO, current: '0.0.1', fetchFn: fakeFetch(200, [rel(TAG)]) });
  assert.strictEqual(none.latest.installer, null);
});

test('parseSums lee "huella  archivo" y no confunde nombres', () => {
  const h1 = 'a'.repeat(64);
  const h2 = 'B'.repeat(64);
  const txt = `${h1}  ${EXE}\r\n${h2} *other.deb\n`;
  assert.strictEqual(U.parseSums(txt, EXE), h1);
  assert.strictEqual(U.parseSums(txt, 'other.deb'), 'b'.repeat(64));
  assert.strictEqual(U.parseSums(txt, 'missing.exe'), null);
  assert.strictEqual(U.parseSums('basura', EXE), null);
});

function netFetch(payload, sumsHash, opts = {}) {
  const calls = [];
  const fn = async (url) => {
    calls.push(url);
    if (url === dl('SHA256SUMS.txt')) return { ok: true, status: 200, text: async () => `${sumsHash}  ${EXE}\n` };
    const body = new ReadableStream({
      start(c) {
        c.enqueue(payload.subarray(0, 4));
        c.enqueue(payload.subarray(4));
        c.close();
      },
    });
    return { ok: opts.status ? false : true, status: opts.status || 200, body };
  };
  fn.calls = calls;
  return fn;
}

test('downloadVerified guarda el instalador cuando la huella coincide y avisa del avance', async () => {
  const payload = Buffer.from('0123456789');
  const good = crypto.createHash('sha256').update(payload).digest('hex');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tm-upd-'));
  const installer = U.pickInstaller(assets(payload.length), REPO, TAG);
  const seen = [];
  const f = await U.downloadVerified({ fetchFn: netFetch(payload, good), installer, destDir: path.join(dir, 'x'), onProgress: (p) => seen.push(p.done) });
  assert.strictEqual(fs.readFileSync(f).toString(), '0123456789');
  assert.deepStrictEqual(seen, [4, 10]);
});

test('downloadVerified rechaza y borra si la huella no coincide, el tamaño no es el anunciado o GitHub falla', async () => {
  const payload = Buffer.from('0123456789');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tm-upd-'));
  const dest = path.join(dir, 'x');
  const installer = U.pickInstaller(assets(payload.length), REPO, TAG);
  await assert.rejects(U.downloadVerified({ fetchFn: netFetch(payload, 'f'.repeat(64)), installer, destDir: dest }), /SHA-256/);
  assert.strictEqual(fs.existsSync(path.join(dest, EXE)), false);
  const good = crypto.createHash('sha256').update(payload).digest('hex');
  await assert.rejects(U.downloadVerified({ fetchFn: netFetch(payload, good), installer: { ...installer, size: 5 }, destDir: dest }), /bigger/);
  await assert.rejects(U.downloadVerified({ fetchFn: netFetch(payload, good), installer: { ...installer, size: 50 }, destDir: dest }), /incomplete/);
  await assert.rejects(U.downloadVerified({ fetchFn: netFetch(payload, good, { status: 500 }), installer, destDir: dest }), /500/);
  await assert.rejects(U.downloadVerified({ fetchFn: netFetch(payload, good), installer: null, destDir: dest }), /no installer/);
  assert.strictEqual(fs.existsSync(path.join(dest, EXE)), false);
});

test('isInstalledCopy distingue la copia instalada de la portable', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tm-inst-'));
  assert.strictEqual(U.isInstalledCopy(path.join(dir, 'TinyML Studio.exe'), 'TinyML Studio'), false);
  fs.writeFileSync(path.join(dir, 'Uninstall TinyML Studio.exe'), '');
  assert.strictEqual(U.isInstalledCopy(path.join(dir, 'TinyML Studio.exe'), 'TinyML Studio'), true);
});
