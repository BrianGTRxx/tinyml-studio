'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

// security.js requiere 'electron' solo para app.isPackaged; en node puro devuelve una ruta.
const security = require('../src/security');

const RENDERER = path.join(__dirname, '..', 'src', 'renderer');

test('webPreferences endurecidas', () => {
  const p = security.buildWebPreferences('/x/preload.js');
  assert.strictEqual(p.contextIsolation, true);
  assert.strictEqual(p.nodeIntegration, false);
  assert.strictEqual(p.sandbox, true);
  assert.strictEqual(p.webSecurity, true);
  assert.strictEqual(p.webviewTag, false);
  assert.strictEqual(p.allowRunningInsecureContent, false);
});

test('solo se acepta la URL de la propia app', () => {
  const ok = require('node:url').pathToFileURL(path.join(RENDERER, 'index.html')).href;
  assert.ok(security.isAppUrl(ok, RENDERER));
  assert.ok(!security.isAppUrl('https://example.com', RENDERER));
  assert.ok(!security.isAppUrl('file:///etc/passwd', RENDERER));
  assert.ok(!security.isAppUrl(ok.replace('renderer/', 'renderer-evil/'), RENDERER));
  assert.ok(!security.isAppUrl(undefined, RENDERER));
});

test('IPC: remitente no confiable rechazado', () => {
  const good = require('node:url').pathToFileURL(path.join(RENDERER, 'index.html')).href;
  assert.ok(security.isTrustedSender({ senderFrame: { url: good } }, RENDERER));
  assert.ok(!security.isTrustedSender({ senderFrame: { url: 'https://x.com' } }, RENDERER));
  assert.ok(!security.isTrustedSender({}, RENDERER));
});

test('el <meta> CSP del HTML coincide con security.CSP y no hay scripts inline', () => {
  const html = fs.readFileSync(path.join(RENDERER, 'index.html'), 'utf8');
  assert.ok(html.includes(`content="${security.CSP}"`));
  assert.ok(!/<script(?![^>]*\bsrc=)/i.test(html), 'script inline');
  assert.ok(!/\son\w+=/i.test(html), 'manejador inline');
  assert.ok(!/https?:\/\//i.test(html.replace(/<meta[^>]*>/g, '')), 'recurso remoto');
});

test('ningún script del renderer usa innerHTML/eval ni pide recursos remotos', () => {
  const files = fs.readdirSync(RENDERER).filter((f) => f.endsWith('.js'));
  assert.ok(files.length >= 4);
  for (const f of files) {
    const js = fs.readFileSync(path.join(RENDERER, f), 'utf8');
    assert.ok(!/innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval\(|new Function/.test(js), f);
    assert.ok(!/https?:\/\//.test(js.replace(/http:\/\/www\.w3\.org\/2000\/svg/g, '')), `${f}: URL remota`);
    assert.ok(!/setAttribute\(\s*['"]style/.test(js), `${f}: estilo inline por atributo`);
  }
});

test('todos los scripts del HTML existen y se cargan desde el propio directorio', () => {
  const html = fs.readFileSync(path.join(RENDERER, 'index.html'), 'utf8');
  const srcs = [...html.matchAll(/<script src="([^"]+)"/g)].map((m) => m[1]);
  assert.deepStrictEqual(srcs, ['theme.js', 'util.js', 'icons.js', 'components.js', 'charts.js', 'export.js', 'quality.js', 'graph.js', 'arduino.js', 'benchmark.js', 'groups.js', 'tune.js', 'modelinfo.js', 'monitor.js', 'history.js', 'setup.js', 'env.js', 'updates.js', 'report.js', 'tour.js', 'wizard.js', 'renderer.js']);
  for (const s of srcs) assert.ok(fs.existsSync(path.join(RENDERER, s)), s);
});

test('el preload no expone ipcRenderer ni el token', () => {
  const js = fs.readFileSync(path.join(__dirname, '..', 'src', 'preload.js'), 'utf8');
  assert.ok(!/exposeInMainWorld\([^)]*ipcRenderer\s*[,}]/.test(js));
  assert.ok(!/token/i.test(js));
});
