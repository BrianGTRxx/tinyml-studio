'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const bridge = require('../src/bridge');

const ID = 'abcdef012345';

test('lista blanca de rutas: lo permitido', () => {
  const ok = [
    ['GET', '/health'], ['GET', '/hardware'], ['GET', '/boards'], ['GET', '/models'],
    ['POST', '/inspect'], ['POST', '/suggest-cleaning'], ['GET', '/jobs'], ['POST', '/jobs'],
    ['GET', `/jobs/${ID}`], ['DELETE', `/jobs/${ID}`], ['GET', `/jobs/${ID}/result`],
    ['GET', `/jobs/${ID}/files`], ['GET', `/jobs/${ID}/files/results.json`],
    ['GET', `/jobs/${ID}/files/sub/tm_model.c`], ['POST', `/jobs/${ID}/cancel`],
    ['GET', '/hardware/live'], ['GET', `/jobs/${ID}/hw`],
  ];
  for (const [m, r] of ok) assert.ok(bridge.routeAllowed(m, r), `${m} ${r}`);
});

test('lista blanca de rutas: lo prohibido', () => {
  const bad = [
    ['POST', '/roots'], ['GET', '/roots'], ['POST', '/profile'], ['GET', '/'], ['GET', '/docs'],
    ['PUT', '/jobs'], ['DELETE', '/jobs'], ['GET', '/jobs/..%2F..'], ['GET', '/jobs/../../x'],
    ['GET', `/jobs/${ID}/files/../job.json`], ['GET', `/jobs/${ID}/files/a%2e%2e/b`],
    ['GET', `/jobs/${ID}/events`], ['GET', '/jobs/XYZ'], ['GET', '/jobs/abcdef01234'],
    ['GET', '/jobs?x=1'], ['GET', `/jobs/${ID}#x`], ['POST', `/jobs/${ID}`],
    ['GET', 'http://evil/jobs'], [null, '/jobs'], ['GET', null],
    ['POST', '/hardware/live'], ['GET', '/hardware/live/x'], ['POST', `/jobs/${ID}/hw`],
    ['GET', '/jobs/abc/hw'], ['GET', `/jobs/${ID}/hw/x`],
  ];
  for (const [m, r] of bad) assert.ok(!bridge.routeAllowed(m, r), `${m} ${r}`);
});

test('lista blanca: historial, comparación, transferencia de receta y params', () => {
  const ok = [
    ['GET', '/history'], ['GET', '/history?lang=en'], ['POST', '/history/compare'],
    ['POST', '/history/compare?lang=es'], ['POST', '/recipe/transfer'],
    ['GET', `/jobs/${ID}/params`], ['GET', `/jobs/${ID}/params?lang=en`],
  ];
  for (const [m, r] of ok) assert.ok(bridge.routeAllowed(m, r), `${m} ${r}`);
  const bad = [
    ['POST', '/history'], ['DELETE', '/history'], ['GET', '/history/compare'],
    ['GET', '/history/x'], ['GET', '/history?limit=1'], ['PUT', '/recipe/transfer'],
    ['GET', '/recipe/transfer'], ['GET', '/recipe'], ['POST', `/jobs/${ID}/params`],
    ['GET', '/jobs/abc/params'], ['GET', `/jobs/${ID}/params/x`], ['GET', `/jobs/${ID}/params%2f..`],
    ['GET', '/history/../roots'],
  ];
  for (const [m, r] of bad) assert.ok(!bridge.routeAllowed(m, r), `${m} ${r}`);
});

test('cuerpo: límite de tamaño y objetos no serializables', () => {
  assert.ok(bridge.bodyOk({ a: 1 }));
  assert.ok(bridge.bodyOk(undefined));
  assert.ok(!bridge.bodyOk({ big: 'x'.repeat(bridge.MAX_BODY_BYTES + 1) }));
  const cyc = {};
  cyc.me = cyc;
  assert.ok(!bridge.bodyOk(cyc));
});

test('copyJobArtifacts: copia la salida del trabajo y rechaza ids raros', () => {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'tms-'));
  const dest = fs.mkdtempSync(path.join(os.tmpdir(), 'tmd-'));
  const out = path.join(work, 'jobs', ID, 'out');
  fs.mkdirSync(path.join(out, 'sub'), { recursive: true });
  fs.writeFileSync(path.join(out, 'a.txt'), 'hola');
  fs.writeFileSync(path.join(out, 'sub', 'b.bin'), Buffer.from([1, 2, 3]));
  fs.writeFileSync(path.join(work, 'jobs', ID, 'job.json'), JSON.stringify({ kind: 'export_c' }));
  fs.writeFileSync(path.join(work, 'secreto.txt'), 'no copiar');
  const r = bridge.copyJobArtifacts(work, ID, dest);
  assert.strictEqual(r.files, 2);
  assert.strictEqual(path.basename(r.dest), `tinyml-export_c-${ID}`);
  assert.strictEqual(fs.readFileSync(path.join(r.dest, 'a.txt'), 'utf8'), 'hola');
  assert.ok(!fs.existsSync(path.join(r.dest, '..', 'secreto.txt')));
  for (const bad of ['../..', `${ID}/../..`, 'ABCDEF012345', '', null, 'abcdef01234g']) {
    assert.throws(() => bridge.copyJobArtifacts(work, bad, dest), /inválido/);
  }
  assert.throws(() => bridge.copyJobArtifacts(work, '0123456789ab', dest), /sin archivos|no tiene/);
});

test('decodeImage accepts real images and rejects spoofed or active content', () => {
  const { decodeImage } = require('../src/bridge');
  const png = 'data:image/png;base64,' + Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3]).toString('base64');
  assert.strictEqual(decodeImage('png', png).ext, 'png');
  const jpg = 'data:image/jpeg;base64,' + Buffer.from([0xff, 0xd8, 0xff, 0xe0, 1]).toString('base64');
  assert.strictEqual(decodeImage('jpg', jpg).ext, 'jpg');
  assert.strictEqual(decodeImage('jpeg', jpg).ext, 'jpeg');
  assert.throws(() => decodeImage('png', jpg));
  assert.throws(() => decodeImage('exe', png));
  assert.throws(() => decodeImage('png', 'data:image/png;base64,AAAA'));
  assert.strictEqual(decodeImage('svg', '<svg xmlns="http://www.w3.org/2000/svg"></svg>').ext, 'svg');
  assert.throws(() => decodeImage('svg', '<svg><script>alert(1)</script></svg>'));
  assert.throws(() => decodeImage('svg', '<html></html>'));
});

test('hardware framework probe is whitelisted and nothing similar is', () => {
  const { routeAllowed } = require('../src/bridge');
  assert.ok(routeAllowed('GET', '/hardware/frameworks'));
  assert.ok(!routeAllowed('POST', '/hardware/frameworks'));
  assert.ok(!routeAllowed('GET', '/hardware/frameworks/x'));
});
