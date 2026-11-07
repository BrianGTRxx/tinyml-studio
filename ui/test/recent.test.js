'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const recent = require('../src/recent');

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'recent-'));

test('guarda el último primero, sin duplicados y con tope', () => {
  let list = [];
  for (const n of ['a', 'b', 'c', 'd', 'e', 'f']) list = recent.add(list, [`/x/${n}.csv`]);
  assert.strictEqual(list.length, recent.MAX);
  assert.deepStrictEqual(list[0], ['/x/f.csv']);
  list = recent.add(list, ['/x/d.csv']);
  assert.deepStrictEqual(list[0], ['/x/d.csv']);
  assert.strictEqual(list.filter((e) => e[0] === '/x/d.csv').length, 1);
});

test('el archivo guardado se lee y descarta basura', () => {
  const f = path.join(tmp(), 'recent.json');
  recent.save(f, [['/a/b.csv', '/a/c.csv']]);
  assert.deepStrictEqual(recent.load(f), [['/a/b.csv', '/a/c.csv']]);
  fs.writeFileSync(f, JSON.stringify([['/ok.csv'], 'raro', [1, 2], []]));
  assert.deepStrictEqual(recent.load(f), [['/ok.csv']]);
  assert.deepStrictEqual(recent.load(path.join(tmp(), 'no-existe.json')), []);
});

test('describe marca lo que ya no existe y resolve solo devuelve rutas válidas', () => {
  const d = tmp();
  const ok = path.join(d, 'ok.csv');
  fs.writeFileSync(ok, 'a\n1\n');
  const list = [[ok], [path.join(d, 'borrado.csv')], [ok, ok, ok]];
  const info = recent.describe(list);
  assert.deepStrictEqual(info.map((x) => x.exists), [true, false, true]);
  assert.strictEqual(info[2].label, 'ok.csv, ok.csv +1');
  assert.deepStrictEqual(recent.resolve(list, 0), [ok]);
  assert.strictEqual(recent.resolve(list, 1), null); // borrado
  assert.strictEqual(recent.resolve(list, 7), null); // fuera de rango
  assert.strictEqual(recent.resolve(list, '0'), null); // el renderer no puede colar otro tipo
});
