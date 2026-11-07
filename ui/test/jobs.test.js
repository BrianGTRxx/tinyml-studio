'use strict';
// Prueba real: sidecar Python + trabajo de entrenamiento + flujo SSE (sin Electron).
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { Sidecar } = require('../src/sidecar');

const REPO = path.resolve(__dirname, '..', '..');
const VENV_PY =
  process.platform === 'win32'
    ? path.join(REPO, '.venv', 'Scripts', 'python.exe')
    : path.join(REPO, '.venv', 'bin', 'python');
const PYTHON = process.env.TINYML_PYTHON || VENV_PY;

function makeCsv(dir) {
  // 2 clases bien separadas (a: ~0, b: ~10), 150 filas cada una
  const rows = ['f1,f2,label'];
  let s = 12345;
  const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5);
  for (const [lab, base] of [['a', 0], ['b', 10]]) {
    for (let i = 0; i < 150; i++) rows.push(`${(base + rnd() * 2).toFixed(4)},${(rnd() * 4).toFixed(4)},${lab}`);
  }
  fs.writeFileSync(path.join(dir, 'd.csv'), rows.join('\n') + '\n');
}

async function waitState(sc, id, timeoutMs = 120000) {
  const t0 = Date.now();
  for (;;) {
    const j = await sc.request('GET', `/jobs/${id}`);
    if (['succeeded', 'failed', 'cancelled'].includes(j.state)) return j;
    assert.ok(Date.now() - t0 < timeoutMs, 'timeout');
    await new Promise((r) => setTimeout(r, 200));
  }
}

test('trabajo de entrenamiento con eventos SSE desde Node', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tmsjobs-'));
  const data = path.join(dir, 'data');
  fs.mkdirSync(data);
  makeCsv(data);
  const sc = new Sidecar({
    python: PYTHON, cwd: REPO, allowedRoots: [data], env: { TINYML_WORKDIR: path.join(dir, 'ws') },
  });
  await sc.start();
  try {
    const rep = await sc.request('POST', '/inspect', { paths: [data] });
    assert.strictEqual(rep.task.target, 'label');
    const job = await sc.request('POST', '/jobs', {
      kind: 'train', params: { recipe: rep.recipe, models: ['decision_tree'], seed: 0 },
    });
    assert.match(job.id, /^[0-9a-f]{12}$/);
    const events = [];
    const s = sc.stream(`/jobs/${job.id}/events`, (ev) => events.push(ev));
    await s.done;
    assert.ok(events.length >= 3);
    assert.ok(events.some((e) => e.stage === 'train'));
    assert.strictEqual(events[events.length - 1].stage, 'end');
    assert.strictEqual(events[events.length - 1].state, 'succeeded');
    const fin = await waitState(sc, job.id);
    assert.strictEqual(fin.state, 'succeeded');
    const res = await sc.request('GET', `/jobs/${job.id}/result`);
    assert.ok(res.results.models.decision_tree.test.f1_macro > 0.9);
    // seguridad: archivos fuera de las raíces y rutas con '..'
    await assert.rejects(
      sc.request('POST', '/jobs', {
        kind: 'train', params: { recipe: { ...rep.recipe, files: ['/etc/passwd'] } },
      }),
      (e) => e.status === 403,
    );
    // cancelar un trabajo en cola/en marcha desde Node
    const long = await sc.request('POST', '/jobs', {
      kind: 'train', params: { recipe: rep.recipe, models: ['random_forest'], seed: 0, params: { random_forest: { n_estimators: 400 } } },
    });
    const c = await sc.request('POST', `/jobs/${long.id}/cancel`);
    assert.strictEqual(c.state, 'cancelled');
  } finally {
    await sc.stop();
  }
});
