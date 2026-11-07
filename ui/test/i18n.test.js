'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const RENDERER = path.join(__dirname, '..', 'src', 'renderer');
const FILES = ['icons.js', 'components.js', 'export.js', 'monitor.js', 'history.js', 'setup.js', 'env.js', 'updates.js', 'report.js', 'tour.js', 'benchmark.js', 'wizard.js', 'quality.js', 'groups.js', 'tune.js', 'modelinfo.js', 'graph.js', 'arduino.js', 'charts.js', 'renderer.js', 'util.js'];

// English is the source language: any accented/Spanish literal must live inside t('en', 'es').
test('no hard-coded Spanish text outside t(en, es)', () => {
  for (const f of FILES) {
    const lines = fs.readFileSync(path.join(RENDERER, f), 'utf8').split(/\r?\n/); // Git en Windows puede dejar CRLF
    lines.forEach((line, i) => {
      if (/^\s*\/\//.test(line)) return;
      // remove every t('..','..') / t(`..`,`..`) call (strings may contain escaped quotes)
      const stripped = line.replace(/\bt\(\s*(['`])(?:\\.|(?!\1).)*\1\s*,\s*(['`])(?:\\.|(?!\2).)*\2/g, 't(');
      assert.ok(!/[áéíóúñ¿¡]/.test(stripped.replace(/'Español'/g, '').replace(/\/\/.*$/, '')), `${f}:${i + 1}: ${line.trim().slice(0, 100)}`);
    });
  }
});

test('theme and language helpers exist and persist safely', () => {
  const src = fs.readFileSync(path.join(RENDERER, 'theme.js'), 'utf8');
  assert.match(src, /prefers-color-scheme/);
  assert.match(src, /try \{/); // localStorage access is guarded
  assert.match(src, /TS\.t = function/);
});

test('stylesheet defines both themes through variables only', () => {
  const css = fs.readFileSync(path.join(RENDERER, 'style.css'), 'utf8');
  assert.match(css, /:root\[data-theme="dark"\]/);
  assert.ok(!/@keyframes/.test(css), 'no keyframe animations');
});
