'use strict';
// Benchmark panel: sortable leaderboard, metric bars, size-vs-quality (Pareto) chart, test comparison.
(function () {
  const TS = window.TS;
  const { h, svg, fmt, table, callout, charts } = TS;
  const ui = TS.ui;
  const t = TS.t;

  const COLS = (task) => (task === 'regression'
    ? [['r2', 'R²', true], ['mae', 'MAE', false], ['rmse', 'RMSE', false]]
    : [['f1_macro', 'F1 macro', true], ['accuracy', t('accuracy', 'exactitud'), true], ['mcc', 'MCC', true], ['mAP', 'mAP', true], ['roc_auc_macro', 'ROC-AUC', true]]);

  const val = (row, split, k) => (row.metrics && row.metrics[split] ? row.metrics[split][k] : null);

  function pareto(rows, metric) {
    const pts = rows.filter((r) => r.size_bytes && val(r, 'val', metric) !== null && val(r, 'val', metric) !== undefined);
    const w = 460; const hh = 280; const L = 44; const B = 32;
    const el = svg('svg', { viewBox: `0 0 ${w} ${hh}`, class: 'chart xy', role: 'img' });
    const title = svg('text', { x: L, y: 12, class: 'chart-title' });
    title.textContent = t('Quality (validation) vs size', 'Calidad (validación) vs tamaño');
    el.append(title);
    if (!pts.length) return el;
    const xs = pts.map((r) => Math.log10(r.size_bytes));
    const ys = pts.map((r) => val(r, 'val', metric));
    const x0 = Math.min(...xs) - 0.1; const x1 = Math.max(...xs) + 0.1;
    const y0 = Math.min(...ys); const y1 = Math.max(...ys);
    const sx = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (w - L - 12);
    const sy = (v) => hh - B - ((v - y0) / (y1 - y0 || 1)) * (hh - B - 24);
    const lab = (x, y, s, anchor) => { const q = svg('text', { x, y, class: 'chart-axis', ...(anchor ? { 'text-anchor': anchor } : {}) }); q.textContent = s; el.append(q); };
    lab(L, hh - 8, fmt.bytes(10 ** x0), 'start'); lab(w - 12, hh - 8, fmt.bytes(10 ** x1), 'end'); lab(2, sy(y0), fmt.num(y0, 3)); lab(2, sy(y1) + 8, fmt.num(y1, 3));
    lab((L + w - 12) / 2, hh - 8, t('size (log scale) →', 'tamaño (escala log) →'), 'middle');
    const placed = []; // labels already drawn: tied models share a point, so shift the later label down instead of overprinting it
    pts.forEach((r, i) => {
      const c = svg('circle', { cx: sx(xs[i]).toFixed(1), cy: sy(ys[i]).toFixed(1), r: r.selected ? 6 : 4, class: `bpt ${r.selected ? 'sel' : ''} ${r.pareto ? 'par' : ''}` });
      const tt = svg('title', {}); tt.textContent = `${r.name} · ${fmt.num(ys[i], 3)} · ${fmt.bytes(r.size_bytes)}`;
      c.append(tt); el.append(c);
      let ly = sy(ys[i]) + 3;
      while (placed.some((q) => Math.abs(q.x - sx(xs[i])) < 70 && Math.abs(q.y - ly) < 10)) ly += 10;
      placed.push({ x: sx(xs[i]), y: ly });
      lab(sx(xs[i]) + 7, ly, r.name.replace('keras_mlp:', 'mlp:'));
    });
    return el;
  }

  TS.benchmarkPanel = function benchmarkPanel(res) {
    const rows = res.leaderboard || [];
    const sel = res.selection || {};
    const cols = COLS(res.task);
    const metric = sel.metric || cols[0][0];
    let sortKey = 'rank'; let asc = true;
    const tbody = h('tbody', {});
    const flags = (r) => [r.selected ? '★' : '', r.pareto ? t('Pareto', 'Pareto') : '', r.statistically_tied_with_best && !r.selected ? t('tied', 'empatado') : '', r.fits_flash === false ? t('does not fit', 'no cabe') : ''].filter(Boolean).join(' · ');
    const get = (r, k) => {
      if (k === 'rank') return r.rank;
      if (k === 'name') return r.name;
      if (k === 'size') return r.size_bytes ?? Infinity;
      if (k === 'fit') return r.fit_seconds;
      if (k === 'us') return r.predict_us_per_row;
      if (k.startsWith('val:')) return val(r, 'val', k.slice(4));
      if (k.startsWith('test:')) return val(r, 'test', k.slice(5));
      return 0;
    };
    const draw = () => {
      const sorted = [...rows].sort((a, b) => {
        const x = get(a, sortKey); const y = get(b, sortKey);
        const c = typeof x === 'string' ? x.localeCompare(y) : (x ?? -Infinity) - (y ?? -Infinity);
        return asc ? c : -c;
      });
      tbody.replaceChildren(...sorted.map((r) => {
        const tr = h('tr', { class: r.selected ? 'sel' : '', 'data-model': r.name },
          h('td', { text: r.rank }),
          h('td', {}, h('b', { text: r.name }), ' ', h('span', { class: 'muted', text: r.family })),
          h('td', { text: r.exportable ? (r.export_job === 'export_tflite' ? 'TFLite' : 'C') : '—', title: r.exportable ? '' : t('No exporter: It can be compared but not flashed.', 'Sin exportador: Se compara pero no se puede grabar.') }),
          ...cols.map(([k]) => h('td', { text: fmt.num(val(r, 'val', k)) })),
          ...cols.map(([k]) => h('td', { text: fmt.num(val(r, 'test', k)) })),
          h('td', { text: r.test_ci ? `[${fmt.num(r.test_ci.lo, 3)}, ${fmt.num(r.test_ci.hi, 3)}]` : '—' }),
          h('td', { text: r.size_bytes ? fmt.bytes(r.size_bytes) : '—' }),
          h('td', { text: r.fit_seconds }), h('td', { text: r.predict_us_per_row }),
          h('td', { text: flags(r) }));
        return tr;
      }));
    };
    const th = (label, key, title) => h('th', { class: 'sortable', title: title || t('Click to sort', 'Clic para ordenar'), 'data-sort': key, onclick: () => { if (sortKey === key) asc = !asc; else { sortKey = key; asc = !(key.includes(':') && !['mae', 'rmse'].some((m) => key.endsWith(m))); } draw(); } }, label);
    const head = h('thead', {}, h('tr', {},
      th('#', 'rank'), th(t('Model', 'Modelo'), 'name'), h('th', { text: t('Export', 'Exportar') }),
      ...cols.map(([k, l]) => th(`${l} val`, `val:${k}`)), ...cols.map(([k, l]) => th(`${l} test`, `test:${k}`)),
      h('th', { text: t('Test 95% CI', 'IC95% test'), title: `${metric}` }), th(t('Size', 'Tamaño'), 'size'), th(t('Fit (s)', 'Ajuste (s)'), 'fit'), th('µs/row', 'us', t('Prediction time per row on the cloud CPU (not the board)', 'Tiempo de predicción por fila en la CPU de la nube (no la placa)')), h('th', { text: '' })));
    draw();

    const barMetric = h('select', { id: 'bench-metric', onchange: () => drawBars() }, cols.map(([k, l]) => h('option', { value: k, text: `${l} (test)` })));
    barMetric.value = cols.some(([k]) => k === metric) ? metric : cols[0][0];
    const barsBox = h('div', {});
    const drawBars = () => {
      const k = barMetric.value;
      const d = {};
      [...rows].sort((a, b) => (val(b, 'test', k) ?? -1) - (val(a, 'test', k) ?? -1)).forEach((r) => { d[r.name] = Number((val(r, 'test', k) ?? 0).toFixed(4)); });
      barsBox.replaceChildren(ui.chart(null, charts.bars(d, `${barMetric.selectedOptions[0].textContent}`), `benchmark-${k}`));
    };
    drawBars();

    const selRow = rows.find((r) => r.selected);
    const warns = res.warnings || [];
    const cm = res.selected_confusion_test;
    return h('div', { id: 'benchmark-results' },
      h('div', { class: 'kpis' },
        h('div', { class: 'kpi' }, h('b', { text: sel.selected || '—' }), h('span', { text: t('recommended model', 'modelo recomendado') })),
        selRow && selRow.test_ci ? h('div', { class: 'kpi' }, h('b', { text: fmt.num(selRow.test_ci.point, 3) }), h('span', { text: `${metric} test · IC95% [${fmt.num(selRow.test_ci.lo, 3)}, ${fmt.num(selRow.test_ci.hi, 3)}]` })) : null,
        h('div', { class: 'kpi' }, h('b', { text: rows.length }), h('span', { text: t('models compared', 'modelos comparados') })),
        h('div', { class: 'kpi' }, h('b', { text: `${res.elapsed_seconds ?? '—'} s` }), h('span', { text: t('total time', 'tiempo total') }))),
      callout('ok', h('b', { text: t('Recommendation: ', 'Recomendación: ') }), sel.reason || ''),
      h('p', { class: 'muted', text: t('Chosen on validation (or CV) with the one-standard-error rule; the test set never chooses. Click a header to sort.', 'Elegido con validación (o CV) y la regla de una desviación estándar; el test nunca elige. Clic en un encabezado para ordenar.') }),
      h('div', { class: 'scroll' }, h('table', { class: 'tbl bench' }, head, tbody)),
      h('div', { class: 'xyrow' },
        h('div', {}, h('label', { class: 'row' }, h('span', { class: 'muted', text: t('Metric', 'Métrica') }), barMetric), barsBox),
        h('div', {}, ui.chart(null, pareto(rows, metric), 'quality-vs-size'), h('p', { class: 'muted', text: t('Filled = on the Pareto front (nothing is both smaller and better). Size = C tables or int8 weights, without TFLM arena.', 'Relleno = en el frente de Pareto (nada es a la vez más pequeño y mejor). Tamaño = tablas de C o pesos int8, sin la arena de TFLM.') }))),
      cm && cm.matrix ? h('div', {}, h('h3', { text: t('Confusion matrix of the recommended model (test)', 'Matriz de confusión del modelo recomendado (test)') }), ui.chart(null, charts.confusion(cm.matrix, cm.classes), 'confusion-selected')) : null,
      warns.length ? h('div', {}, h('h3', { text: t('Read before deciding', 'Léelo antes de decidir') }), h('ul', { class: 'warns' }, warns.map((w) => h('li', { text: w })))) : null,
      (res.models_skipped || []).length ? callout('info', t('Skipped: ', 'Omitidos: '), JSON.stringify(res.models_skipped)) : null);
  };
})();
