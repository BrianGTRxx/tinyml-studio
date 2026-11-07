'use strict';
// Minimal SVG charts (no libraries, CSP stays strict). Every chart is a real <svg> so it can be exported
// to PNG / JPG / SVG and embedded in the PDF report. Colors come from CSS variables / classes.
(function () {
  const { svg, h } = window.TS;
  const t = (...a) => window.TS.t(...a);

  const txt = (parent, x, y, s, cls, anchor, extra) => {
    const el = svg('text', { x, y, class: cls || 'chart-axis', ...(anchor ? { 'text-anchor': anchor } : {}), ...(extra || {}) });
    el.textContent = s;
    parent.append(el);
    return el;
  };
  const short = (s, n) => (String(s).length > n ? `${String(s).slice(0, n - 1)}…` : String(s));
  const num = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toPrecision(d));
  const ticks = (lo, hi, n = 4) => Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n);

  // ---- histogram: before (soft fill) vs after (outline) --------------------------------------------
  function histogram(title, hist) {
    const W = 360;
    const H = 170;
    const P = { l: 8, r: 8, t: 26, b: 26 };
    const el = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart hist', role: 'img', 'aria-label': title });
    const n = hist.before.length;
    const max = Math.max(1, ...hist.before, ...hist.after);
    const bw = (W - P.l - P.r) / Math.max(n, 1);
    const sy = (v) => (v / max) * (H - P.t - P.b);
    el.append(svg('line', { x1: P.l, x2: W - P.r, y1: H - P.b, y2: H - P.b, class: 'axis-line' }));
    hist.before.forEach((v, i) => el.append(svg('rect', { x: P.l + i * bw, y: H - P.b - sy(v), width: Math.max(bw - 0.8, 0.5), height: sy(v), class: 'bar-before' })));
    hist.after.forEach((v, i) => el.append(svg('rect', { x: P.l + i * bw, y: H - P.b - sy(v), width: Math.max(bw - 0.8, 0.5), height: sy(v), class: 'bar-after' })));
    txt(el, P.l, 14, short(title, 38), 'chart-title');
    if (hist.outside_before) txt(el, W - P.r, 14, t('{n} out of range', '{n} fuera de rango', { n: hist.outside_before }), 'chart-axis warn-text', 'end');
    if (hist.edges.length) {
      txt(el, P.l, H - 8, num(hist.edges[0]), 'chart-axis', 'start');
      txt(el, W - P.r, H - 8, num(hist.edges[hist.edges.length - 1]), 'chart-axis', 'end');
    }
    return el;
  }

  // ---- training curves ----------------------------------------------------------------------------
  function lines(title, series, xlabel) {
    const W = 560;
    const H = 230;
    // legend: each entry is as wide as its name (about 5.8 px per character + marker); entries that do not fit
    // on a row wrap to the next one, so a third series or a long Spanish name never gets cut off
    const widths = series.map((s) => 20 + String(s.name).length * 5.8);
    const rows = [[]];
    let used = 0;
    const room = W - 46 - 14;
    widths.forEach((w, k) => {
      if (used + w > room && rows[rows.length - 1].length) { rows.push([]); used = 0; }
      rows[rows.length - 1].push(k);
      used += w;
    });
    const P = { l: 46, r: 14, t: 30 + (rows.length > 1 ? (rows.length - 1) * 14 : 0), b: 30 };
    const el = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart lines', role: 'img', 'aria-label': title });
    const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
    if (!all.length) return el;
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    if (hi === lo) { hi += 1; lo -= 1; }
    const n = Math.max(...series.map((s) => s.values.length));
    const x = (i) => P.l + (i / Math.max(n - 1, 1)) * (W - P.l - P.r);
    const y = (v) => H - P.b - ((v - lo) / (hi - lo)) * (H - P.t - P.b);
    for (const g of ticks(lo, hi)) {
      el.append(svg('line', { x1: P.l, x2: W - P.r, y1: y(g), y2: y(g), class: 'grid' }));
      txt(el, P.l - 6, y(g) + 3, num(g, 2), 'chart-axis', 'end');
    }
    txt(el, P.l, H - 8, '1', 'chart-axis', 'start');
    txt(el, W - P.r, H - 8, xlabel ? '' : String(n), 'chart-axis', 'end');
    txt(el, (P.l + W - P.r) / 2, H - 8, xlabel || t('epoch', 'época'), 'chart-axis', 'middle');
    txt(el, P.l, 16, title, 'chart-title');
    series.forEach((s, k) => {
      const pts = s.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
      el.append(svg('polyline', { points: pts, class: `roc-line c${k}`, fill: 'none' }));
    });
    rows.forEach((row, r) => {
      const total = row.reduce((a, k) => a + widths[k], 0);
      let lx = W - P.r - total; // right-aligned legend
      const ly = 13 + r * 14;
      row.forEach((k) => {
        el.append(svg('rect', { x: lx, y: ly - 5, width: 10, height: 3, class: `cfill c${k}` }));
        txt(el, lx + 14, ly, series[k].name, 'chart-axis');
        lx += widths[k];
      });
    });
    return el;
  }

  // ---- confusion matrix (SVG, row-normalizable) -------------------------------------------------
  function confusion(matrix, labels, normalized) {
    const k = labels.length;
    const cell = Math.max(26, Math.min(52, Math.floor(430 / k)));
    const left = Math.min(150, 14 + Math.max(...labels.map((l) => Math.min(l.length, 20))) * 6);
    const top = 20 + Math.min(110, Math.max(...labels.map((l) => Math.min(l.length, 20))) * 4.2);
    const right = Math.min(120, 10 + Math.max(...labels.map((l) => Math.min(l.length, 20))) * 4);
    const W = left + cell * k + right;
    const H = top + cell * k + 28;
    const el = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart confusion', role: 'img', 'aria-label': t('Confusion matrix', 'Matriz de confusión') });
    const rowSum = matrix.map((r) => Math.max(1e-9, r.reduce((a, b) => a + b, 0)));
    labels.forEach((l, j) => {
      const q = txt(el, left + j * cell + cell / 2, top - 6, short(l, 20), 'chart-axis', 'start');
      q.setAttribute('transform', `rotate(-40 ${left + j * cell + cell / 2} ${top - 6})`);
    });
    labels.forEach((l, i) => txt(el, left - 6, top + i * cell + cell / 2 + 3, short(l, 20), 'chart-axis', 'end'));
    matrix.forEach((r, i) => r.forEach((v, j) => {
      const f = v / rowSum[i];
      const ok = i === j;
      const c = svg('rect', { x: left + j * cell, y: top + i * cell, width: cell - 1, height: cell - 1, class: ok ? 'cm-ok' : 'cm-bad', 'fill-opacity': Math.max(v ? 0.08 : 0, Math.min(0.9, f * 0.9)).toFixed(3) });
      const tt = svg('title', {}); tt.textContent = `${labels[i]} → ${labels[j]}: ${v}${normalized ? '' : ` (${window.TS.fmt.dec(f * 100, 1)} %)`}`;
      c.append(tt);
      el.append(c);
      if (v || cell >= 34) txt(el, left + j * cell + cell / 2, top + i * cell + cell / 2 + 4, normalized ? `${Math.round(f * 100)}` : String(v), f > 0.55 ? 'cm-num on' : 'cm-num', 'middle');
    }));
    txt(el, left + (cell * k) / 2, H - 8, t('predicted →', 'predicho →'), 'chart-axis', 'middle');
    const yl = txt(el, 10, top + (cell * k) / 2, t('true ↓', 'real ↓'), 'chart-axis', 'middle');
    yl.setAttribute('transform', `rotate(-90 10 ${top + (cell * k) / 2})`);
    return el;
  }

  // ---- horizontal bars from {label: value} ------------------------------------------------------
  function bars(dist, title) {
    const entries = Object.entries(dist || {});
    const max = Math.max(1e-9, ...entries.map(([, v]) => v));
    const left = Math.min(170, 10 + Math.max(4, ...entries.map(([k]) => Math.min(k.length, 26))) * 6);
    const rowH = 20;
    const W = 420;
    const H = (title ? 26 : 8) + entries.length * rowH + 6;
    const el = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart bars', role: 'img', 'aria-label': title || 'bars' });
    if (title) txt(el, 4, 14, title, 'chart-title');
    entries.forEach(([k, v], i) => {
      const y = (title ? 26 : 8) + i * rowH;
      txt(el, left - 6, y + 12, short(k, 26), 'chart-axis', 'end');
      el.append(svg('rect', { x: left, y: y + 2, width: W - left - 56, height: 12, rx: 3, class: 'bar-track' }));
      el.append(svg('rect', { x: left, y: y + 2, width: Math.max(1, ((W - left - 56) * v) / max), height: 12, rx: 3, class: 'bar-main' }));
      txt(el, W - 4, y + 12, Number.isInteger(v) ? window.TS.fmt.int(v) : window.TS.fmt.dec(v, 3), 'chart-axis', 'end');
    });
    return el;
  }

  // ---- grouped bars: rows [{label, values:[...]}], names [...] ----------------------------------
  function grouped(rows, names) {
    const left = Math.min(170, 10 + Math.max(4, ...rows.map((r) => Math.min(r.label.length, 26))) * 6);
    const rowH = 12 * names.length + 10;
    const W = 460;
    const H = 26 + rows.length * rowH + 6;
    const el = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart grouped', role: 'img', 'aria-label': names.join(', ') });
    names.forEach((nm, k) => {
      el.append(svg('rect', { x: left + k * 90, y: 6, width: 10, height: 8, class: `cfill c${k}` }));
      txt(el, left + 14 + k * 90, 14, nm, 'chart-axis');
    });
    rows.forEach((r, i) => {
      const y0 = 26 + i * rowH;
      txt(el, left - 6, y0 + (rowH - 10) / 2 + 3, short(r.label, 26), 'chart-axis', 'end');
      r.values.forEach((v, k) => {
        const w = Math.max(0, Math.min(1, v || 0)) * (W - left - 48);
        const b = svg('rect', { x: left, y: y0 + k * 12, width: Math.max(1, w), height: 9, rx: 2, class: `cfill c${k}` });
        const tt = svg('title', {}); tt.textContent = `${r.label} · ${names[k]}: ${v === null || v === undefined ? '—' : window.TS.fmt.dec(v, 3)}`;
        b.append(tt);
        el.append(b);
        txt(el, left + w + 4, y0 + k * 12 + 8, v === null || v === undefined ? '—' : window.TS.fmt.dec(v, 2), 'chart-axis');
      });
    });
    return el;
  }

  // ---- x-y curves in [0,1] (ROC / PR) -----------------------------------------------------------
  const AX = { l: 44, r: 12, t: 26, b: 34 };
  function xy(title, series, xlabel, ylabel, diagonal) {
    const W = 380;
    const H = 320;
    const el = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart xy', role: 'img', 'aria-label': title });
    const sx = (v) => AX.l + v * (W - AX.l - AX.r);
    const sy = (v) => H - AX.b - v * (H - AX.t - AX.b);
    for (const g of [0, 0.25, 0.5, 0.75, 1]) {
      el.append(svg('line', { x1: sx(0), x2: sx(1), y1: sy(g), y2: sy(g), class: 'grid' }));
      const lab = g === 0 ? '0' : g === 1 ? '1' : String(g);
      txt(el, AX.l - 6, sy(g) + 3, lab, 'chart-axis', 'end');
      txt(el, sx(g), H - AX.b + 13, lab, 'chart-axis', 'middle');
    }
    if (diagonal) el.append(svg('line', { x1: sx(0), y1: sy(0), x2: sx(1), y2: sy(1), class: 'stroke-diag' }));
    txt(el, AX.l, 14, title, 'chart-title');
    txt(el, sx(0.5), H - 6, xlabel, 'chart-axis', 'middle');
    const yl = txt(el, 11, sy(0.5), ylabel, 'chart-axis', 'middle');
    yl.setAttribute('transform', `rotate(-90 11 ${sy(0.5)})`);
    series.forEach((s, k) => {
      const pts = s.x.map((v, i) => `${sx(v).toFixed(1)},${sy(s.y[i]).toFixed(1)}`).join(' ');
      el.append(svg('polyline', { points: pts, class: `roc-line c${k % 8}`, fill: 'none' }));
    });
    return el;
  }

  function legend(items) {
    return h('div', { class: 'legend' }, items.map((it, k) => h('span', { class: 'chip' }, h('i', { class: `dot c${k % 8}` }), ` ${it}`)));
  }

  function scatter(title, xs, ys) {
    const all = [...xs, ...ys];
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const W = 380;
    const H = 320;
    const el = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart xy', role: 'img', 'aria-label': title });
    const f = (v) => (v - lo) / (hi - lo || 1);
    const sx = (v) => AX.l + f(v) * (W - AX.l - AX.r);
    const sy = (v) => H - AX.b - f(v) * (H - AX.t - AX.b);
    for (const g of ticks(lo, hi)) {
      el.append(svg('line', { x1: sx(lo), x2: sx(hi), y1: sy(g), y2: sy(g), class: 'grid' }));
      txt(el, AX.l - 6, sy(g) + 3, num(g, 2), 'chart-axis', 'end');
      txt(el, sx(g), H - AX.b + 13, num(g, 2), 'chart-axis', 'middle');
    }
    el.append(svg('line', { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi), class: 'stroke-diag' }));
    xs.forEach((x, i) => el.append(svg('circle', { cx: sx(x).toFixed(1), cy: sy(ys[i]).toFixed(1), r: 2.4, class: 'pt' })));
    txt(el, AX.l, 14, title, 'chart-title');
    return el;
  }

  function hist1(title, edges, counts) {
    const W = 380;
    const H = 180;
    const el = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart hist', role: 'img', 'aria-label': title });
    const max = Math.max(1, ...counts);
    const bw = (W - 24) / counts.length;
    el.append(svg('line', { x1: 12, x2: W - 12, y1: H - 26, y2: H - 26, class: 'axis-line' }));
    counts.forEach((c, i) => {
      const bh = (c / max) * (H - 56);
      el.append(svg('rect', { x: 12 + i * bw, y: H - 26 - bh, width: Math.max(bw - 1, 0.5), height: bh, class: 'bar-main' }));
    });
    txt(el, 12, 14, title, 'chart-title');
    txt(el, 12, H - 8, num(edges[0]), 'chart-axis', 'start');
    txt(el, W - 12, H - 8, num(edges[edges.length - 1]), 'chart-axis', 'end');
    return el;
  }

  window.TS.charts = { histogram, lines, confusion, bars, xy, legend, scatter, hist1, grouped };
})();
