'use strict';
// Netron-like model viewer: layered graph (TFLite / Keras / MLP), decision trees, forests, linear models.
// Data comes from export/graph.py (JSON); nothing is inserted as HTML.
(function () {
  const TS = window.TS;
  const { h, svg, fmt, table, callout, charts } = TS;
  const t = TS.t;
  const NW = 176;
  const NH = 46;
  const GX = 28;
  const GY = 44;

  const shapeStr = (s) => (s && s.length ? `[${s.join('×')}]` : '[]');

  function layout(g) {
    const tensors = new Map(g.tensors.map((x) => [x.id, x]));
    const producer = new Map();
    g.nodes.forEach((n) => n.outputs.forEach((o) => producer.set(o, n.id)));
    const nodes = new Map(g.nodes.map((n) => [n.id, { ...n, kind: 'op' }]));
    // virtual input nodes
    const edges = [];
    g.inputs.forEach((tid, i) => {
      const id = `in${i}`;
      nodes.set(id, { id, op: 'input', name: tensors.get(tid) ? tensors.get(tid).name : 'input', outputs: [tid], inputs: [], kind: 'io', options: {} });
      producer.set(tid, id);
    });
    g.nodes.forEach((n) => n.inputs.forEach((tid) => {
      const p = producer.get(tid);
      if (p !== undefined && !(tensors.get(tid) || {}).is_const) edges.push([p, n.id, tid]);
    }));
    g.outputs.forEach((tid, i) => {
      const id = `out${i}`;
      nodes.set(id, { id, op: 'output', name: tensors.get(tid) ? tensors.get(tid).name : 'output', outputs: [], inputs: [tid], kind: 'io', options: {} });
      if (producer.has(tid)) edges.push([producer.get(tid), id, tid]);
    });
    const level = new Map();
    const lv = (id, seen = new Set()) => {
      if (level.has(id)) return level.get(id);
      if (seen.has(id)) return 0;
      seen.add(id);
      const ins = edges.filter((e) => e[1] === id).map((e) => e[0]);
      const v = ins.length ? 1 + Math.max(...ins.map((x) => lv(x, seen))) : 0;
      level.set(id, v);
      return v;
    };
    [...nodes.keys()].forEach((id) => lv(id));
    const rows = new Map();
    [...nodes.keys()].forEach((id) => {
      const l = level.get(id);
      if (!rows.has(l)) rows.set(l, []);
      rows.get(l).push(id);
    });
    const maxCols = Math.max(...[...rows.values()].map((r) => r.length));
    const width = maxCols * (NW + GX) + GX;
    const pos = new Map();
    for (const [l, ids] of rows) {
      const w = ids.length * (NW + GX) - GX;
      ids.forEach((id, i) => pos.set(id, { x: (width - w) / 2 + i * (NW + GX), y: GY / 2 + l * (NH + GY) }));
    }
    return { nodes, edges, pos, width, height: (rows.size + 0.5) * (NH + GY), tensors };
  }

  function tensorRow(tn) {
    if (!tn) return null;
    const q = tn.quant ? ` · scale ${Array.isArray(tn.quant.scale) ? `${tn.quant.scale.length} ch` : Number(tn.quant.scale).toPrecision(4)}, zp ${Array.isArray(tn.quant.zero_point) ? '…' : tn.quant.zero_point}` : '';
    return h('li', {}, h('b', { text: tn.name || `#${tn.id}` }), ` ${tn.dtype} ${shapeStr(tn.shape)}${q}`,
      tn.data ? h('div', { class: 'muted', text: `min ${fmt.num(tn.data.min, 3)} · max ${fmt.num(tn.data.max, 3)} · mean ${fmt.num(tn.data.mean, 3)} · σ ${fmt.num(tn.data.std, 3)} · ${fmt.num(tn.data.zeros_pct, 1)} % ${t('zeros', 'ceros')}` }) : null,
      tn.data && tn.data.hist ? charts.hist1(t('weights', 'pesos'), tn.data.hist.edges, tn.data.hist.counts) : null,
      tn.data && tn.data.values ? h('div', { class: 'muted', text: `[${tn.data.values.map((v) => Number(v).toPrecision(3)).join(', ')}]` }) : null);
  }

  function details(g, L, n) {
    const ts = (ids) => ids.map((i) => tensorRow(L.tensors.get(i))).filter(Boolean);
    return h('div', {},
      h('h3', { text: n.op }), h('p', { class: 'muted', text: n.name || '' }),
      Object.keys(n.options || {}).length ? table(['option', 'value'], Object.entries(n.options).map(([k, v]) => [k, typeof v === 'object' ? JSON.stringify(v) : String(v)])) : null,
      n.inputs.length ? h('div', {}, h('b', { text: t('Inputs', 'Entradas') }), h('ul', {}, ts(n.inputs))) : null,
      n.outputs.length ? h('div', {}, h('b', { text: t('Outputs', 'Salidas') }), h('ul', {}, ts(n.outputs))) : null);
  }

  function layered(g) {
    const L = layout(g);
    const side = h('div', { class: 'side' }, h('p', { class: 'muted', text: t('Click a node to see its tensors, quantization and weights.', 'Haz clic en un nodo para ver sus tensores, cuantización y pesos.') }));
    const el = svg('svg', { viewBox: `0 0 ${L.width} ${L.height}`, class: 'gsvg', role: 'img', 'aria-label': 'model graph' });
    const root = svg('g', {});
    el.append(root);
    for (const [a, b] of L.edges) {
      const pa = L.pos.get(a);
      const pb = L.pos.get(b);
      const x1 = pa.x + NW / 2; const y1 = pa.y + NH; const x2 = pb.x + NW / 2; const y2 = pb.y;
      root.append(svg('path', { d: `M${x1},${y1} C${x1},${y1 + GY / 2} ${x2},${y2 - GY / 2} ${x2},${y2}`, class: 'gedge' }));
    }
    let selected = null;
    for (const [id, n] of L.nodes) {
      const p = L.pos.get(id);
      const outT = L.tensors.get(n.outputs[0]) || L.tensors.get(n.inputs[0]);
      const gn = svg('g', { class: `gnode ${n.kind}`, transform: `translate(${p.x},${p.y})`, 'data-node': id });
      gn.append(svg('rect', { width: NW, height: NH, rx: 6 }));
      const a = svg('text', { x: 10, y: 19 }); a.textContent = n.op;
      const b = svg('text', { x: 10, y: 35, class: 'sub' }); b.textContent = outT ? `${outT.dtype} ${shapeStr(outT.shape)}` : '';
      gn.append(a, b);
      gn.addEventListener('click', () => {
        if (selected) selected.classList.remove('sel');
        selected = gn; gn.classList.add('sel');
        side.replaceChildren(n.kind === 'io' ? h('div', {}, h('h3', { text: n.op }), tensorRow(outT)) : details(g, L, n));
      });
      root.append(gn);
    }
    // pan (drag) and zoom (wheel) by editing the viewBox: no animation, no style attribute (CSP)
    let vb = [0, 0, L.width, L.height];
    const apply = () => el.setAttribute('viewBox', vb.map((v) => v.toFixed(1)).join(' '));
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const f = e.deltaY < 0 ? 0.88 : 1.14;
      const mx = vb[0] + ((e.clientX - r.left) / r.width) * vb[2];
      const my = vb[1] + ((e.clientY - r.top) / r.height) * vb[3];
      vb = [mx - (mx - vb[0]) * f, my - (my - vb[1]) * f, vb[2] * f, vb[3] * f];
      apply();
    }, { passive: false });
    let drag = null;
    el.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, vb: [...vb] }; });
    window.addEventListener('pointerup', () => { drag = null; });
    el.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const r = el.getBoundingClientRect();
      vb = [drag.vb[0] - ((e.clientX - drag.x) / r.width) * vb[2], drag.vb[1] - ((e.clientY - drag.y) / r.height) * vb[3], vb[2], vb[3]];
      apply();
    });
    const reset = h('button', { type: 'button', class: 'ghost', text: t('Reset view', 'Restablecer vista'), onclick: () => { vb = [0, 0, L.width, L.height]; apply(); } });
    return h('div', {}, h('div', { class: 'row' }, reset, h('span', { class: 'muted', text: t('Wheel = zoom · drag = pan', 'Rueda = zoom · arrastrar = mover') })),
      h('div', { class: 'graph' }, h('div', { class: 'canvas' }, el), side));
  }

  // ---- decision tree (depth-limited) -------------------------------------------------------
  function treeView(root, g) {
    const TW = 150;
    const TH = 52;
    const levels = [];
    const walk = (n, d, ix) => { (levels[d] = levels[d] || []).push(n); n._d = d; if (n.left) walk(n.left, d + 1); if (n.right) walk(n.right, d + 1); };
    walk(root, 0);
    // x positions: leaves get consecutive slots, parents centered above children
    let slot = 0;
    const place = (n) => {
      if (!n.left && !n.right) { n._x = slot++; return; }
      [n.left, n.right].filter(Boolean).forEach(place);
      const kids = [n.left, n.right].filter(Boolean);
      n._x = kids.reduce((s, k) => s + k._x, 0) / kids.length;
    };
    place(root);
    const W = Math.max(1, slot) * (TW + 14) + 14;
    const Ht = levels.length * (TH + 34) + 10;
    const el = svg('svg', { viewBox: `0 0 ${W} ${Ht}`, width: W, height: Ht, class: 'gsvg tree', role: 'img' });
    const side = h('div', { class: 'side' }, h('p', { class: 'muted', text: t('Click a node for its class distribution.', 'Haz clic en un nodo para ver su distribución de clases.') }));
    const draw = (n) => {
      const x = 7 + n._x * (TW + 14);
      const y = 8 + n._d * (TH + 34);
      [n.left, n.right].filter(Boolean).forEach((k) => {
        const kx = 7 + k._x * (TW + 14);
        const ky = 8 + k._d * (TH + 34);
        el.append(svg('path', { d: `M${x + TW / 2},${y + TH} C${x + TW / 2},${y + TH + 16} ${kx + TW / 2},${ky - 16} ${kx + TW / 2},${ky}`, class: 'gedge' }));
        draw(k);
      });
      const gn = svg('g', { class: `gnode ${n.leaf ? 'leaf' : ''}`, transform: `translate(${x},${y})` });
      gn.append(svg('rect', { width: TW, height: TH, rx: 6 }));
      const a = svg('text', { x: 8, y: 20 }); a.textContent = n.leaf || n.truncated ? `${n.prediction ?? (n.value !== undefined ? fmt.num(n.value, 3) : '…')}${n.truncated ? ' …' : ''}` : `${n.feature} ≤ ${Number(n.threshold).toPrecision(4)}`;
      const b = svg('text', { x: 8, y: 38, class: 'sub' }); b.textContent = `n=${n.samples}`;
      gn.append(a, b);
      gn.addEventListener('click', () => {
        const dist = n.distribution;
        side.replaceChildren(h('h3', { text: n.leaf ? t('Leaf', 'Hoja') : t('Split', 'División') }),
          h('p', { text: n.leaf || n.truncated ? '' : `${n.feature} ≤ ${n.threshold}` }),
          h('p', { class: 'muted', text: `${t('samples', 'muestras')}: ${n.samples} · ${t('impurity', 'impureza')}: ${fmt.num(n.impurity, 3)}` }),
          dist ? charts.bars(dist, t('class distribution', 'distribución de clases')) : null,
          n.truncated ? h('p', { class: 'muted', text: t('Subtree hidden (depth limit).', 'Subárbol oculto (límite de profundidad).') }) : null);
      });
      el.append(gn);
    };
    draw(root);
    const canvas = h('div', { class: 'canvas scrollx' }, el);
    // a wide tree starts scrolled to its middle so the root is in view
    [60, 400].forEach((ms) => setTimeout(() => { canvas.scrollLeft = Math.max(0, (canvas.scrollWidth - canvas.clientWidth) / 2); }, ms)); // after it is attached
    return h('div', { class: 'graph' }, canvas, side);
  }

  function linearView(m) {
    const boxes = m.classes.map((c) => {
      const maxAbs = Math.max(1e-9, ...c.coefficients.map((x) => Math.abs(x.value)));
      const rows = c.coefficients.map((x) => {
        const fill = h('div', { class: `bar-fill ${x.value < 0 ? 'neg' : ''}` });
        fill.style.width = `${(Math.abs(x.value) / maxAbs) * 100}%`;
        return h('div', { class: 'bar-row' }, h('span', { class: 'bar-label', text: x.feature, title: x.feature }), h('div', { class: 'bar' }, fill), h('span', { class: 'bar-val', text: fmt.num(x.value, 3) }));
      });
      const single = m.classes.length === 1 && /^-?\d+$/.test(String(c.class)); // regression / one output: "0" says nothing
      return h('div', { class: 'card' }, h('b', { text: single ? t('Predicted value', 'Valor predicho') : String(c.class) }), h('div', { class: 'muted', text: `${t('intercept', 'intercepto')} ${fmt.num(c.intercept, 4)}` }), h('div', { class: 'bars' }, rows));
    });
    return h('div', { class: 'cards' }, boxes);
  }

  // g: model graph JSON. Returns an element.
  TS.graphView = function graphView(g) {
    if (!g || g.error) return callout('warn', t('The structure could not be extracted for this model.', 'No se pudo extraer la estructura de este modelo.'), g && g.error ? ` (${g.error})` : '');
    const s = g.summary || {};
    const st = g.stats || {};
    const head = h('div', { class: 'kpis' },
      g.nodes && g.nodes.length ? h('div', { class: 'kpi' }, h('b', { text: g.nodes.length }), h('span', { text: t('nodes', 'nodos') })) : null,
      st.param_count !== undefined ? h('div', { class: 'kpi' }, h('b', { text: st.param_count }), h('span', { text: t('parameters', 'parámetros') })) : null,
      st.total_const_bytes !== undefined ? h('div', { class: 'kpi' }, h('b', { text: fmt.bytes(st.total_const_bytes) }), h('span', { text: t('constant data', 'datos constantes') })) : null,
      s.estimator ? h('div', { class: 'kpi' }, h('b', { text: s.estimator }), h('span', { text: t('estimator', 'estimador') })) : null,
      s.quantized !== undefined ? h('div', { class: 'kpi' }, h('b', { text: s.quantized ? 'int8' : 'float' }), h('span', { text: t('precision', 'precisión') })) : null);
    const m = g.model;
    let body;
    if (m && m.type === 'tree') body = treeView(m.root, g);
    else if (m && m.type === 'forest') body = h('div', {}, callout('info', t('Random forest with {n} trees. Showing the first tree; the table lists every tree.', 'Random forest con {n} árboles. Se muestra el primero; la tabla lista todos.', { n: m.n_trees })), treeView(m.first_tree.root, g),
      table(['#', t('depth', 'profundidad'), t('leaves', 'hojas')], m.per_tree.slice(0, 50).map((x, i) => [i, x.depth, x.n_leaves])));
    else if (m && m.type === 'linear') body = linearView(m);
    else if (m && m.type === 'svm') body = h('div', {}, table(['', ''], [['kernel', m.kernel], [t('support vectors', 'vectores de soporte'), m.n_support_total], ['gamma', String(m.gamma)]]), callout('info', t('SVM support vectors are not drawn; a kernel SVM has no readable layer structure.', 'Los vectores de soporte de la SVM no se dibujan; una SVM con kernel no tiene una estructura de capas legible.')));
    else if (g.nodes && g.nodes.length) body = layered(g);
    else body = h('p', { class: 'muted', text: '—' });
    return h('div', { class: 'model-graph', id: 'model-graph' }, head, body);
  };
})();
