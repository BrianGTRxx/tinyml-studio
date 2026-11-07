'use strict';
// 7-step wizard (docs/VISION.md). State lives in TS.S; each step is redrawn from it. English is the source language: t(en, es).
(function () {
  const { h, fmt, badge, table, callout, charts } = window.TS;
  const TS = window.TS;
  const t = TS.t;
  const ui = TS.ui;

  const S = (TS.S = {
    step: 0,
    paths: [],
    report: null, // salida de /inspect
    recipe: null, // receta editable
    clean: null, // resultado del trabajo 'clean'
    family: 'classical',
    classicModels: [],
    bench: { models: null, presets: ['small', 'medium'], cv: 0, selection: 'f1_macro', maxFlash: '', epochs: 60, patience: 8 },
    benchPick: null,
    hp: {},
    device: 'auto',
    backend: 'tensorflow',
    gpuProbe: null,
    train: null, // {job, result}
    export: null,
    boardId: 'esp32-s3',
    flashKb: '',
    ramKb: '',
    int8io: false,
    hardware: null,
    boards: null,
    catalog: null,
    busy: false,
  });

  // step names are functions so they follow the language
  TS.STEP_NAMES = () => [
    t('Dataset', 'Dataset'), t('Task', 'Tarea'), t('Cleaning', 'Limpieza'), t('Augmentation', 'Aumento'),
    t('Split', 'Partición'), t('Model & training', 'Modelo y entrenamiento'), t('Board & export', 'Placa y exportación'),
  ];

  const err = (e) => callout('bad', `Error: ${e && e.message ? e.message : e}`);
  const textInput = (value, on, extra) =>
    h('input', { type: 'text', value: value ?? '', oninput: (e) => on(e.target.value), ...(extra || {}) });
  const numInput = (value, on, extra) =>
    h('input', { type: 'number', value: value ?? '', oninput: (e) => on(e.target.value === '' ? null : Number(e.target.value)), ...(extra || {}) });
  const select = (options, value, on) => {
    // a current value that is not in the list (e.g. an internal column) is shown instead of leaving the box blank
    const opts = value && !options.some(([v]) => v === value) ? [...options, [value, String(value)]] : options;
    const el = h('select', { onchange: (e) => on(e.target.value) }, opts.map(([v, label]) => h('option', { value: v, text: label })));
    el.value = value;
    return el;
  };
  const field = (label, control, hint, help) => ui.field(label, control, hint, help);
  const needData = () => ui.empty('folder', t('Open a dataset first', 'Primero abre un dataset'), t('Go to step 1 and load your CSV files.', 'Ve al paso 1 y carga tus archivos CSV.'));

  // model picker: a dropdown with search, grouped by family, multi-select, with quick presets
  const FAM = () => ({ tree: t('Trees', 'Árboles'), linear: t('Linear', 'Lineales'), kernel: t('Kernel / margin', 'Kernel / margen'), neighbors: t('Neighbors', 'Vecinos'), bayes: t('Probabilistic', 'Probabilísticos'), ensemble: t('Ensembles', 'Ensambles'), neural: t('Neural networks', 'Redes neuronales') });
  const DEFAULT_CLASSIC = ['decision_tree', 'random_forest', 'logreg', 'linear', 'svm', 'mlp'];
  function benchBoardSelect(B) {
    const boards = (S.boards && S.boards.boards) || [];
    if (!boards.length) return h('p', { class: 'muted', text: t('Loading boards…', 'Cargando placas…') });
    const groups = [];
    for (const b of boards) { const g = b.family || b.vendor || ''; let x = groups.find((y) => y.label === g); if (!x) { x = { label: g, items: [] }; groups.push(x); } x.items.push(b); }
    const sel = h('select', { id: 'bench-board', onchange: (e) => { S.boardId = e.target.value; const nb = boards.find((b) => b.id === S.boardId); if (nb) { applyBoardDefaults(nb); B.maxFlash = (nb.usable || {}).flash_kb ?? ''; } TS.render(); } },
      groups.map((g) => h('optgroup', { label: g.label }, g.items.map((b) => h('option', { value: b.id, text: b.name || b.id })))));
    sel.value = S.boardId;
    if (B.maxFlash === '' || B.maxFlash === null || B.maxFlash === undefined) { const cur = boards.find((b) => b.id === S.boardId); B.maxFlash = ((cur || {}).usable || {}).flash_kb ?? ''; }
    return sel;
  }

  function modelPicker(avail, selected, onChange, { slowOff = false } = {}) {
    const meta = ((S.catalog && S.catalog.classical) || {}).meta || {};
    const options = avail.map((m) => ({ value: m, label: m, group: FAM()[(meta[m] || {}).family] || '', note: (meta[m] || {}).note || '', badges: [meta[m] ? (meta[m].exportable_c ? 'C' : t('no C', 'sin C')) : null, (meta[m] || {}).needs_scaling ? t('scale', 'escalar') : null].filter(Boolean) }));
    const rec = avail.filter((m) => (slowOff ? !['gradient_boosting', 'svm', 'svr'].includes(m) : DEFAULT_CLASSIC.includes(m)));
    return ui.multiSelect({ options, selected, onChange, id: 'model-picker', placeholder: t('Choose models…', 'Elige modelos…'),
      presets: [{ label: t('Recommended', 'Recomendados'), values: rec }, { label: t('Exportable to C', 'Exportables a C'), values: avail.filter((m) => (meta[m] || {}).exportable_c) }, { label: t('All', 'Todos'), values: avail }, { label: t('None', 'Ninguno'), values: [] }] });
  }

  const featureCols = () => (S.report ? S.report.columns.filter((c) => (S.recipe.roles[c.name] || c.role) === 'feature') : []);
  const numericCols = () => featureCols().filter((c) => ['numeric_continuous', 'numeric_discrete', 'binary'].includes(c.kind));

  // ------------------------------------------------------------------ 1. Dataset
  async function inspectPaths(paths, box) {
    box.replaceChildren(h('p', { class: 'muted', text: t('Analyzing the files…', 'Analizando los archivos…') }));
    try {
      const rep = await TS.api('POST', '/inspect', { paths });
      S.paths = paths;
      S.report = rep;
      S.recipe = JSON.parse(JSON.stringify(rep.recipe));
      S.clean = S.train = S.export = S.tuned = S.rejectBelow = null;
      document.getElementById('btn-save-recipe').disabled = false;
      TS.render();
    } catch (e) {
      box.replaceChildren(err(e));
    }
  }

  // ---- can these files be used together? (same structure merges; different domains are offered apart) ----
  function compatCard(rep, box) {
    const c = rep.compatibility;
    if (!c || c.verdict === 'single') return null;
    const kind = { same: 'ok', similar: 'warn', incompatible: 'bad' }[c.verdict];
    const sets = c.verdict === 'incompatible'
      ? h('div', { class: 'stack' }, ...c.sets.map((g, i) => h('div', { class: 'row wrap' },
        h('span', { text: `${t('Group', 'Grupo')} ${i + 1}: ${g.files.join(', ')}` }),
        h('button', { type: 'button', class: 'ghost', 'data-set': String(i), text: t('Use only this group', 'Usar solo este grupo'), onclick: () => inspectPaths(g.paths, box) }))))
      : null;
    return callout(kind, h('b', { text: t('Do these files belong together?', '¿Estos archivos van juntos?') }), ` ${c.advice}`, sets);
  }

  function viewDataset(root) {
    const box = h('div', { id: 'dataset-box' });
    const pick = (mode) => async () => { const r = await window.tinyml.pickDataset(mode); if (!r.canceled) inspectPaths(r.paths, box); };
    const rep0 = S.report;
    const recentBox = h('div', { class: 'row wrap', id: 'recent-box', 'data-tight': '1' });
    TS.add(root, ui.page(t('Dataset', 'Dataset'), t('Open one or more CSV files (or a folder). They may have different columns: The app understands them and proposes how to organize them.', 'Abre uno o varios CSV (o una carpeta). Pueden tener columnas distintas: La app los entiende y propone cómo organizarlos.'),
      h('div', { class: rep0 ? 'row wrap' : 'drop' },
        rep0 ? null : h('p', { text: t('Supports InfluxDB CSV exports, decimal comma and latin-1 encoding.', 'Admite CSV exportados de InfluxDB, coma decimal y codificación latin-1.') }),
        h('div', { class: 'row wrap', 'data-center': rep0 ? null : '1' },
          h('button', { id: 'btn-pick-files', type: 'button', class: rep0 ? 'ghost' : '', text: rep0 ? t('Change CSV files…', 'Cambiar archivos CSV…') : t('Open CSV…', 'Abrir CSV…'), onclick: pick('files') }),
          h('button', { id: 'btn-pick-folder', type: 'button', class: 'ghost', text: t('Open folder…', 'Abrir carpeta…'), onclick: pick('folder') })),
        recentBox),
      box));
    if (window.tinyml.recentDatasets) {
      window.tinyml.recentDatasets().then((list) => {
        const ok = list.filter((x) => x.exists).slice(0, 3);
        if (!ok.length) return;
        recentBox.replaceChildren(h('span', { class: 'muted', text: `${t('Recent: ', 'Recientes: ')}` }), ...ok.map((x) => h('button', { type: 'button', class: 'ghost sm', 'data-recent': x.id, title: x.label, text: x.label.length > 34 ? `${x.label.slice(0, 32)}…` : x.label, onclick: async () => { const r = await window.tinyml.openRecentDataset(x.id); if (!r.canceled) inspectPaths(r.paths, box); } })));
      }).catch(() => {});
    }
    const rep = S.report;
    if (!rep) return;
    const high = rep.alerts.filter((a) => a.severity === 'high').length;
    const review = rep.column_groups.filter((g) => new Set(Object.values(g.members)).size > 1);
    box.replaceChildren(
      h('div', { id: 'inspect-done' }),
      ui.stats(
        ui.stat(rep.files.length, t('files', 'archivos')), ui.stat(TS.fmt.int(rep.rows_total), t('rows', 'filas')), ui.stat(rep.columns.length, t('unified columns', 'columnas unificadas')),
        ui.stat(rep.alerts.length, t('alerts', 'alertas'), high ? 'bad' : rep.alerts.length ? 'warn' : 'ok'), ui.stat(review.length, t('merges to review', 'uniones por revisar'), review.length ? 'warn' : 'ok')),
      high ? callout('warn', t('{n} high-severity alerts: Review them below; the cleaning step (3) already proposes fixes.', '{n} alertas de severidad alta: Revísalas abajo; el paso de limpieza (3) ya propone arreglos.', { n: high })) : null,
      compatCard(rep, box),
      ui.collapse(t('Files', 'Archivos'), ui.pagedTable([t('File', 'Archivo'), t('Rows', 'Filas'), t('Columns', 'Columnas'), t('Encoding', 'Codificación'), t('Separator', 'Separador')], rep.files.map((f) => [f.file, TS.fmt.int(f.rows), f.cols, f.encoding, JSON.stringify(f.separator)]), { pageSize: 6 }), rep.files.length <= 6, rep.files.length),
      review.length
        ? ui.collapse(t('Columns merged across files (review)', 'Columnas unidas entre archivos (revisa)'), ui.pagedTable([t('Unified column', 'Columna unificada'), t('Original names', 'Nombres originales'), t('Reason', 'Motivo')], review.map((g) => [g.canonical, [...new Set(Object.values(g.members))].join(', '), g.reason]), { pageSize: 6 }), true, review.length)
        : h('p', { class: 'muted', text: t('All columns matched by exact name across files.', 'Todas las columnas coincidieron por nombre exacto entre archivos.') }),
      ui.section(t('Alerts', 'Alertas'), alertsView(rep.alerts)),
      ui.section(t('Columns', 'Columnas'), ui.pagedTable([t('Column', 'Columna'), t('Type', 'Tipo'), t('Suggested role', 'Rol sugerido'), t('Nulls', 'Nulos'), t('Coverage', 'Cobertura')], rep.columns.map((c) => [
        h('span', { title: c.kind_reason, text: c.name }), h('span', { class: 'tag', text: ui.kindLabel(c.kind) }), h('span', { title: c.role_reason, text: ui.roleLabel(c.role) }),
        fmt.pct((c.stats.nulls_pct || 0) / 100), fmt.pct(c.coverage, 1)]), { pageSize: 10, rowText: (r) => r.map((c) => (c && c.nodeType ? c.textContent : String(c))).join(' ') })),
    );
  }

  function alertsView(alerts) {
    if (!alerts.length) return h('p', { class: 'muted', text: t('No alerts.', 'Sin alertas.') });
    const by = { high: [], medium: [], info: [] };
    alerts.forEach((a) => by[a.severity].push(a));
    return h('div', { class: 'alerts' }, ['high', 'medium', 'info'].filter((s) => by[s].length).map((s) =>
      ui.collapse(ui.sevLabel(s), ui.pagedTable([t('Severity', 'Severidad'), t('Message', 'Mensaje')], by[s].map((a) => [h('span', { class: `tag ${s}`, text: ui.sevLabel(s) }), a.message]), { pageSize: 6 }), s === 'high', by[s].length)));
  }

  // ------------------------------------------------------------------ 2. Tarea
  const ROLES = () => ['feature', 'target', 'group', 'time', 'meta', 'drop'].map((r) => [r, ui.roleLabel(r)]);

  // Changing a role keeps `recipe.target` consistent: exactly one dependent variable.
  function setRole(name, v) {
    const r = S.recipe;
    const prev = r.roles[name] || ((S.report.columns.find((c) => c.name === name) || {}).role);
    r.roles[name] = v;
    if (v === 'target') {
      if (r.target && r.target !== name && r.roles[r.target] === 'target') r.roles[r.target] = 'feature';
      r.target = name;
      if (r.label_from_file) r.label_from_file.enabled = false;
    } else if (prev === 'target' && r.target === name) {
      r.target = null;
    }
    r.drop = Object.entries(r.roles).filter(([k, x]) => x === 'drop' && k !== 'source').map(([k]) => k);
    S.cleanStale = true;
    S.train = null;
    TS.render();
  }
  // "Only these columns": tick the inputs you want; every other usable column becomes "unused" (metadata).
  function pickColumnsPanel() {
    const r = S.recipe;
    const usable = S.report.columns.filter((c) => {
      const cur = r.roles[c.name] || c.role;
      return !['target', 'time', 'group'].includes(cur) && ['numeric_continuous', 'numeric_discrete', 'binary', 'categorical'].includes(c.kind);
    });
    const chosen = new Set(usable.filter((c) => (r.roles[c.name] || c.role) === 'feature').map((c) => c.name));
    const boxes = usable.map((c) => h('label', { class: 'chk', 'data-col': c.name },
      h('input', { type: 'checkbox', checked: chosen.has(c.name), onchange: (e) => { e.target.checked ? chosen.add(c.name) : chosen.delete(c.name); } }),
      h('span', { text: ` ${c.name}` })));
    const setAll = (on) => { for (const c of usable) { if (on) chosen.add(c.name); else chosen.delete(c.name); } for (const b of boxes) b.querySelector('input').checked = on; };
    const apply = h('button', { type: 'button', id: 'roles-only-apply', text: t('Use only the ticked columns', 'Usar solo las columnas marcadas'), onclick: () => {
      if (!chosen.size) return;
      for (const c of usable) r.roles[c.name] = chosen.has(c.name) ? 'feature' : 'meta';
      r.drop = Object.entries(r.roles).filter(([k, x]) => x === 'drop' && k !== 'source').map(([k]) => k);
      S.cleanStale = true; S.train = null;
      TS.render();
    } });
    return ui.collapse(t('Choose exactly which columns to use…', 'Elegir exactamente qué columnas usar…'),
      h('div', {}, h('p', { class: 'muted', text: t('Tick the columns the model should read; all the others are set to "unused". The target is not affected.', 'Marca las columnas que debe leer el modelo; todas las demás pasan a "no se usa". El objetivo no se toca.') }),
        h('div', { class: 'row', 'data-tight': '1' },
          h('button', { type: 'button', class: 'ghost sm', id: 'cols-select-all', text: t('Select all', 'Seleccionar todo'), onclick: () => setAll(true) }),
          h('button', { type: 'button', class: 'ghost sm', id: 'cols-select-none', text: t('Deselect all', 'Deseleccionar todo'), onclick: () => setAll(false) })),
        h('div', { class: 'cols-pick' }, boxes), h('div', { class: 'row' }, apply)), false);
  }
  function bulkRoles(kind) {
    const r = S.recipe;
    for (const c of S.report.columns) {
      const cur = r.roles[c.name] || c.role;
      if (cur === 'target' || cur === 'time' || cur === 'group') continue;
      if (kind === 'all' && ['numeric_continuous', 'numeric_discrete', 'binary', 'categorical'].includes(c.kind)) r.roles[c.name] = 'feature';
      if (kind === 'numeric') r.roles[c.name] = ['numeric_continuous', 'numeric_discrete', 'binary'].includes(c.kind) ? 'feature' : (cur === 'feature' ? 'meta' : cur);
      if (kind === 'reset') r.roles[c.name] = c.role;
    }
    r.drop = Object.entries(r.roles).filter(([k, x]) => x === 'drop' && k !== 'source').map(([k]) => k);
    S.cleanStale = true; S.train = null;
    TS.render();
  }

  function viewTask(root) {
    const rep = S.report;
    if (!rep) return TS.add(root, ui.page(t('Task', 'Tarea'), '', needData()));
    const r = S.recipe;
    const task = rep.task;
    const taskSel = select([['classification', t('Classification (pick a category)', 'Clasificación (elegir una categoría)')], ['regression', t('Regression (predict a number)', 'Regresión (predecir un número)')], ['unsupervised', t('No label (not trainable yet)', 'Sin etiqueta (no entrenable aún)')]], r.task, (v) => { r.task = v; TS.render(); });
    const targets = rep.columns.map((c) => [c.name, `${c.name}${(r.roles[c.name] || c.role) === 'drop' ? ` (${t('currently discarded', 'ahora descartada')})` : ''}`]);
    const lff = r.label_from_file.enabled;
    TS.add(root, ui.page(t('Task', 'Tarea'), t('Tell the app what to predict and what each column is. Everything here is a suggestion you can change.', 'Dile a la app qué predecir y qué es cada columna. Todo aquí es una sugerencia que puedes cambiar.'),
      S.transferNotes && S.transferNotes.length ? callout('info', h('b', { text: t('Configuration reused from another run. Please review:', 'Configuración reutilizada de otra corrida. Revísala:') }), h('ul', {}, S.transferNotes.slice(0, 12).map((n) => h('li', { text: n })))) : null,
      callout(task.needs_user_confirmation ? 'warn' : 'ok', h('b', { text: t('Suggestion: ', 'Sugerencia: ') }), task.reason, task.needs_user_confirmation ? t(' — confirm or correct it.', ' — confírmalo o corrígelo.') : ''),
      h('div', { class: 'grid2' },
        field(t('Task type', 'Tipo de tarea'), taskSel),
        field(t('Label source', 'Origen de la etiqueta'), h('label', { class: 'chip' }, h('input', { type: 'checkbox', id: 'lff', checked: lff, onchange: (e) => { r.label_from_file.enabled = e.target.checked; if (e.target.checked) r.target = null; TS.render(); } }), ` ${t('The label is the file name', 'La etiqueta es el nombre del archivo')}`), t('Useful when each file is one class (e.g. one appliance).', 'Útil cuando cada archivo es una clase (p. ej. un aparato).')),
        lff ? null : field(t('Dependent variable (target)', 'Variable dependiente (objetivo)'), select([['', t('(none)', '(ninguna)')], ...targets], r.target || '', (v) => { if (v) setRole(v, 'target'); else { if (r.target) setRole(r.target, 'feature'); r.target = null; TS.render(); } }),
          rep.target_candidates.length ? t('Candidates: {c}', 'Candidatas: {c}', { c: rep.target_candidates.map((c) => `${c.column} (${c.score})`).join(', ') }) : t('No clear candidate.', 'No hay candidata clara.'))),
      lff ? ui.collapse(t('Labels per file (editable)', 'Etiquetas por archivo (editables)'), ui.pagedTable([t('File', 'Archivo'), t('Label', 'Etiqueta')], Object.entries(r.label_from_file.mapping).map(([f, l]) => [f, textInput(l, (v) => { r.label_from_file.mapping[f] = v; })]), { pageSize: 8, rowText: (row) => String(row[0]) }), true, Object.keys(r.label_from_file.mapping).length) : null,
      ui.section(t('Variables: Independent (inputs) and dependent (target)', 'Variables: Independientes (entradas) y dependiente (objetivo)'),
        ui.stats(
          ui.stat(rep.columns.filter((c) => (r.roles[c.name] || c.role) === 'feature').length, t('independent variables (X)', 'variables independientes (X)'), 'ok'),
          ui.stat(lff ? t('file name', 'nombre del archivo') : (r.target || '—'), t('dependent variable (y)', 'variable dependiente (y)'), r.target || lff ? 'ok' : 'warn'),
          ui.stat(rep.columns.filter((c) => ['drop', 'meta'].includes(r.roles[c.name] || c.role)).length, t('not used', 'sin usar'))),
        h('p', { class: 'muted', text: t('Pick which columns the model reads (independent, X) and the one it predicts (dependent, y) in the table. One dependent variable per model.', 'Elige en la tabla qué columnas lee el modelo (independientes, X) y cuál predice (dependiente, y). Una variable dependiente por modelo.') }),
        h('div', { class: 'row wrap', 'data-tight': '1' },
          h('button', { type: 'button', class: 'ghost sm', id: 'roles-all', text: t('Use all usable columns as inputs', 'Usar todas las columnas útiles como entradas'), onclick: () => bulkRoles('all') }),
          h('button', { type: 'button', class: 'ghost sm', id: 'roles-numeric', text: t('Numeric columns only', 'Solo columnas numéricas'), onclick: () => bulkRoles('numeric') }),
          h('button', { type: 'button', class: 'ghost sm', id: 'roles-reset', text: t('Back to suggestions', 'Volver a las sugerencias'), onclick: () => bulkRoles('reset') })),
        pickColumnsPanel(),
        h('p', { class: 'muted', text: t('Hover a name to see why that role was suggested. Changing roles affects the suggested cleaning (step 3).', 'Pasa el ratón por el nombre para ver por qué se sugirió ese rol. Cambiar roles afecta a la limpieza sugerida (paso 3).') }),
        ui.pagedTable([t('Column', 'Columna'), t('Type', 'Tipo'), t('Role', 'Rol')], rep.columns.map((c) => [
          h('span', { title: c.role_reason, text: c.name }), h('span', { class: 'tag', text: ui.kindLabel(c.kind) }),
          select(ROLES(), S.recipe.roles[c.name] || c.role, (v) => setRole(c.name, v))]), { pageSize: 10, key: 'task-roles', rowText: (row) => `${row[0].textContent} ${row[1].textContent}` })),
      encodingCard(),
      r.task === 'unsupervised' && !lff ? callout('warn', t('Without a target column there is nothing to train with supervised models. Pick a target or enable "the label is the file name".', 'Sin columna objetivo no hay nada que entrenar con modelos supervisados. Elige un objetivo o activa "la etiqueta es el nombre del archivo".')) : null));
  }

  // Categorical encoding (recipe.encoding): one-hot / dummy (k-1) / ordinal, per default and per column.
  function encodingCard() {
    const r = S.recipe;
    r.encoding = r.encoding || { default: 'onehot', overrides: {}, max_categories: 20 };
    const e = r.encoding;
    e.overrides = e.overrides || {};
    const cats = S.report.columns.filter((c) => c.kind === 'categorical' && (r.roles[c.name] || c.role) === 'feature');
    const M = [['onehot', t('one-hot (a 0/1 column per category)', 'one-hot (una columna 0/1 por categoría)')], ['dummy', t('dummy (k−1 columns, first category is the reference)', 'dummy (k−1 columnas, la primera categoría es la referencia)')], ['ordinal', t('ordinal (one column with the category index)', 'ordinal (una columna con el índice de la categoría)')]];
    if (!cats.length) return h('div', { id: 'encoding-card' }, ui.section(t('Categorical encoding', 'Codificación de categóricas'), h('p', { class: 'muted', text: t('No categorical features detected: Nothing to encode.', 'No hay features categóricas: No hay nada que codificar.') })));
    return h('div', { id: 'encoding-card', class: 'sec' },
      h('h3', { text: t('Categorical encoding', 'Codificación de categóricas') }),
      h('p', { class: 'muted', text: t('Models need numbers. One-hot is the safe default. Dummy drops one column to avoid perfect collinearity in linear models. Ordinal is compact but imposes an order: Fine for trees, risky for linear/KNN/SVM/MLP. Fitted on train only; reproduced in the C code.', 'Los modelos necesitan números. One-hot es lo seguro. Dummy elimina una columna para evitar colinealidad perfecta en modelos lineales. Ordinal es compacto pero impone un orden: Bien para árboles, riesgoso para lineales/KNN/SVM/MLP. Se ajusta solo con train y se reproduce en el código C.') }),
      h('div', { class: 'grid2' },
        field(t('Default method', 'Método por defecto'), select(M, e.default, (v) => { e.default = v; S.train = null; })),
        field(t('Max. categories per column', 'Máx. categorías por columna'), numInput(e.max_categories, (v) => { e.max_categories = v || 20; }, { min: 2, max: 200 }))),
      ui.pagedTable([t('Column', 'Columna'), t('Categories', 'Categorías'), t('Method', 'Método')], cats.map((c) => [c.name, (c.stats && c.stats.unique) ?? '—',
        select([['', t('(default)', '(por defecto)')], ...M], e.overrides[c.name] || '', (v) => { if (v) e.overrides[c.name] = v; else delete e.overrides[c.name]; S.train = null; })]), { pageSize: 6, rowText: (row) => String(row[0]) }));
  }

  // ------------------------------------------------------------------ 3. Limpieza
  // Scaling without centering for the chosen columns: a negative value stays negative (e.g. reactive power).
  function keepSignPanel(p) {
    const cols = p.columns || [];
    const keep = new Set(p.keep_sign || []);
    const sync = () => { p.keep_sign = cols.filter((c) => keep.has(c)); };
    const info = Object.fromEntries(S.report.columns.map((c) => [c.name, c]));
    const negative = cols.filter((c) => ((info[c] || {}).stats || {}).min < 0);
    const boxes = cols.map((c) => h('label', { class: 'chk' },
      h('input', { type: 'checkbox', checked: keep.has(c), onchange: (e) => { e.target.checked ? keep.add(c) : keep.delete(c); sync(); } }),
      h('span', { text: ` ${c}${negative.includes(c) ? ' (−)' : ''}` })));
    const setSet = (list) => { keep.clear(); list.forEach((c) => keep.add(c)); sync(); boxes.forEach((b, i) => { b.querySelector('input').checked = keep.has(cols[i]); }); };
    return h('div', { class: 'keep-sign', id: 'keep-sign' },
      h('b', { text: t('Keep the sign (do not center)', 'Conservar el signo (no centrar)') }),
      h('p', { class: 'muted', text: t('Ticked columns are only divided by their spread, never shifted by the mean, so a negative value stays negative (e.g. reactive power). Columns marked (−) have negative values.', 'Las columnas marcadas solo se dividen por su dispersión, sin restarles la media, así un valor negativo sigue siendo negativo (p. ej. potencia reactiva). Las marcadas con (−) tienen valores negativos.') }),
      h('div', { class: 'row', 'data-tight': '1' },
        h('button', { type: 'button', class: 'ghost sm', id: 'sign-negative', text: t('Only columns with negatives', 'Solo las que tienen negativos'), onclick: () => setSet(negative) }),
        h('button', { type: 'button', class: 'ghost sm', id: 'sign-all', text: t('Select all', 'Seleccionar todo'), onclick: () => setSet(cols) }),
        h('button', { type: 'button', class: 'ghost sm', id: 'sign-none', text: t('Deselect all', 'Deseleccionar todo'), onclick: () => setSet([]) })),
      h('div', { class: 'cols-pick' }, boxes));
  }

  function stepCard(step, idx) {
    const p = step.params;
    const cols = p.columns || [];
    const extra = [];
    if (step.kind === 'impute') extra.push(field(t('Strategy', 'Estrategia'), select([['median', t('median', 'mediana')], ['mean', t('mean', 'media')], ['mode', t('mode', 'moda')], ['ffill', t('forward-fill', 'rellenar con el valor anterior')]], p.strategy, (v) => { p.strategy = v; })));
    if (step.kind === 'scale') extra.push(field(t('Method', 'Método'), select([['standard', t('z-score (mean 0, std 1)', 'z-score (media 0, desv. 1)')], ['minmax', 'min-max (0–1)'], ['robust', t('robust (median / IQR)', 'robusto (mediana / IQR)')]], p.method, (v) => { p.method = v; })));
    if (step.kind === 'scale') extra.push(keepSignPanel(p));
    if (step.kind === 'absurd_to_nan') extra.push(field(t('Factor (× q01–q99 range)', 'Factor (× rango q01–q99)'), numInput(p.factor, (v) => { p.factor = v || 20; }), null, t('Values farther than this many times the central range become "missing".', 'Los valores más lejos que tantas veces el rango central pasan a "faltante".')));
    return h('div', { class: 'card' },
      h('div', { class: 'row', 'data-tight': '1' }, h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: !step._off, 'aria-label': ui.stepLabel(step.kind), onchange: (e) => { step._off = !e.target.checked; } }), h('b', { text: ui.stepLabel(step.kind) })), h('span', { class: 'spacer' }), badge(`#${idx + 1}`)),
      h('p', { class: 'muted', text: step.why }),
      cols.length ? h('p', { class: 'cols', title: cols.join(', '), text: `${t('Columns', 'Columnas')} (${cols.length}): ${cols.slice(0, 6).join(', ')}${cols.length > 6 ? ` +${cols.length - 6}` : ''}` }) : null,
      ...extra);
  }

  const activeRecipe = () => {
    const r = JSON.parse(JSON.stringify(S.recipe));
    r.cleaning = (r.cleaning || []).filter((s) => !s._off).map((s) => { const c = { ...s }; delete c._off; return c; });
    return r;
  };

  async function runClean(box, btn) {
    btn.disabled = true;
    const prog = TS.progressPanel();
    box.replaceChildren(prog.el);
    try {
      const { job, result } = await TS.runJob('clean', { recipe: activeRecipe(), seed: 42 }, prog.update);
      if (job.state !== 'succeeded') throw new Error(job.error || job.state);
      S.clean = result;
      TS.render();
    } catch (e) {
      box.replaceChildren(err(e));
    } finally {
      btn.disabled = false;
    }
  }

  function viewClean(root) {
    if (!S.report) return TS.add(root, ui.page(t('Cleaning', 'Limpieza'), '', needData()));
    const out = h('div', { id: 'clean-out' });
    const btn = h('button', { id: 'run-clean', type: 'button', text: t('Apply and see before/after', 'Aplicar y ver antes/después'), onclick: () => runClean(out, btn) });
    const regen = h('button', { type: 'button', class: 'ghost', text: t('Regenerate suggestion', 'Regenerar sugerencia'), onclick: async () => {
      const columns = S.report.columns.map((c) => ({ ...c, role: S.recipe.roles[c.name] || c.role }));
      const r = await TS.api('POST', '/suggest-cleaning', { columns: columns.filter((c) => c.role !== 'meta'), alerts: S.report.alerts, multi_file: S.report.files.length > 1 });
      S.recipe.cleaning = r.steps; S.cleanStale = false; TS.render();
    } });
    const steps = S.recipe.cleaning || [];
    TS.add(root, ui.page(t('Cleaning', 'Limpieza'), t('Each step is suggested from an alert and fitted ONLY on the training set (no leakage). Untick what you do not want.', 'Cada paso se sugiere a partir de una alerta y se ajusta SOLO con el conjunto de entrenamiento (sin fuga). Desmarca lo que no quieras.'),
      S.cleanStale ? callout('warn', t('You changed roles: The suggested cleaning may be outdated. Press "Regenerate suggestion".', 'Cambiaste roles: La limpieza sugerida puede estar desactualizada. Pulsa "Regenerar sugerencia".')) : null,
      steps.length ? h('div', { class: 'cards' }, steps.map(stepCard)) : ui.empty('sparkle', t('Nothing to clean', 'Nada que limpiar'), t('No cleaning steps were suggested for this dataset.', 'No se sugirieron pasos de limpieza para este dataset.')),
      h('div', { class: 'row' }, btn, regen), out));
    if (S.clean) out.replaceChildren(cleanResults(S.clean));
  }

  function cleanResults(c) {
    return h('div', { id: 'clean-results', class: 'sec' },
      h('h3', { text: t('Before (red) vs after (blue outline) — training set', 'Antes (rojo) vs después (contorno azul) — conjunto de entrenamiento') }),
      c.units_note ? h('p', { class: 'muted', text: c.units_note }) : null,
      h('div', { class: 'hists' }, c.columns.map((col) => ui.chart(null, charts.histogram(col, c.histograms[col]), `before-after-${col}`))),
      ui.section(t('Statistics', 'Estadísticas'), ui.pagedTable([t('Column', 'Columna'), t('Nulls before→after', 'Nulos antes→después'), t('Min before→after', 'Mín antes→después'), t('Max before→after', 'Máx antes→después'), t('Mean before→after', 'Media antes→después')], c.stats.map((s) => [s.column, `${s.nulls_before}→${s.nulls_after}`, `${fmt.num(s.min_before, 2)}→${fmt.num(s.min_after, 2)}`, `${fmt.num(s.max_before, 2)}→${fmt.num(s.max_after, 2)}`, `${fmt.num(s.mean_before, 2)}→${fmt.num(s.mean_after, 2)}`]), { pageSize: 8, rowText: (r) => String(r[0]) })));
  }

  // ------------------------------------------------------------------ 4. Aumento
  // NEW rows made from the real training rows (not to be confused with filling missing values, step 3).
  // a tiny worked example so nobody has to guess what the two methods do
  function augExample() {
    return ui.collapse(t('Example: What do the two methods do?', 'Ejemplo: ¿Qué hacen los dos métodos?'), h('div', {},
      h('p', { text: t('You have 3 real students (study hours, score): A (10, 60), B (12, 66), C (30, 90).', 'Tienes 3 estudiantes reales (horas de estudio, nota): A (10, 60), B (12, 66), C (30, 90).') }),
      h('p', { text: t('Copies with a little noise: A becomes (10.3, 59.4), then (9.8, 60.5)… Each new row stays very close to a real one.', 'Copias con un poco de ruido: A pasa a (10,3; 59,4), luego a (9,8; 60,5)… Cada fila nueva queda muy cerca de una real.') }),
      h('p', { text: t('Between neighbouring rows: A and B are close, so a new student (11, 63) appears between them. C is far from both, so it is not mixed with them.', 'Entre filas vecinas: A y B están cerca, así que aparece un estudiante nuevo (11, 63) entre los dos. C está lejos de ambos, así que no se mezcla con ellos.') }),
      h('p', { class: 'muted', text: t('In both cases only the training set grows. Validation and test keep the real rows, which is why their score is the one to trust.', 'En los dos casos solo crece el entrenamiento. Validación y test conservan las filas reales, por eso su nota es la que hay que creer.') })), false);
  }

  function augRisks() {
    const items = [
      [t('New rows add no information that is not already in your data.', 'Las filas nuevas no añaden información que no estuviera ya en tus datos.')],
      [t('They can make a model look better than it is. Trust only validation and test, which stay real; cross-validation inside training becomes optimistic.', 'Pueden hacer que un modelo parezca mejor de lo que es. Fíate solo de validación y test, que siguen siendo reales; la validación cruzada dentro del entrenamiento sale optimista.')],
      [t('Noise or interpolation can create rows that could never happen in real life, for example between two very different working modes of a machine.', 'El ruido o la interpolación pueden crear filas que en la vida real no ocurrirían, por ejemplo entre dos modos de trabajo muy distintos de una máquina.')],
      [t('Outliers and mistakes in your data are copied too. Clean first (step 3).', 'Los valores atípicos y los errores de tus datos también se copian. Limpia primero (paso 3).')],
      [t('Interpolating between classes that overlap blurs the border between them. Use few neighbours and check the confusion matrix.', 'Interpolar entre clases que se solapan difumina la frontera entre ellas. Usa pocos vecinos y mira la matriz de confusión.')],
      [t('When you can, measure more real data: it is always better than creating new rows.', 'Cuando puedas, mide más datos reales: siempre es mejor que crear filas nuevas.')],
      [t('It helps most with very few rows, strongly unbalanced classes, or a model that memorizes the training set (training score much higher than validation).', 'Ayuda más con muy pocas filas, clases muy desbalanceadas o un modelo que memoriza el entrenamiento (nota de entrenamiento mucho mayor que la de validación).')],
    ];
    return ui.collapse(t('Read before using: risks and when it helps', 'Léelo antes de usarlo: riesgos y cuándo ayuda'), h('ol', { class: 'risks' }, items.map(([x]) => h('li', { text: x }))), false);
  }

  function synthCard() {
    const r = S.recipe;
    r.augment = r.augment || { method: 'off', mode: 'ratio', ratio: 1, noise_frac: 0.05, k: 5 };
    const a = r.augment;
    const cls = r.task === 'classification';
    const on = a.method && a.method !== 'off';
    const change = () => { S.cleanStale = true; S.train = null; TS.render(); };
    const method = select([
      ['off', t('Do not create new rows', 'No crear filas nuevas')],
      ['noise', t('Copies with a little noise', 'Copias con un poco de ruido')],
      ['interpolate', t('Between neighbouring rows (SMOTE style)', 'Entre filas vecinas (estilo SMOTE)')]], a.method || 'off', (v) => { a.method = v; change(); });
    method.id = 'aug-method';
    const mode = select([
      ['ratio', t('A number of new rows per real row', 'Un número de filas nuevas por cada fila real')],
      ['balance', t('Only enough to equalize the classes', 'Solo las necesarias para igualar las clases')]], a.mode || 'ratio', (v) => { a.mode = v; change(); });
    mode.id = 'aug-mode';
    return h('div', { class: 'sec', id: 'synth-card' },
      h('h3', { text: t('Create new rows from your data', 'Crear filas nuevas a partir de tus datos') }),
      h('p', { class: 'muted', text: t('Not the same as filling missing values (step 3): There the gaps of existing rows are filled; here whole NEW rows are made from the real training rows. Only the training set grows; validation and test stay real.', 'No es lo mismo que rellenar faltantes (paso 3): Allí se rellenan huecos de filas que ya existen; aquí se fabrican filas NUEVAS completas a partir de las filas reales de entrenamiento. Solo crece el entrenamiento; validación y test quedan reales.') }),
      callout('warn', h('b', { text: t('Under your responsibility: ', 'Bajo tu responsabilidad: ') }), t('Creating data is not recommended in general. If you use it, you are the one who decides that the result is valid. TinyML Studio does not check it for you.', 'Crear datos no es recomendable en general. Si lo usas, eres tú quien decide que el resultado es válido. TinyML Studio no lo comprueba por ti.')),
      augExample(),
      augRisks(),
      r.task === 'unsupervised' ? callout('info', t('Pick a classification or regression task first (step 2).', 'Primero elige una tarea de clasificación o regresión (paso 2).')) : null,
      h('div', { class: 'grid2' },
        field(t('Method', 'Método'), method, a.method === 'noise'
          ? t('Each new row is a real row plus a small random noise in every numeric column.', 'Cada fila nueva es una fila real más un pequeño ruido aleatorio en cada columna numérica.')
          : a.method === 'interpolate' ? t('Each new row lies between a real row and one of its closest neighbours of the same class (regression: The target is interpolated too).', 'Cada fila nueva queda entre una fila real y uno de sus vecinos más cercanos de la misma clase (en regresión también se interpola el objetivo).') : ''),
        on && cls ? field(t('How many', 'Cuántas'), mode) : null,
        on && (a.mode !== 'balance' || !cls) ? field(t('New rows per real row', 'Filas nuevas por fila real'), numInput(a.ratio ?? 1, (v) => { a.ratio = v ?? 1; S.cleanStale = true; }, { step: 0.25, min: 0, max: 20 })) : null,
        a.method === 'noise' ? field(t('Noise (fraction of each column deviation)', 'Ruido (fracción de la desviación de cada columna)'), numInput(a.noise_frac ?? 0.05, (v) => { a.noise_frac = v ?? 0.05; S.cleanStale = true; }, { step: 0.01, min: 0, max: 1 })) : null,
        a.method === 'interpolate' ? field(t('Neighbours to choose from', 'Vecinos entre los que elegir'), numInput(a.k ?? 5, (v) => { a.k = v || 5; S.cleanStale = true; }, { step: 1, min: 1, max: 50 })) : null),
      on ? callout('warn', t('New rows add no information that was not already in your data: They help against overfitting or unbalanced classes, they do not replace real measurements. Cross-validation inside training (step 6) becomes optimistic with them; trust validation and test, which stay real.', 'Las filas nuevas no añaden información que no estuviera ya en tus datos: Ayudan contra el sobreajuste o las clases desbalanceadas, no sustituyen medir más. La validación cruzada dentro del entrenamiento (paso 6) sale optimista con ellas; fíate de validación y test, que siguen siendo reales.')) : null);
  }

  function viewAugment(root) {
    if (!S.report) return TS.add(root, ui.page(t('Augmentation', 'Aumento'), '', needData()));
    TS.add(root, ui.page(t('Augmentation (optional)', 'Aumento (opcional)'), t('Make more training data when you have little: New rows created from your real rows. Optional, and at your own responsibility.', 'Crea más datos de entrenamiento cuando tienes pocos: Filas nuevas hechas a partir de tus filas reales. Opcional y bajo tu responsabilidad.'), synthCard()));
  }

  // ------------------------------------------------------------------ 5. Split
  function viewSplit(root) {
    if (!S.report) return TS.add(root, ui.page(t('Split', 'Partición'), '', needData()));
    const sp = S.recipe.split;
    const cols = S.report.columns.map((c) => [c.name, c.name]);
    const out = h('div', { id: 'split-out' });
    const btn = h('button', { id: 'run-split', type: 'button', text: t('Compute sizes', 'Calcular tamaños'), onclick: () => runClean(out, btn) });
    const sum = (sp.ratios || []).reduce((a, b) => a + (Number(b) || 0), 0);
    TS.add(root, ui.page(t('Split', 'Partición'), t('Separate the data into train (learn), validation (choose) and test (final grade). Wrong splits are the #1 cause of fake accuracy.', 'Separa los datos en train (aprender), validación (elegir) y test (nota final). Una mala partición es la causa #1 de precisión falsa.'),
      callout('info', h('b', { text: t('Suggestion: ', 'Sugerencia: ') }), S.report.split.reason),
      h('div', { class: 'grid2' },
        field(t('Strategy', 'Estrategia'), select([['temporal_blocks', t('temporal blocks', 'bloques temporales')], ['group', t('by group', 'por grupo')], ['stratified', t('stratified', 'estratificada')], ['random', t('random', 'aleatoria')]], sp.strategy, (v) => { sp.strategy = v; TS.render(); })),
        sp.strategy === 'temporal_blocks' || sp.strategy === 'group' ? field(t('Group column', 'Columna de grupo'), select([['', t('(none)', '(ninguna)')], ...(S.report.files.length > 1 || sp.group_column === 'source' ? [['source', t('file of origin (source)', 'archivo de origen (source)')]] : []), ...cols], sp.group_column || '', (v) => { sp.group_column = v || null; })) : null,
        sp.strategy === 'temporal_blocks' ? field(t('Time column', 'Columna de tiempo'), select([['', t('(pick one)', '(elige una)')], ...cols], sp.time_column || '', (v) => { sp.time_column = v || null; })) : null,
        field(t('Proportions train / val / test', 'Proporciones train / val / test'), h('div', { class: 'row', 'data-tight': '1' }, [0, 1, 2].map((i) => numInput(sp.ratios[i], (v) => { sp.ratios[i] = v ?? 0; }, { step: 0.05, min: 0, max: 1 }))), Math.abs(sum - 1) > 0.001 ? t('Must add up to 1 (now {s}).', 'Deben sumar 1 (ahora {s}).', { s: sum.toFixed(2) }) : t('Must add up to 1.', 'Deben sumar 1.'))),
      balanceCard(sp),
      sp.strategy === 'random' || sp.strategy === 'stratified' ? callout('warn', t('With very similar consecutive rows (time series) splitting at random leaks information from train to test. Prefer temporal blocks or by group.', 'Con filas consecutivas muy parecidas (series de tiempo) partir al azar filtra información de train a test. Prefiere bloques temporales o por grupo.')) : null,
      h('div', { class: 'row' }, btn), out));
    if (S.clean) out.replaceChildren(splitResults(S.clean));
  }

  // Class balance: the big class is cut down to the size of the small one (never the other way round).
  function balanceCard(sp) {
    if (S.recipe.task !== 'classification') return null;
    const imb = (S.report.alerts || []).find((a) => a.code === 'imbalance');
    const sel = select([
      ['off', t('Do not balance (keep every row)', 'No equilibrar (conservar todas las filas)')],
      ['train', t('Balance train only (recommended)', 'Equilibrar solo train (recomendado)')],
      ['all', t('Balance train, validation and test', 'Equilibrar train, validación y test')]], sp.balance || 'off', (v) => { sp.balance = v; S.cleanStale = true; S.train = null; TS.render(); });
    sel.id = 'balance-mode';
    return h('div', { class: 'sec', id: 'balance-card' },
      imb ? callout('warn', h('b', { text: t('Unbalanced classes: ', 'Clases desbalanceadas: ') }), imb.message) : null,
      field(t('Balance the classes', 'Equilibrar las clases'), sel,
        t('The class with more rows is cut down to the size of the smallest one (rows spread evenly over time); the small class is never repeated or invented. "Train only" keeps validation and test with the real proportion, so the final grade is honest; use F1 macro or balanced accuracy.', 'La clase con más filas se recorta al tamaño de la más pequeña (filas repartidas de forma pareja en el tiempo); la pequeña nunca se repite ni se inventa. "Solo train" deja validación y test con la proporción real, así la nota final es honesta; usa F1 macro o exactitud balanceada.')));
  }

  function splitResults(c) {
    const s = c.split;
    const tot = Object.values(s.sizes).reduce((a, b) => a + b, 0) || 1;
    return h('div', { id: 'split-results', class: 'sec' },
      ui.stats(...Object.entries(s.sizes).map(([k, v]) => ui.stat(TS.fmt.int(v), `${k} · ${(100 * v / tot).toFixed(0)} %`))),
      h('table', { class: 'tbl', hidden: true }, h('tbody', {}, Object.entries(s.sizes).map(([k, v]) => h('tr', {}, h('td', { text: k }), h('td', { text: v }))))),
      Object.keys(s.class_distribution).length
        ? h('div', { class: 'grid3' }, Object.entries(s.class_distribution).map(([k, d]) => ui.chart(null, charts.bars(d, t('Classes in {k}', 'Clases en {k}', { k })), `classes-${k}`)))
        : null);
  }

  // ------------------------------------------------------------------ 6. Modelo y entrenamiento
  const noKeras = () => Boolean(S.catalog && S.catalog.environment && S.catalog.environment.keras === false);
  function families() {
    const all = [['benchmark', t('Benchmark: Compare many models', 'Banco de pruebas: Comparar muchos modelos')], ['classical', t('Classical models (scikit-learn)', 'Modelos clásicos (scikit-learn)')], ['keras_mlp', 'MLP (Keras)'], ['keras_cnn1d', t('1D-CNN (Keras) on windows', '1D-CNN (Keras) sobre ventanas')]];
    // lite build (no TensorFlow/Keras): only offer what the core can actually run
    return noKeras() ? all.filter(([id]) => !/keras/.test(id)) : all;
  }

  // optimizer / schedule / backend controls shared by the Keras models
  function optimizerFields(hp) {
    const k = S.catalog.keras_mlp || {};
    const opts = k.optimizers || [{ id: 'adam' }];
    hp.optimizer = hp.optimizer || 'adam'; hp.lr_schedule = hp.lr_schedule || 'constant';
    const cur = opts.find((o) => o.id === hp.optimizer) || {};
    const sch = (k.lr_schedules || []).find((o) => o.id === hp.lr_schedule) || {};
    const backends = ((k.compute || {}).backends) || [{ id: 'tensorflow' }];
    return h('div', { class: 'sec', id: 'optimizer-box' },
      h('div', { class: 'grid2' },
        field(t('Optimizer', 'Optimizador'), select(opts.map((o) => [o.id, o.id]), hp.optimizer, (v) => { hp.optimizer = v; TS.render(); }), cur.note),
        (cur.params || []).includes('weight_decay') ? field(t('Weight decay', 'Decaimiento de pesos'), numInput(hp.weight_decay ?? 0.004, (v) => { hp.weight_decay = v ?? 0; }, { step: 0.001, min: 0 })) : null,
        (cur.params || []).includes('nesterov') ? field('Nesterov', h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: Boolean(hp.nesterov), onchange: (e) => { hp.nesterov = e.target.checked; } }), h('span', { text: t('Nesterov momentum', 'Momento de Nesterov') }))) : null,
        field(t('Learning-rate schedule', 'Programación de la tasa'), select((k.lr_schedules || [{ id: 'constant' }]).map((o) => [o.id, o.id]), hp.lr_schedule, (v) => { hp.lr_schedule = v; TS.render(); }), sch.note),
        (sch.params || []).includes('lr_final_frac') ? field(t('Final rate (fraction)', 'Tasa final (fracción)'), numInput(hp.lr_final_frac ?? 0.01, (v) => { hp.lr_final_frac = v || 0.01; }, { step: 0.01, min: 0.0001, max: 1 })) : null,
        field(t('Training backend', 'Backend de entrenamiento'), select(backends.map((b) => [b.id, b.id === 'torch' ? 'PyTorch (CUDA / ROCm)' : 'TensorFlow']), S.backend, (v) => { S.backend = v; TS.render(); }), (backends.find((b) => b.id === S.backend) || {}).note)));
  }

  function hyper(root) {
    const d = (S.catalog.keras_mlp || {}).defaults || {};
    const c = (S.catalog.keras_cnn1d || {}).defaults || {};
    const hp = S.hp;
    if (S.family === 'classical') {
      const avail = (S.recipe.task === 'regression' ? S.catalog.classical.models.regression : S.catalog.classical.models.classification);
      if (!S.classicModels.length || !S.classicModels.every((m) => avail.includes(m))) S.classicModels = avail.filter((m) => ['decision_tree', 'random_forest', 'logreg', 'linear', 'svm', 'mlp'].includes(m));
      TS.add(root, field(t('Models to train', 'Modelos a entrenar'), modelPicker(avail, S.classicModels, (v) => { S.classicModels = v; }),
        t('"C" = can be exported to standalone C for the microcontroller. "scale" = distance/gradient based: Add the scaling step in Cleaning.', '"C" = se puede exportar a C autónomo para el microcontrolador. "escalar" = depende de distancias/gradientes: Añade el paso de escalado en Limpieza.')));
    } else if (S.family === 'benchmark') {
      const avail = S.recipe.task === 'regression' ? S.catalog.classical.models.regression : S.catalog.classical.models.classification;
      const B = S.bench;
      if (!B.models || !B.models.every((m) => avail.includes(m))) B.models = avail.filter((m) => !['gradient_boosting', 'svm', 'svr'].includes(m));
      const presets = noKeras() ? [] : ((S.catalog.keras_mlp || {}).presets) || [];
      TS.add(root,
        h('p', { class: 'muted', text: t('Trains every selected model on the SAME splits and ranks them. The test set never chooses the winner.', 'Entrena todos los modelos elegidos con las MISMAS particiones y los ordena. El test nunca elige al ganador.') }),
        h('div', { class: 'grid2' },
          field(t('Classical models', 'Modelos clásicos'), modelPicker(avail, B.models, (v) => { B.models = v; }, { slowOff: true }), t('Slow ones (svm, gradient_boosting) are off by default; add them if you can wait.', 'Los lentos (svm, gradient_boosting) vienen apagados; agrégalos si puedes esperar.')),
          field(t('Dense networks (Keras MLP)', 'Redes densas (MLP Keras)'), ui.multiSelect({ id: 'preset-picker', placeholder: t('Choose networks…', 'Elige redes…'), selected: B.presets, onChange: (v) => { B.presets = v; },
            options: presets.map((p) => ({ value: p.id, label: p.name, note: `[${(p.params.hidden || []).join(', ')}]${p.params.dropout ? ` · dropout ${p.params.dropout}` : ''}` })),
            presets: [{ label: t('None', 'Ninguna'), values: [] }, { label: t('All', 'Todas'), values: presets.map((p) => p.id) }] }))),
        h('div', { class: 'grid2' },
          field(t('Cross-validation on train (folds)', 'Validación cruzada en train (pliegues)'), select([[0, t('off', 'apagada')], [3, '3'], [5, '5']], B.cv, (v) => { B.cv = Number(v); }), t('Slower but gives a spread (±) per model. By group/blocks when the split is by group/time.', 'Más lenta pero da dispersión (±) por modelo. Por grupo/bloques si la partición es por grupo/tiempo.')),
          field(t('Maximum epochs for the networks', 'Épocas máximas de las redes'), numInput(B.epochs ?? 60, (v) => { B.epochs = v || 60; }, { min: 1, max: 2000 }), t('Upper limit only: Training stops earlier on its own (see the next box).', 'Solo es un tope: El entrenamiento se detiene antes por sí solo (mira el recuadro siguiente).')),
          field(t('Stop early after N epochs without improvement', 'Parar antes tras N épocas sin mejora'), numInput(B.patience ?? 8, (v) => { B.patience = v || 8; }, { min: 1, max: 200 }), t('If the validation loss does not improve for this many epochs, training stops and the best epoch is kept. So 100 maximum epochs never wastes time when epoch 33 was already the best.', 'Si la pérdida de validación no mejora en tantas épocas, el entrenamiento se detiene y se conserva la mejor época. Así, 100 épocas máximas no desperdician tiempo si la época 33 ya fue la mejor.')),
          field(t('Optimizer for the networks', 'Optimizador de las redes'), select(((S.catalog.keras_mlp || {}).optimizers || [{ id: 'adam' }]).map((o) => [o.id, o.id]), B.optimizer || 'adam', (v) => { B.optimizer = v; }), t('Applied to every selected network.', 'Se aplica a todas las redes elegidas.')),
          field(t('Selection metric', 'Métrica de selección'), select([['f1_macro', 'F1 macro'], ['balanced_accuracy', t('balanced accuracy', 'exactitud balanceada')], ['mcc', 'MCC'], ['accuracy', t('accuracy', 'exactitud')], ['r2', 'R² (' + t('regression', 'regresión') + ')']], B.selection, (v) => { B.selection = v; })),
          field(t('Target board', 'Placa destino'), benchBoardSelect(B), t('Models bigger than the board budget are flagged “does not fit” and never recommended.', 'Los modelos más grandes que el presupuesto de la placa se marcan «no cabe» y nunca se recomiendan.')),
          field(t('Flash budget for the model (KB)', 'Presupuesto de flash del modelo (KB)'), numInput(B.maxFlash, (v) => { B.maxFlash = v; }), t('Filled from the board; edit it if your firmware needs more room.', 'Se rellena con la placa; edítalo si tu firmware necesita más espacio.'))));
    } else if (S.family === 'keras_mlp') {
      Object.assign(hp, { hidden: d.hidden, epochs: d.epochs, batch_size: d.batch_size, learning_rate: d.learning_rate, patience: d.patience, ...hp });
      TS.add(root, h('div', { class: 'grid2' },
        field(t('Hidden layers (neurons, comma)', 'Capas ocultas (neuronas, coma)'), textInput((hp.hidden || []).join(','), (v) => { hp.hidden = v.split(',').map(Number).filter((x) => x > 0); })),
        field(t('Activation', 'Activación'), select(((S.catalog.keras_mlp || {}).activations || ['relu']).map((a) => [a, a]), hp.activation || 'relu', (v) => { hp.activation = v; })),
        field('Dropout', numInput(hp.dropout ?? 0, (v) => { hp.dropout = v || 0; }, { step: 0.05, min: 0, max: 0.5 })),
        field('L2', numInput(hp.l2 ?? 0, (v) => { hp.l2 = v || 0; }, { step: 0.0001, min: 0 })),
        field(t('Epochs (max.)', 'Épocas (máx.)'), numInput(hp.epochs, (v) => { hp.epochs = v; })), field('Batch size', numInput(hp.batch_size, (v) => { hp.batch_size = v; })),
        field('Learning rate', numInput(hp.learning_rate, (v) => { hp.learning_rate = v; }, { step: 0.0005 })), field(t('Patience (early stopping)', 'Paciencia (parada temprana)'), numInput(hp.patience, (v) => { hp.patience = v; }))),
        optimizerFields(hp));
    } else {
      Object.assign(hp, { window: c.window, stride: c.stride, filters: c.filters, kernel: c.kernel, head: c.head, epochs: c.epochs, batch_size: c.batch_size, learning_rate: c.learning_rate, patience: c.patience, ...hp });
      TS.add(root, h('div', { class: 'grid2' },
        field(t('Window (rows)', 'Ventana (filas)'), numInput(hp.window, (v) => { hp.window = v; })), field(t('Stride between windows (train)', 'Paso entre ventanas (train)'), numInput(hp.stride, (v) => { hp.stride = v; })),
        field(t('Filters per layer (comma)', 'Filtros por capa (coma)'), textInput((hp.filters || []).join(','), (v) => { hp.filters = v.split(',').map(Number).filter((x) => x > 0); })), field('Kernel', numInput(hp.kernel, (v) => { hp.kernel = v; })),
        field(t('Head', 'Cabeza'), select([['gap', t('global average', 'promedio global')], ['flatten', t('flatten', 'aplanar')]], hp.head, (v) => { hp.head = v; })), field(t('Epochs (max.)', 'Épocas (máx.)'), numInput(hp.epochs, (v) => { hp.epochs = v; })),
        field('Learning rate', numInput(hp.learning_rate, (v) => { hp.learning_rate = v; }, { step: 0.0005 }))),
        callout('info', t('Needs a temporal-block split (step 5) and a time column.', 'Necesita partición por bloques temporales (paso 5) y una columna de tiempo.')),
        optimizerFields(hp));
    }
  }

  async function runTrain(out, btn) {
    btn.disabled = true;
    S.train = null;
    S.benchPick = null;
    const prog = TS.progressPanel();
    const curve = { loss: [], val: [] };
    const chart = h('div', { id: 'live-curve' });
    const hw = TS.hwPanel();
    S.lastHw = hw;
    out.replaceChildren(prog.el, chart, hw.el);
    const base = { recipe: activeRecipe(), seed: 42, device: S.device };
    let kind;
    let params;
    const hp = S.hp;
    if (S.family === 'benchmark') {
      const B = S.bench;
      const pre = ((S.catalog.keras_mlp || {}).presets || []).filter((p) => B.presets.includes(p.id)).map((p) => ({ name: p.id, ...p.params, epochs: B.epochs || 60, patience: B.patience || 8, ...(B.optimizer && B.optimizer !== 'adam' ? { optimizer: B.optimizer } : {}) }));
      kind = 'benchmark';
      params = { ...base, models: B.models, keras: pre, cv_folds: B.cv, selection: B.selection, max_flash_kb: B.maxFlash || null };
      delete params.device;
    }
    else if (S.family === 'classical') { kind = 'train'; params = { ...base, models: S.classicModels, params: S.tuned || null }; }
    else if (S.family === 'keras_mlp') { kind = 'train_keras'; params = { ...base, hidden: hp.hidden, epochs: hp.epochs, batch_size: hp.batch_size, learning_rate: hp.learning_rate, patience: hp.patience, activation: hp.activation || 'relu', dropout: hp.dropout || 0, l2: hp.l2 || 0, optimizer: hp.optimizer || 'adam', lr_schedule: hp.lr_schedule || 'constant', weight_decay: hp.weight_decay ?? 0.004, nesterov: Boolean(hp.nesterov), lr_final_frac: hp.lr_final_frac ?? 0.01, backend: S.backend }; }
    else if (S.family === 'keras_cnn1d') { kind = 'train_cnn1d'; params = { ...base, window: hp.window, stride: hp.stride, filters: hp.filters, kernel: hp.kernel, head: hp.head, epochs: hp.epochs, batch_size: hp.batch_size, learning_rate: hp.learning_rate, patience: hp.patience, optimizer: hp.optimizer || 'adam', lr_schedule: hp.lr_schedule || 'constant', weight_decay: hp.weight_decay ?? 0.004, nesterov: Boolean(hp.nesterov), lr_final_frac: hp.lr_final_frac ?? 0.01, backend: S.backend }; }
    try {
      const { job, result } = await TS.runJob(kind, params, (ev) => {
        prog.update(ev);
        if (TS.currentJob) hw.start(TS.currentJob);
        if (ev.stage === 'epoch') {
          curve.loss.push(ev.loss); curve.val.push(ev.val_loss);
          chart.replaceChildren(charts.lines(t('Loss per epoch', 'Pérdida por época'), [{ name: 'train', values: curve.loss }, { name: t('validation', 'validación'), values: curve.val }]));
        }
      });
      await hw.stop();
      if (job.state !== 'succeeded') throw new Error(job.error || job.state);
      prog.done(t('Training finished', 'Entrenamiento terminado'));
      S.train = { job, result, kind };
      S.export = null;
      S.savedMsg = '';
      TS.render();
    } catch (e) {
      out.replaceChildren(err(e));
    } finally {
      btn.disabled = false;
    }
  }

  // GPU support check (a few seconds): which framework can see a GPU, AMD detection and what to install
  function gpuPanel() {
    const box = h('div', { id: 'gpu-panel' });
    const draw = () => {
      const d = S.gpuProbe;
      if (!d) { box.replaceChildren(h('button', { type: 'button', class: 'ghost sm', id: 'btn-gpu-check', text: t('Check GPU support', 'Comprobar soporte de GPU'), onclick: async () => {
        box.replaceChildren(h('p', { class: 'muted', text: t('Probing frameworks (a few seconds)…', 'Probando frameworks (unos segundos)…') }));
        try { S.gpuProbe = await TS.api('GET', '/hardware/frameworks'); } catch (e) { S.gpuProbe = { error: String(e.message || e) }; }
        draw();
      } })); return; }
      if (d.error) { box.replaceChildren(TS.err(new Error(d.error))); return; }
      const fw = d.frameworks || {};
      const seen = (x) => (x && (x.gpu_visible || (x.devices || []).length)) ? t('sees a GPU', 've una GPU') : t('no GPU', 'sin GPU');
      const rows = [
        [t('NVIDIA GPUs', 'GPUs NVIDIA'), (d.gpus || []).filter((g) => !g.vendor || g.vendor === 'NVIDIA').map((g) => g.name).join(', ') || '—'],
        [t('AMD GPUs', 'GPUs AMD'), (d.amd_gpus || []).map((g) => `${g.name}${g.vram_mb ? ` · ${Math.round(g.vram_mb / 1024)} GB` : ''}`).join(', ') || '—'],
        ['ROCm', d.rocm_present ? t('present', 'presente') : t('not found', 'no encontrado')],
        ['TensorFlow', fw.tensorflow ? seen(fw.tensorflow) : '—'], ['PyTorch', fw.torch ? seen(fw.torch) : '—'],
      ];
      box.replaceChildren(ui.collapse(t('GPU support in this computer', 'Soporte de GPU en este computador'), h('div', {},
        TS.table([t('Item', 'Elemento'), t('Result', 'Resultado')], rows),
        (d.guidance || []).length ? ui.collapse(t('How to enable GPU training', 'Cómo activar el entrenamiento con GPU'), h('ul', { class: 'warns' }, (d.guidance || []).map((g) => h('li', {}, g.text || String(g), g.url ? h('div', {}, h('code', { text: g.url })) : null))), false, d.guidance.length) : null,
        h('p', { class: 'muted', text: t('Verified by the developer: CPU training and PyTorch on an NVIDIA GPU (CUDA, RTX 4050, Windows 11). Untested on real hardware: AMD ROCm and TensorFlow with GPU (WSL2). See docs/GPU_AMD.md.', 'Verificado por el desarrollador: Entrenamiento en CPU y con PyTorch en una GPU NVIDIA (CUDA, RTX 4050, Windows 11). Sin probar en hardware real: AMD ROCm y TensorFlow con GPU (WSL2). Ver docs/GPU_AMD.md.') })), true));
    };
    draw();
    return box;
  }

  function viewTrain(root) {
    if (!S.report) return TS.add(root, needData());
    if (!S.catalog) return TS.add(root, ui.pageHead(t('Model & training', 'Modelo y entrenamiento')), h('p', { class: 'muted', text: t('Loading catalog…', 'Cargando catálogo…') }));
    const fams = families();
    if (!fams.some(([v]) => v === S.family)) S.family = fams[0][0];
    const out = h('div', { id: 'train-out' });
    const btn = h('button', { id: 'run-train', type: 'button', text: t('Train', 'Entrenar'), onclick: () => runTrain(out, btn) });
    const hw = S.hardware;
    TS.add(root, 
      ui.pageHead(t('Model & training', 'Modelo y entrenamiento'), t('Train one model or benchmark many on identical splits, then inspect the final quality.', 'Entrena un modelo o compara muchos con particiones idénticas, y revisa la calidad final.')),
      field(t('Model type', 'Tipo de modelo'), select(fams, S.family, (v) => { S.family = v; S.train = null; TS.render(); })),
      noKeras() ? h('p', { class: 'muted', text: t('Neural networks (Keras/TensorFlow) are not installed in this environment. Classical models and C export work. To use Keras, install the "Neural networks" component in Tools > Environment (about 2 GB).', 'Las redes neuronales (Keras/TensorFlow) no están instaladas en este entorno. Los modelos clásicos y la exportación a C funcionan. Para usar Keras, instala el componente «Redes neuronales» en Herramientas > Entorno (unos 2 GB).') }) : null,
      h('div', { id: 'hp-box' }),
      field(t('Compute device', 'Dispositivo de cómputo'), select([['auto', t('automatic', 'automático')], ['cpu', t('force CPU', 'forzar CPU')], ['gpu', t('use the GPU if available (NVIDIA or AMD)', 'usar la GPU si hay (NVIDIA o AMD)')]], S.device, (v) => { S.device = v; }),
        hw ? `${hw.cpu.name} · ${hw.cpu.logical_cores} ${t('threads', 'hilos')}${hw.ram_gb ? ` · ${TS.fmt.dec(hw.ram_gb, 1)} GB RAM` : ''} · GPU: ${hw.gpus.length ? hw.gpus.map((g) => g.name).join(', ') : t('not detected', 'no detectada')}. ${hw.device_note}` : ''),
      gpuPanel(),
      h('div', { class: 'row' }, btn), out,
      S.recipe.task === 'classification' ? TS.groupValidationCard({ recipe: activeRecipe, model: () => (S.family === 'classical' && S.classicModels && S.classicModels[0]) || 'random_forest', markable: (cols) => (cols || []).filter((c) => (S.recipe.roles[c] || '') === 'feature'), onMark: (cols) => { for (const c of cols) if (S.recipe.roles[c] === 'feature') S.recipe.roles[c] = 'meta'; S.cleanStale = true; S.train = null; const old = document.getElementById('train-results'); if (old) old.hidden = true; /* the trained model used those columns */ } }) : null,
      S.recipe.task === 'classification' ? TS.anomalyCard({ recipe: activeRecipe, classes: () => Object.keys((((S.clean || {}).split || {}).class_distribution || {}).train || {}) }) : null,
      S.family === 'classical' ? TS.tuneCard({ recipe: activeRecipe, model: () => (S.classicModels && S.classicModels[0]) || 'random_forest', budget: () => ({ flash: S.flashKb, ram: S.ramKb }), tuned: () => (S.tuned && Object.keys(S.tuned).length ? S.tuned : null), onUse: (m, p) => { S.tuned = { ...(S.tuned || {}), [m]: p }; if (!(S.classicModels || []).includes(m)) S.classicModels = [...(S.classicModels || []), m]; TS.render(); } }) : null);
    hyper(root.querySelector('#hp-box'));
    if (S.train) out.replaceChildren(trainResults(S.train));
  }

  function modelSelect(names, best, on) {
    const el = select(names.map((n) => [n, n === best ? `${n} ← ${t('best', 'mejor')}` : n]), best, on);
    el.id = 'quality-model';
    return h('label', { class: 'field' }, h('span', { text: t('Model to inspect', 'Modelo a inspeccionar') }), el);
  }

  function trainResults(tr) {
    const r = tr.result.results;
    const box = h('div', { id: 'train-results' }, callout('ok', t('Training finished. The test set was not used to choose the model.', 'Entrenamiento terminado. El test no se usó para elegir el modelo.')));
    const fam = tr.result.family;
    if (fam === 'benchmark') {
      const bx = h('div', { id: 'train-results' }, callout('ok', t('Benchmark finished.', 'Banco de pruebas terminado.')), TS.benchmarkPanel(r));
      TS.add(bx, S.lastHw && S.lastHw.el);
      TS.add(bx, h('details', {}, h('summary', { text: t('Full summary', 'Resumen completo') }), h('pre', { text: tr.result.summary || tr.result.summary_es })));
      return bx;
    }
    const q = h('div', { id: 'quality-box' });
    const showQ = (quality, history, classes, kind) => q.replaceChildren(h('h3', { text: t('Final quality', 'Calidad final') }), TS.qualityPanel({ quality, history, classes, kind }));
    const bestName = r.selection && r.selection.best;
    if (fam === 'classical') {
      const cls = r.task === 'classification';
      const key = cls ? 'f1_macro' : 'r2'; const mname = cls ? 'F1 macro' : 'R²';
      TS.add(box, table([t('Model', 'Modelo'), `${mname} val`, `${mname} test`, cls ? t('accuracy test', 'exactitud test') : 'MAE test', t('Fit (s)', 'Ajuste (s)'), t('Size', 'Tamaño')], Object.entries(r.models).map(([n, m]) => [n === bestName ? h('b', { text: `${n} ← ${t('best', 'mejor')}` }) : n, fmt.num(m.val[key]), fmt.num(m.test[key]), fmt.num(cls ? m.test.accuracy : m.test.mae), fmt.num(m.fit_seconds, 2), Object.entries(m.complexity).map(([k, v]) => `${k}=${v}`).join(' ')])));
      TS.add(box, h('p', { class: 'muted', text: t('Baseline ({key}): Test {v}', 'Línea base ({key}): Test {v}', { key: mname, v: fmt.num(r.baseline.test[key]) }) }));
      TS.add(box, modelSelect(Object.keys(r.models), bestName, (n) => showQ(r.models[n].quality, null, r.classes, cls ? 'classification' : 'regression')));
      showQ(r.models[bestName].quality, null, r.classes, cls ? 'classification' : 'regression');
      const best = r.models[bestName];
      if (best.top_features && best.top_features.length) TS.add(q, h('p', { text: t('Most influential features: {f}', 'Features más influyentes: {f}', { f: best.top_features.slice(0, 6).map((f) => `${f.feature} (${f.importance})`).join(', ') }) }));
    } else {
      const m = r.test;
      TS.add(box, table([t('Metric', 'Métrica'), 'val', 'test'], [['F1 macro', fmt.num(r.val.f1_macro), fmt.num(m.f1_macro)], [m.accuracy !== undefined ? t('Accuracy', 'Exactitud') : t('Exact accuracy', 'Exactitud exacta'), fmt.num(r.val.accuracy ?? r.val.subset_accuracy), fmt.num(m.accuracy ?? m.subset_accuracy)]]));
      if (r.reference_rf_last_row) TS.add(box, h('p', { text: t('Reference: Random forest with only the last row → test macro F1 {v}', 'Referencia: Random forest con solo la última fila → F1 macro test {v}', { v: fmt.num(r.reference_rf_last_row.test.f1_macro) }) }));
      showQ(r.quality, r.history, r.classes, r.task === 'regression' ? 'regression' : 'classification');
    }
    TS.add(box, q);
    const warns = r.warnings || [];
    if (warns.length) TS.add(box, h('h3', { text: t('Warnings (read them)', 'Advertencias (léelas)') }), h('ul', { class: 'warns' }, warns.map((x) => h('li', { text: x }))));
    TS.add(box, S.lastHw && S.lastHw.el);
    TS.add(box, h('details', {}, h('summary', { text: t('Full summary', 'Resumen completo') }), h('pre', { text: tr.result.summary || tr.result.summary_es })));
    return box;
  }

  // ------------------------------------------------------------------ 7. Board and export
  const benchRows = () => (S.train && S.train.result.family === 'benchmark' ? S.train.result.results.leaderboard.filter((r) => r.exportable) : []);
  const benchRow = () => {
    const rows = benchRows();
    const want = S.benchPick || S.train.result.results.selection.selected;
    return rows.find((r) => r.name === want) || rows[0] || null;
  };
  function exportKind() {
    if (S.train && S.train.result.family === 'benchmark') { const r = benchRow(); return r ? r.export_job : 'export_c'; }
    return S.train && ['classical'].includes(S.train.result.family) ? 'export_c' : 'export_tflite';
  }

  async function runExport(out, btn) {
    btn.disabled = true;
    const prog = TS.progressPanel();
    out.replaceChildren(prog.el);
    const kind = exportKind();
    const params = { run_job: S.train.job.id, board: S.boardId, flash_kb: S.flashKb ? Number(S.flashKb) : null, ram_kb: S.ramKb ? Number(S.ramKb) : null, device: S.device === 'cpu' ? 'cpu' : 'auto' }; // export never uses the GPU
    if (kind === 'export_tflite') params.int8_io = S.int8io;
    if (kind === 'export_c') { params.model = S.exportModel || null; params.compact = Boolean(S.compact); }
    if (S.train.result.family === 'benchmark') { const r = benchRow(); params.run_subdir = r.run_subdir; if (kind === 'export_c') params.model = r.name; }
    try {
      const { job, result } = await TS.runJob(kind, params, prog.update);
      if (job.state !== 'succeeded') throw new Error(job.error || job.state);
      prog.done(t('Export finished', 'Exportación terminada'));
      S.savedMsg = '';
      S.export = { job, result, kind };
      TS.render();
    } catch (e) {
      out.replaceChildren(err(e));
    } finally {
      btn.disabled = false;
    }
  }

  // ---- board picker: grouped dropdown, variants, verified spec card, auto-filled budgets ---------------
  const TIER = () => ({ none: [t('Too small for AI', 'Muy pequeña para IA'), 'bad'], classical_only: [t('Classical models only', 'Solo modelos clásicos'), 'warn'], tflm_small: [t('Small neural nets (TFLM)', 'Redes pequeñas (TFLM)'), 'ok'], tflm_medium: [t('Neural nets with headroom (TFLM)', 'Redes con margen (TFLM)'), 'ok'], linux_runtime: [t('Runs Linux AI runtimes', 'Corre runtimes de IA de Linux'), 'ok'] });
  const kb = (v) => (v === null || v === undefined ? '—' : v >= 1048576 ? `${TS.fmt.dec(v / 1048576, v % 1048576 ? 1 : 0)} GB` : v >= 1024 ? `${TS.fmt.dec(v / 1024, v % 1024 ? 1 : 0)} MB` : `${v} KB`);

  function applyBoardDefaults(b) {
    const u = b.usable || {};
    S.variantId = (b.variants && b.variants[0] && b.variants[0].id) || null;
    S.flashKb = u.flash_kb ?? '';
    S.ramKb = u.ram_kb ?? '';
  }

  function boardSection(bd) {
    const boards = bd ? bd.boards : [];
    const board = boards.find((b) => b.id === S.boardId) || boards[0];
    if (!board) return h('p', { class: 'muted', text: '—' });
    const groups = [];
    for (const b of boards) {
      const g = b.family || b.vendor || '';
      let x = groups.find((y) => y.label === g);
      if (!x) { x = { label: g, items: [] }; groups.push(x); }
      x.items.push(b);
    }
    const sel = h('select', { id: 'board-select', onchange: (e) => { S.boardId = e.target.value; const nb = boards.find((b) => b.id === S.boardId); if (nb) applyBoardDefaults(nb); TS.render(); } },
      groups.map((g) => h('optgroup', { label: g.label }, g.items.map((b) => h('option', { value: b.id, text: `${b.name || b.id}${b.verified ? '' : ` · ${t('unverified', 'sin verificar')}`}` })))));
    sel.value = board.id;
    const vars = board.variants || [];
    const vsel = vars.length ? h('select', { id: 'variant-select', onchange: (e) => { S.variantId = e.target.value; TS.render(); } }, vars.map((v) => h('option', { value: v.id, text: v.name || v.id }))) : null;
    if (vsel) vsel.value = S.variantId || vars[0].id;
    const v = vars.find((x) => x.id === (S.variantId || (vars[0] || {}).id)) || null;
    const [tierText, tierTone] = TIER()[board.ai_tier] || [board.ai_tier || '—', 'info'];
    const ram = board.ram_kb ? (v && v.ram_kb ? v.ram_kb : board.ram_kb) : null;
    const spec = [
      [t('Processor', 'Procesador'), [board.soc, board.cpu].filter(Boolean).join(' · ') || '—'],
      [t('Cores · clock', 'Núcleos · reloj'), `${board.cores ?? '—'} · ${board.clock_mhz ? `${board.clock_mhz} MHz` : '—'}`],
      [board.ram_kb ? 'RAM' : t('On-chip RAM (SRAM)', 'RAM del chip (SRAM)'), kb(ram ?? board.sram_kb)],
      [t('Flash', 'Flash'), kb((v && v.flash_kb) ?? board.flash_kb)],
      [t('External RAM (PSRAM/SDRAM)', 'RAM externa (PSRAM/SDRAM)'), v && v.psram_kb ? kb(v.psram_kb) : board.psram_kb_max ? `${t('up to', 'hasta')} ${kb(board.psram_kb_max)}` : '—'],
      [t('Accelerators', 'Aceleradores'), (board.accelerators || []).join(', ') || '—'],
      [t('Runtimes', 'Runtimes'), (board.runtimes || []).map((x) => t(({ 'codigo C (clasicos)': 'C code (classical)', 'codigo C': 'C code' })[String(x).normalize('NFD').replace(/[\u0300-\u036f]/g, '')] || x, x)).join(', ') || '—'],
    ];
    const ev = board.evidence || [];
    const u = board.usable || {};
    // the board note can be a wall of text: show the first sentences (about 140 characters) and fold the rest
    const note = (() => {
      const s = board.ai_note || '';
      for (let i = s.indexOf('. '); i !== -1; i = s.indexOf('. ', i + 1)) {
        if (i + 1 >= 140 && 'ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚ0123456789'.includes(s[i + 2])) return { short: s.slice(0, i + 1), more: s.slice(i + 2) };
      }
      return { short: s, more: '' };
    })();
    return h('div', { id: 'board-section', class: 'sec' },
      h('div', { class: 'grid2' },
        field(t('Board', 'Placa'), sel, t('Grouped by family. Picking a board fills the flash and RAM budgets below.', 'Agrupadas por familia. Al elegir una placa se rellenan los presupuestos de flash y RAM de abajo.')),
        vsel ? field(t('Version / module', 'Versión / módulo'), vsel, t('Memory differs by version: Pick yours.', 'La memoria cambia con la versión: Elige la tuya.')) : null),
      h('div', { class: 'card board-card' },
        h('div', { class: 'row', 'data-tight': '1' }, h('b', { text: board.name || board.id }), h('span', { class: `tag ${tierTone === 'bad' ? 'high' : tierTone === 'warn' ? 'medium' : ''}`, text: tierText }),
          board.verified && board.verification === 'datasheet_read'
            ? h('span', { class: 'badge ok', title: t('Numbers read from the official datasheet (version and page in the sources).', 'Números leídos del datasheet oficial (versión y página en las fuentes).'), text: t('Datasheet read', 'Datasheet leído') })
            : board.verified ? h('span', { class: 'badge ok', title: t('Numbers confirmed by official vendor pages (via search). Not a full datasheet read.', 'Números confirmados por páginas oficiales del fabricante (vía búsqueda). No es una lectura completa del datasheet.'), text: t('Official source', 'Fuente oficial') }) : h('span', { class: 'badge', text: t('Unverified', 'Sin verificar') })),
        note.short ? h('p', { class: 'muted', text: note.short }) : null,
        h('dl', { class: 'specs' }, spec.flatMap(([k, val]) => [h('dt', { text: k }), h('dd', { text: val })])),
        note.more ? ui.collapse(t('More about this board', 'Más sobre esta placa'), h('p', { class: 'muted', text: note.more }), false) : null,
        ev.length ? ui.collapse(t('Sources ({n})', 'Fuentes ({n})', { n: ev.length }), h('ul', { class: 'src' }, ev.map((e) => h('li', {}, h('b', { text: `${e.field}: ${e.value}` }), e.note ? h('div', { class: 'muted', text: e.note }) : null, h('code', { text: e.url })))), false) : null,
        board.retrieved ? h('small', { class: 'muted', text: t('Checked {d}', 'Consultado {d}', { d: board.retrieved }) }) : null),
      board.verified && board.verification === 'datasheet_read'
        ? h('p', { class: 'muted', text: t('The chip numbers were read from the official datasheet (version and page in the sources). Budgets are our suggestion, not a datasheet value.', 'Las cifras del chip se leyeron del datasheet oficial (versión y página en las fuentes). Los presupuestos son nuestra sugerencia, no un valor del datasheet.') })
        : board.verified ? h('p', { class: 'muted', text: t('Numbers come from official vendor pages, confirmed through search (not a full datasheet read): Open the linked datasheet before a production decision. Budgets are our suggestion, not a datasheet value.', 'Los números vienen de páginas oficiales del fabricante, confirmadas por búsqueda (no es una lectura completa del datasheet): Abre el datasheet enlazado antes de decidir para producción. Los presupuestos son nuestra sugerencia, no un valor del datasheet.') }) : null,
      board.ai_tier === 'linux_runtime' ? callout('info', t('This board runs Linux: Copy the .tflite file (or ONNX) and use LiteRT / ONNX Runtime. The C and Arduino code are optional here.', 'Esta placa corre Linux: Copia el archivo .tflite (u ONNX) y usa LiteRT / ONNX Runtime. El código C y Arduino es opcional aquí.')) : null,
      board.ai_tier === 'none' ? callout('bad', t('This board cannot hold a useful model. Pick another board or a much smaller model.', 'Esta placa no puede alojar un modelo útil. Elige otra placa o un modelo mucho más pequeño.')) : null,
      h('div', { class: 'grid2' },
        field(t('Flash available for the model (KB)', 'Flash disponible para el modelo (KB)'), numInput(S.flashKb, (x) => { S.flashKb = x; }), u.flash_kb ? t('Suggested: {n} KB (our heuristic)', 'Sugerido: {n} KB ({r})', { n: u.flash_kb, r: u.rule || t('our heuristic', 'heurística nuestra') }) : t('Not a limit on this board (storage is external).', 'No es un límite en esta placa (el almacenamiento es externo).')),
        field(t('RAM available for the model (KB)', 'RAM disponible para el modelo (KB)'), numInput(S.ramKb, (x) => { S.ramKb = x; }), u.ram_kb ? t('Suggested: {n} KB (our heuristic)', 'Sugerido: {n} KB (heurística nuestra)', { n: u.ram_kb }) : null)),
      u.flash_kb || u.ram_kb ? h('button', { type: 'button', class: 'ghost sm', id: 'use-suggested', text: t('Use suggested budgets', 'Usar presupuestos sugeridos'), onclick: () => { S.flashKb = u.flash_kb ?? ''; S.ramKb = u.ram_kb ?? ''; TS.render(); } }) : null);
  }

  const modelUsed = () => {
    const f = S.train.result.family;
    if (f === 'benchmark') return (benchRow() || {}).name || '—';
    if (f === 'classical') return S.exportModel || S.train.result.results.selection.best;
    return f === 'keras_cnn1d' ? '1D-CNN (Keras)' : 'MLP (Keras)';
  };
  // the "try it" card needs a classical model run (the same cleaning and model the exported C uses)
  const tryParams = () => {
    const f = S.train.result.family;
    if (f === 'classical') return { run: () => ({ run_job: S.train.job.id }), model: () => S.exportModel || null };
    const r = f === 'benchmark' ? benchRow() : null;
    if (r && r.export_job === 'export_c') return { run: () => ({ run_job: S.train.job.id, run_subdir: benchRow().run_subdir }), model: () => benchRow().name };
    return null;
  };

  function viewExport(root) {
    if (!S.train) return TS.add(root, ui.pageHead(t('Board & export', 'Placa y exportación')), ui.empty('brain', t('Train a model first', 'Primero entrena un modelo'), t('Go to step 6, train or benchmark, then come back here.', 'Ve al paso 6, entrena o compara modelos y vuelve aquí.')));
    const bd = S.boards;
    const board = bd && bd.boards.find((b) => b.id === S.boardId);
    const out = h('div', { id: 'export-out' });
    const btn = h('button', { id: 'run-export', type: 'button', text: t('Quantize and export', 'Cuantizar y exportar'), onclick: () => runExport(out, btn) });
    const kind = exportKind();
    const fam = S.train.result.family;
    TS.add(root, 
      ui.pageHead(t('Board & export', 'Placa y exportación'), t('Quantize, check that it fits your board, and get the code to flash.', 'Cuantiza, verifica que cabe en tu placa y obtén el código para grabar.')),
      bd && bd.warning && !board.verified ? callout('warn', bd.warning) : null,
      TS.usedCard({ S, modelName: modelUsed() }),
      boardSection(bd),
      h('div', { class: 'grid2' },
        kind === 'export_c' ? field(t('Compact tables (trees and forests)', 'Tablas compactas (árboles y bosques)'), h('label', { class: 'chk' }, h('input', { type: 'checkbox', id: 'compact', checked: Boolean(S.compact), onchange: (e) => { S.compact = e.target.checked; } }), h('span', { text: t('Smaller flash, same predictions', 'Menos flash, mismas predicciones') })), t('About 2.4× smaller for the default forest, bit-exact with the normal export (measured on a PC; see docs/research/TINYFOREST.md). Ignored for other models.', 'Unas 2,4× más pequeño para el bosque por defecto, idéntico bit a bit al export normal (medido en PC; ver docs/research/TINYFOREST.md). Se ignora en otros modelos.')) : null,
        kind === 'export_tflite' ? field(t('int8 input/output', 'Entrada/salida int8'), h('input', { type: 'checkbox', checked: S.int8io, onchange: (e) => { S.int8io = e.target.checked; } }), t('Float by default (what EloquentTinyML expects).', 'Por defecto float (lo que espera EloquentTinyML).')) : null,
        S.train.result.family === 'benchmark'
          ? field(t('Model to export', 'Modelo a exportar'), select(benchRows().map((r) => [r.name, `${r.name}${r.selected ? ' ★' : ''} (${r.export_job === 'export_tflite' ? 'TFLite' : 'C'})`]), (benchRow() || {}).name || '', (v) => { S.benchPick = v; S.export = null; TS.render(); }), t('Only models that have an exporter are listed.', 'Solo se listan modelos con exportador.'))
          : kind === 'export_c' ? field(t('Model to export', 'Modelo a exportar'), select([['', t('the best ({n})', 'el mejor ({n})', { n: S.train.result.results.selection.best })], ...Object.keys(S.train.result.results.models).map((m) => [m, m])], S.exportModel || '', (v) => { S.exportModel = v; })) : null),
      h('div', { class: 'row' }, btn), out,
      tryParams() ? TS.tryCard({ runParams: () => tryParams().run(), modelName: () => tryParams().model() }) : null,
      fam === 'classical' ? onnxCard() : null,
      fam === 'classical' ? TS.importanceCard({ runJob: () => S.train.job.id, model: () => S.exportModel }) : null,
      fam === 'classical' ? TS.latencyCard({ runJob: () => S.train.job.id, model: () => S.exportModel, boardId: () => S.boardId }) : null,
      fam === 'classical' && S.recipe.task === 'classification' ? TS.rejectCard({ runJob: () => S.train.job.id, model: () => S.exportModel, onUse: (thr) => { S.rejectBelow = thr; TS.render(); } }) : null);
    if (S.export) out.replaceChildren(exportResults(S.export));
  }

  // ---- ONNX export (classical models): preprocessing inside the graph, checked against scikit-learn ----
  function onnxCard() {
    const out = h('div', { id: 'onnx-out' });
    const btn = h('button', { id: 'run-onnx', type: 'button', class: 'secondary', text: t('Export ONNX', 'Exportar ONNX'), onclick: async () => {
      btn.disabled = true;
      const prog = TS.progressPanel();
      out.replaceChildren(prog.el);
      try {
        const { job, result } = await TS.runJob('export_onnx', { run_job: S.train.job.id, model: S.exportModel || null, reject_below: S.rejectBelow || null, device: 'cpu' }, prog.update);
        if (job.state !== 'succeeded') throw new Error(job.error || job.state);
        prog.done(t('ONNX ready', 'ONNX listo'));
        const v = result.results.verification;
        const save = h('p', { class: 'muted', id: 'onnx-saved' });
        out.replaceChildren(callout(v.ok ? 'ok' : 'bad', h('b', { text: v.ok ? t('Verified against scikit-learn', 'Verificado contra scikit-learn') : t('Does NOT match scikit-learn', 'NO coincide con scikit-learn') }), ` — ${result.summary}`),
          h('div', { class: 'row' }, h('button', { type: 'button', id: 'btn-save-onnx', text: t('Save files…', 'Guardar archivos…'), onclick: async () => {
            try { const r = await window.tinyml.saveArtifacts(job.id); save.textContent = r.canceled ? t('Cancelled.', 'Cancelado.') : t('Saved {n} files in {d}', 'Guardados {n} archivos en {d}', { n: r.files, d: r.dest }); } catch (x) { save.textContent = `Error: ${x.message}`; }
          } })), save);
      } catch (e) {
        out.replaceChildren(err(e));
      } finally {
        btn.disabled = false;
      }
    } });
    return h('div', { class: 'card' },
      h('h3', { text: 'ONNX' }),
      S.rejectBelow ? callout('info', t('It will answer "unknown" when the top probability is below {t}.', 'Responderá "desconocido" si la probabilidad máxima es menor que {t}.', { t: S.rejectBelow })) : null,
      h('p', { class: 'muted', text: t('Portable model for PC, server or phone (onnxruntime). The cleaning and encoding go inside the graph, so it takes raw values. Needs onnx, onnxruntime and skl2onnx.', 'Modelo portable para PC, servidor o móvil (onnxruntime). La limpieza y codificación van dentro del grafo, así que recibe valores crudos. Requiere onnx, onnxruntime y skl2onnx.') }),
      h('div', { class: 'row' }, btn), out);
  }

  function verdictCard(b, label) {
    const code = b.verdict_code || ({ cabe: 'fits', 'no cabe': 'no_fit', 'sin presupuesto': 'no_budget' }[b.verdict] || (String(b.verdict).startsWith('justo') ? 'tight' : 'no_budget'));
    const cls = { fits: 'ok', tight: 'warn', no_fit: 'bad', no_budget: 'info' }[code];
    return callout(cls, h('b', { text: `${label}: ${b.verdict_label || b.verdict}` }), b.fraction !== undefined
      ? ` — ${t('{p} of {kb} KB ({u})', '{p} de {kb} KB ({u})', { p: fmt.pct(b.fraction), kb: b.budget_kb, u: fmt.bytes(b.used_bytes) })}`
      : ` — ${t('uses {u} (set a budget above)', 'usa {u} (indica un presupuesto arriba)', { u: fmt.bytes(b.used_bytes) })}`);
  }

  // ---- headline "does it fit?" banner with concrete ways out -------------------------------------------
  function fitBanner(e) {
    const r = e.result.results;
    const b = r.budget || {};
    const parts = [['flash_data', t('Flash', 'Flash')], ['ram_working', t('RAM', 'RAM')]].map(([k, label]) => ({ k, label, v: b[k] })).filter((x) => x.v);
    const code = (v) => v.verdict_code || ({ cabe: 'fits', 'no cabe': 'no_fit' }[v.verdict] || (String(v.verdict).startsWith('justo') ? 'tight' : 'no_budget'));
    const bad = parts.filter((x) => code(x.v) === 'no_fit');
    const tight = parts.filter((x) => code(x.v) === 'tight');
    const board = ((S.boards && S.boards.boards) || []).find((x) => x.id === S.boardId);
    const name = board ? board.name || board.id : S.boardId;
    const line = (x) => `${x.label}: ${fmt.bytes(x.v.used_bytes)}${x.v.budget_kb ? ` / ${x.v.budget_kb} KB (${fmt.pct(x.v.fraction, 0)})` : ''}`;
    if (!bad.length) {
      if (tight.length) return callout('warn', h('b', { text: t('It fits {b}, but only just.', 'Cabe en {b}, pero muy justo.', { b: name }) }), ` ${tight.map(line).join(' · ')}. `, t('Your firmware (Wi-Fi, sensors, stack) also needs memory: Leave margin or pick a smaller model.', 'Tu firmware (Wi-Fi, sensores, pila) también necesita memoria: Deja margen o elige un modelo más pequeño.'));
      if (parts.every((x) => code(x.v) === 'fits')) return callout('ok', h('b', { text: t('It fits {b}.', 'Cabe en {b}.', { b: name }) }), ` ${parts.map(line).join(' · ')}.`);
      return null;
    }
    // ways out
    const needKb = (bad.find((x) => x.k === 'flash_data') || { v: { used_bytes: 0 } }).v.used_bytes / 1024;
    const needRam = (bad.find((x) => x.k === 'ram_working') || { v: { used_bytes: 0 } }).v.used_bytes / 1024;
    const budget = (b.flash_data || {}).budget_kb;
    const ideas = [];
    if (S.train && S.train.result.family === 'benchmark' && budget) {
      const fits = (S.train.result.results.leaderboard || []).filter((m) => m.exportable && m.size_bytes && m.size_bytes <= budget * 1024 * 0.8).sort((a, c) => (c.selection_score ?? 0) - (a.selection_score ?? 0)).slice(0, 3);
      if (fits.length) ideas.push(t('Smaller models from your benchmark that do fit: {m}.', 'Modelos más pequeños de tu banco de pruebas que sí caben: {m}.', { m: fits.map((m) => `${m.name} (${fmt.bytes(m.size_bytes)})`).join(', ') }));
    }
    if (e.kind === 'export_c' && !S.compact) ideas.push(t('Tick “Compact tables” (about 2.4× less flash for trees and forests, same predictions) and export again.', 'Marca «Tablas compactas» (unas 2,4× menos flash en árboles y bosques, mismas predicciones) y exporta de nuevo.'));
    if (e.kind === 'export_c') ideas.push(t('Train a shallower tree / fewer trees, or a linear model.', 'Entrena un árbol menos profundo / menos árboles, o un modelo lineal.'));
    else ideas.push(t('Use a smaller network (fewer neurons/layers); int8 weights are already used.', 'Usa una red más pequeña (menos neuronas/capas); los pesos ya son int8.'));
    const boards = ((S.boards && S.boards.boards) || []).filter((x) => x.ai_tier !== 'none' && x.id !== S.boardId && (x.ai_tier === 'linux_runtime' || ((x.usable || {}).flash_kb ?? 0) >= needKb * 1.25 && ((x.usable || {}).ram_kb ?? 0) >= needRam * 1.25)).sort((a, c) => ((a.usable || {}).flash_kb ?? 1e12) - ((c.usable || {}).flash_kb ?? 1e12)).slice(0, 5);
    if (boards.length) ideas.push(t('Boards in the catalog with enough room: {b}.', 'Placas del catálogo con espacio suficiente: {b}.', { b: boards.map((x) => x.name || x.id).join(', ') }));
    return h('div', { id: 'fit-banner' }, callout('bad', h('b', { text: t('DOES NOT FIT {b}: Do not flash this model.', 'NO CABE en {b}: No grabes este modelo.', { b: name }) }), ` ${bad.map(line).join(' · ')}. `,
      t('The firmware would fail to build or run out of memory.', 'El firmware no compilaría o se quedaría sin memoria.'),
      h('ul', {}, ideas.map((x) => h('li', { text: x })))));
  }

  function exportResults(e) {
    const r = e.result.results;
    const box = h('div', { id: 'export-results' });
    TS.add(box, fitBanner(e));
    TS.add(box, h('h3', { text: t('"Fits / does not fit" report', 'Informe "cabe / no cabe"') }), verdictCard(r.budget.flash_data, t('Flash (model tables)', 'Flash (tablas del modelo)')), verdictCard(r.budget.ram_working, t('Working RAM', 'RAM de trabajo')), h('p', { class: 'muted', text: r.budget.note }));
    if (e.kind === 'export_c') {
      const v = r.verification;
      TS.add(box, h('h3', { text: t('Python ↔ C verification (on PC)', 'Verificación Python ↔ C (en PC)') }), v.compile_ok && v.vectors
        ? table([t('Check', 'Comprobación'), t('Result', 'Resultado')], [[t('Test vectors', 'Vectores del test'), v.vectors], [t('C model == scikit-learn', 'Modelo C == scikit-learn'), fmt.pct(v.model_match, 2)], [t('Raw data → C == Python pipeline', 'Datos crudos → C == pipeline Python'), fmt.pct(v.vs_full_python_pipeline, 2)], [t('Max feature difference', 'Diferencia máx. en features'), v.features_max_rel_diff], [t('Compiles as C++', 'Compila como C++'), String(v.compiles_as_cpp)], [t('Compiler warnings', 'Avisos del compilador'), (v.compile_warnings || []).length]])
        : callout('bad', t('C verification not completed: {r}', 'Verificación en C no completada: {r}', { r: v.reason || v.compile_error || v.run_error || '' }), v.available === false ? h('p', { class: 'muted', text: t('Install a C compiler and make sure cc is in your PATH (Windows: MinGW-w64, e.g. WinLibs via winget; Linux: Gcc; macOS: Xcode command line tools). The exported C code is still valid; only this check was skipped.', 'Instala un compilador de C y comprueba que cc esté en el PATH (Windows: MinGW-w64, p. ej. WinLibs con winget; Linux: Gcc; macOS: Herramientas de línea de comandos de Xcode). El código C exportado sigue siendo válido; solo se omitió esta comprobación.') }) : null));
    } else {
      const c = r.comparison;
      TS.add(box, h('h3', { text: t('Keras ↔ float ↔ int8 ↔ TFLite Micro verification', 'Verificación Keras ↔ float ↔ int8 ↔ TFLite Micro') }),
        table([t('Model', 'Modelo'), t('F1 macro (test)', 'F1 macro (test)')], Object.entries(r.metrics_test).map(([k, m]) => [k, fmt.num(m.f1_macro ?? m.r2)])),
        h('p', { text: t('int8 loss: {d} · TFLite Micro vs tf.lite int8: {a}', 'Pérdida por int8: {d} · TFLite Micro vs tf.lite int8: {a}', { d: fmt.num(r.int8_f1_drop, 4), a: c.agreement_tflm_vs_tflite_int8 !== undefined ? fmt.pct(c.agreement_tflm_vs_tflite_int8, 2) : t('not verified', 'no verificado') }) }),
        h('p', { text: t('Sizes: Float {f} → int8 {i} · Measured arena: {a} (suggested {s})', 'Tamaños: Float {f} → int8 {i} · Arena medida: {a} (sugerida {s})', { f: fmt.bytes(r.sizes_bytes.tflite_float), i: fmt.bytes(r.sizes_bytes.tflite_int8), a: r.arena.measured ? fmt.bytes(r.arena.arena_bytes) : '—', s: r.arena_suggested ? fmt.bytes(r.arena_suggested) : '—' }) }),
        h('p', { text: t('TFLM ops to register: {o}', 'Ops de TFLM a registrar: {o}', { o: r.tflm_resolver.join(', ') }) }));
    }
    const w = r.warnings || [];
    if (w.length) TS.add(box, h('h3', { text: t('Warnings', 'Avisos') }), h('ul', { class: 'warns' }, w.map((x) => h('li', { text: x }))));
    const noFit = [r.budget && r.budget.flash_data, r.budget && r.budget.ram_working].some((x) => x && (x.verdict_code === 'no_fit' || x.verdict === 'no cabe'));
    const ard = TS.arduinoPanel(e.result.arduino, e.job.id);
    if (ard) TS.add(box, noFit ? ui.collapse(t('Arduino sketches (they will NOT fit this board)', 'Sketches de Arduino (NO caben en esta placa)'), ard, false) : ard);
    TS.add(box, h('h3', { text: t('Model structure', 'Estructura del modelo') }), TS.graphView(e.result.graph));
    TS.add(box, callout('warn', t('This was verified on a PC. It still has to be tested on the real board (the .ino and the C were not compiled for the ESP32-S3).', 'Esto se verificó en PC. Falta probar en la placa real (el .ino y el C no se han compilado para el ESP32-S3).')));
    // the message lives in the state: a re-render (e.g. catalogs reloading) must not lose it
    const saved = h('p', { id: 'saved-msg', class: 'muted', text: S.savedMsg || '' });
    TS.add(box, h('div', { class: 'row' }, h('button', { id: 'btn-save-artifacts', type: 'button', text: t('Save files…', 'Guardar archivos…'), onclick: async () => {
      try { const s = await window.tinyml.saveArtifacts(e.job.id); S.savedMsg = s.canceled ? t('Cancelled.', 'Cancelado.') : t('Saved {n} files in {d}', 'Guardados {n} archivos en {d}', { n: s.files, d: s.dest }); } catch (x) { S.savedMsg = `Error: ${x.message}`; }
      saved.textContent = S.savedMsg; const live = document.getElementById('saved-msg'); if (live) live.textContent = S.savedMsg;
    } })), saved, h('details', {}, h('summary', { text: t('Full summary', 'Resumen completo') }), h('pre', { text: e.result.summary || e.result.summary_es })));
    return box;
  }

  // After a language switch: ask the core again for the messages it produced in the old language.
  TS.refreshMessages = async function refreshMessages() {
    if (!S.report || !S.paths.length) return;
    try {
      const rep = await TS.api('POST', '/inspect', { paths: S.paths });
      S.report = rep;
      const columns = rep.columns.map((c) => ({ ...c, role: S.recipe.roles[c.name] || c.role }));
      const sug = await TS.api('POST', '/suggest-cleaning', { columns: columns.filter((c) => c.role !== 'meta'), alerts: rep.alerts, multi_file: rep.files.length > 1 });
      const cur = S.recipe.cleaning || [];
      if (sug.steps.length === cur.length && sug.steps.every((s, i) => s.kind === cur[i].kind)) sug.steps.forEach((s, i) => { cur[i].why = s.why; });
    } catch { /* keep the old texts */ }
  };

  TS.views = [viewDataset, viewTask, viewClean, viewAugment, viewSplit, viewTrain, viewExport];
  TS.err = err;
})();
