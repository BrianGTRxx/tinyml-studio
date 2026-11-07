'use strict';
// Run history: every job the core has stored, with its headline result. Reuse a configuration (same
// recipe, hyper-parameters) on the same or ANOTHER dataset, compare runs side by side.
(function () {
  const TS = window.TS;
  const { h, fmt } = TS;
  const ui = TS.ui;
  const t = TS.t;
  const verdict = (c) => ({ fits: t('fits', 'cabe'), tight: t('tight', 'justo'), no_fit: t('does not fit', 'no cabe'), no_budget: t('no budget', 'sin presupuesto') }[c] || c);
  const S = () => TS.S;

  const KINDS = () => ({
    clean: t('Cleaning', 'Limpieza'), train: t('Classical training', 'Entrenamiento clásico'), train_keras: t('Keras MLP', 'MLP Keras'), train_cnn1d: '1D-CNN',
    benchmark: t('Benchmark', 'Banco de pruebas'), export_c: t('Export C', 'Exportar C'), export_tflite: t('Export TFLite', 'Exportar TFLite'), export_onnx: t('Export ONNX', 'Exportar ONNX'),
  });
  const STATES = () => ({ succeeded: t('done', 'listo'), failed: t('failed', 'falló'), cancelled: t('cancelled', 'cancelado'), interrupted: t('interrupted', 'interrumpido') });
  const when = (iso) => { try { return new Date(typeof iso === 'number' && iso < 1e12 ? iso * 1000 : iso).toLocaleString(TS.lang === 'es' ? 'es' : 'en', { dateStyle: 'short', timeStyle: 'short' }); } catch { return iso || '—'; } };
  const isTrain = (k) => ['train', 'train_keras', 'train_cnn1d', 'benchmark'].includes(k);

  // load the recipe + hyper-parameters of a past job into the wizard
  async function reuse(job, newPaths) {
    const p = await TS.api('GET', `/jobs/${job.id}/params`);
    const st = S();
    let recipe = p.recipe;
    let report;
    st.transferNotes = null;
    if (newPaths) {
      const tr = await TS.api('POST', '/recipe/transfer', { old_recipe: p.recipe, paths: newPaths });
      report = tr.report; recipe = tr.recipe; st.transferNotes = tr.notes || [];
      st.paths = newPaths;
    } else {
      report = await TS.api('POST', '/inspect', { paths: recipe.files });
      st.paths = recipe.files;
    }
    st.report = report; st.recipe = recipe; st.clean = st.train = st.export = null;
    const keras = ['hidden', 'epochs', 'batch_size', 'learning_rate', 'patience', 'activation', 'dropout', 'l2', 'window', 'stride', 'filters', 'kernel', 'head'];
    st.hp = Object.fromEntries(keras.filter((k) => p[k] !== undefined).map((k) => [k, p[k]]));
    if (job.kind === 'train') { st.family = 'classical'; if (p.models) st.classicModels = p.models; }
    if (job.kind === 'train_keras') st.family = 'keras_mlp';
    if (job.kind === 'train_cnn1d') st.family = 'keras_cnn1d';
    if (job.kind === 'benchmark') { st.family = 'benchmark'; if (p.models) st.bench.models = p.models; st.bench.cv = p.cv_folds || 0; st.bench.selection = p.selection || 'f1_macro'; }
    document.getElementById('btn-save-recipe').disabled = false;
    st.reach = 1;
    TS.go(1);
  }

  async function compare(ids, box) {
    box.replaceChildren(h('p', { class: 'muted', text: t('Comparing…', 'Comparando…') }));
    try {
      const d = await TS.api('POST', '/history/compare', { ids });
      const keys = [['f1_macro_val', 'F1 val'], ['f1_macro_test', 'F1 test'], ['accuracy_test', t('accuracy test', 'exactitud test')], ['mcc_test', 'MCC'], ['mAP_test', 'mAP'], ['r2_test', 'R²'], ['size_bytes', t('size', 'tamaño')], ['fit_seconds', t('fit s', 'ajuste s')], ['predict_us_per_row', t('µs/row', 'µs/fila')]];
      const used = keys.filter(([k]) => d.rows.some((r) => r.metrics && r.metrics[k] !== null && r.metrics[k] !== undefined));
      box.replaceChildren(ui.section(t('Comparison', 'Comparación'),
        d.same_dataset === false ? h('div', { class: 'callout warn', text: t('These runs used different datasets: The numbers are not directly comparable.', 'Estas corridas usaron datasets distintos: Los números no son directamente comparables.') }) : null,
        d.same_recipe === false ? h('div', { class: 'callout info', text: t('The cleaning/split recipe differs between runs.', 'La receta de limpieza/partición difiere entre corridas.') }) : null,
        TS.table([t('Run', 'Corrida'), ...used.map(([, l]) => l)], d.rows.map((r) => [r.label || r.id, ...used.map(([k]) => (k === 'size_bytes' && r.metrics[k] ? fmt.bytes(r.metrics[k]) : fmt.num(r.metrics[k], 3)))])),
        (d.notes || []).map((n) => h('p', { class: 'muted', text: n }))));
    } catch (e) { box.replaceChildren(TS.err(e)); }
  }

  TS.historyView = async function historyView(root) {
    const box = h('div', { id: 'history-box' }, h('p', { class: 'muted', text: t('Loading history…', 'Cargando historial…') }));
    const cmpBox = h('div', { id: 'compare-box' });
    TS.add(root, ui.page(t('History', 'Historial'), t('Every run is kept on this computer. Reuse a configuration on the same or another dataset, or compare runs side by side.', 'Cada corrida queda guardada en este computador. Reutiliza una configuración con el mismo u otro dataset, o compara corridas lado a lado.'), box, cmpBox));
    let list;
    try { list = await TS.api('GET', '/history'); } catch (e) { box.replaceChildren(TS.err(e)); return; }
    const rows = Array.isArray(list) ? list : (list.runs || []);
    if (!rows.length) { box.replaceChildren(ui.empty('history', t('No runs yet', 'Aún no hay corridas'), t('Train or benchmark a model and it will appear here.', 'Entrena o compara un modelo y aparecerá aquí.'))); return; }
    const picked = new Set();
    const cmpBtn = h('button', { type: 'button', id: 'btn-compare', class: 'ghost', text: t('Compare selected', 'Comparar seleccionadas'), disabled: true, onclick: () => compare([...picked], cmpBox) });
    const upd = () => { cmpBtn.disabled = picked.size < 2 || picked.size > 6; cmpBtn.title = picked.size < 2 || picked.size > 6 ? t('Tick 2 to 6 runs', 'Marca de 2 a 6 corridas') : ''; };
    const ok = rows.filter((r) => r.state === 'succeeded').length;
    const table = ui.pagedTable([t('Pick', 'Elegir'), t('When', 'Cuándo'), t('Run', 'Corrida'), t('Dataset', 'Dataset'), t('Result', 'Resultado'), t('Time', 'Tiempo'), t('Actions', 'Acciones')], rows.map((r) => {
      const hl = r.headline || {};
      let res = '—';
      if (hl.metric_name === 'export') res = [hl.board, hl.flash_verdict && `flash: ${verdict(hl.flash_verdict)}`, hl.ram_verdict && `RAM: ${verdict(hl.ram_verdict)}`].filter(Boolean).join(' · ') || '—';
      else if (hl.metric_name === 'rows') res = `${fmt.int(hl.value_test)} ${t('test rows', 'filas de test')}`;
      else if (hl.value_test !== null && hl.value_test !== undefined) res = `${hl.selected_model ? `${hl.selected_model} · ` : ''}${hl.metric_name || ''} ${fmt.num(hl.value_test, 3)}`;
      else if (r.error) res = r.error.slice(0, 40);
      return [
        h('input', { type: 'checkbox', 'aria-label': t('Select for comparison', 'Seleccionar para comparar'), disabled: !isTrain(r.kind) || r.state !== 'succeeded', onchange: (e) => { if (e.target.checked) picked.add(r.id); else picked.delete(r.id); upd(); } }),
        when(r.created),
        h('span', {}, h('b', { text: KINDS()[r.kind] || r.kind }), ' ', h('span', { class: `tag ${r.state === 'succeeded' ? '' : 'high'}`, text: STATES()[r.state] || r.state })),
        (r.files || []).slice(0, 2).join(', ') + ((r.files || []).length > 2 ? ` +${r.files.length - 2}` : ''),
        res, r.duration_s !== null && r.duration_s !== undefined ? h('span', { class: 'nowrap', text: fmt.dur(r.duration_s) }) : '—',
        h('span', { class: 'row', 'data-tight': '1' },
          isTrain(r.kind) || r.kind === 'clean' ? h('button', { type: 'button', class: 'ghost sm', 'data-act': 'reuse', text: t('Reuse', 'Reutilizar'), title: t('Load this configuration', 'Cargar esta configuración'), onclick: async () => { try { await reuse(r); } catch (e) { box.prepend(TS.err(e)); } } }) : null,
          isTrain(r.kind) || r.kind === 'clean' ? h('button', { type: 'button', class: 'ghost sm', 'data-act': 'other', text: t('Other dataset…', 'Otro dataset…'), title: t('Apply this configuration to a different dataset', 'Aplicar esta configuración a un dataset distinto'), onclick: async () => { const pk = await window.tinyml.pickDataset('files'); if (pk.canceled) return; try { await reuse(r, pk.paths); } catch (e) { box.prepend(TS.err(e)); } } }) : null),
      ];
    }), { pageSize: 10, rowText: (r) => `${r[1]} ${r[2].textContent} ${r[3]} ${r[4]}` });
    box.replaceChildren(
      ui.stats(ui.stat(rows.length, t('runs', 'corridas')), ui.stat(ok, t('successful', 'exitosas'), 'ok'), ui.stat(rows.length - ok, t('failed / cancelled', 'fallidas / canceladas'), rows.length - ok ? 'warn' : '')),
      h('div', { class: 'row' }, cmpBtn), table);
    upd();
  };
})();
