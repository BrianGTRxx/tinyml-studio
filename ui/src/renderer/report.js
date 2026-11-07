'use strict';
// PDF report: builds a printable document from the current session (data, cleaning, split, models,
// quality, export, hardware, reproducibility, limits) with charts rendered as PNG, then asks the main
// process to print it (webContents.printToPDF). Nothing leaves the machine.
(function () {
  const TS = window.TS;
  const { h, fmt, charts } = TS;
  const t = TS.t;
  const S = () => TS.S;

  const hold = h('div', { id: 'chart-hold', 'aria-hidden': 'true' });
  hold.style.position = 'fixed'; hold.style.left = '-10000px'; hold.style.top = '0'; hold.style.width = '700px';

  // renders a chart off screen and returns an <img> (PNG, light theme)
  async function img(svgEl, caption) {
    hold.replaceChildren(svgEl);
    document.body.append(hold);
    const data = await TS.chartToPng(svgEl, 2);
    hold.replaceChildren();
    return h('figure', {}, h('img', { src: data, alt: caption || '' }), caption ? h('figcaption', { text: caption }) : null);
  }

  const tbl = (head, rows) => h('table', {}, h('thead', {}, h('tr', {}, head.map((x) => h('th', { text: x })))), h('tbody', {}, rows.map((r) => h('tr', {}, r.map((c) => h('td', {}, c === null || c === undefined ? '—' : c))))));
  const kpi = (v, l) => h('span', { class: 'rkpi' }, h('b', { text: v }), l);
  const sec = (title, ...kids) => h('section', { class: 'rsec' }, h('h2', { text: title }), ...kids);
  const warn = (txt) => h('div', { class: 'rwarn', text: txt });

  async function build() {
    const s = S();
    const r = h('div', { class: 'rbody' });
    const now = new Date();
    const files = s.report ? s.report.files.map((f) => f.file) : [];
    r.append(h('div', { class: 'rcover' }, h('h1', { text: t('TinyML Studio report', 'Informe de TinyML Studio') }),
      h('p', { class: 'rmeta', text: `${now.toLocaleString(TS.lang === 'es' ? 'es' : 'en')} · ${t('Dataset', 'Dataset')}: ${files.slice(0, 4).join(', ')}${files.length > 4 ? ` +${files.length - 4}` : ''}` }),
      h('p', { text: t('Everything below was produced locally. Numbers marked as estimates are estimates; hardware claims are limited to what was actually measured (see "Limits").', 'Todo lo de abajo se produjo localmente. Las cifras marcadas como estimaciones son estimaciones; las afirmaciones de hardware se limitan a lo realmente medido (ver "Límites").') })));

    // 1 dataset
    if (s.report) {
      const rep = s.report;
      const high = rep.alerts.filter((a) => a.severity === 'high');
      r.append(sec(t('1. Dataset', '1. Dataset'),
        kpi(rep.files.length, t('files', 'archivos')), kpi(fmt.int(rep.rows_total), t('rows', 'filas')), kpi(rep.columns.length, t('columns', 'columnas')), kpi(rep.alerts.length, t('alerts', 'alertas')),
        tbl([t('File', 'Archivo'), t('Rows', 'Filas'), t('Columns', 'Columnas'), t('Encoding', 'Codificación')], rep.files.map((f) => [f.file, fmt.int(f.rows), f.cols, f.encoding])),
        h('h3', { text: t('Main alerts', 'Alertas principales') }), ...(high.length ? high.slice(0, 8).map((a) => warn(a.message)) : [h('p', { text: t('No high-severity alerts.', 'Sin alertas de severidad alta.') })])));
      const rc = s.recipe;
      r.append(sec(t('2. Task and columns', '2. Tarea y columnas'),
        h('p', { text: `${t('Task', 'Tarea')}: ${({ classification: t('classification', 'clasificación'), regression: t('regression', 'regresión'), unsupervised: t('no label', 'sin etiqueta') })[rc.task] || rc.task} ·${rc.label_from_file && rc.label_from_file.enabled ? t('label = file name', 'etiqueta = nombre del archivo') : `${t('target', 'objetivo')}: ${rc.target || '—'}`}` }),
        tbl([t('Column', 'Columna'), t('Type', 'Tipo'), t('Role', 'Rol'), t('Nulls', 'Nulos')], rep.columns.slice(0, 40).map((c) => [c.name, TS.ui.kindLabel(c.kind), TS.ui.roleLabel(rc.roles[c.name] || c.role), fmt.pct((c.stats.nulls_pct || 0) / 100)])),
        rep.columns.length > 40 ? h('p', { class: 'rmeta', text: t('… {n} more columns omitted.', '… {n} columnas más omitidas.', { n: rep.columns.length - 40 }) }) : null,
        rc.encoding ? h('p', { text: `${t('Categorical encoding', 'Codificación de categóricas')}: ${rc.encoding.default}${Object.keys(rc.encoding.overrides || {}).length ? ` (+${Object.keys(rc.encoding.overrides).length} ${t('overrides', 'excepciones')})` : ''}` }) : null));
      const steps = (rc.cleaning || []).filter((x) => !x._off);
      const clean = [h('h3', { text: t('Steps', 'Pasos') }), steps.length ? tbl(['#', t('Step', 'Paso'), t('Why', 'Por qué')], steps.map((x, i) => [i + 1, TS.ui.stepLabel(x.kind), x.why])) : h('p', { text: t('No cleaning steps.', 'Sin pasos de limpieza.') })];
      if (s.clean) {
        const cols = s.clean.columns.slice(0, 6);
        const grid = h('div', { class: 'rgrid' });
        for (const c of cols) grid.append(await img(charts.histogram(c, s.clean.histograms[c]), t('Before (red) / after (blue outline): {c}', 'Antes (rojo) / después (contorno azul): {c}', { c })));
        clean.push(grid);
        const sizes = s.clean.split.sizes;
        r.append(sec(t('3. Cleaning', '3. Limpieza'), ...clean));
        const dist = h('div', { class: 'rgrid' });
        for (const [k, d] of Object.entries(s.clean.split.class_distribution || {})) dist.append(await img(charts.bars(d, t('Classes in {k}', 'Clases en {k}', { k })), ''));
        r.append(sec(t('4. Split', '4. Partición'), h('p', { text: `${t('Strategy', 'Estrategia')}: ${({ temporal_blocks: t('temporal blocks', 'bloques temporales'), group: t('by group', 'por grupo'), stratified: t('stratified', 'estratificada'), random: t('random', 'aleatoria') })[rc.split.strategy] || rc.split.strategy}` }), tbl([t('Split', 'Conjunto'), t('Rows', 'Filas')], Object.entries(sizes).map(([k, v]) => [k, fmt.int(v)])), dist));
      } else r.append(sec(t('3. Cleaning', '3. Limpieza'), ...clean));
    }

    // models
    const tr = s.train;
    if (tr) {
      const res = tr.result.results;
      const parts = [];
      if (tr.result.family === 'benchmark') {
        const rows = res.leaderboard;
        const task = res.task;
        const cols = task === 'regression' ? ['r2', 'mae', 'rmse'] : ['f1_macro', 'accuracy', 'mcc', 'mAP'];
        parts.push(h('p', { text: res.selection.reason }),
          tbl(['#', t('Model', 'Modelo'), ...cols.map((c) => `${c} val`), ...cols.map((c) => `${c} test`), t('Size', 'Tamaño'), t('Notes', 'Notas')],
            rows.map((x) => [x.rank, x.name, ...cols.map((c) => fmt.num(x.metrics.val && x.metrics.val[c])), ...cols.map((c) => fmt.num(x.metrics.test && x.metrics.test[c])), x.size_bytes ? fmt.bytes(x.size_bytes) : '—', [x.selected ? '★' : '', x.statistically_tied_with_best ? t('tied', 'empatado') : '', x.exportable ? '' : t('no exporter', 'sin exportador')].filter(Boolean).join(' ')])));
        const d = {}; rows.forEach((x) => { d[x.name] = Number(((x.metrics.test || {})[cols[0]] || 0).toFixed(4)); });
        parts.push(await img(charts.bars(d, `${cols[0]} (test)`), ''));
        if (res.selected_confusion_test) parts.push(await img(charts.confusion(res.selected_confusion_test.matrix, res.selected_confusion_test.classes), t('Confusion matrix of the recommended model (test)', 'Matriz de confusión del modelo recomendado (test)')));
        (res.warnings || []).forEach((x) => parts.push(warn(x)));
      } else {
        const q = res.quality || (res.models && res.selection && res.models[res.selection.best] && res.models[res.selection.best].quality);
        if (q && q.splits) {
          const test = q.splits.test || q.splits.test_real || {};
          parts.push(h('div', {}, kpi(fmt.pct(test.accuracy ?? test.subset_accuracy), t('accuracy (test)', 'exactitud (test)')), kpi(fmt.num(test.f1_macro), 'F1 macro'), kpi(fmt.num(test.mAP), 'mAP'), kpi(fmt.num(test.roc_auc_macro), 'ROC-AUC'), kpi(fmt.num(test.mcc), 'MCC')));
          if (test.confusion_matrix && (res.classes || q.classes)) parts.push(await img(charts.confusion(test.confusion_matrix, res.classes || q.classes), t('Confusion matrix (test)', 'Matriz de confusión (test)')));
          if (q.curves && q.curves.roc) {
            const names = Object.keys(q.curves.roc);
            parts.push(await img(charts.xy('ROC', names.map((n) => ({ name: n, x: q.curves.roc[n].fpr, y: q.curves.roc[n].tpr })), 'FPR', 'TPR', true), 'ROC'));
          }
        }
        if (res.history && res.history.loss) parts.push(await img(charts.lines(t('Loss per epoch', 'Pérdida por época'), [{ name: 'train', values: res.history.loss }, { name: t('validation', 'validación'), values: res.history.val_loss || [] }]), ''));
        (res.warnings || []).forEach((x) => parts.push(warn(x)));
      }
      r.append(sec(t('5. Models and quality', '5. Modelos y calidad'), ...parts));
      const hw = tr.job && tr.job.hw_summary;
      if (hw) {
        r.append(sec(t('6. Hardware use during training', '6. Uso de hardware durante el entrenamiento'),
          tbl([t('Metric', 'Métrica'), t('Mean', 'Media'), t('Peak', 'Pico')], [
            [t('Training time', 'Tiempo de entrenamiento'), fmt.dur ? fmt.dur(hw.duration_s || 0) : hw.duration_s, ''],
            [t('CPU (whole system)', 'CPU (todo el sistema)'), `${fmt.num(hw.cpu_mean_pct, 1)} %`, `${fmt.num(hw.cpu_peak_pct, 1)} %`],
            [t('CPU (this training)', 'CPU (este entrenamiento)'), `${fmt.num(hw.proc_cpu_mean_pct, 1)} %`, `${fmt.num(hw.proc_cpu_peak_pct, 1)} %`],
            [t('RAM (this training)', 'RAM (este entrenamiento)'), '', `${fmt.num(hw.rss_peak_mb, 0)} MB`],
            ...((hw.gpus || []).map((g) => [`GPU ${g.name || ''}`, `${fmt.num(g.util_mean_pct, 1)} %`, `${fmt.num(g.util_peak_pct, 1)} % · ${fmt.num(g.mem_peak_mb, 0)} MB · ${g.temp_peak_c ?? '—'} °C`])),
            [t('CPU temperature', 'Temperatura CPU'), '', hw.cpu_temp_peak_c ? `${fmt.num(hw.cpu_temp_peak_c, 0)} °C` : t('not available', 'no disponible')]])));
      }
    }
    // export
    const ex = s.export;
    if (ex) {
      const rep = ex.result.results;
      const b = rep.budget || {};
      r.append(sec(t('7. Export', '7. Exportación'),
        b.flash_data ? h('p', { text: `${t('Flash', 'Flash')}: ${b.flash_data.verdict_label || b.flash_data.verdict} (${fmt.bytes(b.flash_data.used_bytes)}) · ${t('RAM', 'RAM')}: ${(b.ram_working || {}).verdict_label || (b.ram_working || {}).verdict} (${fmt.bytes((b.ram_working || {}).used_bytes || 0)})` }) : null,
        b.note ? h('p', { class: 'rmeta', text: b.note }) : null,
        (ex.result.arduino || []).length ? tbl([t('Arduino variant', 'Variante Arduino'), t('Verification', 'Verificación')], ex.result.arduino.map((a) => [a.variant, a.verification ? `${a.verification.level}: ${a.verification.detail}` : (a.error || '—')])) : null,
        ...((rep.warnings || []).map(warn))));
    }
    // reproducibility + limits
    const rep2 = tr && tr.result.results && tr.result.results.reproducibility;
    r.append(sec(t('8. Reproducibility', '8. Reproducibilidad'),
      rep2 ? tbl([t('Item', 'Elemento'), t('Value', 'Valor')], [['seed', rep2.seed], ['python', rep2.python], ['recipe sha256', String(rep2.recipe_sha256 || '').slice(0, 24)], ...Object.entries(rep2.libraries || {}).map(([k, v]) => [k, v])]) : h('p', { text: t('Train a model to include reproducibility data.', 'Entrena un modelo para incluir datos de reproducibilidad.') }),
      h('p', { class: 'rmeta', text: t('The recipe (JSON) can be saved from the header and reproduces cleaning, split and encoding exactly.', 'La receta (JSON) se puede guardar desde la cabecera y reproduce limpieza, partición y codificación exactamente.') })));
    r.append(sec(t('Limits (read this)', 'Límites (léelo)'),
      h('ul', {}, [
        t('Verification was done on a PC; the board itself was not tested unless you did it.', 'La verificación se hizo en un PC; la placa no se probó salvo que tú lo hayas hecho.'),
        t('The test set may come from the same sessions as train: It does not measure generalization to new devices or households.', 'El test puede venir de las mismas sesiones que train: No mide generalización a dispositivos u hogares nuevos.'),
        t('Prediction time is measured on this computer, not on the microcontroller; energy is not measured.', 'El tiempo de predicción se mide en este computador, no en el microcontrolador; la energía no se mide.'),
        t('Board data from boards.seed.yaml is unverified until checked against the vendor documentation.', 'Los datos de placas de boards.seed.yaml no están verificados hasta contrastarlos con la documentación del fabricante.'),
      ].map((x) => h('li', { text: x })))));
    return r;
  }

  TS.makeReport = async function makeReport() {
    const btn = document.getElementById('tool-report');
    if (btn) btn.disabled = true;
    const root = document.getElementById('report');
    try {
      root.replaceChildren(await build());
      const imgs = [...root.querySelectorAll('img')];
      await Promise.all(imgs.map((i) => (i.complete ? null : new Promise((res) => { i.onload = res; i.onerror = res; }))));
      document.body.classList.add('printing');
      await new Promise((res) => setTimeout(res, 150));
      const r = await window.tinyml.printPDF(`tinyml-report-${new Date().toISOString().slice(0, 10)}`);
      TS.lastReport = r;
      return r;
    } catch (e) {
      TS.lastReport = { error: String(e && e.message ? e.message : e) };
      return TS.lastReport;
    } finally {
      document.body.classList.remove('printing');
      root.replaceChildren();
      if (btn) btn.disabled = false;
    }
  };
})();
