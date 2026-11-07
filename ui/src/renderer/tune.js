'use strict';
// "Find the best settings that still fit my board" card: random search over a small grid per model, every
// candidate measured with the same C emitter as the export (flash + RAM). The test set is measured once, only
// for the winner, so it never steers the choice.
(function () {
  const TS = window.TS;
  const { h, callout, table, fmt } = TS;
  const ui = TS.ui;
  const t = TS.t;
  const MODELS = ['decision_tree', 'random_forest', 'extra_trees', 'gradient_boosting', 'logreg', 'mlp'];
  const kb = (v) => (v === null || v === undefined ? '—' : `${TS.fmt.dec(v / 1024, 1)} KB`);

  function result(res, onUse) {
    const b = res.best;
    const rows = res.trials
      .filter((r) => !r.error)
      .sort((x, y) => (y.fits - x.fits) || (y.val_score - x.val_score))
      .slice(0, 12)
      .map((r) => [JSON.stringify(r.params), fmt.pct(r.val_score, 1), kb(r.flash_bytes), kb(r.ram_bytes), r.fits ? t('fits', 'cabe') : t('does not fit', 'no cabe'), r.chosen ? '★' : r.is_default ? t('default', 'por defecto') : '']);
    return h('div', { id: 'tune-results' },
      callout(b.fits_budget || (!res.budget.flash_kb && !res.budget.ram_kb) ? 'ok' : 'bad', h('pre', { class: 'plain', text: res.summary })),
      h('div', { class: 'row' }, h('button', { id: 'btn-use-tuned', type: 'button', text: t('Use these settings when training', 'Usar estos parámetros al entrenar'), onclick: () => onUse(res.model, b.params) })),
      table([t('Settings', 'Parámetros'), t('Validation', 'Validación'), 'Flash', 'RAM', '', ''], rows));
  }

  TS.tuneCard = function tuneCard({ recipe, model, budget, tuned, onUse }) {
    const out = h('div', { id: 'tune-out' });
    const sel = h('select', { id: 'tune-model' }, ...MODELS.map((m) => h('option', { value: m, text: m, selected: m === model() })));
    const flash = h('input', { id: 'tune-flash', type: 'number', min: '1', value: budget().flash || '' });
    const ram = h('input', { id: 'tune-ram', type: 'number', min: '1', value: budget().ram || '' });
    const trials = h('input', { id: 'tune-trials', type: 'number', min: '1', max: '200', value: '24' });
    const btn = h('button', { id: 'run-tune', type: 'button', class: 'ghost', text: t('Search the best settings', 'Buscar los mejores parámetros'), onclick: async () => {
      btn.disabled = true;
      const prog = TS.progressPanel();
      out.replaceChildren(prog.el);
      try {
        const params = { recipe: recipe(), model: sel.value, n_trials: Number(trials.value) || 24, flash_kb: Number(flash.value) || null, ram_kb: Number(ram.value) || null, seed: 42, device: 'cpu' };
        const { job, result: r } = await TS.runJob('tune', params, prog.update);
        if (job.state !== 'succeeded') throw new Error(job.error || job.state);
        prog.done(t('Search finished', 'Búsqueda terminada'));
        out.replaceChildren(result(r.results, onUse));
      } catch (e) { out.replaceChildren(TS.err(e)); }
      btn.disabled = false;
    } });
    return ui.collapse(
      t('Find settings that fit my board', 'Buscar parámetros que quepan en mi placa'),
      h('div', {},
        h('p', { class: 'muted', text: t('Tries several settings, measures how much flash and RAM each would use on the board (same calculation as the export) and keeps the most accurate one that fits. It is chosen on validation; the test set is measured only once, for the winner.', 'Prueba varias configuraciones, mide cuánta flash y RAM usaría cada una en la placa (el mismo cálculo que la exportación) y se queda con la más precisa que quepa. Se elige con validación; el test se mide una sola vez, solo para la ganadora.') }),
        tuned() ? callout('info', t('Training will use the settings you chose: {p}', 'El entrenamiento usará los parámetros elegidos: {p}', { p: JSON.stringify(tuned()) })) : null,
        h('div', { class: 'grid2' },
          ui.field(t('Model', 'Modelo'), sel),
          ui.field(t('Flash budget (KB)', 'Presupuesto de flash (KB)'), flash),
          ui.field(t('RAM budget (KB)', 'Presupuesto de RAM (KB)'), ram),
          ui.field(t('Settings to try', 'Configuraciones a probar'), trials)),
        h('div', { class: 'row' }, btn), out),
      false);
  };
  // ---- which columns does the model really use? (permutation importance on validation) --------------------
  TS.importanceCard = function importanceCard({ runJob, model }) {
    const out = h('div', { id: 'importance-out' });
    const btn = h('button', { id: 'run-importance', type: 'button', class: 'ghost', text: t('Which columns matter?', '¿Qué columnas importan?'), onclick: async () => {
      btn.disabled = true;
      const prog = TS.progressPanel();
      out.replaceChildren(prog.el);
      try {
        const { job, result: r } = await TS.runJob('importance', { run_job: runJob(), model: model() || null, device: 'cpu' }, prog.update);
        if (job.state !== 'succeeded') throw new Error(job.error || job.state);
        prog.done(t('Importance computed', 'Importancia calculada'));
        const res = r.results;
        const rows = res.features.map((f) => [f.feature, fmt.dec(f.drop, 3), `± ${fmt.dec(f.std, 3)}`, f.useless ? t('adds very little', 'aporta muy poco') : '']);
        out.replaceChildren(callout(res.useless.length ? 'warn' : 'ok', h('pre', { class: 'plain', text: res.summary })),
          table([t('Column', 'Columna'), t('Metric drop when shuffled', 'Caída de la métrica al barajarla'), '', ''], rows));
      } catch (e) { out.replaceChildren(TS.err(e)); }
      btn.disabled = false;
    } });
    return ui.collapse(t('Which columns does the model really use?', '¿Qué columnas usa de verdad el modelo?'),
      h('div', {}, h('p', { class: 'muted', text: t('Shuffles one column at a time on the validation data and measures how much the result drops. Columns that barely matter could be removed from the firmware (fewer sensors, less RAM). Retrain without them to confirm.', 'Baraja una columna a la vez en los datos de validación y mide cuánto empeora el resultado. Las que casi no importan se podrían quitar del firmware (menos sensores, menos RAM). Reentrena sin ellas para confirmarlo.') }), h('div', { class: 'row' }, btn), out),
      false);
  };
  // ---- estimated latency per board (an ESTIMATE: assumed cycles per operation, shown as a range) ----------
  TS.latencyCard = function latencyCard({ runJob, model, boardId }) {
    const out = h('div', { id: 'latency-out' });
    const target = h('input', { id: 'latency-target', type: 'number', min: '0', step: '0.1', placeholder: t('optional', 'opcional') });
    const btn = h('button', { id: 'run-latency', type: 'button', class: 'ghost', text: t('Estimate speed on each board', 'Estimar la velocidad en cada placa'), onclick: async () => {
      btn.disabled = true;
      const prog = TS.progressPanel();
      out.replaceChildren(prog.el);
      try {
        const { job, result: r } = await TS.runJob('latency', { run_job: runJob(), model: model() || null, board: 'all', target_ms: Number(target.value) || null, device: 'cpu' }, prog.update);
        if (job.state !== 'succeeded') throw new Error(job.error || job.state);
        prog.done(t('Estimate ready', 'Estimación lista'));
        const res = r.results;
        const rows = res.boards.map((b) => [b.board === boardId() ? `★ ${b.name}` : b.name, `${fmt.dec(b.ms_lo, 3)} – ${fmt.dec(b.ms_hi, 3)} ms`, `${b.clock_mhz} MHz`, b.fpu_known ? '' : t('FPU not confirmed', 'FPU sin confirmar'), b.meets_target === null ? '' : b.meets_target ? t('meets target', 'cumple') : t('too slow', 'muy lenta')]);
        out.replaceChildren(callout('warn', res.disclaimer), table([t('Board', 'Placa'), t('Time per prediction', 'Tiempo por predicción'), t('Clock', 'Reloj'), '', ''], rows));
      } catch (e) { out.replaceChildren(TS.err(e)); }
      btn.disabled = false;
    } });
    return ui.collapse(t('How fast would it be on each board? (estimate)', '¿Qué tan rápido sería en cada placa? (estimación)'),
      h('div', {}, h('p', { class: 'muted', text: t('Counts the operations of the model and multiplies them by assumed cycles per operation and the board clock. It is a range, not a measurement: Time tm_predict on the real board to know the exact figure.', 'Cuenta las operaciones del modelo y las multiplica por ciclos por operación supuestos y por el reloj de la placa. Es un rango, no una medición: Cronometra tm_predict en la placa real para saber la cifra exacta.') }),
        ui.field(t('Maximum time allowed (ms)', 'Tiempo máximo permitido (ms)'), target), h('div', { class: 'row' }, btn), out),
      false);
  };
  // ---- answer "unknown" when the model is unsure (confidence threshold, honest unknown-class test) ---------
  TS.rejectCard = function rejectCard({ runJob, model, onUse }) {
    const out = h('div', { id: 'reject-out' });
    const goal = h('input', { id: 'reject-goal', type: 'number', min: '60', max: '99', value: '95' });
    const btn = h('button', { id: 'run-reject', type: 'button', class: 'ghost', text: t('Calibrate "unknown"', 'Calibrar "desconocido"'), onclick: async () => {
      btn.disabled = true;
      const prog = TS.progressPanel();
      out.replaceChildren(prog.el);
      try {
        const { job, result: r } = await TS.runJob('reject', { run_job: runJob(), model: model() || null, target_accuracy: (Number(goal.value) || 95) / 100, holdout: true, device: 'cpu' }, prog.update);
        if (job.state !== 'succeeded') throw new Error(job.error || job.state);
        prog.done(t('Threshold calibrated', 'Umbral calibrado'));
        const res = r.results;
        const sg = res.unknown_detection && res.unknown_detection.suggested;
        const use = (thr) => h('button', { type: 'button', class: 'ghost sm', text: t('Use {t} in the ONNX export', 'Usar {t} en la exportación ONNX', { t: thr }), onclick: () => { onUse(thr); } });
        out.replaceChildren(callout(res.reached_target ? 'ok' : 'warn', h('pre', { class: 'plain', text: res.summary })),
          h('div', { class: 'row wrap' }, use(res.threshold), sg ? use(sg.threshold) : null),
          res.unknown_detection ? table([t('Class left out', 'Clase retirada'), t('Rows', 'Filas'), t('Rejected as unknown', 'Rechazadas como desconocido')], res.unknown_detection.classes.map((c) => [c.class, TS.fmt.int(c.rows), fmt.pct(c.rejected, 1)])) : null);
      } catch (e) { out.replaceChildren(TS.err(e)); }
      btn.disabled = false;
    } });
    return ui.collapse(t('Answer "unknown" when unsure', 'Responder "desconocido" cuando dude'),
      h('div', {}, h('p', { class: 'muted', text: t('A classifier always picks one of its classes, even for something it has never seen. This sets a confidence threshold on validation and then checks it for real: It retrains without one class at a time and counts how many of its rows are rejected. Only models with probabilities work (trees, forests, logistic regression, MLP, boosting, Naive Bayes). The C export does not apply it yet; the ONNX export does.', 'Un clasificador siempre elige una de sus clases, incluso ante algo que nunca vio. Esto fija un umbral de confianza en validación y lo comprueba de verdad: Reentrena sin una clase a la vez y cuenta cuántas de sus filas se rechazan. Solo sirven los modelos con probabilidades (árboles, bosques, regresión logística, MLP, boosting, Naive Bayes). La exportación a C aún no lo aplica; la de ONNX sí.') }),
        ui.field(t('Correct among accepted (%)', 'Aciertos entre lo aceptado (%)'), goal), h('div', { class: 'row' }, btn), out),
      false);
  };
  // ---- anomaly detector: learn what is normal, flag what is not -------------------------------------------
  TS.anomalyCard = function anomalyCard({ recipe, classes }) {
    const out = h('div', { id: 'anomaly-out' });
    const normal = h('select', { id: 'anomaly-normal', multiple: '', size: '5' }, ...classes().map((c) => h('option', { value: c, text: c })));
    const model = h('select', { id: 'anomaly-model' }, h('option', { value: 'isolation_forest', text: 'Isolation Forest' }), h('option', { value: 'one_class_svm', text: 'One-class SVM' }));
    const alarm = h('input', { id: 'anomaly-alarm', type: 'number', min: '0.1', max: '20', step: '0.1', value: '1' });
    const btn = h('button', { id: 'run-anomaly', type: 'button', class: 'ghost', text: t('Train anomaly detector', 'Entrenar detector de anomalías'), onclick: async () => {
      btn.disabled = true;
      const prog = TS.progressPanel();
      out.replaceChildren(prog.el);
      try {
        const params = { recipe: recipe(), model: model.value, normal_classes: [...normal.selectedOptions].map((o) => o.value), false_alarm: (Number(alarm.value) || 1) / 100, seed: 42, device: 'cpu' };
        const { job, result: r } = await TS.runJob('anomaly', params, prog.update);
        if (job.state !== 'succeeded') throw new Error(job.error || job.state);
        prog.done(t('Detector ready', 'Detector listo'));
        const res = r.results;
        const d = res.detection;
        out.replaceChildren(callout(res.warnings.length ? 'warn' : 'ok', h('pre', { class: 'plain', text: res.summary })),
          d ? table([t('Anomalous class', 'Clase anómala'), t('Rows', 'Filas'), t('Detected', 'Detectadas')], Object.entries(d.per_class).map(([c, v]) => [c, TS.fmt.int(v.rows), fmt.pct(v.detected, 1)])) : null);
      } catch (e) { out.replaceChildren(TS.err(e)); }
      btn.disabled = false;
    } });
    return ui.collapse(t('Detect anomalies (learn only what is normal)', 'Detectar anomalías (aprende solo lo normal)'),
      h('div', {}, h('p', { class: 'muted', text: t('Trains only on the classes you mark as normal and raises an alarm when something does not look like them. The other classes are used just to measure how many it catches, never to train or to set the threshold. If you mark none, every row counts as normal and only false alarms are measured.', 'Entrena solo con las clases que marques como normales y avisa cuando algo no se parece. Las demás clases se usan solo para medir cuántas detecta, nunca para entrenar ni para fijar el umbral. Si no marcas ninguna, todas las filas cuentan como normales y solo se miden las falsas alarmas.') }),
        h('div', { class: 'grid2' }, ui.field(t('Normal classes (Ctrl+click for several)', 'Clases normales (Ctrl+clic para varias)'), normal), ui.field(t('Method', 'Método'), model), ui.field(t('Allowed false alarms (%)', 'Falsas alarmas permitidas (%)'), alarm)),
        h('div', { class: 'row' }, btn), out),
      false);
  };
})();
