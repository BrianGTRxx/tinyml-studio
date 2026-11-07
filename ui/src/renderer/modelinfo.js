'use strict';
// Two cards for step 7: "What was this model trained with" (files, columns, model, settings, all in one place) and
// "Try the model with your own values" (same cleaning and model as the exported C code, with an out-of-range warning).
(function () {
  const TS = window.TS;
  const { h, callout, fmt } = TS;
  const ui = TS.ui;
  const t = TS.t;

  const list = (items, max = 12) => (items.length > max ? `${items.slice(0, max).join(', ')} +${items.length - max}` : items.join(', ')) || '—';

  /** One place that says what produced the model: files, target, columns, split, cleaning and augmentation. */
  TS.usedCard = function usedCard({ S, modelName }) {
    const r = S.recipe;
    const rep = S.report;
    const cols = rep.columns.filter((c) => (r.roles[c.name] || c.role) === 'feature').map((c) => c.name);
    const lff = r.label_from_file && r.label_from_file.enabled;
    const target = lff
      ? `${t('the file name', 'el nombre del archivo')} (${Object.entries(r.label_from_file.mapping || {}).map(([f, l]) => `${f} → ${l}`).join(', ')})`
      : (r.target || '—');
    const sp = r.split || {};
    const strategies = { temporal_blocks: t('temporal blocks', 'bloques temporales'), group: t('by group', 'por grupo'), stratified: t('stratified', 'estratificada'), random: t('random', 'aleatoria') };
    const aug = r.augment && r.augment.method && r.augment.method !== 'off'
      ? `${r.augment.method === 'noise' ? t('copies with noise', 'copias con ruido') : t('between neighbours', 'entre vecinas')} (${r.augment.mode === 'balance' ? t('equalize classes', 'igualar clases') : `${r.augment.ratio ?? 1}×`})`
      : t('none', 'ninguno');
    const bal = sp.balance && sp.balance !== 'off' ? (sp.balance === 'all' ? t('train, validation and test', 'train, validación y test') : t('train only', 'solo train')) : t('none', 'ninguno');
    const steps = (r.cleaning || []).filter((s) => !s._off).map((s) => ui.stepLabel(s.kind));
    const rows = [
      [t('Files', 'Archivos'), `${list(rep.files.map((f) => f.file), 8)} (${TS.fmt.int(rep.rows_total)} ${t('rows', 'filas')})`],
      [t('Task', 'Tarea'), r.task === 'classification' ? t('classification', 'clasificación') : r.task === 'regression' ? t('regression', 'regresión') : r.task],
      [t('Predicts', 'Predice'), target],
      [t('Input columns', 'Columnas de entrada'), `${cols.length}: ${list(cols)}`],
      [t('Model', 'Modelo'), modelName || '—'],
      [t('Split', 'Partición'), `${strategies[sp.strategy] || sp.strategy} · ${(sp.ratios || []).join(' / ')}`],
      [t('Class balance', 'Equilibrio de clases'), bal],
      [t('Cleaning steps', 'Pasos de limpieza'), list(steps)],
      [t('New rows (augmentation)', 'Filas nuevas (aumento)'), aug],
    ];
    return ui.collapse(t('What was this model trained with?', '¿Con qué se entrenó este modelo?'),
      h('div', {}, h('p', { class: 'muted', text: t('Check this against the data you will measure in real life: Same units, same sign, same columns.', 'Compáralo con los datos que vas a medir en la vida real: Mismas unidades, mismo signo, mismas columnas.') }), TS.table([t('Item', 'Elemento'), t('Value', 'Valor')], rows)),
      true);
  };

  /** Type values by hand and see what the model answers, with a warning when a value is outside the training range. */
  TS.tryCard = function tryCard({ runParams, modelName }) {
    const body = h('div', { id: 'try-body' }, h('p', { class: 'muted', text: t('Open to load the inputs of the model…', 'Ábrelo para cargar las entradas del modelo…') }));
    const d = ui.collapse(t('Try the model with your own values', 'Prueba el modelo con tus propios valores'), body, false);
    let loaded = false;
    const base = () => ({ ...runParams(), model: modelName ? modelName() : null, device: 'cpu' });

    async function loadSchema() {
      body.replaceChildren(h('p', { class: 'muted', text: t('Loading…', 'Cargando…') }));
      try {
        const { job, result: r } = await TS.runJob('predict', { run_job: base().run_job, run_subdir: base().run_subdir || null, model: base().model || null }, () => {});
        if (job.state !== 'succeeded') throw new Error(job.error || job.state);
        form(r.results);
      } catch (e) { body.replaceChildren(TS.err(e)); loaded = false; }
    }

    function form(sch) {
      const inputs = {};
      const out = h('div', { id: 'try-out' });
      const grid = h('div', { class: 'grid2' });
      for (const c of sch.numeric) {
        const range = c.min !== undefined ? t('seen in training: {a} to {b}', 'visto al entrenar: {a} a {b}', { a: fmt.num(c.min), b: fmt.num(c.max) }) : '';
        inputs[c.name] = h('input', { type: 'text', inputmode: 'decimal', id: `try-${c.name}`, placeholder: c.median !== undefined ? String(fmt.num(c.median)) : '' });
        grid.append(ui.field(c.name, inputs[c.name], range));
      }
      for (const c of sch.categorical) {
        inputs[c.name] = h('select', { id: `try-${c.name}` }, [h('option', { value: '', text: t('(missing)', '(falta)') }), ...c.values.map((v) => h('option', { value: v, text: v }))]);
        grid.append(ui.field(c.name, inputs[c.name]));
      }
      const btn = h('button', { type: 'button', id: 'try-run', text: t('Predict', 'Predecir'), onclick: async () => {
        btn.disabled = true;
        out.replaceChildren(h('p', { class: 'muted', text: t('Predicting…', 'Prediciendo…') }));
        const values = {};
        for (const [k, el] of Object.entries(inputs)) values[k] = el.value;
        try {
          const { job, result: r } = await TS.runJob('predict', { run_job: base().run_job, run_subdir: base().run_subdir || null, model: base().model || null, values }, () => {});
          if (job.state !== 'succeeded') throw new Error(job.error || job.state);
          out.replaceChildren(answer(r.results));
        } catch (e) { out.replaceChildren(TS.err(e)); }
        btn.disabled = false;
      } });
      body.replaceChildren(
        h('p', { class: 'muted', text: t('Leave a box empty to say "missing" (the model uses the filler it learned). It uses the same cleaning and model as the exported C code, which was checked equal on test vectors; it is not a test on the board.', 'Deja una casilla vacía para decir "falta" (el modelo usa el relleno que aprendió). Usa la misma limpieza y el mismo modelo que el código C exportado, que se verificó igual en los vectores de prueba; no es una prueba en la placa.') }),
        grid, h('div', { class: 'row' }, btn), out);
    }

    function answer(r) {
      const parts = [];
      if (r.task === 'classification') {
        parts.push(h('p', { class: 'try-answer' }, h('b', { text: t('Answer: ', 'Respuesta: ') }), h('span', { id: 'try-class', text: r.class })));
        if (r.probabilities) {
          const rows = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, `${(v * 100).toFixed(1)} %`]);
          parts.push(TS.table([t('Class', 'Clase'), t('Confidence', 'Confianza')], rows));
        }
      } else {
        parts.push(h('p', { class: 'try-answer' }, h('b', { text: t('Answer: ', 'Respuesta: ') }), h('span', { id: 'try-value', text: fmt.num(r.value) })));
      }
      if (r.out_of_range && r.out_of_range.length) {
        parts.push(callout('warn', h('b', { text: t('Outside the range seen in training: ', 'Fuera del rango visto al entrenar: ') }),
          r.out_of_range.map((w) => t('{c} = {v} (seen {a} to {b})', '{c} = {v} (visto {a} a {b})', { c: w.column, v: fmt.num(w.value), a: fmt.num(w.min), b: fmt.num(w.max) })).join('; '),
          ` ${t('The model extrapolates there and the answer may be unreliable.', 'El modelo extrapola ahí y la respuesta puede no ser fiable.')}`));
      }
      if (r.missing && r.missing.length) parts.push(callout('info', t('Missing (filled as in training): {l}', 'Faltan (se rellenan como en el entrenamiento): {l}', { l: r.missing.join(', ') })));
      if (r.unknown_category && r.unknown_category.length) parts.push(callout('warn', t('Unknown category in: {l}', 'Categoría desconocida en: {l}', { l: r.unknown_category.join(', ') })));
      return h('div', {}, parts);
    }

    d.addEventListener('toggle', () => { if (d.open && !loaded) { loaded = true; loadSchema(); } });
    return d;
  };
})();
