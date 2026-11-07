'use strict';
// "Does it work on a NEW session?" card (leave-one-session-out). The usual test split stays inside the same
// recordings, so it measures "later in the same capture"; this leaves a whole session out and says plainly
// which classes cannot be measured because they exist in a single session.
(function () {
  const TS = window.TS;
  const { h, callout, table, fmt } = TS;
  const ui = TS.ui;
  const t = TS.t;

  const VERDICT = () => ({
    measurable: ['ok', t('Every class appears in 2 or more sessions: The figures below measure generalization.', 'Todas las clases aparecen en 2 sesiones o más: Las cifras de abajo miden la generalización.')],
    partial: ['warn', t('Only some classes can be measured on a new session.', 'Solo algunas clases se pueden medir con una sesión nueva.')],
    not_measurable: ['bad', t('This data cannot tell you how the model behaves on a new session.', 'Con estos datos no se puede saber cómo se comporta el modelo con una sesión nueva.')],
  });

  function result(res, ctx) {
    const full = res.with_all_features;
    const [kind, text] = VERDICT()[res.verdict] || ['info', res.verdict];
    const rows = full.folds.map((f) => [
      f.group,
      f.classes.join(', '),
      TS.fmt.int(f.rows),
      fmt.pct(f.accuracy, 1),
      f.overlap >= 0.5 ? h('span', { class: 'tag high', text: t('copy of another session', 'copia de otra sesión'), title: t('At least half of its rows already exist in the training sessions: It does not measure generalization.', 'Al menos la mitad de sus filas ya están en las sesiones de entrenamiento: No mide generalización.') }) : '',
    ]);
    const alt = res.without_session_columns;
    return h('div', { id: 'group-results' },
      callout(kind, text),
      full.mean_accuracy === null ? null : ui.stats(
        ui.stat(fmt.pct(full.mean_accuracy, 1), t('mean accuracy on an unseen session', 'exactitud media en una sesión que no vio'), 'ok'),
        ui.stat(fmt.pct(full.min_accuracy, 1), t('worst session', 'peor sesión')),
        ui.stat(`${full.classes_measurable.length}/${res.n_classes}`, t('classes measurable', 'clases medibles'), full.classes_measurable.length === res.n_classes ? 'ok' : 'warn'),
        alt && alt.mean_accuracy !== null ? ui.stat(fmt.pct(alt.mean_accuracy, 1), t('without the session columns', 'sin las columnas de sesión')) : null),
      rows.length ? table([t('Session left out', 'Sesión dejada fuera'), t('Class', 'Clase'), t('Rows', 'Filas'), t('Accuracy', 'Exactitud'), ''], rows) : null,
      full.skipped_groups.length ? h('p', { class: 'muted', text: `${t('Not measured (their class exists in no other session)', 'Sin medir (su clase no existe en otra sesión)')}: ${full.skipped_groups.join(', ')}` }) : null,
      ...res.warnings.map((w) => callout('warn', w)),
      ctx && ctx.markable.length ? h('div', { class: 'row wrap' },
        h('button', { id: 'btn-mark-meta', type: 'button', text: t('Mark {n} session column(s) as metadata and re-run', 'Marcar {n} columna(s) de sesión como metadatos y repetir', { n: ctx.markable.length }), title: ctx.markable.join(', '), onclick: ctx.mark }),
        h('span', { class: 'muted', text: `${t('Takes them out of the features (they identify the recording, not the device)', 'Las saca de las features (identifican la grabación, no el aparato)')}: ${ctx.markable.join(', ')}` })) : null,
      ctx && ctx.before !== null && ctx.before !== undefined && full.mean_accuracy !== null
        ? callout('info', t('Before: {a} · now: {b}', 'Antes: {a} · ahora: {b}', { a: fmt.pct(ctx.before, 1), b: fmt.pct(full.mean_accuracy, 1) })) : null);
  }

  TS.groupValidationCard = function groupValidationCard({ recipe, model, markable, onMark }) {
    const out = h('div', { id: 'group-out' });
    let lastMean = null;
    let before = null;
    const run = async () => {
      btn.disabled = true;
      const prog = TS.progressPanel();
      out.replaceChildren(prog.el);
      try {
        const { job, result: r } = await TS.runJob('validate_groups', { recipe: recipe(), model: model(), seed: 42 }, prog.update);
        if (job.state !== 'succeeded') throw new Error(job.error || job.state);
        prog.done(t('Validation finished', 'Validación terminada'));
        const res = r.results;
        const mark = () => { before = lastMean; onMark(res.session_columns); run(); };
        out.replaceChildren(result(res, { markable: markable(res.session_columns), before, mark }));
        lastMean = res.with_all_features.mean_accuracy;
      } catch (e) { out.replaceChildren(TS.err(e)); }
      btn.disabled = false;
    };
    const btn = h('button', { id: 'run-groups', type: 'button', class: 'ghost', text: t('Validate on unseen sessions', 'Validar con sesiones que no vio'), onclick: () => { before = null; lastMean = null; run(); } });
    return ui.collapse(
      t('Does it work on a NEW session? (honest validation)', '¿Funciona con una sesión NUEVA? (validación honesta)'),
      h('div', {},
        h('p', { class: 'muted', text: t('The normal test comes from the same recordings as the training, so it measures "later in the same capture". Here each session (file) is left out in turn, the model is trained on the others and measured on the one it never saw. A class can only be measured if it exists in at least TWO sessions.', 'El test normal sale de las mismas grabaciones que el entrenamiento, así que mide "más tarde en la misma captura". Aquí se deja fuera cada sesión (archivo) por turnos, se entrena con las demás y se mide en la que no vio. Una clase solo se puede medir si existe en al menos DOS sesiones.') }),
        h('div', { class: 'row' }, btn), out),
      false);
  };
})();
