'use strict';
// Final-quality panel: overview, training curves, confusion matrix, per-class metrics, ROC/PR curves,
// regression diagnostics. Everything is drawn from the "quality" block of the training result.
(function () {
  const TS = window.TS;
  const { h, fmt, table, callout, charts } = TS;
  const ui = TS.ui;
  const t = TS.t;

  const SPLIT_NAMES = () => ({
    train: t('train', 'entrenamiento'), val: t('validation', 'validación'), test: 'test',
    test_real: t('test (real)', 'test (real)'),
  });
  const sname = (k) => SPLIT_NAMES()[k] || k;
  const kpi = (label, value) => h('div', { class: 'kpi' }, h('b', { text: value }), h('span', { text: label }));

  function tabs(defs) {
    const bar = h('div', { class: 'tabs', role: 'tablist' });
    const body = h('div', { class: 'tabbody' });
    let cur = 0;
    const draw = () => {
      bar.replaceChildren(...defs.map((d, i) => h('button', { type: 'button', role: 'tab', class: i === cur ? 'on' : '', 'data-tab': d.id, text: d.title, onclick: () => { cur = i; draw(); } })));
      body.replaceChildren(defs[cur].render());
    };
    draw();
    return h('div', {}, bar, body);
  }

  function overview(q, kind) {
    const sp = q.splits || {};
    const keys = Object.keys(sp);
    const main = sp.test || sp.test_real || sp[keys[keys.length - 1]] || {};
    const cards = [];
    if (kind === 'regression') {
      cards.push(kpi('R²', fmt.num(main.r2)), kpi('MAE', fmt.num(main.mae)), kpi('RMSE', fmt.num(main.rmse)));
    } else {
      cards.push(kpi(t('Accuracy', 'Exactitud'), fmt.pct(main.accuracy)), kpi(t('Balanced acc.', 'Exactitud balanceada'), fmt.pct(main.balanced_accuracy)), kpi(t('F1 macro', 'F1 macro'), fmt.num(main.f1_macro)), kpi('mAP', fmt.num(main.mAP)), kpi('ROC-AUC', fmt.num(main.roc_auc_macro)));
    }
    const cols = kind === 'regression' ? ['r2', 'mae', 'rmse'] : ['accuracy', 'f1_macro', 'mAP', 'roc_auc_macro'];
    const head = { r2: 'R²', mae: 'MAE', rmse: 'RMSE', subset_accuracy: t('exact', 'exacta'), f1_macro: 'F1 macro', mAP: 'mAP', roc_auc_macro: 'ROC-AUC', accuracy: t('accuracy', 'exactitud') };
    const notes = [...(q.comparison ? q.comparison.notes || [] : []), ...(main.notes || [])];
    return h('div', {},
      h('div', { class: 'kpis' }, cards),
      h('p', { class: 'muted', text: t('Headline numbers use the test set (never used to choose the model).', 'Las cifras grandes usan el conjunto de test (nunca se usó para elegir el modelo).') }),
      table([t('Split', 'Conjunto'), ...cols.map((c) => head[c]), 'n'], keys.map((k) => [sname(k), ...cols.map((c) => (c === 'subset_accuracy' || c === 'accuracy' ? fmt.pct(sp[k][c]) : fmt.num(sp[k][c]))), fmt.int(sp[k].n)])),
      notes.map((n) => callout('info', n)));
  }

  function trainingCurves(history) {
    if (!history || !history.loss || !history.loss.length) {
      return callout('info', t('This model is not trained epoch by epoch, so there is no loss curve. Scikit-learn models are fitted in one step; use the Overview tab to check overfitting (train vs validation vs test).', 'Este modelo no se entrena por épocas, así que no hay curva de pérdida. Los modelos de scikit-learn se ajustan de una vez; usa la pestaña Resumen para ver el sobreajuste (train vs validación vs test).'));
    }
    const best = history.val_loss && history.val_loss.length ? history.val_loss.indexOf(Math.min(...history.val_loss)) + 1 : null;
    return h('div', {},
      ui.chart(null, charts.lines(t('Loss per epoch', 'Pérdida por época'), [{ name: 'train', values: history.loss }, { name: t('validation', 'validación'), values: history.val_loss || [] }]), 'loss-curve'),
      h('p', { class: 'muted', text: best ? t('Best validation loss at epoch {e} of {n}. If validation rises while train keeps falling, the model is overfitting.', 'Mejor pérdida de validación en la época {e} de {n}. Si la validación sube mientras la de train baja, hay sobreajuste.', { e: best, n: history.loss.length }) : '' }));
  }

  function confusionTab(m, classes) {
    if (!m || !m.confusion_matrix) return h('p', { class: 'muted', text: '—' });
    const box = h('div', {});
    let norm = false;
    const draw = () => box.replaceChildren(
      h('label', { class: 'row' }, h('input', { type: 'checkbox', checked: norm, onchange: (e) => { norm = e.target.checked; draw(); } }), ` ${t('Normalize by row (recall per class)', 'Normalizar por fila (recall por clase)')}`),
      ui.chart(norm ? t('Confusion matrix (row %)', 'Matriz de confusión (% por fila)') : t('Confusion matrix (counts)', 'Matriz de confusión (conteos)'), charts.confusion(m.confusion_matrix, classes, norm), 'confusion-matrix'));
    // the normalized matrix holds fractions: draw them as percentages with row-max 1
    draw();
    return box;
  }

  function perClassTab(m) {
    const pc = m.per_class;
    const names = Object.keys(pc);
    return h('div', {},
      ui.chart(t('Precision / recall / F1 per class', 'Precisión / recall / F1 por clase'), charts.grouped(names.map((n) => ({ label: n, values: [pc[n].precision, pc[n].recall, pc[n].f1] })), [t('precision', 'precisión'), 'recall', 'F1']), 'per-class-metrics'),
      table([t('Class', 'Clase'), t('precision', 'precisión'), 'recall', 'F1', 'AUC', 'AP', t('support', 'soporte')], names.map((n) => [n, fmt.num(pc[n].precision), fmt.num(pc[n].recall), fmt.num(pc[n].f1), fmt.num(pc[n].auc), fmt.num(pc[n].ap), fmt.int(pc[n].support)])));
  }

  function curvesTab(curves) {
    if (!curves || !curves.roc) return callout('info', t('This model does not provide class probabilities, so ROC/PR curves are not available.', 'Este modelo no entrega probabilidades por clase, así que no hay curvas ROC/PR.'));
    const names = Object.keys(curves.roc);
    const roc = names.map((n) => ({ name: n, x: curves.roc[n].fpr, y: curves.roc[n].tpr }));
    const pr = names.map((n) => ({ name: n, x: curves.pr[n].recall, y: curves.pr[n].precision }));
    return h('div', {},
      charts.legend(names.map((n) => `${n} · AUC ${fmt.num(curves.roc[n].auc)} · AP ${fmt.num(curves.pr[n].ap)}`)),
      h('div', { class: 'xyrow' }, ui.chart(null, charts.xy('ROC (one-vs-rest)', roc, t('false positive rate', 'tasa de falsos positivos'), t('true positive rate', 'tasa de verdaderos positivos'), true), 'roc'),
        ui.chart(null, charts.xy(t('Precision–recall', 'Precisión–recall'), pr, 'recall', t('precision', 'precisión'), false), 'precision-recall')));
  }

  function regressionTabs(m) {
    return [
      { id: 'scatter', title: t('Predicted vs true', 'Predicho vs real'), render: () => (m.scatter && m.scatter.y_true.length ? ui.chart(null, charts.scatter(t('true (x) vs predicted (y)', 'real (x) vs predicho (y)'), m.scatter.y_true, m.scatter.y_pred), 'predicted-vs-true') : h('p', { class: 'muted', text: '—' })) },
      { id: 'resid', title: t('Residuals', 'Residuos'), render: () => (m.residual_hist ? ui.chart(null, charts.hist1(t('residual = true − predicted', 'residuo = real − predicho'), m.residual_hist.bins, m.residual_hist.counts), 'residuals') : h('p', { class: 'muted', text: '—' })) },
    ];
  }

  // opts: {quality, history, classes, kind: 'classification'|'regression'}
  TS.qualityPanel = function qualityPanel(opts) {
    const q = opts.quality;
    if (!q || q.error) return callout('warn', t('Quality diagnostics are not available for this model.', 'Los diagnósticos de calidad no están disponibles para este modelo.'), q && q.error ? ` (${q.error})` : '');
    const kind = q.kind || opts.kind;
    const sp = q.splits || {};
    const test = sp.test || sp.test_real || {};
    const defs = [{ id: 'overview', title: t('Overview', 'Resumen'), render: () => overview(q, kind) },
      { id: 'training', title: t('Training curves', 'Curvas de entrenamiento'), render: () => trainingCurves(opts.history) }];
    if (kind === 'classification') {
      defs.push({ id: 'confusion', title: t('Confusion matrix', 'Matriz de confusión'), render: () => confusionTab(test, opts.classes || q.classes || test.classes) },
        { id: 'perclass', title: t('Per class', 'Por clase'), render: () => perClassTab(test) },
        { id: 'curves', title: 'ROC / PR', render: () => curvesTab(q.curves) });
    } else {
      defs.push(...regressionTabs(test));
    }
    return h('div', { class: 'quality', id: 'quality-panel' }, tabs(defs));
  };
})();
