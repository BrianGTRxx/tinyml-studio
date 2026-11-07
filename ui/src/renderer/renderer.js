'use strict';
// App shell: sidebar navigation (steps + tools), sticky footer, header controls, core status, catalogs.
(function () {
  const { h, S } = window.TS;
  const TS = window.TS;
  const view = document.getElementById('view');
  const nav = document.getElementById('steps');
  const MAX_STEP = 6;

  function reachable(i) {
    if (i === 0) return true;
    if (!S.report) return false;
    if (i === 6) return Boolean(S.train);
    return true;
  }
  const lockedWhy = (i) => (!S.report ? TS.t('Open a dataset first', 'Primero abre un dataset') : i === 6 ? TS.t('Train a model first', 'Primero entrena un modelo') : '');

  const SUBS = () => [
    TS.t('Load and understand your CSV files', 'Carga y entiende tus CSV'), TS.t('What to predict and what each column is', 'Qué predecir y qué es cada columna'),
    TS.t('Fix missing/absurd values', 'Arregla faltantes y valores absurdos'), TS.t('Optional: Make more training data', 'Opcional: Crea más datos de entrenamiento'),
    TS.t('Train / validation / test without leakage', 'Entrenamiento / validación / test sin fuga'), TS.t('Train and compare models', 'Entrena y compara modelos'),
    TS.t('Quantize, check it fits, get the code', 'Cuantiza, verifica que cabe y obtén el código'),
  ];

  TS.isSetup = () => Boolean(S.setup);

  TS.render = function render() {
    const t = TS.t;
    if (S.setup) return renderSetup();
    const names = TS.STEP_NAMES();
    const subs = SUBS();
    const tool = S.tool;
    S.reach = Math.max(S.reach || 0, S.step);
    const stepBtn = (i) => h('button', {
      type: 'button', 'data-step': i, title: reachable(i) ? subs[i] : lockedWhy(i),
      class: `step ${!tool && i === S.step ? 'active' : ''} ${reachable(i) ? '' : 'locked'} ${reachable(i) && i < S.reach && S.report ? 'done' : ''}`,
      'aria-current': !tool && i === S.step ? 'step' : null,
      disabled: !reachable(i), onclick: () => TS.go(i),
    }, h('span', { class: 'num' }, reachable(i) && i < S.reach && S.report ? TS.icon('check', 14) : String(i + 1)), h('span', { class: 'nm', text: names[i] }), h('span', { class: 'sub', text: subs[i] }));
    const toolBtn = (id, icon, label, on, extra, offTitle) => h('button', { type: 'button', id: `tool-${id}`, class: `tool ${tool === id ? 'active' : ''}`, onclick: on, disabled: extra === false, title: extra === false ? (offTitle || t('Open a dataset first', 'Primero abre un dataset')) : null, 'aria-current': tool === id ? 'page' : null }, TS.icon(icon, 18), label);
    nav.replaceChildren(
      h('div', { class: 'group', text: t('Workflow', 'Flujo de trabajo') }),
      ...names.map((_, i) => stepBtn(i)),
      h('div', { class: 'sep' }),
      h('div', { class: 'group', text: t('Tools', 'Herramientas') }),
      toolBtn('history', 'history', t('History', 'Historial'), () => { S.tool = 'history'; TS.render(); }),
      toolBtn('env', 'cpu', t('Environment', 'Entorno'), () => { S.tool = 'env'; TS.render(); }),
      toolBtn('updates', 'download', t('Updates', 'Actualizaciones'), () => { S.tool = 'updates'; TS.render(); }),
      toolBtn('report', 'file', t('PDF report', 'Informe PDF'), () => TS.makeReport && TS.makeReport(), S.train ? true : false, t('Train a model first: The report is about the trained models', 'Primero entrena un modelo: El informe trata de los modelos entrenados')),
      toolBtn('tour', 'compass', t('Tutorial', 'Tutorial'), () => TS.startTour && TS.startTour(true)),
    );
    nav.setAttribute('aria-label', t('Navigation', 'Navegación'));
    // keep the scroll position when the same screen is simply redrawn (a role changed, etc.)
    const where = `${tool || ''}:${S.step}`;
    const keepY = TS._lastWhere === where ? window.scrollY : 0;
    TS._lastWhere = where;
    view.replaceChildren();
    try {
      if (tool === 'history' && TS.historyView) TS.historyView(view);
      else if (tool === 'env' && TS.envView) TS.envView(view);
      else if (tool === 'updates' && TS.updatesView) TS.updatesView(view);
      else TS.views[S.step](view);
    } catch (e) {
      view.append(TS.err(e));
    }
    if (keepY) window.scrollTo(0, keepY);
    const prev = document.getElementById('prev');
    const next = document.getElementById('next');
    const footer = document.querySelector('footer.actions');
    footer.hidden = Boolean(tool);
    prev.disabled = S.step === 0;
    prev.textContent = t('← Back', '← Atrás');
    const ok = S.step < MAX_STEP && reachable(S.step + 1);
    next.disabled = !ok;
    next.textContent = S.step < MAX_STEP ? t('Next: {n} →', 'Siguiente: {n} →', { n: names[S.step + 1] }) : t('Done', 'Listo');
    next.title = ok ? '' : S.step < MAX_STEP ? lockedWhy(S.step + 1) : '';
    document.getElementById('where').textContent = t('Step {i} of 7', 'Paso {i} de 7', { i: S.step + 1 });
  };
  // No Python core yet (the app ships without packages) or the chosen environment failed to start:
  // show the first-run setup instead of the wizard.
  function renderSetup() {
    const t = TS.t;
    S.tool = 'setup'; // sin botón de ayuda de paso en la cabecera
    nav.replaceChildren(
      h('div', { class: 'group', text: t('Setup', 'Configuración') }),
      h('div', { class: 'sep' }),
      h('div', { class: 'group', text: t('Tools', 'Herramientas') }),
      h('button', { type: 'button', id: 'tool-tour', class: 'tool', onclick: () => TS.startTour && TS.startTour(true) }, TS.icon('compass', 18), t('Tutorial', 'Tutorial')),
    );
    nav.setAttribute('aria-label', t('Navigation', 'Navegación'));
    view.replaceChildren();
    document.querySelector('footer.actions').hidden = true;
    document.getElementById('where').textContent = t('First run', 'Primer arranque');
    TS.setupView(view, refreshCore).catch((e) => view.append(TS.err(e)));
  }
  async function refreshCore() {
    const info = await window.tinyml.envGet();
    S.setup = info.core === 'none' || Boolean(info.startError);
    if (!S.setup && S.tool === 'setup') S.tool = null;
    await initStatus();
    if (!S.setup) loadCatalogs();
    TS.render();
  }
  TS.go = (step) => { S.tool = null; S.step = step; TS.render(); window.scrollTo(0, 0); };

  // ---- header controls: language and theme (plain attributes, no animation) ----
  function seg(id, options, current, on) {
    const box = document.getElementById(id);
    box.replaceChildren(...options.map(([v, label, title]) => h('button', {
      type: 'button', class: v === current ? 'on' : '', title, 'aria-label': title, 'aria-pressed': v === current ? 'true' : 'false', 'data-v': v, onclick: () => on(v),
    }, ['sun', 'moon', 'auto'].includes(label) ? TS.icon(label, 16) : label)));
  }
  function chrome() {
    const t = TS.t;
    seg('lang-seg', [['en', 'EN', 'English'], ['es', 'ES', 'Español']], TS.lang, (v) => { TS.setLang(v); TS.relocalize(); });
    seg('theme-seg', [['light', 'sun', t('Light', 'Claro')], ['dark', 'moon', t('Dark', 'Oscuro')], ['system', 'auto', t('Follow system', 'Seguir al sistema')]],
      TS.themePref, (v) => { TS.setTheme(v); chrome(); });
    document.getElementById('lang-seg').setAttribute('aria-label', t('Language', 'Idioma'));
    document.getElementById('theme-seg').setAttribute('aria-label', t('Theme', 'Tema'));
    document.getElementById('btn-load-recipe').textContent = t('Load recipe…', 'Cargar receta…');
    document.getElementById('btn-save-recipe').textContent = t('Save recipe…', 'Guardar receta…');
    document.getElementById('btn-load-recipe').title = t('Load a saved recipe (JSON)', 'Cargar una receta guardada (JSON)');
    document.getElementById('btn-save-recipe').title = t('Save the configuration as a recipe (needs an open dataset)', 'Guardar la configuración como receta (requiere un dataset abierto)');
  }
  // language change: redraw and ask the core again for the messages that it produced in the old language
  TS.relocalize = async function relocalize() {
    chrome();
    // results the core already wrote in the previous language: translate them from the pairs it returned
    for (const k of ['clean', 'train', 'export', 'bench', 'gpuProbe']) if (S[k]) TS.retranslate(S[k], TS.lang);
    if (S.recipe) TS.retranslate(S.recipe.cleaning, TS.lang);
    if (S.lastHw && S.lastHw.refresh) await S.lastHw.refresh();
    await TS.refreshMessages();
    await initStatus();
    // the GPU probe also carries texts from the core (guidance): probe again in the new language if it had been run
    if (S.gpuProbe && !S.gpuProbe.error) { S.gpuProbe = null; }
    loadCatalogs();
    TS.render();
  };

  document.getElementById('prev').addEventListener('click', () => TS.go(Math.max(0, S.step - 1)));
  document.getElementById('next').addEventListener('click', () => TS.go(Math.min(MAX_STEP, S.step + 1)));
  document.getElementById('btn-save-recipe').addEventListener('click', async () => {
    if (!S.recipe) return;
    const r = JSON.parse(JSON.stringify(S.recipe));
    r.cleaning = (r.cleaning || []).filter((s) => !s._off).map((s) => { const c = { ...s }; delete c._off; return c; });
    await window.tinyml.saveText('recipe.json', JSON.stringify(r, null, 2));
  });
  document.getElementById('btn-load-recipe').addEventListener('click', async () => {
    try {
      const res = await window.tinyml.loadRecipe();
      if (res.canceled) return;
      const rep = await TS.api('POST', '/inspect', { paths: res.recipe.files });
      S.report = rep; S.recipe = res.recipe; S.paths = res.recipe.files; S.clean = S.train = S.export = null;
      document.getElementById('btn-save-recipe').disabled = false;
      S.tool = null; S.step = 1; TS.render();
    } catch (e) { view.replaceChildren(TS.err(e)); }
  });

  async function initStatus() {
    const status = document.getElementById('status');
    try {
      const hh = await window.tinyml.health();
      status.textContent = TS.t('core {v} running', 'núcleo {v} activo', { v: hh.version });
      status.className = 'badge ok';
    } catch (e) {
      status.textContent = S.setup ? TS.t('setup required', 'falta configurar') : TS.t('core unavailable', 'núcleo no disponible');
      status.className = S.setup ? 'badge warn' : 'badge bad';
    }
  }
  function loadCatalogs() {
    // catalogs do not block the UI. A response that arrives after a language switch is stale (its texts are in the
    // old language) and must not overwrite the newer one; each arrival redraws so the new texts show up.
    const lang = TS.lang;
    const keep = (set, always) => (x) => { if (lang !== TS.lang) return; set(x); if (always || (!S.tool && S.step >= 5)) TS.render(); };
    TS.api('GET', '/models').then(keep((c) => { S.catalog = c; }, true)).catch(() => {});
    TS.api('GET', '/hardware').then(keep((x) => { S.hardware = x; })).catch(() => {});
    TS.api('GET', '/boards').then(keep((x) => { S.boards = x; })).catch(() => {});
  }
  async function init() {
    chrome();
    try {
      const info = await window.tinyml.envGet();
      S.setup = info.core === 'none' || Boolean(info.startError);
    } catch { S.setup = false; }
    await initStatus();
    TS.render();
    if (!S.setup) loadCatalogs();
    if (window.tinyml.onCoreChanged) window.tinyml.onCoreChanged(() => refreshCore());
    if (TS.startTour && !S.setup) TS.startTour(false); // first run only
    if (TS.updatesInit && !S.setup) TS.updatesInit(); // asks once whether to tell about new versions; never installs
  }
  init();
})();
