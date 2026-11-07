'use strict';
/**
 * Prueba de extremo a extremo con la ventana REAL (Electron): recorre el asistente como un
 * usuario (clics sobre la interfaz), con el dataset de ejemplo y un modelo clásico, y comprueba
 * resultados y seguridad. Activar con TINYML_SMOKE=1. Capturas en TINYML_SMOKE_SHOTS (opcional).
 */
const fs = require('node:fs');
const path = require('node:path');

async function run({ win, sidecar, app, core }) {
  const out = { ok: false, steps: [] };
  // Modo empaquetado: el núcleo viene del paquete (congelado), no de TINYML_PYTHON; el dataset, de TINYML_SMOKE_PATHS.
  out.packaged = Boolean(app.isPackaged);
  out.core = core ? { kind: core.kind, command: core.command } : null;
  const shots = process.env.TINYML_SMOKE_SHOTS;
  const wc = win.webContents;
  const js = (code) => wc.executeJavaScript(code);
  const waitFor = async (cond, what, timeout = 240000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeout) {
      if (await js(cond)) return;
      await new Promise((r) => setTimeout(r, 250));
    }
    throw new Error(`timeout esperando: ${what}`);
  };
  const click = (sel) => js(`document.querySelector(${JSON.stringify(sel)}).click()`);
  const shot = async (name) => {
    if (!shots) return;
    fs.mkdirSync(shots, { recursive: true });
    await new Promise((r) => setTimeout(r, 400)); // deja pintar antes de capturar
    const img = await wc.capturePage();
    fs.writeFileSync(path.join(shots, `${name}.png`), img.toPNG());
  };
  const step = (name) => out.steps.push(name);
  try {
    if (process.env.TINYML_SMOKE_EXPECT_PACKAGED === '1' && process.env.TINYML_SMOKE_SETUP !== '1') {
      if (!out.packaged || !core || core.kind !== 'frozen') throw new Error('se esperaba la app empaquetada con el núcleo congelado');
      if (!require('node:fs').existsSync(core.command)) throw new Error(`no existe el núcleo empaquetado: ${core.command}`);
      step('app empaquetada + núcleo congelado');
    }
    win.setSize(1280, 1500);
    await new Promise((r) => wc.once('did-finish-load', r));
    await js(`window.TS.setLang(${JSON.stringify(process.env.TINYML_SMOKE_LANG || 'en')}); window.TS.setTheme(${JSON.stringify(process.env.TINYML_SMOKE_THEME || 'light')}); window.TS.relocalize()`);
    if (process.env.TINYML_SMOKE_SETUP === '1') {
      // App "cruda": sin núcleo ni paquetes. Se instala de verdad (uv + PyPI) desde la pantalla de configuración.
      if (!core || core.kind !== 'none') throw new Error('se esperaba la app cruda (sin núcleo)');
      await waitFor(`!!document.getElementById('inst-start')`, 'pantalla de configuración');
      out.setup = { twoOptions: await js(`!!document.getElementById('env-pick') && !!document.getElementById('inst-start')`) };
      if (!out.setup.twoOptions) throw new Error('la pantalla de configuración debe ofrecer los dos caminos');
      await shot('00-setup');
      // TINYML_SMOKE_COMPONENTS=keras,tui: marca componentes opcionales antes de instalar
      for (const id of (process.env.TINYML_SMOKE_COMPONENTS || '').split(',').filter(Boolean)) await js(`document.getElementById('inst-c-${id}').click()`);
      await click('#inst-start');
      await waitFor(`!!document.getElementById('inst-go')`, 'aviso de consentimiento');
      await shot('00-consent');
      const t0 = Date.now();
      await click('#inst-go');
      // captura a mitad de la instalación (barra de progreso y bytes descargados), si dura lo bastante
      await new Promise((r) => setTimeout(r, Number(process.env.TINYML_SMOKE_MIDSHOT_MS || 0)));
      if (process.env.TINYML_SMOKE_MIDSHOT_MS) {
        out.setup.midProgress = await js(`(() => { const b = document.getElementById('inst-progress'); const s = document.getElementById('inst-status'); return { value: b && b.value, status: s && s.textContent }; })()`);
        await shot('00-installing');
      }
      await waitFor(`document.getElementById('status').classList.contains('ok')`, 'instalación y arranque del núcleo', 1500000);
      out.setup.installSeconds = Math.round((Date.now() - t0) / 1000);
      out.setup.kindAfter = await js(`window.tinyml.envGet().then((e) => e.core)`);
      if (out.setup.kindAfter !== 'managed') throw new Error(`tras instalar se esperaba el entorno gestionado, no ${out.setup.kindAfter}`);
      step('instalación desde cero + núcleo gestionado activo');
    }
    out.security = await js(`(async () => ({
      health: (await window.tinyml.health()).status,
      hasRequire: typeof require !== 'undefined',
      hasProcess: typeof process !== 'undefined',
      bridgeKeys: Object.keys(window.tinyml).sort(),
      forbidden: await window.tinyml.request('POST', '/roots', { path: '/' }).then(() => 'permitido', (e) => e.message),
      forbidden2: await window.tinyml.request('GET', '/jobs/../../etc/passwd').then(() => 'permitido', (e) => e.message),
    }))()`);
    out.cspBlocksRemote = await js(`fetch('https://example.com').then(() => false, () => true)`);

    await waitFor(`document.getElementById('status').classList.contains('ok')`, 'núcleo');
    step('núcleo activo');
    if (process.env.TINYML_SMOKE_FIXTURE === '1') {
      // piezas de interfaz con datos de prueba (nombres largos, 3 series, caja de código): sin recorrer el asistente
      const f = FIXTURE_JS;
      for (const lang of ['es', 'en']) {
        await js(`window.TS.setLang(${JSON.stringify(lang)}); window.TS.relocalize()`);
        await new Promise((r) => setTimeout(r, 800));
        for (const width of [1280, 900]) {
          win.setContentSize(width, 1100);
          await js(f);
          await new Promise((r) => setTimeout(r, 500));
          out[`fixture_${lang}_${width}`] = await js(PAGE_AUDIT);
          await shot(`fx-${lang}-${width}`);
        }
      }
      const done = new Error('fixture');
      done.fixtureDone = true;
      throw done;
    }
    // Entorno Python: la pantalla corre el doctor con el núcleo y muestra el informe (sin cambiar nada)
    await click('#tool-env');
    await waitFor(`!!document.getElementById('env-exe') && document.querySelectorAll('#env-box table').length > 0`, 'entorno');
    out.env = await js(`({ exe: document.getElementById('env-exe').textContent, stats: document.querySelectorAll('#env-box .stat').length, pick: !!document.getElementById('env-pick') })`);
    if (!out.env.exe || out.env.stats < 3 || !out.env.pick) throw new Error('pantalla Entorno incompleta');
    step('entorno (doctor)');
    // Pythons ya instalados en el equipo (búsqueda de solo lectura): la lista sale con veredicto y botón "Usar este"
    await click('#py-scan-btn');
    await waitFor(`!!document.querySelector('#py-scan-out table, #py-scan-out .callout')`, 'búsqueda de Python');
    out.pyScan = await js(`document.getElementById('py-scan-out').innerText.slice(0, 500)`);
    out.pyScanUseButtons = await js(`document.querySelectorAll('#py-scan-out button[data-id]').length`);
    await js(`document.getElementById('py-scan-out').scrollIntoView()`);
    step('Pythons detectados');
    // aviso de versión nueva: la primera vez pregunta (sin salir a internet) y hay una página de Actualizaciones
    await waitFor(`!!document.querySelector('#update-banner:not([hidden]) #upd-yes')`, 'pregunta de avisos');
    out.updateAsk = await js(`document.getElementById('update-banner').innerText.slice(0, 160)`);
    await click('#tool-updates');
    await waitFor(`!!document.getElementById('upd-check')`, 'página de actualizaciones');
    out.updatesVersion = await js(`document.getElementById('upd-version').textContent`);
    await click('#upd-no'); // el usuario dice que no: el aviso desaparece y no se guarda nada más
    await waitFor(`document.getElementById('update-banner').hidden`, 'aviso cerrado');
    step('avisos de versión');
    await shot('0-env');
    await js(`document.querySelector('#steps [data-step="0"]').click()`);
    await click('#btn-pick-folder');
    await waitFor(`!!document.getElementById('inspect-done')`, 'inspect');
    out.inspect = await js(`document.querySelectorAll('#dataset-box table')[0].querySelectorAll('tbody tr').length`);
    step('dataset analizado');
    // "Recientes": vuelve a abrir el último conjunto con un clic (el renderer solo manda el número)
    await js(`window.TS.go(1)`);
    await js(`window.TS.go(0)`);
    await waitFor(`document.querySelectorAll('#recent-box button[data-recent]').length > 0`, 'recientes', 20000);
    await click('#recent-box button[data-recent]');
    await waitFor(`!!document.getElementById('inspect-done')`, 'inspect desde recientes');
    out.recent = await js(`document.querySelectorAll('#recent-box button[data-recent]').length`);
    step('abrir reciente');
    await shot('1-dataset');

    await click('#next');
    await waitFor(`!!document.getElementById('lff')`, 'paso tarea');
    out.lff = await js(`document.getElementById('lff').checked`);
    out.rolesBefore = await js(`document.querySelectorAll('#view .stat b')[0].textContent`);
    await click('#roles-numeric');
    out.rolesNumeric = await js(`document.querySelectorAll('#view .stat b')[0].textContent`);
    await click('#roles-reset');
    out.rolesAfter = await js(`document.querySelectorAll('#view .stat b')[0].textContent`);
    out.roleLabels = await js(`[...document.querySelectorAll('#view .stat span')].map((x) => x.textContent)`);
    out.encodingCard = await js(`!!document.querySelector('#encoding-card select')`);
    if (process.env.TINYML_SMOKE_SCENARIO === 'benchmark') {
      await js(`(() => { const s = document.querySelector('#encoding-card select'); s.value = 'dummy'; s.dispatchEvent(new Event('change')); })()`);
      out.encoding = await js(`window.TS.S.recipe.encoding.default`);
    }
    await shot('2-tarea');
    if (process.env.TINYML_SMOKE_PICKCOLS) {
      // "Elegir exactamente qué columnas usar": marca solo las pedidas y comprueba los roles resultantes
      const want = process.env.TINYML_SMOKE_PICKCOLS.split(',');
      await js(`(() => { document.querySelector('.cols-pick').closest('details').open = true;
        for (const l of document.querySelectorAll('.cols-pick label')) { const i = l.querySelector('input'); const on = ${JSON.stringify(want)}.includes(l.dataset.col); if (i.checked !== on) i.click(); } })()`);
      await shot('2b-columnas');
      await click('#roles-only-apply');
      await new Promise((r) => setTimeout(r, 600));
      out.pickCols = await js(`(() => { const r = window.TS.S.recipe.roles; return { features: Object.keys(r).filter((k) => r[k] === 'feature'), target: Object.keys(r).filter((k) => r[k] === 'target'), unused: Object.keys(r).filter((k) => ['meta', 'drop'].includes(r[k])).length }; })()`);
    }
    step('tarea');

    await click('#next');
    await waitFor(`!!document.getElementById('run-clean')`, 'paso limpieza');
    await click('#run-clean');
    await waitFor(`!!document.getElementById('clean-results')`, 'limpieza');
    out.hists = await js(`document.querySelectorAll('#clean-results svg.hist').length`);
    await shot('3-limpieza');
    step('limpieza con gráficas');

    await click('#next'); // aumento
    await shot('4-aumento');
    await click('#next');
    await waitFor(`!!document.getElementById('run-split')`, 'paso partición');
    await click('#run-split');
    await waitFor(`!!document.getElementById('split-results')`, 'tamaños');
    out.splitRows = await js(`document.querySelectorAll('#split-results table tbody tr').length`);
    await shot('5-particion');
    step('partición');

    const scenario = process.env.TINYML_SMOKE_SCENARIO || 'classical';
    out.scenario = scenario;
    if (scenario === 'benchmark') {
      await js(`(() => {
        const S = window.TS.S;
        S.family = 'benchmark';
        for (const p of S.catalog.keras_mlp.presets) p.params.epochs = 5;
        S.bench = { models: ['decision_tree', 'logreg', 'knn', 'gaussian_nb'], presets: ['small'], cv: 0, selection: 'f1_macro', maxFlash: '' };
      })()`);
    } else if (scenario !== 'classical') {
      // esquemas distintos: se preparan igual que lo haría el usuario en los pasos 4 y 6
      await js(`(() => {
        const S = window.TS.S;
        S.family = { keras: 'keras_mlp', keras_torch: 'keras_mlp', cnn1d: 'keras_cnn1d' }[${JSON.stringify(scenario)}];
        if (${JSON.stringify(scenario)} === 'keras_torch') { S.backend = 'torch'; S.device = 'gpu'; }
        S.hp = { epochs: 3, batch_size: 256, hidden: [16, 8], window: 8, stride: 8, filters: [8, 8], kernel: 3, head: 'gap', learning_rate: 0.002, patience: 3, ...(${JSON.stringify(scenario)} === 'keras_torch' ? { optimizer: 'sgd', nesterov: true, lr_schedule: 'cosine' } : {}) };
      })()`);
    }
    await click('#next');
    await waitFor(`!!document.getElementById('run-train')`, 'paso entrenamiento');
    if (scenario === 'benchmark') await js(`window.TS.render()`);
    if (scenario === 'classical') {
      // solo árbol y random forest para que la prueba sea rápida
      await js(`(() => { for (const c of document.querySelectorAll('#hp-box .ms-item input[type=checkbox]')) { const n = c.parentElement.dataset.value; if (!['decision_tree','random_forest'].includes(n) === c.checked) c.click(); } })()`);
    }
    if (scenario === 'classical') { await click('#btn-gpu-check'); await waitFor(`!!document.querySelector('#gpu-panel table')`, 'gpu probe', 120000); out.gpuPanel = await js(`document.querySelector('#gpu-panel').innerText.slice(0, 400)`); await shot('6-gpu'); }
    if (scenario === 'classical') { await js(`document.querySelector('#model-picker .ms-btn').click()`); await shot('6-picker-open'); await js(`document.querySelector('#model-picker .ms-btn').click()`); }
    await click('#run-train');
    await waitFor(`!!document.getElementById('train-results')`, 'entrenamiento');
    out.trainText = await js(`document.getElementById('train-results').innerText.slice(0, 400)`);
    if (scenario === 'benchmark') {
      await waitFor(`!!document.getElementById('benchmark-results')`, 'benchmark');
      out.benchRows = await js(`document.querySelectorAll('#benchmark-results tbody tr').length`);
      out.benchSelected = await js(`document.querySelector('#benchmark-results tr.sel') && document.querySelector('#benchmark-results tr.sel').dataset.model`);
      await js(`document.querySelector('#benchmark-results th[data-sort="test:mcc"]').click()`);
      out.benchSorted = await js(`document.querySelector('#benchmark-results tbody tr').dataset.model`);
      await shot('6-benchmark');
      step('benchmark');
    } else {
    await click('#quality-panel [data-tab=training]');
    out.curve = await js(`!!document.querySelector('#train-results svg.lines')`);
    out.qualityTabs = await js(`[...document.querySelectorAll('#quality-panel [data-tab]')].map((b) => b.dataset.tab)`);
    await click('#quality-panel [data-tab=overview]');
    out.overview = await js(`document.querySelector('#quality-panel .kpis').innerText`);
    for (const tab of out.qualityTabs) {
      await click(`#quality-panel [data-tab=${tab}]`);
      out['tab_' + tab] = await js(`document.querySelectorAll('#quality-panel .tabbody svg, #quality-panel .tabbody table').length`);
    }
    await click('#quality-panel [data-tab=overview]');
    if (scenario === 'classical') {
      // chart export to PNG / JPG / SVG through the real save path (smoke writes into TINYML_SMOKE_SAVE_DIR)
      await click('#quality-panel [data-tab=confusion]');
      out.chartExports = await js(`(async () => { const el = document.querySelector('#quality-panel .chartbox svg'); const r = {}; for (const f of ['png', 'jpg', 'svg']) r[f] = await window.TS.exportChart(el, 'smoke-confusion', f); return r; })()`);
      await shot('6-confusion');
      await click('#quality-panel [data-tab=overview]');
    }
    }
    out.compute = await js(`(() => { const r = window.TS.S.train.result.results; return r && r.compute ? { backend: r.compute.backend, device: r.compute.device_requested, notes: r.compute.notes, transfer: r.compute.torch_to_tf_transfer, opt: r.hyperparameters && r.hyperparameters.optimizer, sched: r.hyperparameters && r.hyperparameters.lr_schedule } : null; })()`);
    await shot('6-entrenamiento');
    step('entrenamiento');
    if (scenario === 'classical') {
      // validación por sesiones: abre la tarjeta, la ejecuta y recoge el veredicto
      await js(`document.getElementById('run-groups').closest('details').open = true`);
      await click('#run-groups');
      await waitFor(`!!document.getElementById('group-results')`, 'validación por sesiones', 240000);
      out.groupValidation = await js(`document.getElementById('group-results').innerText.slice(0, 600)`);
      await js(`document.getElementById('group-results').scrollIntoView()`);
      await shot('6-sesiones');
      step('validación por sesiones');
      // botón "marcar columnas de sesión como metadatos": solo existe si la validación encontró alguna
      if (await js(`!!document.getElementById('btn-mark-meta')`)) {
        out.markMeta = await js(`document.getElementById('btn-mark-meta').innerText`);
        await click('#btn-mark-meta');
        await waitFor(`/Antes:|Before:/.test(document.getElementById('group-results') ? document.getElementById('group-results').innerText : '')`, 'repetir validación', 240000);
        out.groupValidationAfter = await js(`document.getElementById('group-results').innerText.slice(0, 400)`);
        await shot('6b-sesiones-sin-columnas');
        step('columnas de sesión como metadatos');
        await click('#run-train'); // el modelo entrenado usaba esas columnas: se reentrena para seguir con la exportación
        await waitFor(`!!window.TS.S.train`, 'reentrenamiento sin columnas de sesión');
      }
    }

    await click('#next');
    await waitFor(`!!document.getElementById('run-export')`, 'paso exportación');
    if (!process.env.TINYML_SMOKE_BOARD) await js(`(() => { const i = document.querySelectorAll('#view input[type=number]'); i[0].value = 256; i[0].dispatchEvent(new Event('input')); i[1].value = 32; i[1].dispatchEvent(new Event('input')); })()`);
    if (scenario === 'classical') { await click('#compact'); out.compactTicked = await js(`document.getElementById('compact').checked`); }
    if (process.env.TINYML_SMOKE_BOARD) {
      await js(`(() => { const s = document.getElementById('board-select'); s.value = ${JSON.stringify(process.env.TINYML_SMOKE_BOARD)}; s.dispatchEvent(new Event('change')); })()`);
    }
    out.boardGroups = await js(`[...document.querySelectorAll('#board-select optgroup')].map((g) => g.label + ':' + g.children.length)`);
    if (!process.env.TINYML_SMOKE_BOARD) {
      await js(`(() => { const s = document.getElementById('board-select'); s.value = 'arduino-uno-r3'; s.dispatchEvent(new Event('change')); })()`);
      await shot('7-board-uno');
      await js(`(() => { const s = document.getElementById('board-select'); s.value = 'esp32-s3'; s.dispatchEvent(new Event('change')); })()`);
    }
    out.boardBudget = await js(`[...document.querySelectorAll('#view input[type=number]')].slice(0, 2).map((i) => i.value)`);
    await click('#run-export');
    await waitFor(`!!document.getElementById('export-results')`, 'exportación');
    out.fitBanner = await js(`(() => { const b = document.getElementById('fit-banner') || document.querySelector('#export-results .callout'); return b ? b.innerText.slice(0, 700) : null; })()`);
    out.exportText = await js(`document.getElementById('export-results').innerText.slice(0, 600)`);
    out.arduino = await js(`[...document.querySelectorAll('#arduino-panel .arduino')].map((c) => ({ v: c.dataset.variant, lines: c.querySelector('pre').textContent.split('\\n').length, ok: c.querySelector('.callout').className }))`);
    out.graphNodes = await js(`document.querySelectorAll('#model-graph .gnode').length`);
    if (out.graphNodes) {
      await js(`document.querySelector('#model-graph .gnode').dispatchEvent(new MouseEvent('click'))`);
      out.graphDetails = await js(`document.querySelector('#model-graph .side').innerText.slice(0, 200)`);
    }
    // tarjetas nuevas del paso 7: con qué se entrenó el modelo y la prueba con valores escritos
    out.usedCard = await js(`(() => { const s = [...document.querySelectorAll('#view details > summary')].find((x) => /trained with|se entren/.test(x.textContent)); return s ? s.parentElement.innerText.slice(0, 300) : null; })()`);
    if (scenario === 'classical') {
      await js(`[...document.querySelectorAll('#view details > summary')].find((x) => /Try the model|Prueba el modelo/.test(x.textContent)).click()`);
      await waitFor(`!!document.getElementById('try-run')`, 'formulario de prueba', 60000);
      await click('#try-run'); // sin valores: todo "falta", el modelo usa el relleno aprendido
      await waitFor(`!!document.getElementById('try-class') || !!document.getElementById('try-value')`, 'respuesta de la prueba', 60000);
      out.tryAnswer = await js(`(document.getElementById('try-class') || document.getElementById('try-value')).textContent`);
      await js(`window.scrollTo(0, 0)`);
    }
    await shot('7-exportacion');
    await js(`(document.getElementById('arduino-panel') || document.body).scrollIntoView()`);
    await shot('8-arduino');
    await js(`(document.getElementById('model-graph') || document.body).scrollIntoView()`);
    await shot('9-estructura');
    await js(`window.scrollTo(0, 0)`);
    // same screen in the other theme + memory footprint of every Electron process
    const dark = (process.env.TINYML_SMOKE_THEME || 'light') !== 'dark';
    await js(`window.TS.setTheme(${dark ? "'dark'" : "'light'"}); window.TS.relocalize()`);
    await shot(dark ? '7-exportacion-dark' : '7-exportacion-light');
    out.theme = await js(`document.documentElement.getAttribute('data-theme')`);
    out.memoryMB = Object.fromEntries(app.getAppMetrics().map((p) => [`${p.type}:${p.pid}`, Math.round(p.memory.workingSetSize / 1024)]));
    out.memoryTotalMB = Object.values(out.memoryMB).reduce((a, b) => a + b, 0);
    await click('#btn-save-artifacts');
    await waitFor(`document.getElementById('saved-msg').textContent.length > 0`, 'guardado', 30000);
    out.saved = await js(`document.getElementById('saved-msg').textContent`);
    step('exportación y guardado');
    // botón propio del sketch de Arduino: guarda y dice la ruta EXACTA del .ino
    if (await js(`!!document.getElementById('btn-save-arduino')`)) {
      await click('#btn-save-arduino');
      await waitFor(`document.getElementById('arduino-saved').textContent.length > 0`, 'guardado del sketch', 30000);
      out.arduinoSaved = await js(`document.getElementById('arduino-saved').textContent`);
      const ino = out.arduinoSaved.split(': ').pop();
      out.arduinoInoExists = fs.existsSync(ino);
      step('sketch de Arduino guardado');
    }

    // guided tour: opens, moves Next/Back and can be skipped
    await js(`window.TS.startTour(true)`);
    out.tourOpen = await js(`!!document.getElementById('tour-pop')`);
    await shot('10-tour-1');
    await click('#tour-next'); await click('#tour-next');
    out.tourTitle = await js(`document.querySelector('#tour-pop h4').textContent`);
    await shot('10-tour-3');
    await click('#tour-back');
    await click('#tour-skip');
    out.tourClosed = await js(`!document.getElementById('tour-pop')`);
    // PDF report
    out.report = await js(`window.TS.makeReport()`);
    // history view
    await click('#tool-history');
    await new Promise((r) => setTimeout(r, 600));
    await shot('11-history');
    // la pantalla Entorno con la lista de Pythons detectados (al final: la ventana ya está pintada)
    await click('#tool-env');
    await waitFor(`!!document.getElementById('py-scan-btn')`, 'entorno');
    await click('#py-scan-btn');
    await waitFor(`!!document.querySelector('#py-scan-out table')`, 'lista de Pythons');
    await js(`document.getElementById('py-scan-out').scrollIntoView()`);
    await shot('12-entorno-pythons');
    if (process.env.TINYML_SMOKE_AUDIT === '1') out.audit = await runAudit(win, wc, js, click);
    if (process.env.TINYML_SMOKE_MANUAL === '1') out.manual = await runManual(win, js, click, shot);
    out.sidecarPort = sidecar.port;
    out.ok = true;
  } catch (e) {
    if (e && e.fixtureDone) out.ok = true;
    else out.error = String(e && e.stack ? e.stack : e).slice(0, 1500);
    try {
      await shot('error');
    } catch {
      /* sin captura */
    }
  }
  try {
    const dir = process.env.TINYML_SMOKE_SAVE_DIR;
    if (dir && out.chartExports) {
      const head = (f) => fs.readFileSync(path.join(dir, f)).subarray(0, 8);
      out.chartFiles = { png: head('smoke-confusion.png').equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])), jpg: head('smoke-confusion.jpg')[0] === 0xff, svg: fs.readFileSync(path.join(dir, 'smoke-confusion.svg'), 'utf8').startsWith('<svg') };
    }
    if (out.report && out.report.path) out.pdf = { magic: fs.readFileSync(out.report.path).subarray(0, 5).toString(), bytes: out.report.bytes };
  } catch (e) { out.verifyError = String(e); }
  console.log('SMOKE_RESULT ' + JSON.stringify(out));
  // En Windows un exe de GUI no escribe en la consola: el CI lee el resultado de este archivo.
  if (process.env.TINYML_SMOKE_OUT) {
    try {
      fs.writeFileSync(process.env.TINYML_SMOKE_OUT, JSON.stringify(out), 'utf8');
    } catch { /* sin archivo de resultado */ }
  }
  app.exit(out.ok ? 0 : 1);
}


const FIXTURE_JS = `(() => {
  const TS = window.TS, h = TS.h;
  const view = document.getElementById('view');
  const long = ['hours_studied','attendance','parental_education_level=High School','parental_education_level=Postgraduate','distance_from_home=Moderate','extracurricular_activities','access_to_resources=Medium'];
  const coefficients = long.map((feature, i) => ({ feature, value: (i % 2 ? -1 : 1) * (0.5 - i * 0.06) }));
  const graph = TS.graphView({ format: 'sklearn', model: { type: 'linear', classes: [{ class: '0', intercept: -0.4772, coefficients }] }, summary: {} });
  const series = [{ name: 'CPU %', values: [0, 10, 9, 18, 5] }, { name: TS.t('job CPU %', 'CPU trabajo %'), values: [0, 5, 5, 8, 0] }, { name: 'GPU %', values: [0, 0, 0, 0, 0] }];
  const lines = TS.ui.chart(TS.t('Usage history', 'Historial de uso'), TS.charts.lines(TS.t('Load (%)', 'Carga (%)'), series, TS.t('time (last {s} s)', 'tiempo (últimos {s} s)', { s: 4 })), 'fx-lines');
  const code = h('div', { class: 'codebox' }, h('button', { type: 'button', text: TS.t('Copy', 'Copiar') }), h('pre', { class: 'code', text: Array.from({ length: 60 }, (_, i) => 'static const float SAMPLE_NUM[' + i + '] = {19.0f, 64.0f};').join('\\n') }));
  view.replaceChildren(h('div', { class: 'sec' }, graph, lines, code));
})()`;

// ---- capturas para el manual (TINYML_SMOKE_MANUAL=1): recorre el asistente ya completado y fotografía cada pantalla ----
// se ejecuta DENTRO de la página (se serializa con toString): sin rutas ni nombre de usuario en las capturas
function manualClean() {
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = w.nextNode(); n; n = w.nextNode()) n.nodeValue = n.nodeValue.replace(/[A-Za-z]:\Users\[^\\s]+/g, 'C:\Users\you').replace(/\/home\/[^\/\s]+/g, '/home/you').replace(/AMD Ryzen[^·]*/g, 'CPU ').replace(/NVIDIA GeForce[^,.)·]*/g, 'NVIDIA GPU').replace(/AMD Radeon[^,.)·]*/g, 'AMD GPU');
}
const MANUAL_CLEAN = `(${manualClean.toString()})()`;
async function runManual(win, js, click, shot) {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const out = [];
  await js(`window.TS.setLang('en'); window.TS.setTheme('light'); window.TS.relocalize()`);
  win.setContentSize(1280, 1500);
  await wait(1500);
  const snap = async (name, open) => {
    await js(`window.scrollTo(0, 0); document.querySelectorAll('#view details').forEach((d) => { d.open = ${open ? 'true' : 'false'}; })`);
    await wait(700);
    await js(MANUAL_CLEAN);
    await shot(name);
    out.push(name);
  };
  for (let i = 0; i < 7; i += 1) {
    await js(`window.TS.go(${i})`);
    await wait(1200);
    await snap(`m-step${i + 1}`, false);
    if ([0, 1, 5, 6].includes(i)) await snap(`m-step${i + 1}-open`, true);
  }
  await click('#tool-env');
  await wait(1500);
  await snap('m-env', false);
  await click('#tool-history');
  await wait(1500);
  await snap('m-history', false);
  await js(`window.TS.go(0); window.TS.startTour(true)`);
  await wait(1200);
  await shot('m-tour');
  out.push('m-tour');
  return out;
}

// ---- auditoría visual: textos/botones que se salen de su caja y textos en el idioma equivocado --------
const PAGE_AUDIT = `(() => {
  const found = { overflow: [], offscreen: [], svgclip: [], lang: [], a11y: [], contrast: [] };
  const vw = document.documentElement.clientWidth;
  const view = document.getElementById('view') || document.body;
  const vis = (e) => { const r = e.getBoundingClientRect(); const c = getComputedStyle(e); return r.width > 0 && r.height > 0 && c.visibility !== 'hidden' && c.display !== 'none'; };
  const sel = (e) => (e.id ? '#' + e.id : e.tagName.toLowerCase() + (typeof e.className === 'string' && e.className.trim() ? '.' + e.className.trim().split(/\s+/).slice(0, 2).join('.') : ''));
  const snip = (e) => (e.innerText || e.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 60);
  const scrollable = (e) => { for (let p = e.parentElement; p && p !== document.body; p = p.parentElement) { const o = getComputedStyle(p).overflowX; if (o === 'auto' || o === 'scroll' || o === 'hidden') return true; } return false; };
  for (const e of view.querySelectorAll('*')) {
    if (e.closest('svg') || e.closest('pre') || !vis(e)) continue;
    const c = getComputedStyle(e);
    if (e.clientWidth > 0 && e.scrollWidth > e.clientWidth + 2 && c.overflowX === 'visible' && c.display !== 'inline' && !scrollable(e) && snip(e)) found.overflow.push(sel(e) + ' | ' + snip(e) + ' | ' + e.scrollWidth + '>' + e.clientWidth);
    const r = e.getBoundingClientRect();
    if (r.right > vw + 1 && !scrollable(e) && snip(e)) found.offscreen.push(sel(e) + ' | ' + snip(e) + ' | right=' + Math.round(r.right) + ' vw=' + vw);
  }
  for (const sv of view.querySelectorAll('svg[viewBox]')) {
    if (!vis(sv)) continue;
    const vb = sv.viewBox.baseVal;
    for (const t of sv.querySelectorAll('text')) { try { const b = t.getBBox(); if (b.x < vb.x - 1 || b.x + b.width > vb.x + vb.width + 1) found.svgclip.push((sv.getAttribute('aria-label') || 'svg') + ' | ' + (t.textContent || '').slice(0, 40) + ' | x=' + Math.round(b.x) + '..' + Math.round(b.x + b.width) + ' of ' + Math.round(vb.width)); } catch {} }
  }
  const SPA = new Set(['el','la','los','las','del','para','con','sin','una','por','que','esta','este','se','al','más','como','tus','tu','sus','su']);
  const ENG = new Set(['the','and','with','for','your','is','are','to','of','this','that','from','not','or','it','be','use','uses','you','on','in','by']);
  const wantEn = document.documentElement.lang === 'en';
  for (const e of view.querySelectorAll('*')) {
    if (e.closest('svg') || e.closest('pre') || e.closest('code') || e.children.length || !vis(e)) continue;
    const txt = (e.textContent || '').trim();
    if (txt.length < 12) continue;
    const words = txt.toLowerCase().match(/[a-záéíóúñü]+/g) || [];
    const distinct = (set) => new Set(words.filter((w) => set.has(w))).size;
    const accent = /[áéíóúñ¿¡]/i.test(txt);
    if (wantEn && (accent || distinct(SPA) >= 2)) found.lang.push('ES-in-EN | ' + sel(e) + ' | ' + txt.slice(0, 90));
    if (!wantEn && distinct(ENG) >= 2 && !accent) found.lang.push('EN-in-ES | ' + sel(e) + ' | ' + txt.slice(0, 90));
  }
  // accesibilidad: controles sin nombre y texto con poco contraste
  const nameOf = (e) => (e.getAttribute('aria-label') || e.getAttribute('title') || (e.labels && e.labels.length ? e.labels[0].textContent : '') || e.textContent || e.value || e.getAttribute('placeholder') || '').trim();
  for (const e of view.querySelectorAll('button, a[href], input:not([type=hidden]), select, textarea, summary, [role=button]')) {
    if (!vis(e)) continue;
    if (!nameOf(e) && !(e.querySelector && e.querySelector('svg[aria-label], img[alt]'))) found.a11y.push('sin nombre | ' + sel(e) + ' | ' + e.outerHTML.slice(0, 110));
  }
  for (const e of view.querySelectorAll('img:not([alt])')) found.a11y.push('img sin alt | ' + sel(e));
  const lum = (c) => { const v = c.map((x) => { x /= 255; return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); }); return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]; };
  const rgba = (str) => { const m = str.match(/rgba?\(([^)]+)\)/); if (!m) return null; const p = m[1].split(',').map((x) => parseFloat(x)); return { c: p.slice(0, 3), a: p.length > 3 ? p[3] : 1 }; };
  const bgOf = (e) => { for (let p = e; p; p = p.parentElement) { const b = rgba(getComputedStyle(p).backgroundColor); if (b && b.a > 0.9) return b.c; } return [255, 255, 255]; };
  const seen = new Set();
  for (const e of view.querySelectorAll('*')) {
    if (e.closest('svg') || e.closest('pre') || e.children.length || !vis(e) || e.disabled) continue;
    const txt = (e.textContent || '').trim();
    if (txt.length < 2 || e.closest('[disabled]') || e.closest('.locked')) continue;
    const cs = getComputedStyle(e);
    const fg = rgba(cs.color);
    if (!fg) continue;
    const bg = bgOf(e);
    const l1 = lum(fg.c); const l2 = lum(bg);
    const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
    const size = parseFloat(cs.fontSize); const bold = parseInt(cs.fontWeight, 10) >= 700;
    const need = size >= 24 || (size >= 18.66 && bold) ? 3 : 4.5;
    const key = sel(e) + cs.color + JSON.stringify(bg);
    if (ratio < need && !seen.has(key) && found.contrast.length < 12) { seen.add(key); found.contrast.push(ratio.toFixed(2) + ' < ' + need + ' | ' + sel(e) + ' | ' + txt.slice(0, 40)); }
  }
  return found;
})()`;
async function runAudit(win, wc, js, click) {
  const res = {};
  const targets = [0, 1, 2, 3, 4, 5, 6, 'env', 'history'];
  for (const [lang, theme] of [['es', 'light'], ['en', 'light'], ['en', 'dark']]) {
    await js(`window.TS.setLang(${JSON.stringify(lang)}); window.TS.setTheme(${JSON.stringify(theme)}); window.TS.relocalize()`);
    await new Promise((r) => setTimeout(r, 1500));
    for (const width of [1280, 900]) {
      win.setContentSize(width, 900);
      await new Promise((r) => setTimeout(r, 500));
      for (const t of targets) {
        if (typeof t === 'number') await js(`window.TS.go(${t})`); else await click(`#tool-${t}`);
        await new Promise((r) => setTimeout(r, 900));
        const f = await js(PAGE_AUDIT);
        const n = f.overflow.length + f.offscreen.length + f.svgclip.length + f.lang.length + f.a11y.length + f.contrast.length;
        if (n) res[`${lang}/${width}/${theme}/${t}`] = f;
      }
    }
  }
  win.setContentSize(1280, 860);
  return res;
}

module.exports = { run };
