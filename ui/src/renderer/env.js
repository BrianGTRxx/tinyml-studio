'use strict';
// Environment: which Python runs the core, what it is missing, and the exact command to fix it.
// The path of an own interpreter (venv / conda) only comes from the native dialog in the main process.
(function () {
  const TS = window.TS;
  const { h } = TS;
  const ui = TS.ui;
  const t = TS.t;

  const COMPONENT_IDS = ['base', 'tui', 'keras', 'torch', 'tflm'];
  const STATUS = () => ({
    ok: t('OK', 'OK'), missing: t('missing', 'falta'), too_old: t('too old', 'antiguo'), too_new: t('too new', 'muy nuevo'),
  });
  const KIND = () => ({
    frozen: t('built-in core', 'núcleo incluido'), venv: t('virtual environment', 'entorno virtual'),
    conda: 'conda', system: t('system Python', 'Python del sistema'),
  });


  // ---- Pythons already installed on this computer (read-only scan; the app never changes them) -----------
  const VERDICT = () => ({
    recommended: ['', t('recommended', 'recomendado')],
    ok: ['', t('works (3.10 is the minimum)', 'sirve (3.10 es el mínimo)')],
    newer: ['high', t('newer than tested: TensorFlow may not support it yet', 'más nuevo que lo probado: TensorFlow puede no soportarlo aún')],
    too_old: ['high', t('too old: 3.10 or newer is needed', 'muy antiguo: Se necesita 3.10 o más nuevo')],
  });
  const KIND_NAME = () => ({ system: t('system', 'sistema'), venv: t('virtual env', 'entorno virtual'), conda: 'conda' });

  /** Lists the Pythons found on the computer; choosing one shows the usual "what is missing" check. */
  TS.pyScanPanel = function pyScanPanel(show) {
    const out = h('div', { id: 'py-scan-out' });
    const btn = h('button', { type: 'button', id: 'py-scan-btn', class: 'ghost', text: t('Look for Python on this computer', 'Buscar Python en este equipo'), onclick: async () => {
      btn.disabled = true;
      out.replaceChildren(h('p', { class: 'muted', text: t('Looking…', 'Buscando…') }));
      try {
        const list = await window.tinyml.envScan();
        if (!list.length) {
          out.replaceChildren(TS.callout('info', t('No Python found. That is fine: The app can install its own private Python (option 1), without touching anything else on your computer.', 'No se encontró Python. No pasa nada: La app puede instalar su propio Python privado (opción 1), sin tocar nada más de tu equipo.')));
        } else {
          const rows = list.map((p) => {
            const [cls, label] = VERDICT()[p.verdict] || ['', p.verdict];
            const use = h('button', { type: 'button', class: 'ghost sm', 'data-id': p.id, disabled: p.verdict === 'too_old', text: t('Use this one', 'Usar este'), onclick: async () => {
              use.disabled = true;
              try {
                const r = await window.tinyml.envUseScanned(p.id, TS.lang);
                if (r.error) show({ error: `${r.path}: ${r.error}` });
                else show({ result: r });
              } catch (e) { show({ error: e.message || String(e) }); }
              use.disabled = false;
            } });
            return [p.version, KIND_NAME()[p.kind] || p.kind, p.executable, h('span', { class: `tag ${cls}`, text: label }), use];
          });
          out.replaceChildren(
            ui.pagedTable([t('Version', 'Versión'), t('Type', 'Tipo'), t('Location', 'Ubicación'), t('Status', 'Estado'), ''], rows, { pageSize: 8, rowText: (r) => `${r[0]} ${r[2]}` }),
            TS.callout('warn', t('The app never upgrades or changes your Python. Adding packages to an existing environment can change library versions that your other projects use: A clean virtual environment, or the private Python that the app installs, avoids that.', 'La app nunca actualiza ni cambia tu Python. Añadir paquetes a un entorno que ya usas puede cambiar versiones de librerías de tus otros proyectos: Un entorno virtual limpio, o el Python privado de la app, lo evita.')));
        }
      } catch (e) { out.replaceChildren(TS.err(e)); }
      btn.disabled = false;
    } });
    return h('div', {},
      h('div', { class: 'row' }, btn, h('span', { class: 'muted', text: t('Node.js is not needed: The app already includes it.', 'Node.js no hace falta: La app ya lo incluye.') })),
      out);
  };

  function commandText(cmd) {
    return cmd.map((a) => (/[\s<>]/.test(a) ? `"${a}"` : a)).join(' ');
  }

  /** Button that installs what the check found missing in the environment in use (asks first). */
  function fixPanel(fix) {
    const err = h('div');
    const btn = h('button', { type: 'button', id: 'env-fix-btn', text: t('Install what is missing in this environment', 'Instalar lo que falta en este entorno'), onclick: async () => {
      btn.disabled = true; err.replaceChildren();
      try { await fix(); } catch (e) { err.append(TS.err(e)); }
      btn.disabled = false;
    } });
    return h('div', {}, h('div', { class: 'row' }, btn, h('span', { class: 'muted', text: t('You will be asked to confirm. Packages that already fit are not touched.', 'Te pediremos confirmar. Los paquetes que ya encajan no se tocan.') })), TS.consolePanel ? TS.consolePanel('own') : null, err);
  }

  function report(rep, box, fix) {
    const py = rep.python;
    box.replaceChildren(...[
      ui.stats(
        ui.stat(py.version, t('Python', 'Python'), py.ok ? 'ok' : 'warn'),
        ui.stat(KIND()[py.kind] || py.kind, t('environment', 'entorno')),
        ui.stat(rep.ok ? t('all good', 'todo en orden') : t('needs fixing', 'requiere arreglo'), t('status', 'estado'), rep.ok ? 'ok' : 'warn')),
      h('p', { class: 'muted', id: 'env-exe', text: py.executable }),
      ...Object.entries(rep.components).map(([id, c]) => ui.section(`${c.label}`,
        ui.pagedTable([t('Package', 'Paquete'), t('Installed', 'Instalado'), t('Required', 'Requerido'), t('Status', 'Estado')],
          c.packages.map((r) => [r.package, r.have || '—', r.required, h('span', { class: `tag ${r.status === 'ok' ? '' : 'high'}`, text: STATUS()[r.status] || r.status })]),
          { pageSize: 12, rowText: (r) => `${r[0]}` }),
        c.note ? h('p', { class: 'muted', text: c.note }) : null)),
      ...rep.warnings.map((w) => TS.callout('warn', w)),
      rep.commands.length ? ui.section(fix ? t('Fix it', 'Arréglalo') : t('To fix it, run in a terminal', 'Para arreglarlo, ejecuta en una terminal'),
        fix ? fixPanel(fix) : null,
        h('details', {}, h('summary', { text: t('Prefer to do it yourself?', '¿Prefieres hacerlo tú?') }),
          ...rep.commands.map((c) => h('pre', { class: 'code', id: 'env-fix', text: commandText(c) }))),
        h('p', { class: 'muted', text: t('Nothing is installed without your confirmation.', 'No se instala nada sin tu confirmación.') })) : null,
    ].filter(Boolean));
  }

  TS.envView = async function envView(root) {
    const box = h('div', { id: 'env-box' }, h('p', { class: 'muted', text: t('Checking the environment…', 'Revisando el entorno…') }));
    const picked = h('div', { id: 'env-picked' });
    // wrapper: if the app re-renders while we await, this (now detached) wrapper is filled harmlessly
    const wrap = h('div');
    root.append(wrap);
    const info = await window.tinyml.envGet();
    const selected = new Set(['base']);
    const checks = COMPONENT_IDS.map((id) => h('label', { class: 'row', 'data-tight': '1' },
      h('input', { type: 'checkbox', id: `env-c-${id}`, checked: selected.has(id), disabled: id === 'base', onchange: (e) => { e.target.checked ? selected.add(id) : selected.delete(id); } }),
      ` ${{ base: t('Core', 'Núcleo'), tui: t('Terminal UI', 'Interfaz de terminal'), keras: 'Keras / TensorFlow', torch: 'PyTorch', tflm: 'TFLite Micro' }[id]}`));
    // only an environment of the user's own can be completed from here (the managed one has its installer)
    const fixAction = info.core === 'external' ? async () => {
      const out = await window.tinyml.envFixOwn(TS.lang, [...selected]);
      if (out.report) report(out.report, box, fixAction);
    } : null;
    const recheck = async () => {
      box.replaceChildren(h('p', { class: 'muted', text: t('Checking the environment…', 'Revisando el entorno…') }));
      try { report(await window.tinyml.envDoctor([...selected], TS.lang), box, fixAction); } catch (e) { box.replaceChildren(TS.err(e)); }
    };
    const pick = async () => {
      picked.replaceChildren();
      const r = await window.tinyml.envPick(TS.lang);
      if (r.canceled) return;
      if (r.error) { picked.append(TS.callout('bad', `${r.path}: ${r.error}`)); return; }
      picked.append(TS.ownEnvResult(r, { onChange: () => TS.render(), label: t('Interpreter: ', 'Intérprete: ') }));
    };
    const showScanned = ({ result, error }) => {
      picked.replaceChildren();
      if (error) picked.append(TS.callout('bad', error));
      else picked.append(TS.ownEnvResult(result, { onChange: () => TS.render(), label: t('Interpreter: ', 'Intérprete: ') }));
    };
    TS.add(wrap, ui.page(t('Environment', 'Entorno'), t('Use the built-in core or your own Python (venv, conda, miniconda).', 'Usa el núcleo incluido o tu propio Python (venv, conda, miniconda).'),
      info.missing ? TS.callout('warn', t('The saved interpreter no longer exists; using the built-in core.', 'El intérprete guardado ya no existe; se usa el núcleo incluido.')) : null,
      ui.section(t('Which Python runs the core', 'Qué Python ejecuta el núcleo'),
        h('p', { id: 'env-mode', text: info.core === 'external' ? `${t('Your environment: ', 'Tu entorno: ')}${info.python}` : info.core === 'managed' ? t('Environment installed by the app (private Python).', 'Entorno instalado por la app (Python privado).') : info.core === 'dev' ? t('Development Python.', 'Python de desarrollo.') : t('Built-in core (works offline).', 'Núcleo incluido (funciona sin conexión).') }),
        h('div', { class: 'row wrap' },
          h('button', { type: 'button', id: 'env-pick', class: 'ghost', text: t('Choose Python…', 'Elegir Python…'), onclick: pick })),
        TS.pyScanPanel(showScanned),
        h('div', { class: 'row wrap' },
          info.core === 'external' ? h('button', { type: 'button', id: 'env-reset', class: 'ghost', text: info.installer.installed || info.frozenAvailable ? t('Stop using my environment', 'Dejar de usar mi entorno') : t('Stop using my environment', 'Dejar de usar mi entorno'), onclick: async () => { await window.tinyml.envReset(); TS.render(); } }) : null),
        picked),
      info.installer && (info.installer.uvAvailable || info.installer.installed) ? ui.section(t('Components installed by the app', 'Componentes instalados por la app'), TS.componentsPanel(info, () => TS.render())) : null,
      ui.section(t('What to check', 'Qué revisar'), h('div', { class: 'row wrap' }, ...checks, h('button', { type: 'button', id: 'env-check', class: 'ghost', text: t('Check', 'Revisar'), onclick: recheck }))),
      box));
    recheck();
  };
})();
