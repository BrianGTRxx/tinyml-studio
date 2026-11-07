'use strict';
// First-run setup and component manager. The app ships WITHOUT Python packages; here the user picks
// one of two paths: (1) let the app install a private Python + packages (uv), or (2) use their own
// environment (venv / conda / miniconda). The renderer only ever sends component ids from the manifest.
(function () {
  const TS = window.TS;
  const { h } = TS;
  const ui = TS.ui;
  const t = TS.t;
  const lbl = (o) => (TS.lang === 'es' ? o.es : o.en);

  /** Live console fed by installer events (stays alive across re-renders through a module-level buffer). */
  const newDl = () => ({ total: 0, done: 0, sizes: {} });
  const bufs = { managed: { lines: [], step: null, state: 'idle', error: null, dl: newDl() }, own: { lines: [], step: null, state: 'idle', error: null, dl: newDl() } };
  const MAX_LINES = 400;
  // uv prints "Downloading name (35.0MiB)" and later "Downloaded name": the bar advances by bytes
  const UNIT = { B: 1, KiB: 1024, MiB: 1048576, GiB: 1073741824 };
  function trackDownload(b, line) {
    const s = String(line).trim();
    if (s.startsWith('Downloading ')) {
      const open = s.lastIndexOf('(');
      const close = s.lastIndexOf(')');
      const m = open > 0 && close > open ? s.slice(open + 1, close).match(/^([0-9.]+) ?([KMG]?i?B)$/) : null;
      if (m && UNIT[m[2]]) {
        const name = s.slice(12, open).trim();
        b.dl.sizes[name] = parseFloat(m[1]) * UNIT[m[2]];
        b.dl.total += b.dl.sizes[name];
      }
    } else if (s.startsWith('Downloaded ')) {
      b.dl.done += b.dl.sizes[s.slice(11).trim()] || 0;
    }
  }
  let unsub = null;
  let listeners = new Set();
  function ensureListener() {
    if (unsub || !window.tinyml.onInstallerEvent) return;
    unsub = window.tinyml.onInstallerEvent((ev) => {
      const consoleBuf = bufs[ev.scope === 'own' ? 'own' : 'managed'];
      if (ev.type === 'log') {
        consoleBuf.lines.push(ev.line);
        trackDownload(consoleBuf, ev.line);
        if (consoleBuf.lines.length > MAX_LINES) consoleBuf.lines.splice(0, consoleBuf.lines.length - MAX_LINES);
      } else if (ev.type === 'step') {
        consoleBuf.step = ev;
        consoleBuf.dl = newDl();
        consoleBuf.lines.push(`── [${ev.index + 1}/${ev.total}] ${ev.label}`);
      } else if (ev.type === 'state') {
        consoleBuf.state = ev.state;
        consoleBuf.error = ev.error || null;
        if (ev.state === 'running') { consoleBuf.lines = []; consoleBuf.step = null; consoleBuf.dl = newDl(); }
      }
      listeners.forEach((fn) => fn(ev));
    });
  }

  const STATE_TEXT = () => ({
    running: t('Installing…', 'Instalando…'), done: t('Installed', 'Instalado'),
    failed: t('Failed', 'Falló'), cancelled: t('Cancelled', 'Cancelado'),
  });

  function consolePanel(scope = 'managed') {
    const consoleBuf = bufs[scope];
    const pre = h('pre', { class: 'code', id: 'inst-console', 'aria-live': 'polite', style: 'max-height:260px;overflow:auto', text: '' });
    const bar = h('progress', { id: 'inst-progress', max: 1, value: 0, 'aria-label': t('Installation progress', 'Progreso de la instalación'), style: 'width:100%' });
    const status = h('p', { class: 'muted', id: 'inst-status' });
    const box = h('div', { class: 'console' }, status, bar, pre);
    const paint = () => {
      const s = consoleBuf.step;
      box.hidden = consoleBuf.state === 'idle';
      pre.textContent = consoleBuf.lines.join('\n');
      pre.scrollTop = pre.scrollHeight;
      const dl = consoleBuf.dl;
      const part = dl.total > 0 ? Math.min(dl.done / dl.total, 0.98) : 0; // fraction of the current step already downloaded
      bar.value = s ? (consoleBuf.state === 'done' ? 1 : (s.index + part) / s.total) : 0;
      const bytes = consoleBuf.state === 'running' && dl.total > 0 ? `${TS.fmt.bytes(Math.min(dl.done, dl.total))} / ${TS.fmt.bytes(dl.total)}` : '';
      status.textContent = [STATE_TEXT()[consoleBuf.state] || '', s && consoleBuf.state === 'running' ? `${s.label} (${s.index + 1}/${s.total})` : '', bytes, consoleBuf.error || ''].filter(Boolean).join(' · ');
      status.className = `muted ${consoleBuf.state === 'failed' ? 'bad' : ''}`;
    };
    ensureListener();
    const onEvent = () => { if (box.isConnected) paint(); else listeners.delete(onEvent); };
    listeners.add(onEvent);
    paint();
    return box;
  }

  /**
   * Components table + consent + install / cancel / remove. `onChange` runs after the core changed
   * (an installation finished and the core restarted).
   */
  TS.componentsPanel = function componentsPanel(info, onChange) {
    const st = info.installer;
    const picked = new Set(['base']);
    const box = h('div', { id: 'components-box' });
    const consent = h('div', { id: 'inst-consent' });
    const go = async () => {
      consent.replaceChildren();
      try { await window.tinyml.installerStart([...picked]); } catch (e) { consent.append(TS.err(e)); }
    };
    const ask = () => {
      const comps = st.components.filter((c) => picked.has(c.id)).map((c) => lbl(c.label));
      consent.replaceChildren(TS.callout('warn',
        h('b', { text: t('This will download and install:', 'Esto descargará e instalará:') }),
        h('ul', {}, ...comps.map((c) => h('li', { text: c })), h('li', { text: t('A private Python 3.12 (python-build-standalone builds from github.com/astral-sh)', 'Un Python 3.12 privado (builds de python-build-standalone de github.com/astral-sh)') })),
        h('p', { class: 'muted', text: t('Packages come from pypi.org with pinned versions and checksums. Everything stays inside the app folder; nothing is installed system-wide and you can remove it later. Needs an internet connection.', 'Los paquetes vienen de pypi.org con versiones fijadas y sumas de verificación. Todo queda dentro de la carpeta de la app; no se instala nada en el sistema y puedes quitarlo después. Necesita conexión a internet.') }),
        h('div', { class: 'row' }, h('button', { type: 'button', id: 'inst-go', text: t('Download and install', 'Descargar e instalar'), onclick: go }), h('button', { type: 'button', class: 'ghost', text: t('Cancel', 'Cancelar'), onclick: () => consent.replaceChildren() }))));
    };
    const rows = st.components.map((c) => {
      const cb = h('input', { type: 'checkbox', id: `inst-c-${c.id}`, checked: c.installed || c.required, disabled: c.installed || c.required || !c.available || st.busy,
        'aria-label': lbl(c.label), onchange: (e) => { e.target.checked ? picked.add(c.id) : picked.delete(c.id); } });
      const tag = c.installed ? h('span', { class: 'tag', text: t('installed', 'instalado') }) : !c.available ? h('span', { class: 'tag high', text: t('not available on this system', 'no disponible en este sistema') }) : h('span', { class: 'tag', text: t('not installed', 'sin instalar') });
      return [cb, h('span', {}, h('b', { text: lbl(c.label) }), c.note ? h('div', { class: 'muted', text: lbl(c.note) }) : null, c.diskMb ? h('div', { class: 'muted', text: t('≈ {n} MB on disk', '≈ {n} MB en disco', { n: c.diskMb.toLocaleString() }) }) : null), tag];
    });
    const busy = st.busy;
    const missing = st.components.filter((c) => c.available && !c.installed);
    box.append(...[
      ui.pagedTable(['', t('Component', 'Componente'), t('Status', 'Estado')], rows, { pageSize: 12, rowText: (r) => r[1].textContent }),
      h('div', { class: 'row wrap' },
        h('button', { type: 'button', id: 'inst-start', text: st.installed ? t('Install selected / repair', 'Instalar lo elegido / reparar') : t('Install', 'Instalar'), disabled: busy || !st.uvAvailable || (st.installed && missing.length === 0 && false), onclick: ask }),
        busy ? h('button', { type: 'button', id: 'inst-cancel', class: 'ghost', text: t('Cancel installation', 'Cancelar instalación'), onclick: () => window.tinyml.installerCancel() }) : null,
        st.installed && !busy ? h('button', { type: 'button', id: 'inst-reset', class: 'ghost', text: t('Remove the installed environment', 'Quitar el entorno instalado'), title: t('Deletes the private Python and packages (not your data)', 'Borra el Python privado y los paquetes (no tus datos)'),
          onclick: async () => { await window.tinyml.installerReset(); if (onChange) onChange(); } }) : null),
      !st.uvAvailable ? TS.callout('bad', t('The installer (uv) was not found in this build. Use your own environment instead.', 'No se encontró el instalador (uv) en esta versión. Usa tu propio entorno.')) : null,
      consent, consolePanel()].filter(Boolean));
    // when an installation ends the main process restarts the core; refresh the screen afterwards
    const fn = (ev) => { if (ev.scope === 'own') return; /* completing the user's own environment does not restart the core */ if (ev.type === 'state' && ev.state !== 'running') { listeners.delete(fn); setTimeout(() => onChange && onChange(), 600); } };
    listeners.add(fn);
    return box;
  };

  TS.consolePanel = consolePanel;

  /**
   * Result of checking an own environment (shared by the first-run screen and the Environment tab):
   * what is missing, a button that installs it (the app does the work, after asking), and "Use it".
   */
  TS.ownEnvResult = function ownEnvResult(r, { onChange, label } = {}) {
    const box = h('div', { id: 'own-env-result' });
    const q = (a) => (/[\s<>]/.test(a) ? `"${a}"` : a);
    const draw = (res) => {
      const rep = res.report;
      const missingStudio = !rep.tinyml_studio;
      const missing = (rep.fixes || []).length;
      const canFix = missingStudio || missing > 0;
      const wheelCmd = missingStudio && res.wheel ? `${q(r.path)} -m pip install --no-deps ${q(res.wheel)}` : null;
      const fix = async () => {
        fixBtn.disabled = true;
        errBox.replaceChildren();
        try {
          const out = await window.tinyml.envFixOwn(TS.lang);
          if (out.report) draw({ report: out.report, wheel: out.wheel });
        } catch (e) { errBox.append(TS.err(e)); }
        fixBtn.disabled = false;
      };
      const errBox = h('div');
      const fixBtn = h('button', { type: 'button', id: 'env-fix', text: t('Install what is missing in this environment', 'Instalar lo que falta en este entorno'), onclick: fix });
      box.replaceChildren(...[
        TS.callout(rep.ok && !missingStudio ? 'ok' : 'warn', label || '', r.path, ' — ', rep.ok && !missingStudio ? t('ready to use', 'listo para usar') : t('something is missing', 'falta algo')),
        missingStudio ? TS.callout('warn', t('tinyml-studio is not installed in this environment. This app ships it and can install it for you.', 'tinyml-studio no está instalado en este entorno. Esta app lo trae y puede instalarlo por ti.')) : null,
        missing ? h('div', {}, h('p', { class: 'muted', text: t('Missing or out of range:', 'Faltan o están fuera de rango:') }), h('ul', {}, ...rep.fixes.map((f) => h('li', { text: `${f.requirement}${f.status === 'missing' ? '' : ` (${f.status === 'too_old' ? t('too old', 'muy antiguo') : t('too new', 'muy nuevo')})`}` })))) : null,
        ...rep.warnings.filter((w) => !/tinyml-studio/.test(w)).map((w) => TS.callout('warn', w)),
        canFix ? h('div', { class: 'row' }, fixBtn, h('span', { class: 'muted', text: t('You will be asked to confirm. Packages that already fit are not touched.', 'Te pediremos confirmar. Los paquetes que ya encajan no se tocan.') })) : null,
        canFix ? TS.consolePanel('own') : null,
        errBox,
        wheelCmd ? h('details', {}, h('summary', { text: t('Prefer to do it yourself?', '¿Prefieres hacerlo tú?') }), h('pre', { class: 'code', id: 'own-env-wheel', text: wheelCmd })) : null,
        h('div', { class: 'row' }, h('button', { type: 'button', id: 'env-apply', disabled: missingStudio, title: missingStudio ? t('Install tinyml-studio in that environment first', 'Primero instala tinyml-studio en ese entorno') : '', text: t('Use this environment', 'Usar este entorno'), onclick: async () => { await window.tinyml.envApply(); if (onChange) onChange(); } })),
      ].filter(Boolean));
    };
    draw(r);
    return box;
  };

  /** Path 2: pick the Python of an own environment (venv, conda). */
  function ownEnvPanel(onChange) {
    const out = h('div', { id: 'own-env-out' });
    const pick = async () => {
      out.replaceChildren();
      const r = await window.tinyml.envPick(TS.lang);
      if (r.canceled) return;
      if (r.error) { out.append(TS.callout('bad', `${r.path}: ${r.error}`)); return; }
      out.append(TS.ownEnvResult(r, { onChange }));
    };
    const showScanned = ({ result, error }) => {
      out.replaceChildren();
      if (error) out.append(TS.callout('bad', error));
      else out.append(TS.ownEnvResult(result, { onChange }));
    };
    return h('div', {}, TS.pyScanPanel(showScanned), h('p', { class: 'muted', text: t('Pick the python executable of your virtual environment or conda environment (for example .../envs/tml/bin/python). The app checks what is missing and can install it for you.', 'Elige el ejecutable python de tu entorno virtual o de conda (por ejemplo .../envs/tml/bin/python). La app revisa qué falta y puede instalarlo por ti.') }),
      h('div', { class: 'row' }, h('button', { type: 'button', id: 'env-pick', class: 'ghost', text: t('Choose Python…', 'Elegir Python…'), onclick: pick })), out);
  }

  /** Full-page first-run screen (shown when the core cannot start because nothing is installed). */
  TS.setupView = async function setupView(root, onChange) {
    const wrap = h('div');
    root.append(wrap);
    const info = await window.tinyml.envGet();
    TS.add(wrap, ui.page(t('Set up TinyML Studio', 'Configura TinyML Studio'),
      t('This app ships without Python packages to stay small. Choose how to get them.', 'Esta app viene sin paquetes de Python para ser pequeña. Elige cómo conseguirlos.'),
      info.startError ? TS.callout('bad', t('The selected environment could not start: ', 'No arrancó el entorno elegido: '), info.startError) : null,
      ui.section(t('Option 1 — Install for me (recommended)', 'Opción 1 — Instalar por mí (recomendado)'), TS.componentsPanel(info, onChange)),
      ui.section(t('Option 2 — Use my own environment', 'Opción 2 — Usar mi propio entorno'), ownEnvPanel(onChange))));
  };
})();
