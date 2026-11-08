'use strict';
// New version notice. It tells the user. Only when they press "Update now" (Windows, installed copy) the main
// process downloads the installer from the official release, checks its SHA-256 and opens it; every other path
// (portable copy, Linux) just opens the download page. Nothing happens without a click.
(function () {
  const TS = window.TS;
  const { h, callout } = TS;
  const ui = TS.ui;
  const t = TS.t;
  const api = () => window.tinyml;

  const versionLabel = (l) => `${l.version}${l.prerelease ? t(' (pre-release)', ' (versión preliminar)') : ''}`;

  function notice(res, onDone, canInstall) {
    const l = res.latest;
    const status = h('p', { class: 'muted', id: 'upd-status', hidden: true });
    const go = canInstall && l.installer ? h('button', { type: 'button', id: 'upd-install', text: t('Update now', 'Actualizar ahora'), onclick: async () => {
      for (const b of box.querySelectorAll('button')) b.disabled = true;
      status.hidden = false;
      status.textContent = t('Downloading…', 'Descargando…');
      const off = api().onUpdatesProgress((m) => {
        const pct = m.total ? Math.min(100, Math.floor((100 * m.done) / m.total)) : 0;
        status.textContent = t('Downloading… {p} %', 'Descargando… {p} %', { p: pct });
      });
      try {
        await api().updatesInstall();
        status.textContent = t('Verified. Opening the installer; this app will close.', 'Verificado. Abriendo el instalador; esta app se cerrará.');
      } catch (e) {
        status.replaceChildren(callout('warn', t('The update could not be installed: {e}', 'No se pudo instalar la actualización: {e}', { e: String((e && e.message) || e).replace(/^Error invoking remote method '[^']*': (Error: )?/, '') })));
        for (const b of box.querySelectorAll('button')) b.disabled = false;
      } finally { off(); }
    } }) : null;
    const open = h('button', { type: 'button', id: 'upd-open', text: t('See what is new and download', 'Ver novedades y descargar'), onclick: async () => { try { await api().updatesOpen(); } catch { /* sin navegador */ } } });
    const later = h('button', { type: 'button', class: 'ghost', id: 'upd-later', text: t('Not now', 'Ahora no'), onclick: () => onDone('later') });
    const skip = h('button', { type: 'button', class: 'ghost', id: 'upd-skip', text: t('Skip this version', 'Omitir esta versión'), onclick: async () => { await api().updatesSkip(); onDone('skipped'); } });
    const box = h('div', { class: 'update-notice' },
      h('p', {}, h('b', { text: t('A new version is available: {v}', 'Hay una versión nueva: {v}', { v: versionLabel(l) }) }), ` ${t('You have {c}.', 'Tienes la {c}.', { c: res.current })}`),
      l.notes ? h('p', { class: 'muted', text: l.notes.split('\n').slice(0, 4).join(' ').slice(0, 260) }) : null,
      h('p', { class: 'muted', text: go
        ? t('"Update now" downloads the installer from the official release of this project, checks its SHA-256 against the list published with it and opens it. Nothing is installed until you confirm in the installer. The installers are not signed, so Windows may warn you.', '"Actualizar ahora" descarga el instalador del release oficial de este proyecto, comprueba su SHA-256 contra la lista publicada con él y lo abre. No se instala nada hasta que lo confirmes en el instalador. Los instaladores no están firmados, así que Windows puede avisarte.')
        : t('Nothing is downloaded or installed by this app. The button opens the download page in your browser; you decide whether to update.', 'Esta app no descarga ni instala nada. El botón abre la página de descarga en tu navegador; tú decides si actualizas.') }),
      status,
      h('div', { class: 'row wrap' }, go, open, later, skip));
    return box;
  }

  /** Top bar: the one time question, and the notice when the automatic check finds something newer. */
  function bar() { return document.getElementById('update-banner'); }
  function show(node) {
    const b = bar();
    if (!b) return;
    b.replaceChildren(node);
    b.hidden = false;
  }
  function hide() {
    const b = bar();
    if (b) { b.replaceChildren(); b.hidden = true; }
  }

  TS.updatesInit = async function updatesInit() {
    try {
      const st = await api().updatesGet();
      if (st.auto === null && st.repo) {
        show(h('div', { class: 'update-notice' },
          h('p', {}, h('b', { text: t('Tell me when a new version is available?', '¿Avisarte cuando haya una versión nueva?' ) }), ' ',
            t('The app would ask GitHub about once a day whether a newer release exists. Nothing about you or your files is sent, and nothing is downloaded or installed by itself.', 'La app preguntaría a GitHub, más o menos una vez al día, si hay una versión más nueva. No se envía nada tuyo ni de tus archivos, y no se descarga ni instala nada sola.')),
          h('div', { class: 'row wrap' },
            h('button', { type: 'button', id: 'upd-yes', text: t('Yes, tell me', 'Sí, avísame'), onclick: async () => { await api().updatesSetAuto(true); hide(); TS.updatesInit(); } }),
            h('button', { type: 'button', class: 'ghost', id: 'upd-no', text: t('No, thanks', 'No, gracias'), onclick: async () => { await api().updatesSetAuto(false); hide(); } }))));
        return;
      }
      if (st.auto === true) {
        const res = await api().updatesAuto();
        if (res && res.status === 'newer') show(notice(res, hide, st.canInstall));
      }
    } catch { /* offline or no repository: the notice is optional */ }
  };

  TS.updatesView = async function updatesView(root) {
    const wrap = h('div');
    root.append(wrap);
    const st = await api().updatesGet();
    const out = h('div', { id: 'upd-out' });
    const auto = h('input', { type: 'checkbox', id: 'upd-auto', checked: st.auto === true, onchange: async (e) => { await api().updatesSetAuto(e.target.checked); } });
    const btn = h('button', { type: 'button', id: 'upd-check', text: t('Check for updates', 'Buscar actualizaciones'), onclick: async () => {
      btn.disabled = true;
      out.replaceChildren(h('p', { class: 'muted', text: t('Asking GitHub…', 'Preguntando a GitHub…') }));
      try {
        const res = await api().updatesCheck();
        if (res.status === 'newer') out.replaceChildren(notice(res, () => out.replaceChildren(callout('info', t('Understood. You can check again at any time.', 'Entendido. Puedes volver a buscar cuando quieras.'))), st.canInstall));
        else if (res.status === 'current') out.replaceChildren(callout('ok', t('You have the latest version ({v}).', 'Tienes la última versión ({v}).', { v: res.current })));
        else if (res.status === 'none') out.replaceChildren(callout('info', t('No published versions were found. The project may not have released one yet.', 'No se encontraron versiones publicadas. Puede que el proyecto aún no haya publicado ninguna.')));
        else out.replaceChildren(callout('warn', t('Could not check: {e}', 'No se pudo comprobar: {e}', { e: res.error || '' })));
      } catch (e) { out.replaceChildren(TS.err(e)); }
      btn.disabled = false;
    } });
    TS.add(wrap, ui.page(t('Updates', 'Actualizaciones'), t('Find out whether a newer version has been published. You decide whether to update.', 'Averigua si hay una versión más nueva publicada. Tú decides si actualizas.'),
      ui.section(t('This version', 'Esta versión'), h('p', { id: 'upd-version', text: st.version })),
      ui.section(t('Check', 'Comprobar'),
        h('p', { class: 'muted', text: t('Pressing the button contacts api.github.com (the public page of the project releases). Nothing about you or your files is sent.', 'Al pulsar el botón se contacta con api.github.com (la página pública de versiones del proyecto). No se envía nada tuyo ni de tus archivos.') }),
        h('div', { class: 'row wrap' }, btn),
        h('label', { class: 'chk' }, auto, h('span', { text: ` ${t('Also tell me automatically, about once a day', 'Avisarme también automáticamente, más o menos una vez al día')}` })),
        out),
      ui.section(t('How to update', 'Cómo actualizar'), h('p', { text: st.canInstall
        ? t('When a new version appears, press "Update now". The app downloads the installer, checks it and opens it over the old version. It replaces the program and keeps your settings. You do not need to uninstall first.', 'Cuando aparezca una versión nueva, pulsa "Actualizar ahora". La app descarga el instalador, lo comprueba y lo abre sobre la versión anterior. Reemplaza el programa y conserva tus ajustes. No hace falta desinstalar antes.')
        : t('Download the new installer from the page the notice opens and run it over the old version. It replaces the program and keeps your settings. You do not need to uninstall first.', 'Descarga el instalador nuevo desde la página que abre el aviso y ejecútalo sobre la versión anterior. Reemplaza el programa y conserva tus ajustes. No hace falta desinstalar antes.') }))));
  };
})();
