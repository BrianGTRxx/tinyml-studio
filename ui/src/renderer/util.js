'use strict';
// Utilidades de la interfaz. Todo se construye con createElement/textContent: los datos de los
// CSV y de los informes nunca se insertan como HTML.
(function () {
  const TS = (window.TS = window.TS || {});

  TS.h = function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else if (['value', 'checked', 'disabled', 'selected', 'hidden', 'min', 'max', 'step'].includes(k))
        el[k] = v === true ? true : v;
      else el.setAttribute(k, v === true ? '' : String(v));
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  };

  // append que ignora null/undefined/false (Element.append escribiría "null")
  TS.add = function add(parent, ...kids) {
    for (const kid of kids.flat(Infinity)) {
      if (kid === null || kid === undefined || kid === false) continue;
      parent.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return parent;
  };

  TS.svg = function svg(tag, attrs, ...kids) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, String(v));
    for (const kid of kids.flat()) if (kid) el.append(kid);
    return el;
  };

  TS.fmt = {
    // decimal mark follows the UI language (never the OS locale): 78,7 in Spanish, 78.7 in English
    dec: (x, d = 1) => { const s = Number(x).toFixed(d); return TS.lang === 'es' ? s.replace('.', ',') : s; },
    pct: (x, d = 1) => (x === null || x === undefined ? '—' : `${TS.fmt.dec(x * 100, d)} %`),
    num: (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? '—' : TS.fmt.dec(x, d)),
    // integer with thousands separators following the UI language (never the OS locale)
    int: (n) => (n === null || n === undefined || Number.isNaN(Number(n)) ? '—' : Number(n).toLocaleString(TS.lang === 'es' ? 'es' : 'en')),
    bytes: (n) => (n < 1024 ? `${Math.round(n)} B` : n < 1048576 ? `${TS.fmt.dec(n / 1024, 1)} KB` : `${TS.fmt.dec(n / 1048576, 2)} MB`),
  };

  TS.badge = (text, kind) => TS.h('span', { class: `badge ${kind || ''}`, text });

  TS.table = function table(headers, rows, cls) {
    return TS.h(
      'table',
      { class: cls || 'tbl' },
      TS.h('thead', {}, TS.h('tr', {}, headers.map((x) => TS.h('th', { text: x })))),
      TS.h('tbody', {}, rows.map((r) => TS.h('tr', {}, r.map((c) => TS.h('td', {}, c))))),
    );
  };

  TS.callout = (kind, ...kids) => TS.h('div', { class: `callout ${kind}` }, ...kids);

  // The language travels with every call that produces messages (query ?lang= for GET, field `lang` for POST).
  TS.api = (method, route, body) => {
    let r = route;
    let b = body;
    if (method === 'GET' && /^\/(hardware(\/(frameworks|live))?|boards|models|jobs\/[\w:-]+\/hw)$/.test(route)) r = `${route}?lang=${TS.lang}`;
    if (method === 'POST' && (route === '/inspect' || route === '/suggest-cleaning')) b = { ...body, lang: TS.lang };
    if (method === 'POST' && route === '/jobs') b = { ...body, params: { ...body.params, lang: TS.lang } };
    return window.tinyml.request(method, r, b);
  };

  // ---- trabajos: crear, escuchar eventos y esperar el final ------------------------------------
  // Messages the core wrote in the language that was active when a job ran. The job also returns every
  // [Spanish, English] pair it said, so a language switch can translate the stored results (warnings,
  // reports, Arduino steps...) without running the job again.
  const PAIRS = new Map();
  TS.i18nAdd = (list) => { for (const p of list || []) if (Array.isArray(p) && p.length === 2) PAIRS.set(p[0], p[1]); };
  function translateString(s, to, ordered) {
    if (to === 'en' ? PAIRS.has(s) : false) return PAIRS.get(s);
    let out = s;
    for (const [es, en] of ordered) {
      const from = to === 'en' ? es : en;
      if (out.includes(from)) out = out.split(from).join(to === 'en' ? en : es);
    }
    return out;
  }
  TS.retranslate = function retranslate(root, to) {
    if (!PAIRS.size) return root;
    const ordered = [...PAIRS.entries()].sort((a, b) => b[0].length - a[0].length); // longest first
    const seen = new WeakSet();
    const walk = (v) => {
      if (typeof v === 'string') return translateString(v, to, ordered);
      if (!v || typeof v !== 'object' || seen.has(v)) return v;
      seen.add(v);
      if (Array.isArray(v)) { for (let i = 0; i < v.length; i += 1) v[i] = walk(v[i]); return v; }
      for (const k of Object.keys(v)) v[k] = walk(v[k]);
      return v;
    };
    return walk(root);
  };

  TS.runJob = async function runJob(kind, params, onEvent) {
    const job = await TS.api('POST', '/jobs', { kind, params });
    TS.currentJob = job.id;
    let finished = false;
    const final = new Promise((resolve) => {
      const off = window.tinyml.onJobEvent(({ id, ev }) => {
        if (id !== job.id) return;
        if (onEvent) onEvent(ev);
        if (ev.stage === 'end' && !finished) {
          finished = true;
          off();
          resolve();
        }
      });
      // respaldo por si se pierde el evento final (reinicio del flujo SSE)
      const poll = setInterval(async () => {
        if (finished) return clearInterval(poll);
        try {
          const j = await TS.api('GET', `/jobs/${job.id}`);
          if (['succeeded', 'failed', 'cancelled', 'interrupted'].includes(j.state)) {
            finished = true;
            clearInterval(poll);
            off();
            resolve();
          }
        } catch {
          /* se reintenta */
        }
      }, 3000);
    });
    await window.tinyml.watchJob(job.id);
    await final;
    const info = await TS.api('GET', `/jobs/${job.id}`);
    let result = null;
    if (info.state === 'succeeded') {
      result = await TS.api('GET', `/jobs/${job.id}/result`);
      TS.i18nAdd(result && result._i18n);
      if (result) delete result._i18n;
    }
    TS.currentJob = null;
    return { job: info, result };
  };

  TS.cancelCurrent = async () => {
    if (TS.currentJob) await TS.api('POST', `/jobs/${TS.currentJob}/cancel`).catch(() => {});
  };

  // reusable progress panel with elapsed time and a rough ETA (stages are not linear: it is an estimate)
  const dur = (s) => (s < 60 ? `${Math.round(s)} s` : s < 3600 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`);
  TS.fmt.dur = dur;
  TS.progressPanel = function progressPanel() {
    const bar = TS.h('div', { class: 'bar-fill' });
    const msg = TS.h('span', { class: 'muted', text: TS.t('Preparing…', 'Preparando…') });
    const time = TS.h('span', { class: 'muted', text: '' });
    const cancel = TS.h('button', { type: 'button', class: 'ghost sm', text: TS.t('Cancel', 'Cancelar'), onclick: TS.cancelCurrent });
    const el = TS.h('div', { class: 'progress-wrap' }, TS.h('div', { class: 'progress' }, TS.h('div', { class: 'bar', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100 }, bar), time, cancel), msg);
    const state = { width: 0, t0: Date.now(), eta: null };
    const tick = () => {
      const el_s = (Date.now() - state.t0) / 1000;
      const eta = state.width >= 5 && state.width < 100 ? (el_s * (100 - state.width)) / state.width : null;
      state.eta = eta === null ? state.eta : (state.eta === null ? eta : 0.7 * state.eta + 0.3 * eta);
      if (state.epochEta !== undefined) state.eta = Math.max(0, state.epochEta - (Date.now() - state.epochAt) / 1000);
      time.textContent = `${dur(el_s)}${state.eta !== null && state.width < 100 ? ` · ${TS.t('~{e} left', 'faltan ~{e}', { e: dur(state.eta) })}` : ''}`;
    };
    const timer = setInterval(() => { if (!el.isConnected) return clearInterval(timer); tick(); }, 1000);
    return {
      el,
      update(ev) {
        if (ev.pct !== null && ev.pct !== undefined) {
          state.width = Math.max(state.width, Math.min(100, ev.pct));
          bar.style.width = `${state.width}%`;
          bar.parentElement.setAttribute('aria-valuenow', String(Math.round(state.width)));
        }
        if (ev.eta_s !== undefined && ev.eta_s !== null) { state.epochEta = ev.eta_s; state.epochAt = Date.now(); } // measured per-epoch pace beats the linear guess
        if (ev.msg) msg.textContent = ev.epoch_s !== undefined ? `${ev.msg} · ${TS.fmt.dec(ev.epoch_s, 1)} ${TS.t('s/epoch', 's/época')}` : ev.msg;
        tick();
      },
      done(text) {
        clearInterval(timer);
        bar.style.width = '100%';
        msg.textContent = text || TS.t('Done', 'Terminado');
        time.textContent = dur((Date.now() - state.t0) / 1000);
        cancel.hidden = true;
      },
    };
  };
})();
