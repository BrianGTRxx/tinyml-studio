'use strict';
// Theme (light/dark/system) and language (en/es). Preferences live in localStorage (try/catch:
// it can be unavailable); everything else is a plain attribute on <html>, so there is no animation
// and no extra memory cost.
(function () {
  const TS = (window.TS = window.TS || {});
  const store = {
    get(k) { try { return window.localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch { /* ignore */ } },
  };
  const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  TS.themePref = store.get('ts.theme') || 'system';
  TS.applyTheme = function applyTheme() {
    const dark = TS.themePref === 'dark' || (TS.themePref === 'system' && mq && mq.matches);
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  };
  TS.setTheme = function setTheme(p) {
    TS.themePref = ['light', 'dark', 'system'].includes(p) ? p : 'system';
    store.set('ts.theme', TS.themePref);
    TS.applyTheme();
  };
  if (mq && mq.addEventListener) mq.addEventListener('change', () => TS.themePref === 'system' && TS.applyTheme());
  TS.applyTheme();

  const nav = (window.navigator && window.navigator.language) || 'en';
  TS.lang = ['en', 'es'].includes(store.get('ts.lang')) ? store.get('ts.lang') : nav.toLowerCase().startsWith('es') ? 'es' : 'en';
  document.documentElement.setAttribute('lang', TS.lang);
  // English is the source language: t('English', 'Español', {x: 1}) replaces {x}.
  TS.t = function t(en, es, kw) {
    let s = TS.lang === 'es' && es !== undefined ? es : en;
    if (kw) for (const [k, v] of Object.entries(kw)) s = s.split(`{${k}}`).join(String(v));
    return s;
  };
  TS.setLang = function setLang(l) {
    TS.lang = l === 'es' ? 'es' : 'en';
    store.set('ts.lang', TS.lang);
    document.documentElement.setAttribute('lang', TS.lang);
  };
})();
