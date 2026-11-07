'use strict';
/**
 * Aviso de versión nueva. SOLO avisa: nunca descarga ni instala nada por su cuenta; el usuario decide.
 *
 * Qué se contacta: la API pública de GitHub del repositorio del proyecto (homepage de package.json), con una
 * petición GET sin datos del usuario. Cuándo: solo si el usuario lo activó (una pregunta la primera vez) o
 * cuando pulsa "Buscar actualizaciones". La dirección a abrir sale de la respuesta y se valida: tiene que ser
 * una página de releases de ESE repositorio. El renderer nunca elige la dirección.
 * Sin dependencias de Electron para poder probarlo con node puro (`fetchFn` se inyecta).
 */
const fs = require('node:fs');
const path = require('node:path');

const DAY_MS = 24 * 60 * 60 * 1000;

function parseRepo(homepage) {
  const m = /^https:\/\/github\.com\/([\w.-]+)\/([\w.-]+?)(?:\.git)?\/?$/.exec(String(homepage || ''));
  return m ? `${m[1]}/${m[2]}` : null;
}

/** Compara x.y.z (con o sin "v" delante y sufijo): -1, 0 o 1. */
function compareVersions(a, b) {
  const nums = (v) => String(v).replace(/^v/i, '').split(/[-+]/)[0].split('.').map((n) => parseInt(n, 10) || 0);
  const x = nums(a);
  const y = nums(b);
  for (let i = 0; i < Math.max(x.length, y.length); i += 1) {
    const d = (x[i] || 0) - (y[i] || 0);
    if (d) return d < 0 ? -1 : 1;
  }
  return 0;
}

function validReleaseUrl(url, repo) {
  return typeof url === 'string' && url.startsWith(`https://github.com/${repo}/releases`);
}

/**
 * Busca la versión publicada más nueva (incluye prelanzamientos: las primeras versiones lo son).
 * @returns {Promise<{status:'newer'|'current'|'none'|'error', current:string, latest?:object, error?:string}>}
 */
async function checkForUpdate({ repo, current, fetchFn = globalThis.fetch, timeoutMs = 8000 }) {
  if (!repo) return { status: 'error', current, error: 'no repository' };
  try {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    let res;
    try {
      res = await fetchFn(`https://api.github.com/repos/${repo}/releases?per_page=10`, {
        headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'TinyML-Studio-update-check' },
        signal: ctl.signal,
      });
    } finally {
      clearTimeout(timer);
    }
    if (res.status === 404) return { status: 'none', current }; // sin releases públicos (o repositorio privado)
    if (!res.ok) return { status: 'error', current, error: `GitHub answered ${res.status}` };
    const list = (await res.json()).filter((r) => r && !r.draft && typeof r.tag_name === 'string');
    if (!list.length) return { status: 'none', current };
    list.sort((p, q) => compareVersions(q.tag_name, p.tag_name));
    const top = list[0];
    const url = validReleaseUrl(top.html_url, repo) ? top.html_url : `https://github.com/${repo}/releases`;
    const latest = {
      version: String(top.tag_name).replace(/^v/i, ''),
      tag: top.tag_name,
      prerelease: Boolean(top.prerelease),
      url,
      notes: String(top.body || '').slice(0, 1200),
      published: top.published_at || null,
    };
    return { status: compareVersions(latest.version, current) > 0 ? 'newer' : 'current', current, latest };
  } catch (e) {
    return { status: 'error', current, error: String((e && e.message) || e) };
  }
}

// ---- preferencias del aviso (userData/updates.json) ------------------------------------------------------
function loadPrefs(file, fsMod = fs) {
  try {
    const raw = JSON.parse(fsMod.readFileSync(file, 'utf8'));
    return {
      auto: typeof raw.auto === 'boolean' ? raw.auto : null, // null = todavía no se le preguntó
      lastCheck: Number(raw.lastCheck) || 0,
      skipped: typeof raw.skipped === 'string' ? raw.skipped : '',
    };
  } catch {
    return { auto: null, lastCheck: 0, skipped: '' };
  }
}

function savePrefs(file, prefs, fsMod = fs) {
  fsMod.mkdirSync(path.dirname(file), { recursive: true });
  fsMod.writeFileSync(file, JSON.stringify({ auto: prefs.auto, lastCheck: prefs.lastCheck, skipped: prefs.skipped }, null, 2), 'utf8');
}

/** ¿Toca la comprobación automática? Solo si el usuario la activó y pasó un día desde la última. */
function dueForAutoCheck(prefs, now = Date.now()) {
  return prefs.auto === true && now - prefs.lastCheck >= DAY_MS;
}

module.exports = { parseRepo, compareVersions, validReleaseUrl, checkForUpdate, loadPrefs, savePrefs, dueForAutoCheck, DAY_MS };
