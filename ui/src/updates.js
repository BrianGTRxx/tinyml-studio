'use strict';
/**
 * Aviso de versión nueva y, solo si el usuario pulsa "Actualizar ahora", descarga del instalador (Windows).
 * Nunca descarga ni instala nada por su cuenta: el usuario decide cada paso.
 *
 * Qué se contacta: la API pública de GitHub del repositorio del proyecto (homepage de package.json), con una
 * petición GET sin datos del usuario. Cuándo: solo si el usuario lo activó (una pregunta la primera vez) o
 * cuando pulsa "Buscar actualizaciones". La dirección a abrir sale de la respuesta y se valida: tiene que ser
 * una página de releases de ESE repositorio. El renderer nunca elige la dirección.
 * Sin dependencias de Electron para poder probarlo con node puro (`fetchFn` se inyecta).
 */
const crypto = require('node:crypto');
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
      installer: pickInstaller(top.assets, repo, top.tag_name),
    };
    return { status: compareVersions(latest.version, current) > 0 ? 'newer' : 'current', current, latest };
  } catch (e) {
    return { status: 'error', current, error: String((e && e.message) || e) };
  }
}

// ---- descarga verificada del instalador (Windows, solo a petición del usuario) --------------------------------
const MAX_INSTALLER_BYTES = 600 * 1024 * 1024;

/**
 * Busca en los archivos del release el instalador de Windows y su lista de huellas. Solo se aceptan direcciones
 * de descarga de ESE repositorio y de ESA etiqueta; cualquier otra cosa se ignora.
 * @returns {{name:string,url:string,size:number,sumsUrl:string}|null}
 */
function pickInstaller(assets, repo, tag) {
  if (!Array.isArray(assets) || !repo || typeof tag !== 'string') return null;
  const base = `https://github.com/${repo}/releases/download/${encodeURIComponent(tag)}/`;
  const ok = (a) => a && typeof a.name === 'string' && typeof a.browser_download_url === 'string' && a.browser_download_url === base + encodeURIComponent(a.name);
  const exe = assets.find((a) => ok(a) && /^TinyML-Studio-raw-Setup-[\w.+-]+\.exe$/.test(a.name));
  const sums = assets.find((a) => ok(a) && a.name === 'SHA256SUMS.txt');
  const size = exe ? Number(exe.size) : 0;
  if (!exe || !sums || !(size > 0) || size > MAX_INSTALLER_BYTES) return null;
  return { name: exe.name, url: exe.browser_download_url, size, sumsUrl: sums.browser_download_url };
}

/** Huella SHA-256 de `name` dentro de un SHA256SUMS.txt (formato "huella  archivo"), o null. */
function parseSums(text, name) {
  for (const line of String(text).split(/\r?\n/)) {
    const m = /^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$/.exec(line);
    if (m && m[2] === name) return m[1].toLowerCase();
  }
  return null;
}

/**
 * Descarga el instalador a `destDir`, calcula su SHA-256 mientras llega y lo compara con la lista de huellas del
 * mismo release. Si no coincide (o el tamaño no es el anunciado) borra el archivo y falla.
 * @returns {Promise<string>} ruta del instalador verificado
 */
async function downloadVerified({ fetchFn, installer, destDir, onProgress, fsMod = fs, timeoutMs = 30 * 60 * 1000 }) {
  if (!installer) throw new Error('no installer in this release');
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  const dest = path.join(destDir, installer.name);
  try {
    const sumsRes = await fetchFn(installer.sumsUrl, { signal: ctl.signal });
    if (!sumsRes.ok) throw new Error(`SHA256SUMS: GitHub answered ${sumsRes.status}`);
    const expected = parseSums(await sumsRes.text(), installer.name);
    if (!expected) throw new Error('SHA256SUMS has no entry for the installer');

    fsMod.rmSync(destDir, { recursive: true, force: true });
    fsMod.mkdirSync(destDir, { recursive: true });
    const res = await fetchFn(installer.url, { signal: ctl.signal });
    if (!res.ok || !res.body) throw new Error(`installer: GitHub answered ${res.status}`);
    const hash = crypto.createHash('sha256');
    const out = fsMod.openSync(dest, 'w');
    let done = 0;
    try {
      const reader = res.body.getReader();
      for (;;) {
        const { value, done: end } = await reader.read();
        if (end) break;
        done += value.length;
        if (done > installer.size || done > MAX_INSTALLER_BYTES) throw new Error('the download is bigger than announced');
        hash.update(value);
        fsMod.writeSync(out, value);
        if (onProgress) onProgress({ done, total: installer.size });
      }
    } finally {
      fsMod.closeSync(out);
    }
    if (done !== installer.size) throw new Error('the download is incomplete');
    if (hash.digest('hex') !== expected) throw new Error('the SHA-256 does not match the release list');
    return dest;
  } catch (e) {
    try { fsMod.rmSync(dest, { force: true }); } catch { /* nada que borrar */ }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

/** ¿Esta copia está instalada con el instalador de Windows (hay desinstalador al lado)? Si no, es portable. */
function isInstalledCopy(execPath, productName, fsMod = fs) {
  try {
    return fsMod.existsSync(path.join(path.dirname(execPath), `Uninstall ${productName}.exe`));
  } catch {
    return false;
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

module.exports = { parseRepo, compareVersions, validReleaseUrl, checkForUpdate, pickInstaller, parseSums, downloadVerified, isInstalledCopy, MAX_INSTALLER_BYTES, loadPrefs, savePrefs, dueForAutoCheck, DAY_MS };
