'use strict';
/**
 * Busca intérpretes de Python ya instalados (PATH, lanzador `py`, carpetas habituales, conda/pyenv) para
 * ofrecerlos al usuario en vez de obligarle a buscarlos. Solo LEE: ejecuta cada candidato con `-I` para
 * preguntarle su versión; nunca instala ni modifica nada.
 *
 * Seguridad: la ruta de un intérprete NO la fija el renderer. El renderer recibe ids; el proceso principal
 * recuerda id -> ruta y solo él arranca el candidato elegido.
 * Sin dependencias de Electron para poder probarlo con node puro (`execFileFn`/`fsMod` se inyectan).
 */
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFile } = require('node:child_process');

// Versiones: 3.10 es el mínimo del núcleo; 3.11–3.12 son las que se prueban (la app instala 3.12 propio).
const MIN = [3, 10];
const RECOMMENDED = [3, 11, 12]; // minors recomendados
const PROBE = 'import sys,json;print(json.dumps({"v":list(sys.version_info[:3]),"exe":sys.executable,"venv":sys.prefix!=getattr(sys,"base_prefix",sys.prefix),"conda":("conda-meta" in __import__("os").listdir(sys.prefix))}))';

function verdictFor(v) {
  const [maj, min] = v;
  if (maj < MIN[0] || (maj === MIN[0] && min < MIN[1])) return 'too_old';
  if (maj === 3 && RECOMMENDED.includes(min)) return 'recommended';
  if (maj === 3 && min === MIN[1]) return 'ok';
  return 'newer'; // 3.13+: algunas librerías (p. ej. TensorFlow) pueden no tener paquetes todavía
}

function candidatePaths({ platform = process.platform, env = process.env, home = os.homedir(), fsMod = fs } = {}) {
  const win = platform === 'win32';
  const exeNames = win ? ['python.exe'] : ['python3', 'python', 'python3.10', 'python3.11', 'python3.12', 'python3.13'];
  const out = [];
  const add = (p) => { if (p && !out.includes(p)) out.push(p); };
  const exists = (p) => { try { return fsMod.statSync(p).isFile(); } catch { return false; } };
  // 1) PATH
  for (const dir of String(env.PATH || env.Path || '').split(win ? ';' : ':').filter(Boolean)) {
    for (const n of exeNames) { const p = path.join(dir, n); if (exists(p)) add(p); }
  }
  // 2) carpetas habituales
  const dirs = [];
  if (win) {
    const local = env.LOCALAPPDATA || path.join(home, 'AppData', 'Local');
    const pf = env.ProgramFiles || 'C:\\Program Files';
    for (const base of [path.join(local, 'Programs', 'Python'), pf, 'C:\\']) {
      try { for (const d of fsMod.readdirSync(base)) if (/^Python3\d+/i.test(d)) dirs.push(path.join(base, d)); } catch { /* no existe */ }
    }
  } else {
    for (const d of ['/usr/bin', '/usr/local/bin', '/opt/homebrew/bin', path.join(home, '.local', 'bin')]) dirs.push(d);
    try { // pyenv
      const v = path.join(home, '.pyenv', 'versions');
      for (const d of fsMod.readdirSync(v)) dirs.push(path.join(v, d, 'bin'));
    } catch { /* sin pyenv */ }
  }
  for (const d of dirs) for (const n of exeNames) { const p = path.join(d, n); if (exists(p)) add(p); }
  // 3) conda / miniconda / anaconda (base y entornos)
  for (const root of ['miniconda3', 'anaconda3', 'miniforge3', 'mambaforge'].map((n) => path.join(home, n))) {
    const bin = (r) => (win ? path.join(r, 'python.exe') : path.join(r, 'bin', 'python'));
    if (exists(bin(root))) add(bin(root));
    try { for (const e of fsMod.readdirSync(path.join(root, 'envs'))) { const p = bin(path.join(root, 'envs', e)); if (exists(p)) add(p); } } catch { /* sin envs */ }
  }
  return out;
}

function probe(exe, { execFileFn = execFile, timeoutMs = 8000 } = {}) {
  return new Promise((resolve) => {
    execFileFn(exe, ['-I', '-c', PROBE], { timeout: timeoutMs, windowsHide: true }, (err, stdout) => {
      if (err) return resolve(null); // p. ej. el atajo de la Microsoft Store: no es un Python real
      try {
        const j = JSON.parse(String(stdout).trim().split(/\r?\n/).pop());
        if (!Array.isArray(j.v) || j.v.length < 2) return resolve(null);
        resolve({ version: j.v.join('.'), v: j.v, executable: j.exe || exe, venv: Boolean(j.venv), conda: Boolean(j.conda) });
      } catch { resolve(null); }
    });
  });
}

/** @returns {Promise<Array<{version:string, executable:string, kind:string, verdict:string}>>} sin duplicados, mejor primero */
async function scanPythons(opts = {}) {
  const paths = candidatePaths(opts);
  const found = [];
  const seen = new Set();
  const real = (p) => { try { return (opts.fsMod || fs).realpathSync(p).toLowerCase(); } catch { return p.toLowerCase(); } };
  for (const p of paths) {
    const info = await probe(p, opts);
    if (!info) continue;
    const key = real(info.executable);
    if (seen.has(key)) continue;
    seen.add(key);
    found.push({
      version: info.version,
      executable: info.executable,
      kind: info.conda ? 'conda' : info.venv ? 'venv' : 'system',
      verdict: verdictFor(info.v),
      _v: info.v,
    });
  }
  const rank = { recommended: 0, ok: 1, newer: 2, too_old: 3 };
  found.sort((a, b) => rank[a.verdict] - rank[b.verdict] || b._v[1] - a._v[1]);
  return found.map(({ _v, ...rest }) => rest);
}

module.exports = { scanPythons, candidatePaths, verdictFor, probe, MIN, RECOMMENDED };
