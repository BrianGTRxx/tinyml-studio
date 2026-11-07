'use strict';
/**
 * Gestor de instalación del entorno Python propio de la app ("entorno gestionado").
 *
 * La app llega SIN paquetes de Python: este módulo usa `uv` (binario incluido en la app) para
 *   1. crear un entorno con su propio Python (uv descarga un CPython 3.12 aparte, no toca el del sistema),
 *   2. instalar los componentes que el usuario elija desde los locks con hashes (packaging/env/locks),
 *   3. instalar la rueda de tinyml-studio (incluida en la app).
 * Sin depender de Electron (se prueba con node puro y un `uv` falso).
 *
 * Seguridad: los ids de componente se validan contra el manifiesto (el renderer nunca manda nombres de
 * paquete ni rutas); solo se instala desde PyPI con `--require-hashes --only-binary :all:` (sin ejecutar
 * setup.py de terceros); no se lee la configuración de uv del usuario (UV_NO_CONFIG).
 */
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');
const { EventEmitter } = require('node:events');

function platformKey(platform = process.platform) {
  return platform === 'win32' ? 'windows' : platform === 'darwin' ? 'macos' : 'linux';
}

function loadManifest(resourcesDir, fsMod = fs) {
  return JSON.parse(fsMod.readFileSync(path.join(resourcesDir, 'manifest.json'), 'utf8'));
}

function pythonBin(envDir, platform = process.platform) {
  return platform === 'win32' ? path.win32.join(envDir, 'Scripts', 'python.exe') : path.posix.join(envDir, 'bin', 'python');
}

/** Componentes pedidos -> ids válidos y disponibles en esta plataforma, con `base` siempre primero. */
function resolveComponents(manifest, wanted, platform = process.platform) {
  const plat = platformKey(platform);
  const byId = new Map(manifest.components.map((c) => [c.id, c]));
  const ids = ['base'];
  for (const id of Array.isArray(wanted) ? wanted : []) {
    if (typeof id !== 'string' || !byId.has(id)) throw new Error(`componente desconocido: ${String(id).slice(0, 40)}`);
    if (!byId.get(id).platforms.includes(plat)) throw new Error(`componente no disponible en ${plat}: ${id}`);
    if (!ids.includes(id)) ids.push(id);
  }
  return ids;
}

// Requisitos que el doctor propone ("scipy>=1.10,<2"): nombre + rangos de versión, nada más
// (sin URLs, rutas ni opciones de pip).
const REQ_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]*(\[[A-Za-z0-9_,-]+\])?((>=|<=|==|!=|~=|<|>)[A-Za-z0-9.*+!]+(,(>=|<=|==|!=|~=|<|>)[A-Za-z0-9.*+!]+)*)?$/;

/** Devuelve un manejador de chunks que llama a onLine por cada línea no vacía (\n o \r\n). */
function lineReader(onLine) {
  let buf = '';
  return (chunk) => {
    buf += chunk.toString('utf8');
    let i;
    while ((i = buf.search(/\r?\n/)) >= 0) {
      const line = buf.slice(0, i);
      buf = buf.slice(buf[i] === '\r' ? i + 2 : i + 1);
      if (line.trim()) onLine(line);
    }
  };
}

class Installer extends EventEmitter {
  /**
   * @param {object} o
   * @param {string} o.root          carpeta propia del entorno gestionado (userData/python-env)
   * @param {string} o.uv            ruta del binario uv
   * @param {string} o.resourcesDir  carpeta con manifest.json, locks/ y wheels/
   * @param {string} [o.repoRoot]    desarrollo: instala tinyml-studio en modo editable si no hay rueda
   * @param {string} [o.platform]
   * @param {Function} [o.spawnFn]
   */
  constructor({ root, uv, resourcesDir, repoRoot = null, platform = process.platform, spawnFn = spawn, env = process.env }) {
    super();
    Object.assign(this, { root, uv, resourcesDir, repoRoot, platform, spawnFn, baseEnv: env });
    this.envDir = path.join(root, 'env');
    this.stateFile = path.join(root, 'installed.json');
    this.running = null;
    this.cancelled = false;
    this.proc = null;
    this.scope = 'managed'; // 'own' mientras se completa el entorno propio del usuario
  }

  get python() {
    return pythonBin(this.envDir, this.platform);
  }

  manifest() {
    return loadManifest(this.resourcesDir);
  }

  status() {
    let state = { components: {} };
    try {
      state = JSON.parse(fs.readFileSync(this.stateFile, 'utf8'));
    } catch { /* aún sin instalar */ }
    const installed = fs.existsSync(this.python) && Boolean(state.components && state.components.base);
    const plat = platformKey(this.platform);
    return {
      installed,
      busy: Boolean(this.running),
      python: installed ? this.python : null,
      uvAvailable: fs.existsSync(this.uv),
      components: this.manifest().components.map((c) => ({
        id: c.id,
        label: c.label,
        note: c.note || null,
        required: Boolean(c.required),
        diskMb: c.disk_mb || null,
        available: c.platforms.includes(plat),
        installed: installed && Boolean(state.components[c.id]),
        time: (state.components[c.id] || {}).time || null,
      })),
    };
  }

  _uvEnv() {
    return {
      ...this.baseEnv,
      UV_NO_CONFIG: '1',
      UV_PYTHON_INSTALL_DIR: path.join(this.root, 'python'),
      UV_CACHE_DIR: path.join(this.root, 'cache'),
      UV_PYTHON_DOWNLOADS: 'automatic',
      UV_NO_PROGRESS: '1',
      NO_COLOR: '1',
      PYTHONUTF8: '1',
    };
  }

  /** Pasos (sin ejecutar) para instalar `ids`; separado de la ejecución para poder probarlo. */
  plan(ids) {
    const manifest = this.manifest();
    const wanted = resolveComponents(manifest, ids, this.platform);
    const plat = platformKey(this.platform);
    const steps = [];
    if (!fs.existsSync(this.python)) {
      steps.push({
        label: 'Python',
        args: ['venv', '--python', manifest.python, '--python-preference', 'only-managed', this.envDir],
      });
    }
    for (const id of wanted) {
      const lock = path.join(this.resourcesDir, 'locks', `${id}-${plat}.txt`);
      if (!fs.existsSync(lock)) throw new Error(`falta el lock de ${id} para ${plat}`);
      steps.push({
        label: id,
        component: id,
        args: ['pip', 'install', '--python', this.python, '--require-hashes', '--only-binary', ':all:', '-r', lock],
      });
    }
    const wheelsDir = path.join(this.resourcesDir, 'wheels');
    const wheel = fs.existsSync(wheelsDir) ? fs.readdirSync(wheelsDir).find((f) => /^tinyml_studio-.*\.whl$/.test(f)) : null;
    if (wheel) {
      const wheelPath = path.join(wheelsDir, wheel);
      steps.push({ label: 'tinyml-studio', wheelPath, args: ['pip', 'install', '--python', this.python, '--no-deps', '--reinstall', wheelPath] });
    } else if (this.repoRoot) {
      steps.push({ label: 'tinyml-studio (dev)', args: ['pip', 'install', '--python', this.python, '--no-deps', '-e', this.repoRoot] });
    } else {
      throw new Error('falta la rueda de tinyml-studio');
    }
    return steps;
  }

  /**
   * Pasos (sin ejecutar) para completar el entorno PROPIO del usuario (`python`): la rueda de
   * tinyml-studio (sin dependencias) y solo los paquetes que faltan o están fuera de rango. No usa los
   * locks con hashes: fijar versiones exactas rompería las librerías que el usuario ya tiene.
   */
  planExternal({ python, wheel = null, requirements = [] }) {
    if (typeof python !== 'string' || !path.isAbsolute(python)) throw new Error('ruta de Python no válida');
    const steps = [];
    if (wheel) {
      if (!/^tinyml_studio-[^/\\]*\.whl$/.test(path.basename(wheel))) throw new Error('rueda no válida');
      steps.push({ label: 'tinyml-studio', args: ['pip', 'install', '--python', python, '--no-deps', wheel] });
    }
    const reqs = [...new Set(requirements)];
    for (const r of reqs) if (typeof r !== 'string' || !REQ_RE.test(r)) throw new Error(`requisito no válido: ${String(r).slice(0, 40)}`);
    if (reqs.length) steps.push({ label: reqs.map((r) => r.split(/[<>=!~[]/)[0]).join(', '), args: ['pip', 'install', '--python', python, ...reqs] });
    return steps;
  }

  /** Ejecuta `planExternal` con los mismos eventos que `install` (log / step / state). */
  async installExternal(opts) {
    if (this.running) throw new Error('ya hay una instalación en curso');
    if (!fs.existsSync(this.uv)) throw new Error(`no se encuentra uv: ${this.uv}`);
    const steps = this.planExternal(opts);
    if (!steps.length) return;
    this.cancelled = false;
    this.scope = 'own';
    this.running = (async () => {
      fs.mkdirSync(this.root, { recursive: true });
      this.emit('state', { state: 'running', steps: steps.map((st) => st.label), scope: this.scope });
      for (let i = 0; i < steps.length; i += 1) {
        this.emit('step', { index: i, total: steps.length, label: steps[i].label, scope: this.scope });
        await this._run(steps[i]);
      }
    })();
    try {
      await this.running;
      this.emit('state', { state: 'done', scope: this.scope });
    } catch (e) {
      this.emit('state', { state: this.cancelled ? 'cancelled' : 'failed', error: e.message, scope: this.scope });
      throw e;
    } finally {
      this.running = null;
      this.scope = 'managed';
    }
  }

  _run(step) {
    return new Promise((resolve, reject) => {
      const proc = this.spawnFn(this.uv, step.args, { env: this._uvEnv(), windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
      this.proc = proc;
      const tail = [];
      const reader = (stream) => lineReader((line) => {
        tail.push(line);
        if (tail.length > 30) tail.shift();
        this.emit('log', { line, stream, step: step.label, scope: this.scope });
      });
      if (proc.stdout) proc.stdout.on('data', reader('out'));
      if (proc.stderr) proc.stderr.on('data', reader('err'));
      proc.once('error', (e) => reject(new Error(`no se pudo ejecutar uv: ${e.message}`)));
      proc.once('close', (code) => {
        this.proc = null;
        if (this.cancelled) return reject(new Error('cancelado'));
        if (code === 0) return resolve();
        reject(new Error(`uv ${step.args.slice(0, 2).join(' ')} falló (código ${code}): ${tail.slice(-4).join(' | ')}`.slice(0, 600)));
      });
    });
  }

  async install(ids) {
    if (this.running) throw new Error('ya hay una instalación en curso');
    if (!fs.existsSync(this.uv)) throw new Error(`no se encuentra uv: ${this.uv}`);
    const steps = this.plan(ids);
    this.cancelled = false;
    this.running = (async () => {
      fs.mkdirSync(this.root, { recursive: true });
      this.emit('state', { state: 'running', steps: steps.map((s) => s.label), scope: this.scope });
      const done = [];
      let wheelSha = null;
      for (let i = 0; i < steps.length; i += 1) {
        this.emit('step', { index: i, total: steps.length, label: steps[i].label, scope: this.scope });
        await this._run(steps[i]);
        if (steps[i].component) done.push(steps[i].component);
        if (steps[i].wheelPath) wheelSha = sha256File(steps[i].wheelPath);
      }
      let state = { components: {} };
      try { state = JSON.parse(fs.readFileSync(this.stateFile, 'utf8')); } catch { /* nuevo */ }
      const now = new Date().toISOString();
      for (const id of done) state.components[id] = { time: now };
      state.python = this.python;
      if (wheelSha) state.wheelSha = wheelSha; // para detectar una rueda nueva tras actualizar la app
      fs.writeFileSync(this.stateFile, JSON.stringify(state, null, 2), 'utf8');
    })();
    try {
      await this.running;
      this.emit('state', { state: 'done', scope: this.scope });
    } catch (e) {
      this.emit('state', { state: this.cancelled ? 'cancelled' : 'failed', error: e.message, scope: this.scope });
      throw e;
    } finally {
      this.running = null;
    }
  }

  /** Rueda de tinyml-studio que trae esta versión de la app (o null: desarrollo, que usa el código del repo). */
  bundledWheel() {
    const dir = path.join(this.resourcesDir, 'wheels');
    try {
      const f = fs.readdirSync(dir).find((n) => /^tinyml_studio-.*\.whl$/.test(n));
      return f ? path.join(dir, f) : null;
    } catch { return null; }
  }

  /**
   * Tras actualizar la app, el entorno gestionado que ya existe conserva la rueda ANTERIOR de tinyml-studio
   * (la versión del paquete no cambia entre compilaciones). Compara el hash de la rueda incluida con el que
   * se instaló y, si difiere, reinstala SOLO esa rueda (sin dependencias, ~1 MB). No toca nada más.
   * @returns {Promise<{refreshed: boolean, reason?: string}>}
   */
  async refreshWheel() {
    if (this.running) return { refreshed: false, reason: 'busy' };
    const wheel = this.bundledWheel();
    if (!wheel || !this.status().installed || !fs.existsSync(this.uv)) return { refreshed: false, reason: 'n/a' };
    const sha = sha256File(wheel);
    let state = { components: {} };
    try { state = JSON.parse(fs.readFileSync(this.stateFile, 'utf8')); } catch { return { refreshed: false, reason: 'n/a' }; }
    if (state.wheelSha === sha) return { refreshed: false, reason: 'current' };
    this.cancelled = false;
    this.running = this._run({ label: 'tinyml-studio', args: ['pip', 'install', '--python', this.python, '--no-deps', '--reinstall', wheel] });
    try {
      await this.running;
    } finally {
      this.running = null;
    }
    state.wheelSha = sha;
    fs.writeFileSync(this.stateFile, JSON.stringify(state, null, 2), 'utf8');
    return { refreshed: true };
  }

  cancel() {
    if (!this.running || !this.proc) return false;
    this.cancelled = true;
    const pid = this.proc.pid;
    if (this.platform === 'win32') spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true });
    else try { process.kill(pid, 'SIGTERM'); } catch { /* ya terminó */ }
    return true;
  }

  /** Borra el entorno gestionado (Python incluido). No toca nada fuera de `root`. */
  reset() {
    if (this.running) throw new Error('hay una instalación en curso');
    for (const sub of ['env', 'python', 'cache']) fs.rmSync(path.join(this.root, sub), { recursive: true, force: true });
    fs.rmSync(this.stateFile, { force: true });
  }
}

function sha256File(file) {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

/** Dónde está `uv`: TINYML_UV, el incluido en la app (resources/uv) o, en desarrollo, el del PATH/.venv. */
function findUv({ isPackaged, resourcesPath, repoRoot, env = process.env, platform = process.platform, existsSync = fs.existsSync }) {
  const exe = platform === 'win32' ? 'uv.exe' : 'uv';
  if (env.TINYML_UV) return env.TINYML_UV;
  if (isPackaged && resourcesPath) return path.join(resourcesPath, 'uv', exe);
  const venvBin = platform === 'win32' ? path.win32.join(repoRoot, '.venv', 'Scripts', exe) : path.posix.join(repoRoot, '.venv', 'bin', exe);
  if (existsSync(venvBin)) return venvBin;
  for (const dir of String(env.PATH || '').split(path.delimiter)) {
    if (dir && existsSync(path.join(dir, exe))) return path.join(dir, exe);
  }
  return path.join(repoRoot, '.venv', 'bin', exe); // no existe: status().uvAvailable lo dirá
}

module.exports = { findUv, lineReader, Installer, platformKey, loadManifest, pythonBin, resolveComponents };
