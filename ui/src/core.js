'use strict';
/**
 * Resolución del núcleo Python (sin depender de Electron, para poder probarla con node puro).
 *  - Empaquetado (app.isPackaged) o TINYML_CORE_EXE definido: se lanza el ejecutable congelado
 *    (PyInstaller) `resources/core/tinyml-core[.exe]` sin argumentos, con cwd = su carpeta.
 *  - Desarrollo: `python -m tinyml_studio.api` desde la raíz del repo (TINYML_PYTHON sin cambios).
 * El protocolo del sidecar (token por entorno, línea "ready", cierre al cerrarse stdin) es idéntico.
 */
const os = require('node:os');
const path = require('node:path');

const CORE_DIR_NAME = 'core';

function coreExeName(platform = process.platform) {
  return platform === 'win32' ? 'tinyml-core.exe' : 'tinyml-core';
}

/**
 * @param {object} o
 * @param {boolean} o.isPackaged     app.isPackaged
 * @param {string}  [o.resourcesPath] process.resourcesPath (solo si está empaquetado)
 * @param {string}  o.repoRoot       raíz del repo (modo desarrollo)
 * @param {string}  [o.platform]     process.platform
 * @param {object}  [o.env]          process.env
 * @param {string|null} [o.userPython] intérprete propio elegido por el usuario (venv/conda): modo 'external'
 * @param {string|null} [o.managedPython] Python del entorno que instaló la app (uv): modo 'managed'
 * @param {boolean} [o.hasFrozen] existe resources/core (variante "completa sin conexión"); si no, 'none'
 * @returns {{kind:'frozen'|'dev'|'external'|'managed'|'none', command:string, args:string[], cwd:string}}
 */
function resolveCore({ isPackaged, resourcesPath, repoRoot, platform = process.platform, env = process.env, userPython = null, managedPython = null, hasFrozen = true }) {
  const pathMod = platform === 'win32' ? path.win32 : path.posix;
  const override = env.TINYML_CORE_EXE;
  if (override) {
    return { kind: 'frozen', command: override, args: [], cwd: pathMod.dirname(override) };
  }
  if (userPython) {
    // Su entorno: necesita tinyml-studio instalado (lo comprueba `tinyml-studio doctor`).
    return { kind: 'external', command: userPython, args: ['-m', 'tinyml_studio.api'], cwd: os.homedir() };
  }
  if (managedPython) {
    return { kind: 'managed', command: managedPython, args: ['-m', 'tinyml_studio.api'], cwd: os.homedir() };
  }
  if (env.TINYML_FORCE_RAW === '1') {
    // pruebas: se comporta como la app "cruda" (sin núcleo ni paquetes) aunque sea desarrollo
    return { kind: 'none', command: '', args: [], cwd: os.homedir() };
  }
  if (isPackaged) {
    if (!hasFrozen) {
      // app "cruda": sin núcleo ni paquetes; hay que instalarlos (instalador) o elegir un entorno propio
      return { kind: 'none', command: '', args: [], cwd: os.homedir() };
    }
    if (!resourcesPath) throw new Error('resourcesPath no definido en la app empaquetada');
    const dir = pathMod.join(resourcesPath, CORE_DIR_NAME);
    return { kind: 'frozen', command: pathMod.join(dir, coreExeName(platform)), args: [], cwd: dir };
  }
  return {
    kind: 'dev',
    command: env.TINYML_PYTHON || (platform === 'win32' ? 'python' : 'python3'),
    args: ['-m', 'tinyml_studio.api'],
    cwd: repoRoot,
  };
}

module.exports = { resolveCore, coreExeName, CORE_DIR_NAME };
