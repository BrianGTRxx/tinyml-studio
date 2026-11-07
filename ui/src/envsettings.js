'use strict';
/**
 * Entorno Python elegido por el usuario ("usa tu propio entorno": venv, conda, miniconda).
 * Sin depender de Electron para poder probarlo con node puro.
 *
 * Seguridad: la ruta del intérprete SOLO sale de un diálogo nativo del proceso principal; el renderer
 * nunca puede fijar una ruta (sería ejecución de código arbitrario en el siguiente arranque).
 */
const fs = require('node:fs');
const path = require('node:path');
const { execFile } = require('node:child_process');

const COMPONENTS = ['base', 'tui', 'keras', 'torch', 'tflm'];
const LANGS = ['en', 'es'];

function loadSettings(file, fsMod = fs) {
  try {
    const raw = JSON.parse(fsMod.readFileSync(file, 'utf8'));
    const py = typeof raw.python === 'string' && path.isAbsolute(raw.python) ? raw.python : null;
    return { python: py };
  } catch {
    return { python: null };
  }
}

function saveSettings(file, settings, fsMod = fs) {
  fsMod.mkdirSync(path.dirname(file), { recursive: true });
  fsMod.writeFileSync(file, JSON.stringify({ python: settings.python || null }, null, 2), 'utf8');
}

function cleanComponents(list) {
  const out = (Array.isArray(list) && list.length ? list : ['base']).filter((c) => COMPONENTS.includes(c));
  return out.length ? [...new Set(out)] : ['base'];
}

/**
 * Comando que ejecuta `doctor`.
 *  - Con un intérprete concreto (`python`): lo ejecuta ÉL MISMO sobre `doctor_standalone.py` (solo
 *    biblioteca estándar; no necesita tinyml_studio instalado, así sirve en la app "cruda").
 *  - Sin intérprete: el núcleo incluido (`cli doctor` congelado) o el de desarrollo (`-m`).
 * @param {{kind:string, command:string}} builtin  resolveCore() sin intérprete de usuario
 * @param {string|null} python                      intérprete a revisar (null: el propio núcleo)
 * @param {string} standalone                       ruta de doctor_standalone.py
 */
function doctorCommand(builtin, python, components, lang, standalone) {
  const tail = ['--json', '--components', cleanComponents(components).join(',')];
  const langArgs = LANGS.includes(lang) ? ['--lang', lang] : [];
  if (python) {
    if (!standalone) throw new Error('falta doctor_standalone.py');
    return { command: python, args: ['-I', standalone, ...langArgs, ...tail] };
  }
  if (builtin.kind === 'frozen') return { command: builtin.command, args: ['cli', ...langArgs, 'doctor', ...tail] };
  return { command: builtin.command, args: ['-m', 'tinyml_studio.cli', ...langArgs, 'doctor', ...tail] };
}

/** Ejecuta el doctor. El código de salida 1 (entorno con problemas) también trae JSON válido. */
function runDoctor(builtin, python, components, lang, { cwd, standalone, timeoutMs = 60000, execFileFn = execFile } = {}) {
  return new Promise((resolve, reject) => {
    const { command, args } = doctorCommand(builtin, python, components, lang, standalone);
    execFileFn(command, args, { cwd, timeout: timeoutMs, windowsHide: true, maxBuffer: 4 * 1024 * 1024, env: { ...process.env, PYTHONUTF8: '1' } },
      (err, stdout, stderr) => {
        const text = String(stdout || '').trim();
        try {
          resolve(JSON.parse(text));
          return;
        } catch { /* sin JSON: error real */ }
        reject(new Error(String(stderr || (err && err.message) || 'doctor sin salida').trim().slice(-400)));
      });
  });
}

module.exports = { COMPONENTS, loadSettings, saveSettings, cleanComponents, doctorCommand, runDoctor };
