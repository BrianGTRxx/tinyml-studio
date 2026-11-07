'use strict';
/**
 * Sandbox de Chromium en Linux empaquetado (AppImage / .deb).
 *
 * Chromium necesita UNO de estos dos mecanismos:
 *   a) el ayudante `chrome-sandbox` con setuid root (el .deb lo instala así; el AppImage no puede), o
 *   b) espacios de nombres de usuario sin privilegios (userns).
 * Muchas distribuciones los restringen (Debian: kernel.unprivileged_userns_clone=0; Ubuntu 24.04:
 * kernel.apparmor_restrict_unprivileged_userns=1; contenedores con max_user_namespaces=0) y entonces
 * Electron aborta al arrancar ("The SUID sandbox helper binary was found, but is not configured
 * correctly"). Decisión (ADR-011): SOLO en Linux empaquetado, si ninguno de los dos está disponible,
 * se arranca con --no-sandbox y una línea visible en el log. En Windows nunca. contextIsolation,
 * CSP, sandbox:true por ventana y la navegación bloqueada no se tocan. TINYML_FORCE_SANDBOX=1
 * desactiva el respaldo (falla en vez de degradar).
 */
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function readText(file, fsMod = fs) {
  try {
    return fsMod.readFileSync(file, 'utf8').trim();
  } catch {
    return null;
  }
}

function suidHelperOk(execPath, fsMod = fs) {
  try {
    const st = fsMod.statSync(path.join(path.dirname(execPath), 'chrome-sandbox'));
    return st.uid === 0 && (st.mode & 0o4000) !== 0;
  } catch {
    return false;
  }
}

// Prueba real, parecida a lo que hace Chromium (espacios user+pid+net y chroot dentro): ¿puede este usuario? (los sysctl no lo dicen todo:
// AppArmor, seccomp o el runtime del contenedor pueden bloquearlo aunque los sysctl parezcan correctos).
// Si `unshare` no existe, no se concluye nada (null) y manda la lectura de sysctl.
function defaultProbe() {
  try {
    const r = spawnSync('unshare', ['--user', '--map-root-user', '--pid', '--net', '--fork', 'chroot', '/', 'true'], { timeout: 3000, stdio: 'ignore' });
    if (r.error) return null;
    return r.status === 0;
  } catch {
    return null;
  }
}

function usernsUsable(fsMod = fs, probe = fsMod === fs ? defaultProbe : () => null) {
  const clone = readText('/proc/sys/kernel/unprivileged_userns_clone', fsMod); // Debian/Ubuntu antiguos
  if (clone === '0') return false;
  const max = readText('/proc/sys/user/max_user_namespaces', fsMod);
  if (max !== null && Number(max) === 0) return false;
  const aa = readText('/proc/sys/kernel/apparmor_restrict_unprivileged_userns', fsMod); // Ubuntu 23.10+
  if (aa === '1') return false;
  return probe() !== false;
}

/**
 * @returns {{disable:boolean, reason:string}}
 */
function decideSandbox({ platform = process.platform, isPackaged, execPath, env = process.env, argv = process.argv, fsMod = fs }) {
  if (platform !== 'linux') return { disable: false, reason: 'no es Linux: sandbox siempre activo' };
  if (!isPackaged) return { disable: false, reason: 'modo desarrollo: sandbox sin tocar' };
  if (argv.includes('--no-sandbox')) return { disable: false, reason: '--no-sandbox ya indicado por el usuario' };
  if (env.TINYML_FORCE_SANDBOX === '1') return { disable: false, reason: 'TINYML_FORCE_SANDBOX=1' };
  if (suidHelperOk(execPath, fsMod)) return { disable: false, reason: 'chrome-sandbox con setuid root' };
  if (usernsUsable(fsMod)) return { disable: false, reason: 'userns sin privilegios disponible' };
  return {
    disable: true,
    reason: 'ni chrome-sandbox con setuid ni userns sin privilegios disponibles en este sistema',
  };
}

/** Llamar ANTES de app.whenReady(). */
function applyLinuxSandboxPolicy(app, log = console.warn) {
  const d = decideSandbox({ isPackaged: app.isPackaged, execPath: process.execPath });
  if (d.disable) {
    app.commandLine.appendSwitch('no-sandbox');
    log(`[TinyML Studio] AVISO: sandbox de Chromium desactivado (${d.reason}). Solo se carga contenido local; ` +
      'para exigirlo, instala el .deb o habilita userns y define TINYML_FORCE_SANDBOX=1.');
  }
  return d;
}

module.exports = { decideSandbox, applyLinuxSandboxPolicy, suidHelperOk, usernsUsable };
