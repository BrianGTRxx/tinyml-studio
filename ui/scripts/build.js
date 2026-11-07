'use strict';
/**
 * Orquesta una compilación: prepara los recursos (packaging/prepare_resources.py) y llama a electron-builder.
 *   node scripts/build.js <raw|offline> [dir|linux|win]
 *  - raw (por defecto): la app NO lleva paquetes de Python; instala su propio entorno con uv al primer uso.
 *  - offline: lleva dentro el núcleo congelado (PyInstaller), funciona sin conexión y pesa mucho más.
 * Variables: PYTHON (intérprete para el preparador), TINYML_OFFLINE_ELECTRON=1 (usa node_modules/electron/dist
 * sin descargar nada; solo para --dir en máquinas sin red hacia GitHub).
 */
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const [variant = 'raw', target = 'dir'] = process.argv.slice(2);
if (!['raw', 'offline'].includes(variant) || !['dir', 'linux', 'win'].includes(target)) {
  console.error('uso: node scripts/build.js <raw|offline> [dir|linux|win]');
  process.exit(2);
}
const ROOT = path.resolve(__dirname, '..');
const plat = target === 'win' || (target === 'dir' && process.platform === 'win32') ? 'windows' : 'linux';
const python = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');

function run(cmd, args, env = {}) {
  const r = spawnSync(cmd, args, { cwd: ROOT, stdio: 'inherit', env: { ...process.env, ...env }, shell: process.platform === 'win32' });
  if (r.status !== 0) process.exit(r.status || 1);
}

run(python, [path.join('..', 'packaging', 'prepare_resources.py'), '--variant', variant, '--platform', plat]);
const args = ['electron-builder'];
if (target === 'dir') args.push('--dir');
else if (target === 'linux') args.push('--linux', 'AppImage', 'deb', '--x64');
else args.push('--win', 'nsis', 'zip', '--x64');
if (process.env.TINYML_OFFLINE_ELECTRON === '1') args.push('-c.electronDist=node_modules/electron/dist');
run('npx', args, { TINYML_VARIANT: variant });
