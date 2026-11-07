'use strict';
/**
 * electron-builder afterPack (solo Linux): instala un lanzador delante del binario de Electron.
 *
 * Por qué: Chromium comprueba su sandbox de forma NATIVA al arrancar (si `chrome-sandbox` existe pero
 * no es setuid root y no hay user namespaces, aborta con "The SUID sandbox helper binary was found, but
 * is not configured correctly"), antes de que se ejecute nada de JavaScript. Ninguna política dentro de la
 * app puede evitarlo; hay que decidir antes de lanzar Electron. Medido en CI (Ubuntu 24.04, AppArmor
 * restringe los user namespaces): la política en JS decidía bien y la app abortaba igual.
 *
 * Lógica del lanzador (ver launcherScript): sandbox normal si hay `chrome-sandbox` setuid-root o user
 * namespaces utilizables; si no (o si se ejecuta como root), `--no-sandbox` con un aviso visible.
 * TINYML_FORCE_SANDBOX=1 lo desactiva (falla en vez de degradar); TINYML_NO_SANDBOX=1 lo fuerza.
 */
const fs = require('node:fs');
const path = require('node:path');

function launcherScript(binName) {
  return `#!/bin/sh
# TinyML Studio: lanzador (decide el sandbox de Chromium ANTES de arrancar Electron; ver scripts/after-pack.js)
HERE="$(dirname "$(readlink -f "$0")")"
BIN="$HERE/${binName}"
TINYML_LAUNCHER="$HERE/$(basename "$0")"
export TINYML_LAUNCHER
for a in "$@"; do
  [ "$a" = "--no-sandbox" ] && exec "$BIN" "$@"
done
if [ "$TINYML_FORCE_SANDBOX" = "1" ]; then exec "$BIN" "$@"; fi
reason=""
if [ "$TINYML_NO_SANDBOX" = "1" ]; then
  reason="TINYML_NO_SANDBOX=1"
elif [ "$(id -u)" = "0" ]; then
  reason="se ejecuta como root (Chromium lo exige)"
elif [ -u "$HERE/chrome-sandbox" ] && [ "$(stat -c %u "$HERE/chrome-sandbox" 2>/dev/null)" = "0" ]; then
  exec "$BIN" "$@"
elif command -v unshare >/dev/null 2>&1 && unshare --user --map-root-user --pid --net --fork chroot / true >/dev/null 2>&1; then
  exec "$BIN" "$@"
else
  reason="ni chrome-sandbox con setuid ni user namespaces utilizables en este sistema"
fi
echo "[TinyML Studio] AVISO: sandbox de Chromium desactivado ($reason). Solo se carga contenido local; para exigirlo, instala el .deb o habilita user namespaces y define TINYML_FORCE_SANDBOX=1." >&2
exec "$BIN" --no-sandbox "$@"
`;
}

async function afterPack(context) {
  if (context.electronPlatformName !== 'linux') return;
  const name = context.packager.executableName;
  const dir = context.appOutDir;
  const real = path.join(dir, name);
  const bin = `${name}.bin`;
  fs.renameSync(real, path.join(dir, bin));
  fs.writeFileSync(real, launcherScript(bin), { mode: 0o755 });
  fs.chmodSync(real, 0o755);
}

module.exports = afterPack;
module.exports.launcherScript = launcherScript;
