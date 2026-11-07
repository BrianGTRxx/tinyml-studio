'use strict';
/**
 * Endurecimiento de Electron (docs/ARCHITECTURE.md, "Cuidados obligatorios").
 * Funciones puras o que reciben el objeto de Electron, para poder probarlas sin pantalla.
 */
const path = require('node:path');
const { pathToFileURL } = require('node:url');

// Sin scripts/estilos inline, sin red: connect-src 'none' (la UI no habla con el sidecar;
// lo hace el proceso principal vía IPC, así el token nunca llega al renderer).
const CSP = [
  "default-src 'none'",
  "script-src 'self'",
  "style-src 'self'",
  "img-src 'self' data:",
  "connect-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
  "frame-ancestors 'none'",
].join('; ');

function isPackaged() {
  try {
    return Boolean(require('electron').app?.isPackaged);
  } catch {
    return false; // fuera de Electron (pruebas con node puro)
  }
}

function buildWebPreferences(preloadPath) {
  return {
    preload: preloadPath,
    contextIsolation: true,
    nodeIntegration: false,
    nodeIntegrationInWorker: false,
    nodeIntegrationInSubFrames: false,
    sandbox: true,
    webSecurity: true,
    allowRunningInsecureContent: false,
    webviewTag: false,
    navigateOnDragDrop: false,
    spellcheck: false,
    devTools: !isPackaged(),
  };
}

/** Solo se permite cargar la propia página de la UI (file://.../renderer/*). */
function isAppUrl(url, rendererDir) {
  try {
    const base = pathToFileURL(path.resolve(rendererDir) + path.sep).href;
    return typeof url === 'string' && url.startsWith(base);
  } catch {
    return false;
  }
}

/** Una URL de petición es aceptable si es de la app, un data: o interna de DevTools. */
function isAllowedRequestUrl(url, rendererDir) {
  return isAppUrl(url, rendererDir) || url.startsWith('data:') || url.startsWith('devtools:');
}

function hardenSession(session, rendererDir) {
  session.setPermissionRequestHandler((_wc, _perm, cb) => cb(false));
  session.setPermissionCheckHandler(() => false);
  session.webRequest.onBeforeRequest((details, cb) => {
    cb({ cancel: !isAllowedRequestUrl(details.url, rendererDir) });
  });
  session.webRequest.onHeadersReceived((details, cb) => {
    cb({
      responseHeaders: { ...details.responseHeaders, 'Content-Security-Policy': [CSP] },
    });
  });
}

function hardenWebContents(contents, rendererDir) {
  contents.setWindowOpenHandler(() => ({ action: 'deny' }));
  contents.on('will-navigate', (e, url) => {
    if (!isAppUrl(url, rendererDir)) e.preventDefault();
  });
  contents.on('will-attach-webview', (e) => e.preventDefault());
}

/** El remitente de un IPC debe ser nuestra propia página. */
function isTrustedSender(event, rendererDir) {
  const url = event?.senderFrame?.url;
  return isAppUrl(url, rendererDir);
}

module.exports = {
  CSP,
  buildWebPreferences,
  isAppUrl,
  isAllowedRequestUrl,
  hardenSession,
  hardenWebContents,
  isTrustedSender,
};
