'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { app, BrowserWindow, dialog, ipcMain, Menu, session, shell } = require('electron');
const { Sidecar } = require('./sidecar');
const security = require('./security');
const bridge = require('./bridge');
const { resolveCore, coreExeName } = require('./core');
const { Installer, findUv } = require('./installer');
const envsettings = require('./envsettings');
const updates = require('./updates');
const recent = require('./recent');
const { scanPythons } = require('./pyscan');
const { applyLinuxSandboxPolicy } = require('./linux-sandbox');

const RENDERER_DIR = path.join(__dirname, 'renderer');
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const SMOKE = process.env.TINYML_SMOKE === '1';

// Linux empaquetado (AppImage/.deb): respaldo documentado si Chromium no puede crear su sandbox.
// Debe ir antes de app.whenReady(). No hace nada en Windows ni en desarrollo (ver linux-sandbox.js).
applyLinuxSandboxPolicy(app);

// Una sola instancia: evita dos sidecars compitiendo.
if (!app.requestSingleInstanceLock()) app.quit();

const WORKDIR = process.env.TINYML_WORKDIR || path.join(app.getPath('userData'), 'workspace');
// Ajustes del usuario (intérprete propio): userData/settings.json. La ruta solo sale de un diálogo nativo.
const SETTINGS_FILE = path.join(app.getPath('userData'), 'settings.json');
const settings = envsettings.loadSettings(SETTINGS_FILE);
const coreOpts = { isPackaged: app.isPackaged, resourcesPath: process.resourcesPath, repoRoot: REPO_ROOT };
// Variante "completa sin conexión": resources/core/tinyml-core[.exe] (PyInstaller). La app "cruda" no lo trae.
const hasFrozen = !app.isPackaged || fs.existsSync(path.join(process.resourcesPath || '', 'core', coreExeName()));
// Entorno gestionado: la app instala su propio Python y paquetes con `uv` (packaging/env, src/installer.js).
const RES_ENV = app.isPackaged ? path.join(process.resourcesPath, 'env') : path.join(REPO_ROOT, 'packaging', 'env');
const DOCTOR_SCRIPT = path.join(RES_ENV, 'doctor_standalone.py');
const installer = new Installer({
  root: path.join(app.getPath('userData'), 'python-env'),
  uv: findUv(coreOpts),
  resourcesDir: RES_ENV,
  repoRoot: app.isPackaged ? null : REPO_ROOT, // desarrollo: tinyml-studio editable en vez de la rueda
});
// Núcleo incluido o de desarrollo (para el doctor cuando no se revisa un intérprete concreto).
const BUILTIN = resolveCore({ ...coreOpts, hasFrozen });

/** Prioridad: intérprete propio > entorno gestionado > núcleo incluido/desarrollo > ninguno (app cruda). */
function pickCore() {
  const userPython = settings.python && fs.existsSync(settings.python) ? settings.python : null;
  const st = installer.status();
  return resolveCore({ ...coreOpts, hasFrozen, userPython, managedPython: st.installed ? st.python : null });
}
let CORE = pickCore();
let startError = null; // el entorno elegido no arrancó: la interfaz muestra el motivo y deja repararlo
console.log(`[TinyML Studio] núcleo (${CORE.kind}): ${CORE.command} ${CORE.args.join(' ')}`.trim());
const sidecar = new Sidecar({
  python: CORE.command,
  args: CORE.args,
  cwd: CORE.cwd,
  env: { TINYML_WORKDIR: WORKDIR },
  allowedRoots: process.env.TINYML_ALLOWED_ROOTS
    ? process.env.TINYML_ALLOWED_ROOTS.split(path.delimiter)
    : [],
});
let win = null;
const watchers = new Map();
const registeredRoots = new Set(); // carpetas que el usuario eligió: se vuelven a registrar al reiniciar el núcleo

function trusted(handler) {
  return async (event, ...args) => {
    if (!security.isTrustedSender(event, RENDERER_DIR)) throw new Error('remitente no confiable');
    return handler(event, ...args);
  };
}
const send = (channel, payload) => {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
};

ipcMain.handle('sidecar:health', trusted(() => sidecar.health()));

/** Para el sidecar, recalcula qué núcleo usar y lo arranca (sin relanzar la app). */
async function restartCore() {
  await sidecar.stop();
  CORE = pickCore();
  Object.assign(sidecar, { python: CORE.command, args: CORE.args, cwd: CORE.cwd });
  startError = null;
  if (CORE.kind !== 'none') {
    try {
      await sidecar.start();
      for (const d of registeredRoots) await sidecar.request('POST', '/roots', { path: d });
    } catch (e) {
      startError = String(e.message || e);
    }
  }
  console.log(`[TinyML Studio] núcleo (${CORE.kind}): ${CORE.command} ${CORE.args.join(' ')}`.trim());
  send('core:changed', { core: CORE.kind, error: startError });
}

// ---- entorno Python ----
let pendingPython = null; // elegido en el diálogo nativo, aún sin aplicar
const doctorTarget = () => (CORE.kind === 'external' ? settings.python : CORE.kind === 'managed' ? CORE.command : null);
const envOpts = () => ({ cwd: app.getPath('userData'), standalone: DOCTOR_SCRIPT });
ipcMain.handle('env:get', trusted(() => ({
  mode: CORE.kind === 'external' ? 'external' : 'builtin',
  python: settings.python,
  missing: Boolean(settings.python && CORE.kind !== 'external' && !fs.existsSync(settings.python)),
  core: CORE.kind,
  builtin: BUILTIN.kind,
  frozenAvailable: hasFrozen && app.isPackaged,
  startError,
  installer: installer.status(),
})));
ipcMain.handle('env:doctor', trusted((_e, components, lang) => {
  const target = doctorTarget();
  if (!target && BUILTIN.kind === 'none') throw new Error('no hay ningún Python que revisar todavía');
  return envsettings.runDoctor(BUILTIN, target, components, lang, envOpts());
}));
/** Rueda de tinyml-studio que trae la app (para instalarla en el entorno propio del usuario). */
function findWheel() {
  const dir = path.join(RES_ENV, 'wheels');
  try {
    const f = fs.readdirSync(dir).find((n) => /^tinyml_studio-.*\.whl$/.test(n));
    return f ? path.join(dir, f) : null;
  } catch { return null; }
}
/** Revisa un intérprete elegido (diálogo nativo o lista detectada) y lo deja pendiente de aplicar. */
async function checkPicked(picked, lang) {
  try {
    const report = await envsettings.runDoctor(BUILTIN, picked, ['base'], lang, envOpts());
    pendingPython = picked;
    return { path: picked, report, wheel: findWheel() };
  } catch (e) {
    pendingPython = null;
    return { path: picked, error: String(e.message || e) };
  }
}
ipcMain.handle('env:pick', trusted(async (_e, lang) => {
  const r = await dialog.showOpenDialog(win, { properties: ['openFile'], title: 'Python' });
  if (r.canceled || !r.filePaths.length) return { canceled: true };
  return checkPicked(r.filePaths[0], lang);
}));
/** Pythons ya instalados en el equipo. El renderer solo recibe ids: la ruta la guarda este proceso. */
let scanned = new Map();
ipcMain.handle('env:scan', trusted(async () => {
  const list = await scanPythons();
  scanned = new Map(list.map((p, i) => [`py${i}`, p.executable]));
  return list.map((p, i) => ({ id: `py${i}`, ...p }));
}));
ipcMain.handle('env:useScanned', trusted(async (_e, id, lang) => {
  const exe = scanned.get(String(id));
  if (!exe) throw new Error('intérprete desconocido: vuelve a buscar');
  return checkPicked(exe, lang);
}));
/** Completa el entorno propio elegido: instala la rueda de tinyml-studio y lo que falte, tras pedir permiso. */
ipcMain.handle('env:fixOwn', trusted(async (_e, lang, components) => {
  // El entorno recién elegido (aún sin aplicar) o el entorno propio que la app ya está usando.
  const py = pendingPython || (CORE.kind === 'external' ? settings.python : null);
  if (!py) throw new Error('nada que arreglar: elige un Python propio primero');
  if (installer.status().busy) throw new Error('ya hay una instalación en curso');
  const en = lang === 'en';
  const comps = envsettings.cleanComponents(components); // ids conocidos; nunca nombres de paquete del renderer
  const before = await envsettings.runDoctor(BUILTIN, py, comps, lang, envOpts());
  if (!before.python.ok) throw new Error(en ? 'This Python is too old.' : 'Este Python es demasiado antiguo.');
  const needStudio = !before.tinyml_studio;
  const wheel = needStudio ? findWheel() : null;
  if (needStudio && !wheel) throw new Error(en ? 'This build does not ship the tinyml-studio package.' : 'Esta versión no trae el paquete tinyml-studio.');
  const requirements = (before.fixes || []).map((f) => f.requirement);
  const steps = installer.planExternal({ python: py, wheel, requirements }); // valida antes de preguntar
  if (!steps.length) return { report: before, wheel: null };
  const lines = [...(wheel ? ['tinyml-studio (' + (en ? 'package included in this app, without dependencies' : 'paquete incluido en esta app, sin dependencias') + ')'] : []), ...requirements];
  const warn = (before.warnings || []).filter((w) => !/tinyml-studio/.test(w));
  const { response } = await dialog.showMessageBox(win, {
    type: 'question', buttons: [en ? 'Install' : 'Instalar', en ? 'Cancel' : 'Cancelar'], defaultId: 1, cancelId: 1,
    title: 'TinyML Studio',
    message: en ? 'Install the missing pieces in your environment?' : '¿Instalar lo que falta en tu entorno?',
    detail: `${py}\n\n${lines.map((l) => `• ${l}`).join('\n')}\n\n${en ? 'Packages that already satisfy the range are not touched.' : 'Los paquetes que ya cumplen el rango no se tocan.'}${requirements.length ? (en ? ' The listed packages come from pypi.org.' : ' Los paquetes de la lista vienen de pypi.org.') : ''}${warn.length ? `\n\n${warn.join('\n')}` : ''}`,
  });
  if (response !== 0) return { canceled: true };
  await installer.installExternal({ python: py, wheel, requirements });
  const report = await envsettings.runDoctor(BUILTIN, py, comps, lang, envOpts());
  return { report, wheel: report.tinyml_studio ? null : findWheel() };
}));
ipcMain.handle('env:apply', trusted(async () => {
  if (!pendingPython) throw new Error('nada que aplicar');
  settings.python = pendingPython;
  envsettings.saveSettings(SETTINGS_FILE, settings);
  pendingPython = null;
  await restartCore();
  return { core: CORE.kind, error: startError };
}));

// ---- aviso de versión nueva: solo avisa; descargar e instalar lo decide el usuario ----------------------------
const UPDATES_FILE = path.join(app.getPath('userData'), 'updates.json');
const UPDATE_REPO = updates.parseRepo(require('../package.json').homepage);
let lastUpdate = null; // resultado de la última comprobación: la dirección a abrir sale de aquí, nunca del renderer
const updState = () => {
  const p = updates.loadPrefs(UPDATES_FILE);
  return { version: app.getVersion(), auto: p.auto, skipped: p.skipped, repo: UPDATE_REPO };
};
async function runUpdateCheck() {
  // La sesión por defecto bloquea TODO lo que no sea la propia página (security.hardenSession), así que net.fetch
  // fallaba con ERR_BLOCKED_BY_CLIENT. Se usa una sesión aparte, sin cookies y solo en memoria, que no carga ninguna
  // página; la dirección es fija (api.github.com del repositorio) y la fija updates.js, nunca el renderer.
  const updSession = session.fromPartition('tinyml-update-check');
  const r = await updates.checkForUpdate({ repo: UPDATE_REPO, current: app.getVersion(), fetchFn: (...a) => updSession.fetch(...a) });
  const p = updates.loadPrefs(UPDATES_FILE);
  p.lastCheck = Date.now();
  updates.savePrefs(UPDATES_FILE, p);
  lastUpdate = r;
  return r;
}
ipcMain.handle('updates:get', trusted(() => updState()));
ipcMain.handle('updates:setAuto', trusted((_e, on) => {
  const p = updates.loadPrefs(UPDATES_FILE);
  p.auto = Boolean(on);
  updates.savePrefs(UPDATES_FILE, p);
  return updState();
}));
ipcMain.handle('updates:check', trusted(() => runUpdateCheck()));
ipcMain.handle('updates:auto', trusted(async () => {
  if (SMOKE) return null; // las pruebas de ventana no salen a internet
  const p = updates.loadPrefs(UPDATES_FILE);
  if (!updates.dueForAutoCheck(p)) return null;
  const r = await runUpdateCheck();
  return r.status === 'newer' && r.latest.version !== p.skipped ? r : null;
}));
ipcMain.handle('updates:skip', trusted(() => {
  const p = updates.loadPrefs(UPDATES_FILE);
  if (lastUpdate && lastUpdate.latest) p.skipped = lastUpdate.latest.version;
  updates.savePrefs(UPDATES_FILE, p);
  return updState();
}));
ipcMain.handle('updates:open', trusted(async () => {
  const url = lastUpdate && lastUpdate.latest && lastUpdate.latest.url;
  if (!url || !updates.validReleaseUrl(url, UPDATE_REPO)) throw new Error('no hay una versión nueva que abrir');
  if (SMOKE) return { opened: false };
  await shell.openExternal(url);
  return { opened: true };
}));

ipcMain.handle('env:reset', trusted(async () => {
  settings.python = null;
  envsettings.saveSettings(SETTINGS_FILE, settings);
  await restartCore();
  return { core: CORE.kind, error: startError };
}));

// ---- instalador del entorno gestionado: el renderer solo pide ids de componentes del manifiesto ----
installer.on('log', (e) => send('installer:event', { type: 'log', ...e }));
installer.on('step', (e) => send('installer:event', { type: 'step', ...e }));
installer.on('state', (e) => send('installer:event', { type: 'state', ...e }));
ipcMain.handle('installer:status', trusted(() => installer.status()));
ipcMain.handle('installer:start', trusted(async (_e, ids) => {
  if (installer.status().busy) throw new Error('ya hay una instalación en curso');
  installer.plan(ids); // valida (ids del manifiesto, locks, rueda) antes de tocar nada
  if (CORE.kind === 'managed') await sidecar.stop(); // Windows no deja reemplazar archivos en uso
  installer.install(ids).then(
    () => restartCore(),
    () => restartCore(), // si falla o se cancela, vuelve a arrancar lo que hubiera
  );
  return { started: true };
}));
ipcMain.handle('installer:cancel', trusted(() => installer.cancel()));
ipcMain.handle('installer:reset', trusted(async () => {
  await sidecar.stop();
  installer.reset();
  await restartCore();
  return installer.status();
}));

// ---- API: solo rutas de la lista blanca; el token nunca sale de este proceso -------------------
ipcMain.handle(
  'api:request',
  trusted(async (_e, method, route, body) => {
    if (!bridge.routeAllowed(method, route)) throw new Error(`ruta no permitida: ${method} ${route}`);
    if (!bridge.bodyOk(body)) throw new Error('cuerpo demasiado grande o inválido');
    try {
      return await sidecar.request(method, route, body);
    } catch (e) {
      throw new Error(e.status ? `[${e.status}] ${e.message}` : e.message);
    }
  }),
);

// Native dialogs follow the OS language (the renderer has its own EN/ES switch).
const L = (en, es) => (app.getLocale().toLowerCase().startsWith('es') ? es : en);

// ---- Diálogos nativos: aquí se decide qué carpetas puede leer el sidecar -----------------------
async function registerRoots(dirs) {
  for (const d of new Set(dirs)) {
    await sidecar.request('POST', '/roots', { path: d });
    registeredRoots.add(d);
  }
}

ipcMain.handle(
  'dialog:pickDataset',
  trusted(async (_e, mode) => {
    let paths;
    if (SMOKE && process.env.TINYML_SMOKE_PATHS) {
      paths = process.env.TINYML_SMOKE_PATHS.split(path.delimiter);
    } else {
      const r = await dialog.showOpenDialog(win, {
        title: mode === 'folder' ? L('Open CSV folder', 'Abrir carpeta de CSV') : L('Open CSV', 'Abrir CSV'),
        properties: mode === 'folder' ? ['openDirectory'] : ['openFile', 'multiSelections'],
        filters: mode === 'folder' ? [] : [{ name: 'CSV', extensions: ['csv', 'tsv', 'txt'] }],
      });
      if (r.canceled || !r.filePaths.length) return { canceled: true };
      paths = r.filePaths;
    }
    await registerRoots(paths.map((p) => (fs.statSync(p).isDirectory() ? p : path.dirname(p))));
    rememberRecent(paths);
    return { canceled: false, paths };
  }),
);

// ---- últimos CSV abiertos: el renderer solo pide "el número N"; las rutas viven aquí ----------------
const RECENT_FILE = path.join(app.getPath('userData'), 'recent.json');
function rememberRecent(paths) {
  try { recent.save(RECENT_FILE, recent.add(recent.load(RECENT_FILE), paths)); } catch { /* no es esencial */ }
}
ipcMain.handle('dialog:recentList', trusted(() => recent.describe(recent.load(RECENT_FILE))));
ipcMain.handle(
  'dialog:openRecent',
  trusted(async (_e, id) => {
    const paths = recent.resolve(recent.load(RECENT_FILE), id);
    if (!paths) return { canceled: true, missing: true };
    await registerRoots(paths.map((p) => (fs.statSync(p).isDirectory() ? p : path.dirname(p))));
    return { canceled: false, paths };
  }),
);

ipcMain.handle(
  'recipe:load',
  trusted(async () => {
    const r = await dialog.showOpenDialog(win, {
      title: L('Load recipe', 'Cargar receta'),
      properties: ['openFile'],
      filters: [{ name: 'Recipe JSON', extensions: ['json'] }],
    });
    if (r.canceled || !r.filePaths.length) return { canceled: true };
    const file = r.filePaths[0];
    if (fs.statSync(file).size > bridge.MAX_BODY_BYTES) throw new Error('receta demasiado grande');
    const recipe = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (!recipe || typeof recipe !== 'object' || !Array.isArray(recipe.files)) {
      throw new Error('no parece una receta de TinyML Studio (falta "files")');
    }
    await registerRoots(recipe.files.map((f) => path.dirname(String(f))).filter((d) => fs.existsSync(d)));
    return { canceled: false, recipe };
  }),
);

ipcMain.handle(
  'file:saveText',
  trusted(async (_e, suggestedName, text) => {
    if (typeof text !== 'string' || text.length > bridge.MAX_BODY_BYTES) throw new Error('texto inválido');
    const name = String(suggestedName || 'archivo.txt').replace(/[^\w.\-]/g, '_');
    let dest;
    if (SMOKE && process.env.TINYML_SMOKE_SAVE_DIR) {
      dest = path.join(process.env.TINYML_SMOKE_SAVE_DIR, name);
    } else {
      const r = await dialog.showSaveDialog(win, { defaultPath: name });
      if (r.canceled || !r.filePath) return { canceled: true };
      dest = r.filePath;
    }
    fs.writeFileSync(dest, text, 'utf8');
    return { canceled: false, path: dest };
  }),
);

ipcMain.on('app:flags', (e) => { e.returnValue = { smoke: Boolean(SMOKE) }; });

ipcMain.handle(
  'image:save',
  trusted(async (_e, name, fmt, data) => {
    const { ext, buf } = bridge.decodeImage(fmt, data);
    const base = String(name || 'chart').replace(/[^\w\-]/g, '_').slice(0, 60) || 'chart';
    let dest;
    if (SMOKE && process.env.TINYML_SMOKE_SAVE_DIR) {
      fs.mkdirSync(process.env.TINYML_SMOKE_SAVE_DIR, { recursive: true });
      dest = path.join(process.env.TINYML_SMOKE_SAVE_DIR, `${base}.${ext}`);
    } else {
      const exts = ext === 'jpg' || ext === 'jpeg' ? ['jpg', 'jpeg'] : [ext];
      const r = await dialog.showSaveDialog(win, { defaultPath: `${base}.${ext}`, filters: [{ name: ext.toUpperCase(), extensions: exts }] });
      if (r.canceled || !r.filePath) return { canceled: true };
      dest = r.filePath;
    }
    fs.writeFileSync(dest, buf);
    return { canceled: false, path: dest, bytes: buf.length };
  }),
);

// PDF report: the renderer switches to its print layout, then asks the main process to print the page.
ipcMain.handle(
  'report:pdf',
  trusted(async (_e, name) => {
    const base = String(name || 'tinyml-report').replace(/[^\w\-]/g, '_').slice(0, 60) || 'tinyml-report';
    const pdf = await win.webContents.printToPDF({ printBackground: true, pageSize: 'A4', margins: { marginType: 'default' }, preferCSSPageSize: true });
    let dest;
    if (SMOKE && process.env.TINYML_SMOKE_SAVE_DIR) {
      fs.mkdirSync(process.env.TINYML_SMOKE_SAVE_DIR, { recursive: true });
      dest = path.join(process.env.TINYML_SMOKE_SAVE_DIR, `${base}.pdf`);
    } else {
      const r = await dialog.showSaveDialog(win, { defaultPath: `${base}.pdf`, filters: [{ name: 'PDF', extensions: ['pdf'] }] });
      if (r.canceled || !r.filePath) return { canceled: true };
      dest = r.filePath;
    }
    fs.writeFileSync(dest, pdf);
    return { canceled: false, path: dest, bytes: pdf.length };
  }),
);

// ---- Trabajos: progreso por SSE, reenviado a la ventana ----------------------------------------
ipcMain.handle(
  'job:watch',
  trusted(async (_e, id) => {
    if (!bridge.isJobId(id)) throw new Error('id de trabajo inválido');
    if (watchers.has(id)) return { watching: true };
    const s = sidecar.stream(`/jobs/${id}/events`, (ev) => {
      if (win && !win.isDestroyed()) win.webContents.send('job:event', { id, ev });
    });
    watchers.set(id, s);
    s.done.catch(() => {}).finally(() => watchers.delete(id));
    return { watching: true };
  }),
);

ipcMain.handle(
  'artifacts:save',
  trusted(async (_e, jobId) => {
    let destRoot;
    if (SMOKE && process.env.TINYML_SMOKE_SAVE_DIR) {
      destRoot = process.env.TINYML_SMOKE_SAVE_DIR;
    } else {
      const r = await dialog.showOpenDialog(win, {
        title: L('Choose the folder to save the files', 'Elige la carpeta donde guardar los archivos'),
        properties: ['openDirectory', 'createDirectory'],
      });
      if (r.canceled || !r.filePaths.length) return { canceled: true };
      destRoot = r.filePaths[0];
    }
    return { canceled: false, ...bridge.copyJobArtifacts(WORKDIR, jobId, destRoot) };
  }),
);

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 860,
    show: !SMOKE || Boolean(process.env.TINYML_SMOKE_SHOTS), // las capturas necesitan ventana visible (xvfb)
    icon: path.join(__dirname, '..', 'assets', 'icon.png'),
    title: 'TinyML Studio',
    webPreferences: security.buildWebPreferences(path.join(__dirname, 'preload.js')),
  });
  security.hardenWebContents(win.webContents, RENDERER_DIR);
  win.loadFile(path.join(RENDERER_DIR, 'index.html'));
  return win;
}

app.whenReady().then(async () => {
  // Sin la barra File/Edit/View/Window/Help de Electron en la app instalada (no es de TinyML Studio).
  // macOS necesita su menú de aplicación; en desarrollo se deja para poder abrir las herramientas.
  if (app.isPackaged && process.platform !== 'darwin') Menu.setApplicationMenu(null);
  security.hardenSession(session.defaultSession, RENDERER_DIR);
  try {
    fs.mkdirSync(WORKDIR, { recursive: true });
    if (CORE.kind === 'managed') {
      // tras actualizar la app, el entorno existente conserva la rueda anterior de tinyml-studio: se reinstala sola
      try {
        const r = await installer.refreshWheel();
        if (r.refreshed) console.log('[TinyML Studio] rueda de tinyml-studio actualizada en el entorno gestionado');
      } catch (e) { console.error(`[TinyML Studio] no se pudo actualizar la rueda: ${e.message}`); }
    }
    if (CORE.kind !== 'none') await sidecar.start();
  } catch (e) {
    if (['managed', 'external'].includes(CORE.kind)) {
      startError = String(e.message || e); // la interfaz ofrece reparar o elegir otro entorno
    } else {
      dialog.showErrorBox('TinyML Studio', `${L('Could not start the Python core', 'No se pudo iniciar el núcleo Python')}:\n${e.message}`);
      return app.exit(1);
    }
  }
  sidecar.on('failed', (e) =>
    dialog.showErrorBox('TinyML Studio', `${L('The Python core stopped responding', 'El núcleo Python dejó de responder')}:\n${e.message}`),
  );
  createWindow();
  if (SMOKE) require('./smoke').run({ win, sidecar, WORKDIR, app, core: CORE, installer, restartCore });
});

let quitting = false;
app.on('before-quit', (e) => {
  if (quitting) return;
  e.preventDefault();
  quitting = true;
  for (const s of watchers.values()) s.abort();
  sidecar.stop().finally(() => app.quit());
});
app.on('window-all-closed', () => app.quit());
