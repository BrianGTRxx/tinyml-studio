'use strict';
// Preload mínimo: solo funciones concretas (nunca ipcRenderer completo ni secretos de sesión).
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('tinyml', {
  flags: ipcRenderer.sendSync('app:flags'),
  health: () => ipcRenderer.invoke('sidecar:health'),
  // El proceso principal solo reenvía rutas de una lista blanca (src/bridge.js).
  request: (method, route, body) => ipcRenderer.invoke('api:request', method, route, body),
  pickDataset: (mode) => ipcRenderer.invoke('dialog:pickDataset', mode),
  recentDatasets: () => ipcRenderer.invoke('dialog:recentList'),
  openRecentDataset: (id) => ipcRenderer.invoke('dialog:openRecent', id),
  loadRecipe: () => ipcRenderer.invoke('recipe:load'),
  saveText: (suggestedName, text) => ipcRenderer.invoke('file:saveText', suggestedName, text),
  watchJob: (id) => ipcRenderer.invoke('job:watch', id),
  onJobEvent: (cb) => {
    const handler = (_e, msg) => cb(msg);
    ipcRenderer.on('job:event', handler);
    return () => ipcRenderer.removeListener('job:event', handler);
  },
  saveImage: (name, fmt, data) => ipcRenderer.invoke('image:save', name, fmt, data),
  printPDF: (name) => ipcRenderer.invoke('report:pdf', name),
  saveArtifacts: (jobId) => ipcRenderer.invoke('artifacts:save', jobId),
  // Entorno Python (ver src/envsettings.js): el renderer nunca fija rutas, solo pide el diálogo nativo.
  envGet: () => ipcRenderer.invoke('env:get'),
  envDoctor: (components, lang) => ipcRenderer.invoke('env:doctor', components, lang),
  envPick: (lang) => ipcRenderer.invoke('env:pick', lang),
  envScan: () => ipcRenderer.invoke('env:scan'),
  envUseScanned: (id, lang) => ipcRenderer.invoke('env:useScanned', id, lang),
  envFixOwn: (lang, components) => ipcRenderer.invoke('env:fixOwn', lang, components),
  envApply: () => ipcRenderer.invoke('env:apply'),
  envReset: () => ipcRenderer.invoke('env:reset'),
  updatesGet: () => ipcRenderer.invoke('updates:get'),
  updatesSetAuto: (on) => ipcRenderer.invoke('updates:setAuto', on),
  updatesCheck: () => ipcRenderer.invoke('updates:check'),
  updatesAuto: () => ipcRenderer.invoke('updates:auto'),
  updatesSkip: () => ipcRenderer.invoke('updates:skip'),
  updatesOpen: () => ipcRenderer.invoke('updates:open'),
  // Instalador del entorno gestionado: solo ids de componentes del manifiesto (nunca paquetes ni rutas).
  installerStatus: () => ipcRenderer.invoke('installer:status'),
  installerStart: (ids) => ipcRenderer.invoke('installer:start', ids),
  installerCancel: () => ipcRenderer.invoke('installer:cancel'),
  installerReset: () => ipcRenderer.invoke('installer:reset'),
  onInstallerEvent: (cb) => {
    const handler = (_e, msg) => cb(msg);
    ipcRenderer.on('installer:event', handler);
    return () => ipcRenderer.removeListener('installer:event', handler);
  },
  onCoreChanged: (cb) => {
    const handler = (_e, msg) => cb(msg);
    ipcRenderer.on('core:changed', handler);
    return () => ipcRenderer.removeListener('core:changed', handler);
  },
});
