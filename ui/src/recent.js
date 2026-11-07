'use strict';
/**
 * Últimos conjuntos de CSV abiertos, para volver a abrirlos con un clic.
 *
 * Solo el proceso principal conoce las rutas completas: el renderer pide "reabrir el número N" y el principal
 * comprueba que los archivos siguen existiendo y vuelve a registrar sus carpetas. El renderer nunca elige rutas.
 * Sin dependencias de Electron para poder probarlo con node puro.
 */
const fs = require('node:fs');
const path = require('node:path');

const MAX = 5;

function load(file, fsMod = fs) {
  try {
    const raw = JSON.parse(fsMod.readFileSync(file, 'utf8'));
    return Array.isArray(raw) ? raw.filter((e) => Array.isArray(e) && e.length && e.every((p) => typeof p === 'string')).slice(0, MAX) : [];
  } catch {
    return [];
  }
}

function save(file, list, fsMod = fs) {
  fsMod.mkdirSync(path.dirname(file), { recursive: true });
  fsMod.writeFileSync(file, JSON.stringify(list.slice(0, MAX)), 'utf8');
}

/** Pone este conjunto el primero (sin duplicados, máximo MAX). */
function add(list, paths) {
  const key = paths.join('|');
  return [paths, ...list.filter((e) => e.join('|') !== key)].slice(0, MAX);
}

/** Texto corto para el botón: los nombres de los archivos (o de la carpeta). */
function label(paths) {
  const names = paths.map((p) => path.basename(p));
  return names.length > 2 ? `${names.slice(0, 2).join(', ')} +${names.length - 2}` : names.join(', ');
}

function describe(list, fsMod = fs) {
  return list.map((paths, id) => ({ id, label: label(paths), exists: paths.every((p) => fsMod.existsSync(p)) }));
}

/** Rutas del conjunto N si el índice es válido y todo sigue existiendo; si no, null. */
function resolve(list, id, fsMod = fs) {
  if (!Number.isInteger(id) || id < 0 || id >= list.length) return null;
  const paths = list[id];
  return paths.every((p) => fsMod.existsSync(p)) ? paths : null;
}

module.exports = { MAX, load, save, add, label, describe, resolve };
