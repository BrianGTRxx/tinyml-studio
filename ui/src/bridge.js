'use strict';
/**
 * Puente renderer -> sidecar. El renderer NUNCA habla con el sidecar ni conoce el token: pide al
 * proceso principal, que solo reenvía las rutas de esta lista blanca (método + ruta exacta).
 */
const fs = require('node:fs');
const path = require('node:path');

const JOB = '[0-9a-f]{12}';
const RULES = [
  ['GET', /^\/health$/],
  ['GET', /^\/hardware$/],
  ['GET', /^\/hardware\/frameworks$/],
  ['GET', /^\/hardware\/live$/],
  ['GET', /^\/boards$/],
  ['GET', /^\/models$/],
  ['POST', /^\/inspect$/],
  ['POST', /^\/suggest-cleaning$/],
  ['GET', /^\/jobs$/],
  ['POST', /^\/jobs$/],
  ['GET', new RegExp(`^/jobs/${JOB}$`)],
  ['DELETE', new RegExp(`^/jobs/${JOB}$`)],
  ['GET', new RegExp(`^/jobs/${JOB}/(result|files|hw)$`)],
  ['GET', new RegExp(`^/jobs/${JOB}/files/[A-Za-z0-9_.\\-/]+$`)],
  ['POST', new RegExp(`^/jobs/${JOB}/cancel$`)],
  ['GET', /^\/history$/],
  ['POST', /^\/history\/compare$/],
  ['POST', /^\/recipe\/transfer$/],
  ['GET', new RegExp(`^/jobs/${JOB}/params$`)],
];
const MAX_BODY_BYTES = 8 * 1024 * 1024;

function routeAllowed(method, route) {
  if (typeof method !== 'string' || typeof route !== 'string') return false;
  // the only query string allowed is ?lang=en|es (language of the messages)
  route = route.replace(/\?lang=(en|es)$/, '');
  if (route.includes('..') || route.includes('%') || route.includes('?') || route.includes('#')) {
    return false;
  }
  return RULES.some(([m, re]) => m === method && re.test(route));
}

function bodyOk(body) {
  if (body === undefined || body === null) return true;
  try {
    return Buffer.byteLength(JSON.stringify(body)) <= MAX_BODY_BYTES;
  } catch {
    return false;
  }
}

const isJobId = (id) => typeof id === 'string' && new RegExp(`^${JOB}$`).test(id);

/** Copia la carpeta de salida de un trabajo a `destRoot/<nombre>`; nunca fuera de `workdir`. */
function copyJobArtifacts(workdir, jobId, destRoot) {
  if (!isJobId(jobId)) throw new Error('id de trabajo inválido');
  const jobDir = path.join(workdir, 'jobs', jobId);
  const src = path.join(jobDir, 'out');
  if (!fs.existsSync(src)) throw new Error('el trabajo no tiene archivos de salida');
  let kind = 'job';
  try {
    kind = JSON.parse(fs.readFileSync(path.join(jobDir, 'job.json'), 'utf8')).kind || 'job';
  } catch {
    /* sin job.json */
  }
  const safeKind = String(kind).replace(/[^a-z0-9_]/gi, '');
  const dest = path.join(destRoot, `tinyml-${safeKind}-${jobId}`);
  fs.cpSync(src, dest, { recursive: true, errorOnExist: false, force: true });
  let files = 0;
  const walk = (d) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      e.isDirectory() ? walk(path.join(d, e.name)) : (files += 1);
    }
  };
  walk(dest);
  return { dest, files };
}

/**
 * Validates and decodes an image sent by the renderer. fmt: png | jpg | jpeg | svg.
 * Raster images arrive as a data URL; the magic bytes are checked so a renderer bug cannot make the main
 * process write arbitrary content under an image name. Returns { ext, buf }.
 */
function decodeImage(fmt, data) {
  if (!['png', 'jpg', 'jpeg', 'svg'].includes(fmt)) throw new Error('invalid image format');
  if (typeof data !== 'string' || data.length > 40 * 1024 * 1024) throw new Error('invalid image data');
  if (fmt === 'svg') {
    if (!/^<svg[\s>]/.test(data.trimStart()) && !/^<\?xml/.test(data.trimStart())) throw new Error('invalid svg');
    if (/<script|javascript:|on\w+\s*=/i.test(data)) throw new Error('svg with active content');
    return { ext: 'svg', buf: Buffer.from(data, 'utf8') };
  }
  const m = /^data:image\/(png|jpeg);base64,([A-Za-z0-9+/=]+)$/.exec(data);
  if (!m) throw new Error('invalid image data');
  const buf = Buffer.from(m[2], 'base64');
  const png = buf.length > 8 && buf.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  const jpg = buf.length > 3 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff;
  if ((fmt === 'png' && !png) || (fmt !== 'png' && !jpg)) throw new Error('image does not match its format');
  return { ext: fmt === 'png' ? 'png' : fmt, buf };
}

module.exports = { routeAllowed, bodyOk, isJobId, copyJobArtifacts, decodeImage, MAX_BODY_BYTES };
