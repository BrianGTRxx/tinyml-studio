'use strict';
/**
 * Gestión del sidecar Python (FastAPI en 127.0.0.1).
 *  - Token de sesión efímero (32 bytes aleatorios) pasado por variable de entorno.
 *  - Puerto aleatorio elegido por el SO; se recibe en la línea "ready" por stdout.
 *  - stdin del hijo es un pipe: si este proceso muere (incluso con kill -9) el SO lo
 *    cierra y el sidecar se apaga solo. Además se mata el árbol al salir.
 *  - Reinicio automático con límite si el sidecar muere inesperadamente.
 */
const { spawn, spawnSync } = require('node:child_process');
const { EventEmitter } = require('node:events');
const crypto = require('node:crypto');
const http = require('node:http');
const path = require('node:path');
const readline = require('node:readline');

class Sidecar extends EventEmitter {
  constructor({
    python,
    args = ['-m', 'tinyml_studio.api'],
    cwd,
    env = {},
    allowedRoots = [],
    readyTimeoutMs = 30000,
    requestTimeoutMs = 60000,
    maxRestarts = 3,
  }) {
    super();
    Object.assign(this, { python, args, cwd, extraEnv: env, allowedRoots });
    Object.assign(this, { readyTimeoutMs, requestTimeoutMs, maxRestarts });
    this.proc = null;
    this.port = null;
    this.token = null;
    this.restarts = 0;
    this.stopping = false;
  }

  async start() {
    this.stopping = false;
    this.token = crypto.randomBytes(32).toString('hex');
    const env = {
      ...process.env,
      ...this.extraEnv,
      TINYML_SESSION_TOKEN: this.token,
      TINYML_ALLOWED_ROOTS: this.allowedRoots.join(path.delimiter),
      PYTHONUNBUFFERED: '1',
    };
    const proc = spawn(this.python, this.args, {
      cwd: this.cwd,
      env,
      stdio: ['pipe', 'pipe', 'inherit'],
      windowsHide: true,
    });
    this.proc = proc;
    proc.stdin.on('error', () => {});
    proc.on('exit', (code, signal) => this._onExit(proc, code, signal));

    this.port = await new Promise((resolve, reject) => {
      const rl = readline.createInterface({ input: proc.stdout });
      const timer = setTimeout(() => {
        rl.close();
        this._kill(proc);
        reject(new Error('el sidecar no respondió a tiempo'));
      }, this.readyTimeoutMs);
      const fail = (err) => {
        clearTimeout(timer);
        reject(err);
      };
      proc.once('error', fail);
      proc.once('exit', (code) => fail(new Error(`el sidecar salió antes de iniciar (código ${code})`)));
      rl.once('line', (line) => {
        clearTimeout(timer);
        try {
          const msg = JSON.parse(line);
          if (msg.event !== 'ready' || !Number.isInteger(msg.port)) throw new Error('bad');
          resolve(msg.port);
        } catch {
          reject(new Error('handshake inválido del sidecar'));
        }
        rl.on('line', () => {}); // descarta el resto de stdout
      });
    });
    return this.port;
  }

  request(method, route, body) {
    return new Promise((resolve, reject) => {
      if (!this.port) return reject(new Error('sidecar no iniciado'));
      const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
      const req = http.request(
        {
          host: '127.0.0.1',
          port: this.port,
          path: route,
          method,
          // the GPU probe imports TensorFlow and PyTorch in subprocesses: slow the first time after installing
          timeout: route.startsWith('/hardware/frameworks') ? Math.max(this.requestTimeoutMs, 240000) : this.requestTimeoutMs,
          headers: {
            Authorization: `Bearer ${this.token}`,
            ...(payload && {
              'Content-Type': 'application/json',
              'Content-Length': payload.length,
            }),
          },
        },
        (res) => {
          const chunks = [];
          res.on('data', (c) => chunks.push(c));
          res.on('end', () => {
            let data;
            try {
              data = JSON.parse(Buffer.concat(chunks).toString('utf8'));
            } catch {
              data = null;
            }
            if (res.statusCode >= 200 && res.statusCode < 300) return resolve(data);
            const err = new Error(data?.detail || `HTTP ${res.statusCode}`);
            err.status = res.statusCode;
            reject(err);
          });
        },
      );
      req.on('timeout', () => req.destroy(new Error('timeout')));
      req.on('error', reject);
      if (payload) req.write(payload);
      req.end();
    });
  }

  health() {
    return this.request('GET', '/health');
  }

  /**
   * Suscripción SSE (Server-Sent Events). Llama a onEvent(obj) por cada evento `data:` y
   * resuelve cuando el servidor cierra o se aborta. Devuelve { abort, done }.
   */
  stream(route, onEvent, after = 0) {
    let req = null;
    const done = new Promise((resolve, reject) => {
      if (!this.port) return reject(new Error('sidecar no iniciado'));
      req = http.request(
        {
          host: '127.0.0.1',
          port: this.port,
          path: `${route}?after=${after}`,
          method: 'GET',
          headers: { Authorization: `Bearer ${this.token}`, Accept: 'text/event-stream' },
        },
        (res) => {
          if (res.statusCode !== 200) {
            res.resume();
            return reject(new Error(`HTTP ${res.statusCode}`));
          }
          res.setEncoding('utf8');
          let buf = '';
          res.on('data', (chunk) => {
            buf += chunk;
            let i;
            while ((i = buf.indexOf('\n\n')) >= 0) {
              const block = buf.slice(0, i);
              buf = buf.slice(i + 2);
              const data = block
                .split('\n')
                .filter((l) => l.startsWith('data: '))
                .map((l) => l.slice(6))
                .join('\n');
              if (!data) continue;
              try {
                onEvent(JSON.parse(data));
              } catch {
                /* línea mal formada: se ignora */
              }
            }
          });
          res.on('end', resolve);
          res.on('error', reject);
        },
      );
      req.on('error', (e) => (req.destroyed ? resolve() : reject(e)));
      req.end();
    });
    return { abort: () => req && req.destroy(), done };
  }

  async _onExit(proc, code, signal) {
    if (proc !== this.proc) return;
    this.port = null;
    if (this.stopping) return;
    this.emit('crashed', { code, signal });
    if (this.restarts >= this.maxRestarts) {
      this.emit('failed', new Error('el sidecar superó el máximo de reinicios'));
      return;
    }
    this.restarts += 1;
    await new Promise((r) => setTimeout(r, 500 * this.restarts));
    if (this.stopping) return;
    try {
      await this.start();
      this.emit('restarted', { attempt: this.restarts });
    } catch (err) {
      this.emit('failed', err);
    }
  }

  _kill(proc) {
    if (!proc || proc.exitCode !== null || proc.pid === undefined) return;
    if (process.platform === 'win32') {
      // /T mata el árbol (p. ej. el bootloader de PyInstaller + el Python real)
      spawnSync('taskkill', ['/pid', String(proc.pid), '/T', '/F'], { windowsHide: true });
    } else {
      proc.kill('SIGKILL');
    }
  }

  /** Apagado ordenado: cierra stdin (el sidecar sale solo) y, si no, mata el proceso. */
  async stop(graceMs = 3000) {
    this.stopping = true;
    const proc = this.proc;
    if (!proc || proc.exitCode !== null) return;
    const exited = new Promise((r) => proc.once('exit', r));
    try {
      proc.stdin.end();
    } catch {
      /* ya cerrado */
    }
    const timer = setTimeout(() => this._kill(proc), graceMs);
    await exited;
    clearTimeout(timer);
  }
}

module.exports = { Sidecar };
