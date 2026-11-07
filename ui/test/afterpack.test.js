'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const afterPack = require('../scripts/after-pack');

function setup() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tms-launch-'));
  fs.writeFileSync(path.join(dir, 'app.bin'), '#!/bin/sh\necho "ARGS:$*"\necho "LAUNCHER:$TINYML_LAUNCHER"\n', { mode: 0o755 });
  fs.writeFileSync(path.join(dir, 'app'), afterPack.launcherScript('app.bin'), { mode: 0o755 });
  return dir;
}
const run = (dir, args = [], env = {}) => spawnSync(path.join(dir, 'app'), args, { encoding: 'utf8', env: { PATH: process.env.PATH, ...env } });

test('launcher script is valid sh', { skip: process.platform === 'win32' }, () => {
  const dir = setup();
  assert.strictEqual(spawnSync('sh', ['-n', path.join(dir, 'app')]).status, 0);
});

test('launcher: --no-sandbox is forced by env, respected from the user, and refused by TINYML_FORCE_SANDBOX', { skip: process.platform === 'win32' }, () => {
  const dir = setup();
  const forced = run(dir, ['--x'], { TINYML_NO_SANDBOX: '1' });
  assert.match(forced.stdout, /ARGS:--no-sandbox --x/);
  assert.match(forced.stderr, /AVISO: sandbox de Chromium desactivado \(TINYML_NO_SANDBOX=1\)/);
  assert.match(forced.stdout, new RegExp(`LAUNCHER:${dir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}/app`));
  const user = run(dir, ['--no-sandbox', '--y'], { TINYML_NO_SANDBOX: '1' });
  assert.match(user.stdout, /ARGS:--no-sandbox --y/);
  assert.doesNotMatch(user.stderr, /AVISO/);
  const strict = run(dir, ['--z'], { TINYML_NO_SANDBOX: '1', TINYML_FORCE_SANDBOX: '1' });
  assert.match(strict.stdout, /ARGS:--z\n/);
});

test('afterPack renames the binary and installs the launcher only on linux', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tms-ap-'));
  fs.writeFileSync(path.join(dir, 'tinyml-studio'), 'ELF', { mode: 0o755 });
  await afterPack({ electronPlatformName: 'win32', appOutDir: dir, packager: { executableName: 'tinyml-studio' } });
  assert.strictEqual(fs.readFileSync(path.join(dir, 'tinyml-studio'), 'utf8'), 'ELF');
  await afterPack({ electronPlatformName: 'linux', appOutDir: dir, packager: { executableName: 'tinyml-studio' } });
  assert.strictEqual(fs.readFileSync(path.join(dir, 'tinyml-studio.bin'), 'utf8'), 'ELF');
  assert.match(fs.readFileSync(path.join(dir, 'tinyml-studio'), 'utf8'), /^#!\/bin\/sh/);
  if (process.platform !== 'win32') assert.ok(fs.statSync(path.join(dir, 'tinyml-studio')).mode & 0o111); // NTFS no guarda el bit de ejecución
});
