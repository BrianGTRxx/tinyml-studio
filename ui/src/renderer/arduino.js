'use strict';
// "Use it in Arduino / ESP32" panel: steps, input order, ready-to-copy code and honest verification level.
(function () {
  const TS = window.TS;
  const { h, table, callout } = TS;
  const t = TS.t;

  const TITLES = () => ({
    raw_c: [t('Plain C (.h / .c, no library)', 'C puro (.h / .c, sin librería)'), t('Copy the folder into your Arduino sketchbook. Uses the generated tm_model.c/.h directly; no TensorFlow needed.', 'Copia la carpeta a tu carpeta de sketches de Arduino. Usa directamente tm_model.c/.h; no necesita TensorFlow.')],
    eloquent: ['EloquentTinyML / tflm_esp32', t('Uses model.h with the Eloquent library (same pattern as the template sketches).', 'Usa model.h con la librería Eloquent (mismo patrón que los sketches de plantilla).')],
    tflm: [t('TensorFlow Lite Micro (raw)', 'TensorFlow Lite Micro (directo)'), t('Calls the TFLM interpreter yourself; handles int8 input scale/zero-point.', 'Llama tú mismo al intérprete de TFLM; maneja escala/zero-point de entrada int8.')],
  });

  function copyBtn(getText) {
    const b = h('button', { type: 'button', class: 'ghost', text: t('Copy', 'Copiar') });
    b.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(getText()); b.textContent = t('Copied', 'Copiado'); } catch { b.textContent = t('Select and copy manually', 'Selecciona y copia a mano'); }
      setTimeout(() => { b.textContent = t('Copy', 'Copiar'); }, 1500);
    });
    return b;
  }

  function variantCard(v) {
    if (v.error) return callout('info', `${v.error}. `, t('The export folder still contains inference_example.ino (an example, not compiled for the ESP32-S3).', 'La carpeta exportada sí incluye inference_example.ino (un ejemplo, no compilado para el ESP32-S3).'));
    const [title, blurb] = TITLES()[v.variant] || [v.variant, ''];
    const files = Object.keys(v.code || {});
    const ino = files.find((f) => f.endsWith('.ino')) || files[0];
    const pre = h('pre', { class: 'code', text: (v.code || {})[ino] || '' });
    const sel = h('select', { onchange: (e) => { pre.textContent = v.code[e.target.value]; } }, files.map((f) => h('option', { value: f, text: f })));
    sel.value = ino;
    const ok = v.verification && v.verification.level === 'host_compiled';
    return h('div', { class: 'card arduino', 'data-variant': v.variant },
      h('h3', { text: title }), h('p', { class: 'muted', text: blurb }),
      callout(ok ? 'ok' : 'warn', h('b', { text: ok ? t('Compiled and run on the PC with an Arduino shim: ', 'Compilado y ejecutado en el PC con un shim de Arduino: ') : t('Generated, NOT compiled here: ', 'Generado, NO compilado aquí: ') }), v.verification ? v.verification.detail : ''),
      h('ol', { class: 'steps' }, (v.steps || []).map((s) => h('li', { text: s }))),
      v.input_table && v.input_table.length ? h('div', {}, h('b', { text: t('Input order (the order of your sensor values)', 'Orden de entradas (el orden de los valores de tus sensores)') }),
        table(['#', t('feature', 'variable'), t('type', 'tipo')], v.input_table.map((r) => [r.index, r.name + (r.unit ? ` (${r.unit})` : ''), r.array]))) : null,
      (v.warnings || []).length ? h('ul', { class: 'warns' }, v.warnings.map((w) => h('li', { text: w }))) : null,
      h('div', { class: 'row' }, h('span', { class: 'muted nowrap', text: t('Sketch folder (inside what you save):', 'Carpeta del sketch (dentro de lo que guardes):') }), h('code', { text: v.sketch_dir }), files.length > 1 ? sel : null),
      h('div', { class: 'codebox' }, copyBtn(() => pre.textContent), pre));
  }

  TS.arduinoPanel = function arduinoPanel(list, jobId) {
    if (!list || !list.length) return null;
    const msg = h('p', { id: 'arduino-saved', class: 'muted' });
    const sketch = (list.find((v) => v && v.sketch_dir) || {}).sketch_dir || '';
    const save = h('button', { type: 'button', id: 'btn-save-arduino', text: t('Save the Arduino sketch… (you choose where)', 'Guardar el sketch de Arduino… (tú eliges dónde)'), onclick: async () => {
      msg.textContent = '';
      try {
        const s = await window.tinyml.saveArtifacts(jobId);
        if (s.canceled) { msg.textContent = t('Cancelled: Nothing was saved.', 'Cancelado: No se guardó nada.'); return; }
        const sep = s.dest.includes(String.fromCharCode(92)) ? String.fromCharCode(92) : '/';
        msg.textContent = t('Saved. Open this file in the Arduino IDE: ', 'Guardado. Abre este archivo en el IDE de Arduino: ') + [s.dest, ...sketch.split('/').filter(Boolean), `${sketch.split('/').filter(Boolean).pop() || 'sketch'}.ino`].join(sep);
      } catch (e) { msg.textContent = `Error: ${e.message}`; }
    } });
    return h('div', { id: 'arduino-panel' },
      h('h3', { text: t('Use it in Arduino / ESP32', 'Úsalo en Arduino / ESP32') }),
      h('p', { text: t('Nothing is saved on your computer until you press the button below and choose a folder. Then open the .ino in the Arduino IDE (board: ESP32S3 Dev Module).', 'No se guarda nada en tu equipo hasta que pulses el botón de abajo y elijas una carpeta. Luego abre el .ino en el IDE de Arduino (placa: ESP32S3 Dev Module).') }),
      h('div', { class: 'row' }, save), msg,
      ...list.map(variantCard));
  };
})();
