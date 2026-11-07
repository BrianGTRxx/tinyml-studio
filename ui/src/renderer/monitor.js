'use strict';
// Live hardware panel for training: CPU / RAM / GPU load, temperatures, GPU memory, history chart and a
// final summary. Data comes from GET /jobs/<id>/hw (sampled by the core every second). Honest about
// missing sensors: whatever the OS does not expose is shown as "n/a" with the reason from the core.
(function () {
  const TS = window.TS;
  const { h, charts } = TS;
  const ui = TS.ui;
  const t = TS.t;
  const pct = (v) => (v === null || v === undefined ? 'n/a' : `${Number(v).toFixed(0)} %`);
  const deg = (v) => (v === null || v === undefined ? 'n/a' : `${Number(v).toFixed(0)} °C`);
  const mb = (v) => (v === null || v === undefined ? 'n/a' : v >= 1024 ? `${TS.fmt.dec(v / 1024, 1)} GB` : `${Math.round(v)} MB`);

  function gauge(label, value, text, fraction, hotAt, critAt) {
    const fill = h('div', { class: 'bar-fill' });
    fill.style.width = `${Math.max(0, Math.min(100, fraction ?? 0))}%`;
    const f = fraction ?? 0;
    return h('div', { class: `gauge ${f >= (critAt || 95) ? 'crit' : f >= (hotAt || 80) ? 'hot' : ''}`, 'data-gauge': label },
      h('div', { class: 'gv', text }), h('div', { class: 'gl', text: label }), h('div', { class: 'bar' }, fill));
  }

  TS.hwPanel = function hwPanel() {
    const gauges = h('div', { class: 'gauges' });
    const chartBox = h('div', {});
    const notes = h('div', {});
    const sumBox = h('div', {});
    const title = h('h3', { text: t('Hardware during training', 'Hardware durante el entrenamiento') });
    const el = h('div', { id: 'hw-panel', class: 'sec' }, title, gauges, chartBox, sumBox, notes);
    let timer = null;
    let jobId = null;
    const chip = document.getElementById('hwchip');

    const draw = (d) => {
      const s = d.samples || [];
      const last = s[s.length - 1];
      if (!last) { gauges.replaceChildren(h('p', { class: 'muted', text: t('Waiting for the first sample…', 'Esperando la primera muestra…') })); return; }
      const g0 = (last.gpus || [])[0];
      const tiles = [
        gauge(t('CPU (whole system)', 'CPU (todo el sistema)'), last.cpu_pct, pct(last.cpu_pct), last.cpu_pct),
        gauge(t('CPU (this training)', 'CPU (este entrenamiento)'), last.proc_cpu_pct, pct(last.proc_cpu_pct), last.proc_cpu_pct),
        gauge(t('RAM (system)', 'RAM (sistema)'), last.ram_used_mb, `${mb(last.ram_used_mb)} / ${mb(last.ram_total_mb)}`, last.ram_total_mb ? (100 * last.ram_used_mb) / last.ram_total_mb : 0, 85, 95),
        gauge(t('RAM (this training)', 'RAM (este entrenamiento)'), last.proc_rss_mb, mb(last.proc_rss_mb), last.ram_total_mb && last.proc_rss_mb ? (100 * last.proc_rss_mb) / last.ram_total_mb : 0, 60, 85),
        gauge(last.cpu_temp_source === 'Thermal zone' ? t('CPU temperature (system zone)', 'Temperatura CPU (zona del sistema)') : t('CPU temperature', 'Temperatura CPU'), last.cpu_temp_c, deg(last.cpu_temp_c), last.cpu_temp_c ?? 0, 80, 92),
      ];
      if (g0) {
        tiles.push(gauge(`GPU ${g0.index}`, g0.util_pct, pct(g0.util_pct), g0.util_pct));
        tiles.push(gauge(t('GPU memory', 'Memoria GPU'), g0.mem_used_mb, `${mb(g0.mem_used_mb)} / ${mb(g0.mem_total_mb)}`, g0.mem_total_mb ? (100 * g0.mem_used_mb) / g0.mem_total_mb : 0, 85, 95));
        tiles.push(gauge(t('GPU temperature', 'Temperatura GPU'), g0.temp_c, deg(g0.temp_c), g0.temp_c ?? 0, 78, 88));
      }
      gauges.replaceChildren(...tiles);
      const t0 = s[0].t;
      const series = [{ name: 'CPU %', values: s.map((x) => x.cpu_pct ?? 0) }, { name: t('job CPU %', 'CPU trabajo %'), values: s.map((x) => x.proc_cpu_pct ?? 0) }];
      if (g0) series.push({ name: 'GPU %', values: s.map((x) => ((x.gpus || [])[0] || {}).util_pct ?? 0) });
      const span = Math.round(s[s.length - 1].t - t0);
      chartBox.replaceChildren(ui.chart(t('Usage history', 'Historial de uso'), charts.lines(t('Load (%)', 'Carga (%)'), series, t('time (last {s} s)', 'tiempo (últimos {s} s)', { s: span })), 'hardware-usage'));
      if (d.notes && d.notes.length) {
        const tempHint = d.cpu_temp_c === null || d.cpu_temp_c === undefined ? [TS.callout('info', d.notes[0])] : [];
        notes.replaceChildren(...tempHint, ui.collapse(t('About these sensors', 'Sobre estos sensores'), h('ul', {}, d.notes.map((n) => h('li', { text: n })))));
      }
      if (chip) { chip.hidden = !d.running; chip.textContent = `CPU ${pct(last.cpu_pct)} · RAM ${mb(last.ram_used_mb)}${g0 ? ` · GPU ${pct(g0.util_pct)}` : ''}`; }
      if (!d.running && d.summary) summary(d.summary);
    };

    const summary = (m) => {
      const rows = [
        [t('Duration', 'Duración'), TS.fmt.dur ? TS.fmt.dur(m.duration_s || 0) : m.duration_s, ''],
        [t('CPU whole system', 'CPU todo el sistema'), pct(m.cpu_mean_pct), pct(m.cpu_peak_pct)],
        [t('CPU this training', 'CPU este entrenamiento'), pct(m.proc_cpu_mean_pct), pct(m.proc_cpu_peak_pct)],
        [t('RAM this training', 'RAM este entrenamiento'), '', mb(m.rss_peak_mb)],
        [t('CPU temperature', 'Temperatura CPU'), '', deg(m.cpu_temp_peak_c)],
        ...((m.gpus || []).flatMap((g) => [[`GPU ${g.index} ${g.name || ''}`, pct(g.util_mean_pct), pct(g.util_peak_pct)], [t('GPU memory / temperature', 'Memoria / temperatura GPU'), '', `${mb(g.mem_peak_mb)} · ${deg(g.temp_peak_c)}`]])),
      ];
      sumBox.replaceChildren(ui.collapse(t('Summary (mean / peak)', 'Resumen (media / pico)'), TS.table([t('Metric', 'Métrica'), t('Mean', 'Media'), t('Peak', 'Pico')], rows), true));
    };

    const poll = async () => {
      try {
        const d = await TS.api('GET', `/jobs/${jobId}/hw`);
        draw(d);
        if (!d.running && timer) { clearInterval(timer); timer = null; }
      } catch { /* the job may not have started yet */ }
    };
    return {
      el,
      /** After a language switch: new title and the texts of the core (sensor notes) in the new language. */
      async refresh() { title.textContent = t('Hardware during training', 'Hardware durante el entrenamiento'); if (jobId) await poll(); },
      start(id) { if (timer || !id) return; jobId = id; poll(); timer = setInterval(() => { if (!el.isConnected) { clearInterval(timer); timer = null; return; } poll(); }, 1500); },
      async stop() { if (timer) { clearInterval(timer); timer = null; } if (jobId) await poll(); if (chip) chip.hidden = true; },
    };
  };
})();
