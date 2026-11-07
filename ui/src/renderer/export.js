'use strict';
// Chart export (PNG / JPG / SVG). The SVG is cloned and every visual property is copied from the
// computed style into presentation attributes, so the file looks the same outside the app. The
// export is always drawn with the LIGHT theme on a white background (documents, slides, papers).
(function () {
  const TS = window.TS;
  const { h } = TS;
  const t = TS.t;
  const PROPS = ['fill', 'stroke', 'stroke-width', 'stroke-dasharray', 'opacity', 'fill-opacity', 'font-size', 'font-family', 'font-weight'];
  const SVGNS = 'http://www.w3.org/2000/svg';

  function withLightTheme(fn) {
    const root = document.documentElement;
    const prev = root.getAttribute('data-theme');
    root.setAttribute('data-theme', 'light');
    try { return fn(); } finally { if (prev) root.setAttribute('data-theme', prev); else root.removeAttribute('data-theme'); }
  }

  function inline(src, dst) {
    const cs = window.getComputedStyle(src);
    for (const p of PROPS) {
      const v = cs.getPropertyValue(p);
      if (v && v !== 'none' && v !== 'normal') dst.setAttribute(p, v);
      else if (p === 'fill' && v === 'none') dst.setAttribute('fill', 'none');
    }
    const a = src.children;
    const b = dst.children;
    for (let i = 0; i < a.length; i += 1) inline(a[i], b[i]);
  }

  // returns {svg: string, width, height}
  TS.chartToSvg = function chartToSvg(el, title) {
    return withLightTheme(() => {
      const vb = (el.getAttribute('viewBox') || '').split(/\s+/).map(Number);
      const rect = el.getBoundingClientRect();
      const w = vb.length === 4 ? vb[2] : rect.width;
      const hh = vb.length === 4 ? vb[3] : rect.height;
      const clone = el.cloneNode(true);
      inline(el, clone);
      clone.setAttribute('xmlns', SVGNS);
      clone.setAttribute('width', String(Math.round(w)));
      clone.setAttribute('height', String(Math.round(hh)));
      if (vb.length === 4) clone.setAttribute('viewBox', `${vb[0]} ${vb[1]} ${vb[2]} ${vb[3]}`);
      const bg = document.createElementNS(SVGNS, 'rect');
      bg.setAttribute('x', String(vb.length === 4 ? vb[0] : 0));
      bg.setAttribute('y', String(vb.length === 4 ? vb[1] : 0));
      bg.setAttribute('width', String(w));
      bg.setAttribute('height', String(hh));
      bg.setAttribute('fill', '#ffffff');
      clone.insertBefore(bg, clone.firstChild);
      if (title) {
        const tt = document.createElementNS(SVGNS, 'title');
        tt.textContent = title;
        clone.insertBefore(tt, clone.firstChild);
      }
      return { svg: new XMLSerializer().serializeToString(clone), width: w, height: hh };
    });
  };

  function toRaster(svg, w, hh, mime, scale) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = Math.round(w * scale);
        c.height = Math.round(hh * scale);
        const ctx = c.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, c.width, c.height);
        ctx.drawImage(img, 0, 0, c.width, c.height);
        resolve(c.toDataURL(mime, 0.95));
      };
      img.onerror = () => reject(new Error('render'));
      img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
    });
  }

  const slug = (s) => String(s || 'chart').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60) || 'chart';

  // fmt: 'png' | 'jpg' | 'svg'; returns the dialog result {canceled, path}
  TS.exportChart = async function exportChart(el, name, fmt) {
    const { svg, width, height } = TS.chartToSvg(el, name);
    let data;
    if (fmt === 'svg') data = svg;
    else data = await toRaster(svg, width, height, fmt === 'png' ? 'image/png' : 'image/jpeg', 2);
    return window.tinyml.saveImage(slug(name), fmt, data);
  };

  // raster data URL for the PDF report (PNG, light theme)
  TS.chartToPng = async function chartToPng(el, scale = 2) {
    const { svg, width, height } = TS.chartToSvg(el);
    return toRaster(svg, width, height, 'image/png', scale);
  };

  // dropdown with the export formats; `<details>` keeps it keyboard accessible and animation free
  TS.exportMenu = function exportMenu(el, name) {
    const d = h('details', { class: 'menu' }, h('summary', { class: 'ghost sm', title: t('Export this chart as an image', 'Exportar esta gráfica como imagen'), 'aria-label': t('Export chart', 'Exportar gráfica') }, TS.icon('download', 15)));
    const msg = h('small', { class: 'muted' });
    const item = (fmt, label) => h('button', { type: 'button', 'data-fmt': fmt, class: 'ghost sm', text: label, onclick: async () => {
      try {
        const r = await TS.exportChart(el, name, fmt);
        msg.textContent = r && !r.canceled ? t('Saved', 'Guardado') : '';
      } catch (e) { msg.textContent = t('Could not export', 'No se pudo exportar'); }
      d.open = false;
    } });
    d.append(h('div', { class: 'menu-pop' }, item('png', 'PNG'), item('jpg', 'JPG / JPEG'), item('svg', t('SVG (vector)', 'SVG (vectorial)')), msg));
    return d;
  };
})();
