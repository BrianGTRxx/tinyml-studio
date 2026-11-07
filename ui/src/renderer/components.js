'use strict';
// Small UI kit: section headers, cards, stat tiles, collapsibles, searchable/paged tables, chart frames,
// human labels for the technical tokens (kinds, roles, cleaning steps). Everything is built with
// createElement/textContent (CSP-safe: only DOM APIs, no inline styles).
(function () {
  const TS = window.TS;
  const { h } = TS;
  const t = TS.t;

  const ui = (TS.ui = {});

  // ---- human labels (functions: they follow the language) ---------------------------------------
  ui.kindLabel = (k) => ({
    numeric_continuous: t('Number (continuous)', 'Número (continuo)'), numeric_discrete: t('Number (discrete)', 'Número (discreto)'),
    binary: t('Yes/No', 'Sí/No'), categorical: t('Category', 'Categoría'), text: t('Free text', 'Texto libre'),
    datetime: t('Date/time', 'Fecha/hora'), id: t('Identifier', 'Identificador'), constant: t('Constant', 'Constante'), empty: t('Empty', 'Vacía'),
  }[k] || k);
  ui.roleLabel = (r) => ({
    feature: t('Independent · input (X)', 'Independiente · entrada (X)'), target: t('Dependent · target (y)', 'Dependiente · objetivo (y)'), group: t('Group', 'Grupo'),
    time: t('Time', 'Tiempo'), meta: t('Metadata (unused)', 'Metadato (no se usa)'), drop: t('Discard', 'Descartar'),
  }[r] || r);
  ui.stepLabel = (k) => ({
    absurd_to_nan: t('Absurd values → missing', 'Valores absurdos → faltantes'), impute: t('Fill missing values', 'Rellenar faltantes'),
    scale: t('Scale', 'Escalar'), clip: t('Clip outliers', 'Recortar atípicos'), replace_sentinel: t('Replace sentinel values', 'Reemplazar valores centinela'),
    drop_duplicates: t('Remove duplicates', 'Quitar duplicados'), drop_rows: t('Drop rows', 'Quitar filas'), drop_columns: t('Drop columns', 'Quitar columnas'),
    range_nan: t('Out-of-range → missing', 'Fuera de rango → faltantes'),
  }[k] || k);
  const SEV = () => ({ high: t('High', 'Alta'), medium: t('Medium', 'Media'), info: t('Info', 'Info') });
  ui.sevLabel = (s) => SEV()[s] || s;

  // ---- layout blocks --------------------------------------------------------------------------
  ui.page = (title, subtitle, ...kids) => h('div', { class: 'page' }, ui.pageHead(title, subtitle), ...kids);

  ui.pageHead = (title, subtitle) => h('header', { class: 'page-head' }, h('div', { class: 'head-row' }, h('h2', { text: title }), h('span', { class: 'spacer' }), TS.stepHelpButton && TS.S && !TS.S.tool ? TS.stepHelpButton(TS.S.step) : null), subtitle ? h('p', { class: 'lead', text: subtitle }) : null);

  ui.section = (title, ...kids) => h('section', { class: 'sec' }, title ? h('h3', { text: title }) : null, ...kids);

  ui.card = (...kids) => h('div', { class: 'card' }, ...kids);

  ui.stat = (value, label, tone) => h('div', { class: `stat ${tone || ''}` }, h('b', { text: value }), h('span', { text: label }));
  ui.stats = (...tiles) => h('div', { class: 'stats' }, ...tiles.filter(Boolean));

  // collapsible block (native <details>: keyboard accessible, no JS state)
  ui.collapse = (summary, body, open, badge) => {
    const d = h('details', { class: 'collapse' }, h('summary', {}, h('span', { text: summary }), badge !== undefined && badge !== null ? h('span', { class: 'count', text: badge }) : null), h('div', { class: 'collapse-body' }, body));
    if (open) d.open = true;
    return d;
  };

  ui.empty = (icon, title, text) => h('div', { class: 'empty' }, h('div', { class: 'empty-ic' }, TS.icon ? TS.icon(icon, 34) : null), h('b', { text: title }), text ? h('p', { class: 'muted', text }) : null);

  // little "?" that explains a concept on hover/focus (title attribute: no extra DOM, no animation)
  ui.help = (text) => h('span', { class: 'help', tabindex: 0, role: 'img', 'aria-label': text, title: text, text: '?' });

  ui.field = (label, control, hint, help) => h('label', { class: 'field' },
    h('span', { class: 'flabel' }, label, help ? ui.help(help) : null), control, hint ? h('small', { class: 'muted', text: hint }) : null);

  // ---- searchable + paged table (keeps the DOM small whatever the data size) --------------------
  // headers: [string]; rows: [[cell]] where a cell is a string/number/Node; search runs on the text of
  // `rowText(row)` (default: join of string cells).
  ui.pagedTable = (headers, rows, opts = {}) => {
    const size = opts.pageSize || 10;
    // opts.key: the page survives a re-render of the view (e.g. changing a role re-draws the step)
    const mem = (ui._pageMem = ui._pageMem || {});
    let page = opts.key && mem[opts.key] ? mem[opts.key] : 0;
    let q = '';
    const text = (r) => (opts.rowText ? opts.rowText(r) : r.map((c) => (c && c.nodeType ? c.textContent : String(c))).join(' ')).toLowerCase();
    const index = rows.map((r) => text(r));
    const body = h('tbody', {});
    const info = h('span', { class: 'muted' });
    const prev = h('button', { type: 'button', class: 'ghost sm', text: '‹', 'aria-label': t('Previous page', 'Página anterior'), onclick: () => { page -= 1; draw(); } });
    const next = h('button', { type: 'button', class: 'ghost sm', text: '›', 'aria-label': t('Next page', 'Página siguiente'), onclick: () => { page += 1; draw(); } });
    const draw = () => {
      const idx = rows.map((_, i) => i).filter((i) => !q || index[i].includes(q));
      const pages = Math.max(1, Math.ceil(idx.length / size));
      page = Math.min(Math.max(0, page), pages - 1);
      if (opts.key) mem[opts.key] = page;
      body.replaceChildren(...(idx.length ? idx.slice(page * size, page * size + size).map((i) => h('tr', {}, rows[i].map((c) => h('td', {}, c)))) : [h('tr', {}, h('td', { colspan: headers.length, class: 'muted', text: t('No matches', 'Sin coincidencias') }))]));
      info.textContent = t('{a}–{b} of {n}', '{a}–{b} de {n}', { a: idx.length ? page * size + 1 : 0, b: Math.min(idx.length, (page + 1) * size), n: idx.length });
      prev.disabled = page === 0;
      next.disabled = page >= pages - 1;
      pager.hidden = idx.length <= size;
    };
    const pager = h('div', { class: 'pager' }, info, h('span', { class: 'spacer' }), prev, next);
    const search = rows.length > size ? h('input', { type: 'search', class: 'search', placeholder: t('Search…', 'Buscar…'), 'aria-label': t('Search', 'Buscar'), oninput: (e) => { q = e.target.value.trim().toLowerCase(); page = 0; draw(); } }) : null;
    draw();
    return h('div', { class: 'ptable' }, search,
      h('div', { class: 'scroll' }, h('table', { class: 'tbl' }, h('thead', {}, h('tr', {}, headers.map((x) => h('th', { text: x })))), body)), pager);
  };

  // chart frame: title + "export" menu (PNG / JPG / SVG). `el` must be an <svg>.
  ui.chart = (title, el, name) => {
    const bar = h('div', { class: 'chart-bar' }, h('b', { text: title || '' }), h('span', { class: 'spacer' }), TS.exportMenu ? TS.exportMenu(el, name || title || 'chart') : null);
    return h('figure', { class: 'chartbox' }, bar, el);
  };

  // plain stacked form row
  ui.row = (...kids) => h('div', { class: 'row wrap' }, ...kids);

  // ---- multi-select dropdown --------------------------------------------------------------------
  // opts: {options:[{value,label,group?,note?,badges?:[string]}], selected:[values], onChange(values),
  //        presets?:[{label, values}], placeholder, id, searchable}
  // A field-like button that opens a panel with search, quick presets and grouped checkboxes.
  ui.multiSelect = (opts) => {
    let sel = new Set(opts.selected || []);
    let q = '';
    const root = h('div', { class: 'ms', id: opts.id || null });
    const btn = h('button', { type: 'button', class: 'ms-btn', 'aria-haspopup': 'listbox', 'aria-expanded': 'false' });
    const pop = h('div', { class: 'ms-pop', role: 'listbox', 'aria-multiselectable': 'true', hidden: true });
    const summary = () => {
      const labels = opts.options.filter((o) => sel.has(o.value)).map((o) => o.label);
      btn.replaceChildren(
        h('span', { class: 'ms-text', text: labels.length ? `${labels.slice(0, 3).join(', ')}${labels.length > 3 ? ` +${labels.length - 3}` : ''}` : (opts.placeholder || t('Select…', 'Elegir…')) }),
        h('span', { class: 'ms-count', text: `${labels.length}/${opts.options.length}` }), TS.icon ? TS.icon('chevron', 16) : null);
    };
    const emit = () => { summary(); opts.onChange([...sel]); };
    const list = h('div', { class: 'ms-list' });
    const draw = () => {
      const groups = [];
      for (const o of opts.options) {
        if (q && !`${o.label} ${o.group || ''} ${o.note || ''}`.toLowerCase().includes(q)) continue;
        const g = o.group || '';
        let bucket = groups.find((x) => x.g === g);
        if (!bucket) { bucket = { g, items: [] }; groups.push(bucket); }
        bucket.items.push(o);
      }
      list.replaceChildren(...groups.flatMap((b) => [
        b.g ? h('div', { class: 'ms-group', text: b.g }) : null,
        ...b.items.map((o) => h('label', { class: 'ms-item', 'data-value': o.value, title: o.note || '' },
          h('input', { type: 'checkbox', checked: sel.has(o.value), onchange: (e) => { if (e.target.checked) sel.add(o.value); else sel.delete(o.value); emit(); } }),
          h('span', { class: 'ms-label' }, h('b', { text: o.label }), o.note ? h('small', { class: 'muted', text: o.note }) : null),
          ...(o.badges || []).map((x) => h('span', { class: 'tag', text: x })))),
      ]).filter(Boolean));
      if (!groups.length) list.replaceChildren(h('p', { class: 'muted', text: t('No matches', 'Sin coincidencias') }));
    };
    const presets = (opts.presets || []).map((p) => h('button', { type: 'button', class: 'ghost sm', text: p.label, onclick: () => { sel = new Set(p.values); draw(); emit(); } }));
    const search = opts.options.length > 8 ? h('input', { type: 'search', class: 'search', placeholder: t('Search…', 'Buscar…'), oninput: (e) => { q = e.target.value.trim().toLowerCase(); draw(); } }) : null;
    pop.append(search, presets.length ? h('div', { class: 'row wrap', 'data-tight': '1' }, presets) : null, list);
    const close = () => { pop.hidden = true; btn.setAttribute('aria-expanded', 'false'); document.removeEventListener('click', away, true); document.removeEventListener('keydown', esc, true); };
    const away = (e) => { if (!root.contains(e.target)) close(); };
    const esc = (e) => { if (e.key === 'Escape') { close(); btn.focus(); } };
    btn.addEventListener('click', () => {
      if (!pop.hidden) return close();
      pop.hidden = false; btn.setAttribute('aria-expanded', 'true'); draw();
      document.addEventListener('click', away, true); document.addEventListener('keydown', esc, true);
    });
    summary(); draw();
    root.append(btn, pop);
    return root;
  };
})();
