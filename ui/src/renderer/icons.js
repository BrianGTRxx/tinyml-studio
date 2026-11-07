'use strict';
// Monochrome line icons (24x24, stroke = currentColor, no fill): one visual language, black in the light
// theme and light in the dark theme. Own geometry, no icon font and no network.
(function () {
  const TS = window.TS;
  const P = {
    history: ['M3 12a9 9 0 1 0 2.6-6.4', 'M3 4v5h5', 'M12 7.5V12l3 2'],
    file: ['M14 3H7.5A1.5 1.5 0 0 0 6 4.5v15A1.5 1.5 0 0 0 7.5 21h9a1.5 1.5 0 0 0 1.5-1.5V7z', 'M14 3v4h4', 'M9 12.5h6', 'M9 16h6'],
    compass: ['M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z', 'M15.5 8.5l-1.8 4.9-4.9 1.8 1.8-4.9z'],
    folder: ['M3.5 7.5A1.5 1.5 0 0 1 5 6h4l2 2h8a1.5 1.5 0 0 1 1.5 1.5v8A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5z'],
    upload: ['M12 15V4', 'M7.5 8.5L12 4l4.5 4.5', 'M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15'],
    download: ['M12 4v11', 'M7.5 11.5L12 16l4.5-4.5', 'M4 19.5h16'],
    check: ['M5 12.5l4.5 4.5L19 7.5'],
    chevron: ['M6 9l6 6 6-6'],
    search: ['M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13z', 'M15.5 15.5L20 20'],
    help: ['M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z', 'M9.6 9.4a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1.1 1-1.1 1.8', 'M12 16.8h.01'],
    cpu: ['M7 7h10v10H7z', 'M10 10h4v4h-4z', 'M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3'],
    chip: ['M6 6h12v12H6z', 'M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3', 'M9.5 12h5'],
    brain: ['M8.5 4.5a3 3 0 0 0-3 3v.4A3 3 0 0 0 4 10.5c0 1 .5 1.8 1.2 2.3A3 3 0 0 0 6 17a3 3 0 0 0 5 1.4V5.6A3 3 0 0 0 8.5 4.5z', 'M15.5 4.5a3 3 0 0 1 3 3v.4a3 3 0 0 1 1.5 2.6c0 1-.5 1.8-1.2 2.3A3 3 0 0 1 18 17a3 3 0 0 1-5 1.4V5.6a3 3 0 0 1 2.5-1.1z'],
    chart: ['M4 20V4', 'M4 20h16', 'M8 16v-5', 'M12 16V8', 'M16 16v-3'],
    sun: ['M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z', 'M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4'],
    moon: ['M20 14.2A8 8 0 1 1 9.8 4 6.5 6.5 0 0 0 20 14.2z'],
    auto: ['M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z', 'M12 3v18', 'M12 3a9 9 0 0 1 0 18z'],
    sparkle: ['M12 4l1.6 4.4L18 10l-4.4 1.6L12 16l-1.6-4.4L6 10l4.4-1.6z', 'M18.5 16.5l.6 1.4 1.4.6-1.4.6-.6 1.4-.6-1.4-1.4-.6 1.4-.6z'],
    warn: ['M12 4l9 15.5H3z', 'M12 10v4.5', 'M12 17.2h.01'],
    play: ['M8 5.5v13l10-6.5z'],
    save: ['M5 4h11l3 3v12.5a.5.5 0 0 1-.5.5h-13a.5.5 0 0 1-.5-.5z', 'M8 4v5h7V4', 'M8 20v-6h8v6'],
    tune: ['M5 6h14M5 12h14M5 18h14', 'M9 4v4M15 10v4M8 16v4'],
    layers: ['M12 4l9 4.5-9 4.5-9-4.5z', 'M3 12.5L12 17l9-4.5', 'M3 16.5L12 21l9-4.5'],
    x: ['M6 6l12 12M18 6L6 18'],
  };
  TS.icon = function icon(name, size, cls) {
    const s = size || 18;
    const el = TS.svg('svg', { viewBox: '0 0 24 24', width: s, height: s, fill: 'none', stroke: 'currentColor', 'stroke-width': 1.7, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', class: `ico ${cls || ''}`, 'aria-hidden': 'true', focusable: 'false' });
    for (const d of P[name] || P.help) el.append(TS.svg('path', { d }));
    return el;
  };
  TS.icons = Object.keys(P);
})();
