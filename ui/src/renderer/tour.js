'use strict';
// Guided tour (Next / Back / Skip) + contextual "?" help per step. Plain DOM, positioned with CSSOM
// (CSP-safe), no animation. The first-run flag lives in localStorage (guarded).
(function () {
  const TS = window.TS;
  const { h } = TS;
  const t = TS.t;

  const store = {
    get(k) { try { return window.localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch { /* ignore */ } },
  };

  // what each step is for and what you can do there
  const HELP_BASE = () => [
    { title: t('1 · Dataset', '1 · Dataset'), lead: t('Load your CSV files and let the app understand them.', 'Carga tus CSV y deja que la app los entienda.'),
      bullets: [t('Open several CSV files or a whole folder (different columns are fine).', 'Abre varios CSV o una carpeta completa (columnas distintas no son problema).'), t('See rows, columns, quality alerts and which columns were merged across files.', 'Mira filas, columnas, alertas de calidad y qué columnas se unieron entre archivos.'), t('Search and page through the column table instead of scrolling thousands of lines.', 'Busca y pagina la tabla de columnas en vez de recorrer miles de líneas.')] },
    { title: t('2 · Task', '2 · Tarea'), lead: t('Say what to predict and what each column is.', 'Indica qué predecir y qué es cada columna.'),
      bullets: [t('Classification (a category) or regression (a number).', 'Clasificación (una categoría) o regresión (un número).'), t('Choose the target column, or use the file name as the label.', 'Elige la columna objetivo, o usa el nombre del archivo como etiqueta.'), t('Set roles per column and the categorical encoding (one-hot, dummy, ordinal).', 'Define el rol de cada columna y la codificación de categóricas (one-hot, dummy, ordinal).')] },
    { title: t('3 · Cleaning', '3 · Limpieza'), lead: t('Fix missing and absurd values, without leaking test data.', 'Arregla faltantes y valores absurdos sin filtrar datos de test.'),
      bullets: [t('Every step comes from an alert and is fitted on the training set only.', 'Cada paso viene de una alerta y se ajusta solo con el conjunto de entrenamiento.'), t('Untick steps, change strategies, then see before/after charts.', 'Desmarca pasos, cambia estrategias y mira las gráficas antes/después.'), t('Export any chart as PNG, JPG or SVG with the download button above each chart.', 'Exporta cualquier gráfica como PNG, JPG o SVG con el botón de descarga sobre cada gráfica.')] },
    { title: t('4 · Augmentation', '4 · Aumento'), lead: t('Optional: Make more training data from your real rows.', 'Opcional: Crea más datos de entrenamiento a partir de tus filas reales.'),
      bullets: [t('New rows from copies with a little noise, or between neighbouring rows. Not the same as filling gaps (step 3). Only the training set grows.', 'Filas nuevas a partir de copias con un poco de ruido, o entre filas vecinas. No es rellenar huecos (paso 3). Solo crece el entrenamiento.'), t('Creating data is not recommended in general: It is under your responsibility.', 'Crear datos no es recomendable en general: Es bajo tu responsabilidad.'), t('Skip it if you have enough data.', 'Sáltalo si tienes datos suficientes.')] },
    { title: t('5 · Split', '5 · Partición'), lead: t('Train / validation / test, the right way.', 'Entrenamiento / validación / test, de la forma correcta.'),
      bullets: [t('Temporal blocks, by group, stratified or random.', 'Bloques temporales, por grupo, estratificada o aleatoria.'), t('Check sizes and class balance per split.', 'Revisa tamaños y balance de clases por conjunto.'), t('A bad split gives fake accuracy: Follow the suggestion unless you know why not.', 'Una mala partición da precisión falsa: Sigue la sugerencia salvo que sepas por qué no.')] },
    { title: t('6 · Model & training', '6 · Modelo y entrenamiento'), lead: t('Train one model or benchmark many.', 'Entrena un modelo o compara muchos.'),
      bullets: [t('Benchmark: Many models on identical splits, ranked without peeking at the test set.', 'Banco de pruebas: Muchos modelos con particiones idénticas, ordenados sin mirar el test.'), t('Watch CPU/GPU load, temperatures, memory and the time left while training.', 'Mira carga de CPU/GPU, temperaturas, memoria y tiempo restante mientras entrena.'), t('Inspect quality: Confusion matrix, ROC/PR, per-class metrics, loss curves.', 'Revisa la calidad: Matriz de confusión, ROC/PR, métricas por clase, curvas de pérdida.')] },
    { title: t('7 · Board & export', '7 · Placa y exportación'), lead: t('Quantize, check it fits, take the code.', 'Cuantiza, verifica que cabe y llévate el código.'),
      bullets: [t('Pick your board and flash/RAM budget; get a fits / does-not-fit verdict.', 'Elige tu placa y presupuesto de flash/RAM; obtén el veredicto cabe / no cabe.'), t('Get C code, TFLite and ready Arduino sketches (plain C, EloquentTinyML, TFLM).', 'Obtén código C, TFLite y sketches de Arduino listos (C puro, EloquentTinyML, TFLM).'), t('Browse the model structure like Netron and save the files.', 'Explora la estructura del modelo como Netron y guarda los archivos.')] },
  ];
  // what was added after the first version of the tour, per step (same order as HELP_BASE)
  const EXTRA = () => [
    [t('A card tells you whether the files belong together (same structure) or are different kinds of data you should not mix.', 'Una tarjeta te dice si los archivos van juntos (misma estructura) o son datos de distinto tipo que no conviene mezclar.')],
    [t('"Choose exactly which columns to use" lets you tick only the inputs you want; all the others become unused.', '"Elegir exactamente qué columnas usar" te deja marcar solo las entradas que quieres; las demás pasan a no usarse.'), t('Columns that identify a recording (like a sensor id) can make results look better than they are: Leave them out.', 'Las columnas que identifican una grabación (como un id de sensor) pueden hacer que los resultados parezcan mejores de lo que son: Déjalas fuera.')],
    [], [], [],
    [t('"Does it work on a NEW session?" trains without one recording and tests on it, which is the honest check.', '"¿Funciona con una sesión NUEVA?" entrena sin una grabación y prueba con ella, que es la comprobación honesta.'), t('"Find settings that fit my board" searches the best settings under your flash and RAM budget.', '"Buscar parámetros que quepan en mi placa" busca la mejor configuración dentro de tu presupuesto de flash y RAM.'), t('"Detect anomalies" learns only what is normal and flags what is not.', '"Detectar anomalías" aprende solo lo normal y avisa de lo que no se parece.')],
    [t('"Save the Arduino sketch" asks where to save and shows the exact path of the .ino file.', '"Guardar el sketch de Arduino" pregunta dónde guardar y muestra la ruta exacta del archivo .ino.'), t('Extra tools: export to ONNX, which columns matter, estimated speed per board, and answering "unknown" when unsure.', 'Herramientas extra: Exportar a ONNX, qué columnas importan, velocidad estimada por placa y responder "desconocido" cuando dude.'), t('Speed figures are estimates, not measurements: Time the model on your board.', 'Las velocidades son estimaciones, no mediciones: Cronometra el modelo en tu placa.')],
  ];
  const HELP = () => HELP_BASE().map((s, i) => ({ ...s, bullets: [...s.bullets, ...(EXTRA()[i] || [])] }));
  TS.stepHelp = HELP;

  // While the app is still being set up (no Python packages yet) the tour only explains the setup;
  // the rest of the modules do not exist yet, so touring them would be confusing.
  const inSetup = () => Boolean(TS.isSetup && TS.isSetup());
  const SETUP_STOPS = () => [
    { title: t('Setting up TinyML Studio', 'Configurando TinyML Studio'), bullets: [t('This app ships without Python packages to stay small. First you choose how to get them.', 'Esta app viene sin paquetes de Python para ser pequeña. Primero eliges cómo conseguirlos.'), t('This short tour only covers the setup. When it is done, open Tutorial again to tour the rest of the app.', 'Este recorrido corto solo cubre la configuración. Cuando termine, abre Tutorial otra vez para conocer el resto de la app.')] },
    { target: '#components-box', side: 'bottom', title: t('Option 1 — Install for me', 'Opción 1 — Instalar por mí'), bullets: [t('Tick what you need. "Core" is always included; Keras and PyTorch are big, add them only if you will train neural networks.', 'Marca lo que necesites. «Núcleo» siempre va incluido; Keras y PyTorch pesan mucho, añádelos solo si vas a entrenar redes neuronales.'), t('The app downloads a private Python and the packages with checksums. Nothing is installed system-wide.', 'La app descarga un Python privado y los paquetes con sumas de verificación. No se instala nada en todo el sistema.')] },
    { target: '#env-pick', side: 'bottom', title: t('Option 2 — Use my own environment', 'Opción 2 — Usar mi propio entorno'), bullets: [t('Pick the python of your virtual or conda environment. The app tells you what is missing and can install it for you, after asking.', 'Elige el python de tu entorno virtual o de conda. La app te dice qué falta y puede instalarlo por ti, tras pedir permiso.')] },
    { title: t('What happens next', 'Qué pasa después'), bullets: [t('When the installation finishes the workflow appears on the left, starting with your dataset.', 'Al terminar la instalación aparece el flujo de trabajo a la izquierda, empezando por tu dataset.'), t('Then open Tutorial again for the full tour.', 'Entonces abre Tutorial otra vez para el recorrido completo.')] },
  ];

  const STOPS = () => {
    if (inSetup()) return SETUP_STOPS();
    const H = HELP();
    const stops = [
      { title: t('Welcome to TinyML Studio', 'Bienvenido a TinyML Studio'), bullets: [t('Turn your data into a small model that fits a microcontroller, in 7 guided steps.', 'Convierte tus datos en un modelo pequeño que cabe en un microcontrolador, en 7 pasos guiados.'), t('Everything runs on your computer. This tour takes about a minute; you can skip it any time.', 'Todo corre en tu computador. Este recorrido dura un minuto; puedes omitirlo cuando quieras.')] },
      { target: '#steps', side: 'right', title: t('The workflow', 'El flujo de trabajo'), bullets: [t('The seven steps live here. A ✓ means you already went through it.', 'Aquí están los siete pasos. Un ✓ significa que ya pasaste por él.'), t('Locked steps open when their prerequisite is done (dataset first, training before export).', 'Los pasos bloqueados se abren al cumplir su requisito (primero el dataset, entrenar antes de exportar).')] },
    ];
    H.forEach((s, i) => stops.push({ target: `#steps [data-step="${i}"]`, side: 'right', title: s.title, lead: s.lead, bullets: s.bullets }));
    stops.push(
      { target: '#tool-history', side: 'right', title: t('History & PDF report', 'Historial e informe PDF'), bullets: [t('History lists past runs: Compare them, or reuse a configuration on another dataset.', 'El historial lista corridas anteriores: Compáralas o reutiliza una configuración con otro dataset.'), t('The PDF report collects everything you did: Data, cleaning, models, charts, hardware use and limits.', 'El informe PDF reúne todo lo que hiciste: Datos, limpieza, modelos, gráficas, uso de hardware y límites.')] },
      { target: '#tool-env', side: 'right', title: t('Environment', 'Entorno'), bullets: [t('Shows which Python runs the core and what is missing in it.', 'Muestra qué Python ejecuta el núcleo y qué le falta.'), t('"Look for Python on this computer" lists the Pythons you already have, with a verdict for each. The app never changes your Python.', '"Buscar Python en este equipo" lista los Python que ya tienes, con un veredicto para cada uno. La app nunca cambia tu Python.'), t('Node.js is not needed to use the app.', 'Node.js no hace falta para usar la app.')] },
      { target: '#theme-seg', side: 'bottom', title: t('Theme and language', 'Tema e idioma'), bullets: [t('Light, dark or follow your system. English or Spanish at any time.', 'Claro, oscuro o seguir al sistema. Inglés o español en cualquier momento.'), t('Save and load your whole configuration as a recipe (top right).', 'Guarda y carga toda tu configuración como una receta (arriba a la derecha).')] },
      { title: t('You are ready', 'Todo listo'), bullets: [t('Start with step 1: Open your CSV files.', 'Empieza con el paso 1: Abre tus archivos CSV.'), t('Every page has a ? button with tips, and this tour is always under Tutorial.', 'Cada página tiene un botón ? con ayuda, y este recorrido siempre está en Tutorial.')] },
    );
    return stops;
  };

  let layer = null;

  function close(done) {
    if (layer) { layer.remove(); layer = null; }
    document.removeEventListener('keydown', onKey, true);
    if (done && !inSetup()) store.set('ts.tour', '1'); // el recorrido de la configuración no cuenta como el primero
  }
  function onKey(e) {
    if (!layer) return;
    if (e.key === 'Escape') { e.preventDefault(); close(true); }
    if (e.key === 'ArrowRight') { e.preventDefault(); layer.go(1); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); layer.go(-1); }
  }

  function place(pop, hole, stop) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const el = stop.target ? document.querySelector(stop.target) : null;
    if (!el) {
      hole.style.display = 'none';
      pop.style.left = `${Math.max(12, (vw - 360) / 2)}px`;
      pop.style.top = `${Math.max(12, vh / 2 - 130)}px`;
      return;
    }
    el.scrollIntoView({ block: 'nearest' });
    const r = el.getBoundingClientRect();
    const pad = 4;
    hole.style.display = 'block';
    hole.style.left = `${r.left - pad}px`; hole.style.top = `${r.top - pad}px`;
    hole.style.width = `${r.width + 2 * pad}px`; hole.style.height = `${r.height + 2 * pad}px`;
    const pw = Math.min(360, vw - 24);
    const ph = pop.offsetHeight || 220;
    let left = stop.side === 'right' ? r.right + 14 : r.left;
    let top = stop.side === 'right' ? r.top : r.bottom + 12;
    left = Math.min(Math.max(12, left), vw - pw - 12);
    top = Math.min(Math.max(12, top), vh - ph - 12);
    pop.style.left = `${left}px`; pop.style.top = `${top}px`;
  }

  // force=false: only on the very first run
  TS.startTour = function startTour(force, from) {
    if (!force && (store.get('ts.tour') === '1' || (window.tinyml && window.tinyml.flags && window.tinyml.flags.smoke))) return;
    close(false);
    const stops = STOPS();
    let i = from || 0;
    const hole = h('div', { class: 'tour-hole' });
    const pop = h('div', { class: 'tour-pop', role: 'dialog', 'aria-modal': 'true', 'aria-label': t('Guided tour', 'Recorrido guiado'), id: 'tour-pop' });
    const shade = h('div', { class: 'tour-shade' });
    layer = h('div', { id: 'tour' }, shade, hole, pop);
    layer.go = (d) => { i = Math.min(stops.length - 1, Math.max(0, i + d)); draw(); };
    const draw = () => {
      const s = stops[i];
      const last = i === stops.length - 1;
      shade.hidden = Boolean(s.target);
      pop.replaceChildren(
        h('h4', { text: s.title }), ...(s.lead ? [h('p', { class: 'muted', text: s.lead })] : []),
        h('ul', {}, s.bullets.map((b) => h('li', { text: b }))),
        h('div', { class: 'tour-foot' },
          h('span', { class: 'tour-dots', 'aria-hidden': 'true' }, stops.map((_, k) => h('i', { class: k === i ? 'on' : '' }))),
          h('span', { class: 'muted', text: `${i + 1}/${stops.length}` }),
          last ? null : h('button', { type: 'button', class: 'ghost sm', id: 'tour-skip', text: t('Skip', 'Omitir'), onclick: () => close(true) }),
          h('button', { type: 'button', class: 'ghost sm', id: 'tour-back', text: t('Back', 'Atrás'), disabled: i === 0, onclick: () => layer.go(-1) }),
          h('button', { type: 'button', class: 'sm', id: 'tour-next', text: last ? t('Finish', 'Terminar') : t('Next', 'Siguiente'), onclick: () => (last ? close(true) : layer.go(1)) })));
      place(pop, hole, s);
      const nb = pop.querySelector('#tour-next');
      if (nb) nb.focus();
    };
    document.body.append(layer);
    document.addEventListener('keydown', onKey, true);
    window.addEventListener('resize', () => layer && place(pop, hole, stops[i]), { once: true });
    draw();
  };

  // contextual help for the current step: a small popover under the "?" button
  TS.stepHelpButton = function stepHelpButton(step) {
    const btn = h('button', { type: 'button', class: 'ghost sm', id: 'step-help', 'aria-label': t('Help for this step', 'Ayuda de este paso'), title: t('What can I do here?', '¿Qué puedo hacer aquí?') }, TS.icon('help', 15));
    btn.addEventListener('click', () => {
      const s = HELP()[step];
      if (!s) return;
      const idx = step + 2; // position of the step inside the tour
      TS.startTour(true, idx);
    });
    return btn;
  };
})();
