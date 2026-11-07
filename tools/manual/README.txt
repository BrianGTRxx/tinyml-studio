How the manual is built
=======================
1. Generate the synthetic recordings (no real data):
     python tools/manual/make_sample_data.py OUT/data
2. Let the app take its own screenshots (needs the Electron app, see como_ejecutar.txt):
     TINYML_SMOKE=1 TINYML_SMOKE_MANUAL=1 TINYML_SMOKE_LANG=en TINYML_SMOKE_THEME=light
     TINYML_SMOKE_PATHS=OUT/data TINYML_ALLOWED_ROOTS=OUT/data TINYML_SMOKE_SHOTS=OUT/shots
     TINYML_WORKDIR=OUT/ws TINYML_PYTHON=<python with tinyml-studio>  npx electron . --disable-gpu
   (run it from the ui folder; it writes the m-*.png files into OUT/shots)
3. Build the PDF:
     python tools/manual/build_manual.py OUT/shots docs/manual/TinyML-Studio-User-Manual.pdf
4. Check it (blank pages, dash bullets):
     python tools/manual/check_manual.py docs/manual/TinyML-Studio-User-Manual.pdf
Requires fpdf2 and Pillow (build) and pypdf (check).
