"""Revisa un Python SIN tener tinyml_studio instalado: lo ejecuta el propio intérprete a revisar.

    <python> -I doctor_standalone.py --json --components base,keras --lang en

Busca ``doctor.py`` junto a este archivo (app empaquetada: resources/env) o en el repo
(``../../tinyml_studio/doctor.py``). Solo usa la biblioteca estándar.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (
    os.path.join(HERE, "doctor.py"),
    os.path.join(HERE, "..", "..", "tinyml_studio", "doctor.py"),
):
    if os.path.isfile(cand):
        spec = importlib.util.spec_from_file_location("tinyml_doctor", cand)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tinyml_doctor"] = (
            mod  # los dataclasses/anotaciones necesitan el módulo registrado
        )
        spec.loader.exec_module(mod)
        raise SystemExit(mod.main(sys.argv[1:]))
print("error: no se encuentra doctor.py", file=sys.stderr)
raise SystemExit(2)
