"""¿Se pueden usar varios archivos juntos? Compara las columnas de cada archivo (tras armonizar).

Dos archivos son compatibles si comparten al menos ``MIN_SIMILARITY`` de sus columnas (Jaccard).
Los archivos se agrupan en conjuntos compatibles: si sale más de uno (p. ej. datos eléctricos y de
suelo), se avisa y se ofrece usar cada conjunto por separado en vez de mezclarlos.
"""

from __future__ import annotations

from pathlib import Path

from tinyml_studio.i18n import tr

from .harmonize import ColumnGroup, source_name

MIN_SIMILARITY = 0.5
SAME_STRUCTURE = 0.9


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def compatibility(files: list[Path], groups: list[ColumnGroup]) -> dict:
    names = [source_name(p) for p in files]
    cols: dict[str, set[str]] = {n: set() for n in names}
    for g in groups:
        for p in g.members:
            cols[source_name(p)].add(g.canonical)
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    lowest = 1.0
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            s = _jaccard(cols[a], cols[b])
            lowest = min(lowest, s)
            if s >= MIN_SIMILARITY:
                parent[find(a)] = find(b)
    clusters: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        clusters.setdefault(find(n), []).append(i)
    sets = [
        {"files": [names[i] for i in idx], "paths": [str(files[i]) for i in idx]}
        for idx in clusters.values()
    ]
    if len(names) < 2:
        verdict = "single"
    elif len(sets) > 1:
        verdict = "incompatible"
    elif lowest >= SAME_STRUCTURE:
        verdict = "same"
    else:
        verdict = "similar"
    advice = {
        "single": "",
        "same": tr(
            "Todos los archivos tienen la misma estructura: Se unen sin problema.",
            "All files share the same structure: They merge cleanly.",
        ),
        "similar": tr(
            "Los archivos tienen columnas parecidas pero no idénticas: Las que faltan en alguno "
            "quedan vacías. Revisa que midan lo mismo antes de mezclarlos.",
            "The files have similar but not identical columns: Those missing in some files stay "
            "empty. Check that they measure the same thing before mixing them.",
        ),
        "incompatible": tr(
            "Hay {n} grupos de archivos con columnas muy distintas (no es buena idea mezclarlos, "
            "p. ej. datos eléctricos con calidad de suelo). Usa cada grupo por separado, salvo que "
            "tu estudio necesite cruzarlos.",
            "There are {n} groups of files with very different columns (mixing them is rarely a "
            "good idea, e.g. electrical data with soil quality). Use each group separately unless "
            "your study needs to combine them.",
            n=len(sets),
        ),
    }[verdict]
    return {
        "verdict": verdict,
        "min_similarity": round(lowest, 2),
        "sets": sets,
        "advice": advice,
    }
