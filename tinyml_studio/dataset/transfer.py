"""Reutilizar una configuración (receta) en otro dataset.

``transfer_recipe(old_recipe, new_report)`` parte de la receta que el motor sugiere para el
dataset NUEVO (``new_report["recipe"]``) y le traslada las decisiones del usuario de la receta
antigua solo donde la columna existe en el nuevo (mismo nombre, o nombre equivalente tras la
normalización de ``dataset/names.py``). Nunca se arrastra estado ajustado (medianas, escalas,
límites): los pasos de limpieza se ajustan de nuevo con el train del dataset nuevo. No modifica
sus argumentos.
"""

from __future__ import annotations

import copy
from typing import Any

from tinyml_studio.i18n import tr

from .names import label_from_filename, name_tokens, normalize_name

NUMERIC_KINDS = {"numeric_continuous", "numeric_discrete", "binary"}
# pasos que solo tienen sentido sobre columnas numéricas
NUMERIC_STEPS = {"replace_values", "absurd_to_nan", "outliers", "scale", "filter_rule"}
COLUMN_LIST_STEPS = {
    "replace_values",
    "absurd_to_nan",
    "impute",
    "outliers",
    "scale",
    "drop_columns",
}
# orden de ejecución recomendado (solo para colocar pasos AÑADIDOS; los traídos conservan su orden)
STEP_RANK = {
    "replace_values": 0,
    "filter_rule": 1,
    "drop_columns": 1,
    "absurd_to_nan": 2,
    "outliers": 3,
    "drop_duplicates": 4,
    "impute": 5,
    "scale": 6,
}
MAX_LISTED = 8


def _listed(names: list[str]) -> str:
    shown = ", ".join(f"'{n}'" for n in names[:MAX_LISTED])
    if len(names) > MAX_LISTED:
        shown += tr(" y {n} más", " and {n} more", n=len(names) - MAX_LISTED)
    return shown


class _Matcher:
    """Empareja nombres antiguos con nuevos: exacto -> normalizado -> mismos tokens (1 a 1)."""

    def __init__(self, new_names: list[str]):
        self.new_names = list(dict.fromkeys(new_names))
        self.map: dict[str, str | None] = {}
        self.taken: set[str] = set()
        self.how: dict[str, str] = {}

    def prefill(self, old_names: list[str]) -> None:
        old = [o for o in dict.fromkeys(old_names) if o not in self.map]
        for o in old:  # 1) exacto
            if o in self.new_names and o not in self.taken:
                self._set(o, o, "exact")
        for fn, how in (
            (normalize_name, "normalized"),
            (lambda s: frozenset(name_tokens(s)), "tokens"),
        ):
            for o in old:
                if o in self.map:
                    continue
                key = fn(o)
                if not key:
                    continue
                cands = [n for n in self.new_names if n not in self.taken and fn(n) == key]
                if len(cands) == 1:
                    self._set(o, cands[0], how)
        for o in old:
            self.map.setdefault(o, None)

    def _set(self, o: str, n: str, how: str) -> None:
        self.map[o], self.how[o] = n, how
        self.taken.add(n)

    def get(self, o: str | None) -> str | None:
        if o is None:
            return None
        if o not in self.map:
            self.prefill([o])
        return self.map[o]


def _is_numeric(kind: str | None, dtype: str | None) -> bool:
    """Numérica de verdad (un 'binary' de texto como 'sí/no' no lo es)."""
    if kind in ("numeric_continuous", "numeric_discrete"):
        return True
    return kind == "binary" and str(dtype).startswith(("int", "uint", "float", "bool"))


def _old_columns(old: dict[str, Any]) -> list[str]:
    """Todos los nombres de columna que menciona la receta antigua (para emparejarlos juntos)."""
    out: list[str] = list(old.get("roles") or {})
    out += list(old.get("drop") or [])
    if old.get("target"):
        out.append(old["target"])
    sp = old.get("split") or {}
    out += [c for c in (sp.get("group_column"), sp.get("time_column")) if c]
    out += list((old.get("encoding") or {}).get("overrides") or {})
    for s in old.get("cleaning") or []:
        p = s.get("params") or {}
        out += list(p.get("columns") or [])
        out += [p[k] for k in ("column", "by") if isinstance(p.get(k), str)]
    return [c for c in out if isinstance(c, str)]


def _map_columns(
    cols: list[str], m: _Matcher, not_numeric: set[str], numeric_only: bool
) -> tuple[list[str], list[str], list[str]]:
    """-> (columnas nuevas, ausentes, con tipo incompatible)."""
    kept, missing, bad = [], [], []
    for c in cols:
        n = m.get(c)
        if n is None:
            missing.append(c)
        elif numeric_only and n in not_numeric:
            bad.append(n)
        elif n not in kept:
            kept.append(n)
    return kept, missing, bad


def transfer_recipe(old_recipe: dict[str, Any], new_report: dict[str, Any]) -> dict[str, Any]:
    old = copy.deepcopy(old_recipe)
    new_report = copy.deepcopy(new_report)
    new: dict[str, Any] = new_report["recipe"]
    notes: list[str] = []
    ap = {
        "matched_exact": 0,
        "matched_renamed": 0,
        "columns_missing": 0,
        "columns_new": 0,
        "roles": 0,
        "target": 0,
        "split": 0,
        "cleaning_kept": 0,
        "cleaning_adjusted": 0,
        "cleaning_dropped": 0,
        "cleaning_added": 0,
        "encoding_overrides": 0,
    }
    kinds = {c["name"]: c.get("kind") for c in new_report.get("columns", [])}
    new_roles: dict[str, str] = dict(new.get("roles") or {})
    lff_new = new.get("label_from_file") or {}
    pool = list(new_roles) + list(kinds)
    if lff_new.get("enabled") and lff_new.get("column"):
        pool.append(lff_new["column"])
    m = _Matcher(pool)
    m.prefill(_old_columns(old))

    # ---------------------------------------------------------------- emparejamiento
    old_cols = [c for c in dict.fromkeys(_old_columns(old)) if c != "source"]
    for o in old_cols:
        n = m.map.get(o)
        if n is None:
            ap["columns_missing"] += 1
        elif n == o:
            ap["matched_exact"] += 1
        else:
            ap["matched_renamed"] += 1
            notes.append(
                tr(
                    "'{o}' de la configuración anterior se emparejó con '{n}' (nombre equivalente)",
                    "'{o}' from the previous configuration was matched to '{n}' (equivalent name)",
                    o=o,
                    n=n,
                )
            )
    missing = [o for o in old_cols if m.map.get(o) is None]
    if missing:
        notes.append(
            tr(
                "Columnas de la configuración anterior que no existen en el dataset nuevo "
                "(se ignoran): {l}",
                "Columns of the previous configuration that do not exist in the new dataset "
                "(ignored): {l}",
                l=_listed(missing),
            )
        )
    matched_new = set(m.taken)
    fresh = [
        n for n in kinds if n not in matched_new and n != "source" and n != lff_new.get("column")
    ]
    ap["columns_new"] = len(fresh)
    if fresh and old_cols:
        notes.append(
            tr(
                "Columnas nuevas que no estaban en la configuración anterior (rol sugerido por "
                "el análisis del dataset nuevo; revísalas): {l}",
                "New columns that were not in the previous configuration (role suggested by the "
                "analysis of the new dataset; review them): {l}",
                l=_listed(fresh),
            )
        )

    # ---------------------------------------------------------------- roles
    old_roles: dict[str, str] = dict(old.get("roles") or {})
    for c in old.get("drop") or []:
        old_roles.setdefault(c, "drop")
    for o, role in old_roles.items():
        n = m.map.get(o)
        if n is None or n == "source" or role == "target":
            continue  # el objetivo se decide aparte
        if role == "time" and kinds.get(n) not in (
            None,
            "datetime",
            "numeric_continuous",
            "numeric_discrete",
            "id",
        ):
            notes.append(
                tr(
                    "'{n}' era columna de tiempo y en el dataset nuevo es de tipo '{k}': Se "
                    "mantiene el rol sugerido para el nuevo",
                    "'{n}' was a time column and in the new dataset it is of kind '{k}': The "
                    "role suggested for the new dataset is kept",
                    n=n,
                    k=kinds.get(n),
                )
            )
            continue
        if kinds.get(n) == "empty":
            continue
        new_roles[n] = role
        ap["roles"] += 1

    # ---------------------------------------------------------------- objetivo / tarea
    lff_old = old.get("label_from_file") or {}
    target_note_needed = False
    if lff_old.get("enabled"):
        files = list(lff_new.get("mapping") or {})
        if not files:
            files = [f["file"] for f in new_report.get("files", [])]
        old_map = lff_old.get("mapping") or {}
        by_norm = {normalize_name(k): v for k, v in old_map.items()}
        mapping, auto = {}, []
        for f in files:
            if f in old_map:
                mapping[f] = old_map[f]
            elif normalize_name(f) in by_norm:
                mapping[f] = by_norm[normalize_name(f)]
            else:
                mapping[f] = (lff_new.get("mapping") or {}).get(f) or label_from_filename(f)
                auto.append(f)
        col = lff_old.get("column") or lff_new.get("column") or "label"
        new["label_from_file"] = {"enabled": True, "column": col, "mapping": mapping}
        if new.get("target") and new["target"] != col and new_roles.get(new["target"]) == "target":
            new_roles[new["target"]] = "feature"
        new["target"], new["task"] = col, old.get("task") or "classification"
        new_roles[col] = "target"
        ap["target"] = 1
        notes.append(
            tr(
                "La etiqueta sale del archivo (un archivo = una clase), como en la configuración "
                "anterior",
                "The label comes from the file (one file = one class), as in the previous "
                "configuration",
            )
        )
        if auto:
            notes.append(
                tr(
                    "Etiquetas deducidas del nombre de archivo (no estaban en la configuración "
                    "anterior): {l}",
                    "Labels derived from the file name (they were not in the previous "
                    "configuration): {l}",
                    l=_listed(auto),
                )
            )
    elif old.get("target"):
        tgt = m.get(old["target"])
        if tgt is None or tgt not in kinds:
            target_note_needed = True
            notes.append(
                tr(
                    "ATENCIÓN: El objetivo '{t}' no existe en el dataset nuevo; se usa el objetivo "
                    "sugerido para el nuevo ({s}). Elige el correcto antes de entrenar",
                    "WARNING: The target '{t}' does not exist in the new dataset; the target "
                    "suggested for the new one ({s}) is used. Choose the right one before training",
                    t=old["target"],
                    s=new.get("target") or tr("ninguno", "none"),
                )
            )
        else:
            if (
                new.get("target")
                and new["target"] != tgt
                and new_roles.get(new["target"]) == "target"
            ):
                new_roles[new["target"]] = "feature"
            new["target"] = tgt
            new_roles[tgt] = "target"
            lff = dict(new.get("label_from_file") or {})
            lff["enabled"] = False
            new["label_from_file"] = lff
            task, k = old.get("task") or new.get("task"), kinds.get(tgt)
            if task == "classification" and k == "numeric_continuous":
                task = "regression"
                notes.append(
                    tr(
                        "El objetivo '{t}' era de clasificación y en el dataset nuevo es numérico "
                        "continuo: La tarea pasa a regresión",
                        "The target '{t}' was a classification target and in the new dataset it is "
                        "continuous numeric: The task becomes regression",
                        t=tgt,
                    )
                )
            elif task == "regression" and k in ("categorical", "binary"):
                task = "classification"
                notes.append(
                    tr(
                        "El objetivo '{t}' era de regresión y en el dataset nuevo es categórico: "
                        "la tarea pasa a clasificación",
                        "The target '{t}' was a regression target and in the new dataset it is "
                        "categorical: The task becomes classification",
                        t=tgt,
                    )
                )
            new["task"] = task
            ap["target"] = 1
    new["roles"] = new_roles
    new["drop"] = [c for c, r in new_roles.items() if r == "drop" and c != "source"]

    # ---------------------------------------------------------------- partición
    sp_old = old.get("split")
    if sp_old:
        sp = dict(new.get("split") or {})
        strat = sp_old.get("strategy")
        g = m.get(sp_old.get("group_column")) if sp_old.get("group_column") else None
        t = m.get(sp_old.get("time_column")) if sp_old.get("time_column") else None
        problem = None
        if strat == "group" and g is None:
            problem = tr(
                "la columna de grupo '{c}'", "the group column '{c}'", c=sp_old.get("group_column")
            )
        elif strat == "temporal_blocks" and t is None:
            problem = tr(
                "la columna de tiempo '{c}'", "the time column '{c}'", c=sp_old.get("time_column")
            )
        elif strat == "stratified" and new.get("task") != "classification":
            problem = tr("una etiqueta de clasificación", "a classification label")
        if strat and problem is None:
            sp.update(
                strategy=strat,
                group_column=g,
                time_column=t,
                reason=tr(
                    "estrategia traída de la configuración anterior",
                    "strategy carried over from the previous configuration",
                ),
            )
            ap["split"] = 1
        elif strat:
            notes.append(
                tr(
                    "La partición '{s}' necesitaba {p}, que no existe en el dataset nuevo: Se usa "
                    "la estrategia sugerida para el nuevo ({n}) con las mismas proporciones",
                    "The '{s}' split needed {p}, which does not exist in the new dataset: The "
                    "strategy suggested for the new one ({n}) is used with the same ratios",
                    s=strat,
                    p=problem,
                    n=sp.get("strategy"),
                )
            )
        if sp_old.get("ratios"):
            sp["ratios"] = list(sp_old["ratios"])
            ap["split"] = 1
        new["split"] = sp

    # ---------------------------------------------------------------- codificación
    enc_old = old.get("encoding")
    if enc_old:
        enc = dict(new.get("encoding") or {})
        for k in ("default", "max_categories"):
            if k in enc_old:
                enc[k] = enc_old[k]
        ov: dict[str, Any] = {}
        lost: list[str] = []
        for c, meth in (enc_old.get("overrides") or {}).items():
            n = m.get(c)
            if n is None or kinds.get(n) == "numeric_continuous":
                lost.append(c)
            else:
                ov[n] = meth
        if lost:
            notes.append(
                tr(
                    "Codificación personalizada descartada (columna ausente o ahora numérica): {l}",
                    "Custom encoding dropped (column missing or now numeric): {l}",
                    l=_listed(lost),
                )
            )
        enc["overrides"] = ov
        ap["encoding_overrides"] = len(ov)
        new["encoding"] = enc

    # ---------------------------------------------------------------- limpieza
    non_num = {
        c["name"]
        for c in new_report.get("columns", [])
        if not _is_numeric(c.get("kind"), c.get("dtype"))
    }
    steps = _transfer_cleaning(old.get("cleaning") or [], new, new_roles, non_num, m, notes, ap)
    new["cleaning"] = steps

    if not target_note_needed and not new.get("target"):
        notes.append(
            tr(
                "La receta resultante no tiene objetivo: Elige uno antes de entrenar",
                "The resulting recipe has no target: Choose one before training",
            )
        )
    return {"recipe": new, "notes": notes, "applied": ap}


def _transfer_cleaning(old_steps, new, roles, non_num, m, notes, ap) -> list[dict]:
    carried: list[dict] = []
    for s in old_steps:
        kind, p = s.get("kind"), copy.deepcopy(s.get("params") or {})
        why = s.get("why", "")
        if kind in COLUMN_LIST_STEPS and "columns" in p:
            numeric = kind in NUMERIC_STEPS or (
                kind == "impute" and p.get("strategy", "median") in ("median", "mean")
            )
            cols, miss, bad = _map_columns(p["columns"], m, non_num, numeric)
            # solo features reales (no objetivo/descartadas/meta) en transformaciones de valor
            if kind != "drop_columns":
                notf = [c for c in cols if roles.get(c) not in ("feature", None)]
                cols = [c for c in cols if c not in notf]
                bad += notf
            if miss or bad:
                notes.append(
                    tr(
                        "Paso '{k}': Se quitan columnas ausentes ({a}) o con tipo/rol "
                        "incompatible ({b})",
                        "Step '{k}': Columns removed because they are missing ({a}) or have an "
                        "incompatible kind/role ({b})",
                        k=kind,
                        a=_listed([*dict.fromkeys(miss)]) or "-",
                        b=_listed([*dict.fromkeys(bad)]) or "-",
                    )
                )
            if not cols:
                ap["cleaning_dropped"] += 1
                continue
            ap["cleaning_adjusted" if (miss or bad) else "cleaning_kept"] += 1
            p["columns"] = sorted(cols) if kind != "drop_columns" else cols
        elif kind == "filter_rule":
            n = m.get(p.get("column"))
            if n is None or n in non_num:
                ap["cleaning_dropped"] += 1
                notes.append(
                    tr(
                        "Se descarta el filtro sobre '{c}' (columna ausente o no numérica en el "
                        "dataset nuevo)",
                        "The filter on '{c}' is dropped (column missing or not numeric in the new "
                        "dataset)",
                        c=p.get("column"),
                    )
                )
                continue
            p["column"] = n
            ap["cleaning_kept"] += 1
        elif kind == "drop_duplicates":
            ig = [m.get(c) or (c if c == "source" else None) for c in p.get("ignore", [])]
            p["ignore"] = [c for c in ig if c]
            ap["cleaning_kept"] += 1
        else:
            ap["cleaning_kept"] += 1
        if "by" in p and isinstance(p.get("by"), str):
            b = "source" if p["by"] == "source" else m.get(p["by"])
            if b is None:
                notes.append(
                    tr(
                        "Paso '{k}': La columna de agrupación '{c}' no existe; se calcula sin "
                        "agrupar",
                        "Step '{k}': The grouping column '{c}' does not exist; computed without "
                        "grouping",
                        k=kind,
                        c=p["by"],
                    )
                )
                ap["cleaning_adjusted"] += 1
            p["by"] = b
        carried.append({"kind": kind, "params": p, "state": {}, "why": why})

    # pasos que el análisis del dataset nuevo sugiere para columnas que la config. anterior no cubre
    covered: dict[str, set[str]] = {}
    for s in carried:
        covered.setdefault(s["kind"], set()).update(s["params"].get("columns", []))
    for s in new.get("cleaning") or []:
        kind, p = s.get("kind"), copy.deepcopy(s.get("params") or {})
        if "columns" in p:
            rest = [c for c in p["columns"] if c not in covered.get(kind, set())]
            if not rest:
                continue
            tgt = next((c for c in carried if c["kind"] == kind and kind == "scale"), None)
            if tgt is not None:  # mismo método de escalado para las columnas nuevas
                tgt["params"]["columns"] = sorted({*tgt["params"]["columns"], *rest})
                ap["cleaning_adjusted"] += 1
                covered[kind].update(rest)
                notes.append(
                    tr(
                        "Se añadieron al escalado las columnas del dataset nuevo: {l}",
                        "Columns of the new dataset were added to the scaling: {l}",
                        l=_listed(rest),
                    )
                )
                continue
            p["columns"] = rest
        elif any(c["kind"] == kind for c in carried):
            continue
        add = {"kind": kind, "params": p, "state": {}, "why": s.get("why", "")}
        rank = STEP_RANK.get(kind, 9)
        pos = next(
            (i for i, c in enumerate(carried) if STEP_RANK.get(c["kind"], 9) > rank), len(carried)
        )
        carried.insert(pos, add)
        covered.setdefault(kind, set()).update(p.get("columns", []))
        ap["cleaning_added"] += 1
        notes.append(
            tr(
                "Paso de limpieza añadido porque el dataset nuevo lo necesita: '{k}'{c}",
                "Cleaning step added because the new dataset needs it: '{k}'{c}",
                k=kind,
                c=f" ({_listed(p['columns'])})" if p.get("columns") else "",
            )
        )
    return carried

