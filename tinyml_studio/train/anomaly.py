"""Detección de anomalías: aprender cómo es lo NORMAL y avisar cuando algo no se parece.

Se entrena solo con filas normales (sin etiquetas de falla). El umbral se fija en VALIDACIÓN para
que, como mucho, una fracción ``false_alarm`` de lo normal dispare la alarma. El TEST se mide una
sola vez: tasa de falsas alarmas sobre lo normal y, si hay clases marcadas como anómalas (solo para
EVALUAR, nunca para entrenar ni para fijar el umbral), cuántas se detectan y el AUC.

Una anomalía que el modelo nunca vio no tiene por qué parecerse a las clases anómalas de este
ejemplo: esto mide cómo se comporta con ESAS, no con todas las fallas posibles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tinyml_studio.clean.prepare import prepare_splits
from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit

from .features import FeatureEncoder
from .run import feature_columns

MODELS = ("isolation_forest", "one_class_svm")
MAX_TRAIN_ROWS = 20_000


def _build(model: str, seed: int):
    if model == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        return IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1)
    if model == "one_class_svm":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM

        return make_pipeline(StandardScaler(), OneClassSVM(nu=0.05, gamma="scale"))
    raise ValueError(
        tr(
            "modelo de anomalías desconocido: {m}. Disponibles: {a}",
            "unknown anomaly model: {m}. Available: {a}",
            m=model,
            a=", ".join(MODELS),
        )
    )


def _score(est, X: np.ndarray) -> np.ndarray:
    """Mayor = más anómalo."""
    raw = est.score_samples(X) if hasattr(est, "score_samples") else est.decision_function(X)
    return -np.asarray(raw)


def anomaly_detection(
    recipe: dict[str, Any],
    *,
    model: str = "isolation_forest",
    normal_classes: list[str] | None = None,
    false_alarm: float = 0.01,
    seed: int = 42,
    exclude: tuple[str, ...] = (),
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not 0 < false_alarm < 0.5:
        raise ValueError(
            tr("false_alarm debe estar entre 0 y 0.5", "false_alarm must be between 0 and 0.5")
        )
    emit("prepare", 5, tr("Preparando datos…", "Preparing data…"))
    prep = prepare_splits(recipe, seed=seed)
    target = prep.target
    labeled = bool(target) and recipe.get("task") == "classification"
    if normal_classes and not labeled:
        raise ValueError(
            tr(
                "para elegir clases normales la receta necesita una columna objetivo de clases",
                "to pick normal classes the recipe needs a class target column",
            )
        )
    feats = feature_columns(recipe, prep.clean["train"], exclude)
    train, val, test = prep.clean["train"], prep.clean["val"], prep.clean["test"]

    def is_normal(df):
        if not normal_classes:
            return np.ones(len(df), bool)
        return df[target].astype(str).isin([str(c) for c in normal_classes]).to_numpy()

    if normal_classes:
        known = set(map(str, train[target].dropna().unique())) | set(
            map(str, val[target].dropna().unique())
        )
        missing = [c for c in normal_classes if str(c) not in known]
        if missing:
            raise ValueError(
                tr("clases normales inexistentes: {c}", "unknown normal classes: {c}", c=missing)
            )
    tr_n, va_n, te_n = train[is_normal(train)], val[is_normal(val)], test[is_normal(test)]
    if len(tr_n) < 30 or len(va_n) < 10:
        raise ValueError(
            tr(
                "muy pocas filas normales (train {a}, val {b}); hacen falta al menos 30 y 10",
                "too few normal rows (train {a}, val {b}); at least 30 and 10 are needed",
                a=len(tr_n),
                b=len(va_n),
            )
        )
    enc = FeatureEncoder.fit(tr_n, feats, None, "unsupervised", recipe.get("encoding"))
    if len(tr_n) > MAX_TRAIN_ROWS:
        tr_n = tr_n.sample(MAX_TRAIN_ROWS, random_state=seed)
    emit("fit", 30, tr("Aprendiendo lo normal…", "Learning what is normal…"))
    est = _build(model, seed)
    est.fit(enc.transform(tr_n))
    s_val = _score(est, enc.transform(va_n))
    thr = float(np.quantile(s_val, 1 - false_alarm))
    s_te = _score(est, enc.transform(te_n)) if len(te_n) else np.array([])
    res: dict[str, Any] = {
        "model": model,
        "normal_classes": list(normal_classes or []),
        "false_alarm_target": false_alarm,
        "threshold": round(thr, 6),
        "rows": {"train_normal": int(len(tr_n)), "val_normal": int(len(va_n))},
        "val_false_alarm": round(float((s_val > thr).mean()), 4),
        "test_false_alarm": round(float((s_te > thr).mean()), 4) if len(s_te) else None,
        "detection": None,
        "warnings": [],
        "family": "anomaly",
    }
    if normal_classes and labeled:
        bad = test[~is_normal(test)]
        if len(bad):
            s_bad = _score(est, enc.transform(bad))
            per = {}
            for c, g in bad.groupby(bad[target].astype(str)):
                per[c] = {
                    "rows": int(len(g)),
                    "detected": round(float((_score(est, enc.transform(g)) > thr).mean()), 4),
                }
            res["detection"] = {
                "rate": round(float((s_bad > thr).mean()), 4),
                "auc": _auc(s_te, s_bad),
                "per_class": per,
            }
        else:
            res["warnings"].append(
                tr(
                    "el test no tiene filas de las clases anómalas: No se pudo medir la detección",
                    "the test set has no rows of the anomalous classes: Detection was not measured",
                )
            )
    if res["test_false_alarm"] is not None and res["test_false_alarm"] > 3 * false_alarm:
        res["warnings"].append(
            tr(
                "en test hay {t:.1%} de falsas alarmas (objetivo {g:.1%}): Lo normal cambia entre "
                "validación y test (otra sesión o deriva)",
                "the test set has {t:.1%} false alarms (target {g:.1%}): Normal data shifts "
                "between validation and test (another session or drift)",
                t=res["test_false_alarm"],
                g=false_alarm,
            )
        )
    if out_dir is not None:
        import json

        import joblib

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(est, out / "anomaly_model.joblib")
        (out / "anomaly_encoder.json").write_text(json.dumps(enc.to_dict()), encoding="utf-8")
        (out / "anomaly.json").write_text(
            json.dumps({k: v for k, v in res.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    res["summary"] = summary(res)
    emit("done", 100, tr("Detector listo", "Detector ready"))
    return res


def _auc(normal_scores: np.ndarray, bad_scores: np.ndarray) -> float | None:
    if not len(normal_scores) or not len(bad_scores):
        return None
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.zeros(len(normal_scores)), np.ones(len(bad_scores))]
    return round(float(roc_auc_score(y, np.r_[normal_scores, bad_scores])), 4)


def summary(res: dict[str, Any]) -> str:
    lines = [
        tr(
            "Detector {m}: Umbral {th}, pensado para ≤ {g:.1%} de falsas alarmas sobre lo normal.",
            "Detector {m}: Threshold {th}, meant for ≤ {g:.1%} false alarms on normal data.",
            m=res["model"],
            th=res["threshold"],
            g=res["false_alarm_target"],
        ),
        tr(
            "Falsas alarmas: Validación {v:.1%}; test (una vez) {t}.",
            "False alarms: Validation {v:.1%}; test (once) {t}.",
            v=res["val_false_alarm"],
            t="—" if res["test_false_alarm"] is None else f"{res['test_false_alarm']:.1%}",
        ),
    ]
    d = res["detection"]
    if d:
        lines.append(
            tr(
                "Detecta {r:.1%} de las filas de las clases anómalas (AUC {a}).",
                "Detects {r:.1%} of the rows of the anomalous classes (AUC {a}).",
                r=d["rate"],
                a=d["auc"],
            )
        )
        lines += [f"  {c}: {v['detected']:.1%} ({v['rows']})" for c, v in d["per_class"].items()]
        lines.append(
            tr(
                "Solo mide esas clases: Una falla distinta puede pasar inadvertida.",
                "This only measures those classes: A different fault may go unnoticed.",
            )
        )
    elif not res["normal_classes"]:
        lines.append(
            tr(
                "Sin clases anómalas de ejemplo no se puede medir cuánto detecta; solo las falsas "
                "alarmas.",
                "Without example anomalous classes detection cannot be measured; only false "
                "alarms.",
            )
        )
    lines += res["warnings"]
    return "\n".join(lines)
