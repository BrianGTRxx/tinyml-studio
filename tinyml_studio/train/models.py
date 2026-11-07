"""Catálogo de modelos clásicos (scikit-learn) con complejidad medible."""

from __future__ import annotations

from typing import Any

import numpy as np

from tinyml_studio.i18n import tr

CLASSIFIERS = (
    "decision_tree",
    "random_forest",
    "logreg",
    "svm",
    "mlp",
    "knn",
    "gaussian_nb",
    "extra_trees",
    "gradient_boosting",
    "lda",
    "linear_svm",
)
REGRESSORS = (
    "decision_tree",
    "random_forest",
    "linear",
    "mlp",
    "knn",
    "extra_trees",
    "gradient_boosting",
    "svr",
    "lasso",
    "elasticnet",
)
# conjunto por defecto cuando no se piden modelos concretos (el resto es opt-in: knn y SVR son
# lentos con muchas filas y el banco ampliado se elige a propósito en la interfaz)
DEFAULT_SET = {
    "classification": ("decision_tree", "random_forest", "logreg", "svm", "mlp"),
    "regression": ("decision_tree", "random_forest", "linear", "mlp"),
}
# modelos con tope de filas de entrenamiento (muestreo aleatorio; ver train/run.py)
ROW_CAPPED = ("svm", "svr")
MLP_SOLVERS = ("adam", "sgd", "lbfgs")
LINEAR_FAMILY = ("logreg", "linear", "lda", "linear_svm", "lasso", "elasticnet")

DEFAULTS: dict[str, dict[str, Any]] = {
    "decision_tree": {"max_depth": 10, "min_samples_leaf": 5},
    "random_forest": {"n_estimators": 50, "max_depth": 10, "min_samples_leaf": 3},
    "logreg": {"C": 1.0, "max_iter": 500},
    "linear": {"alpha": 1.0},
    "svm": {"C": 1.0, "kernel": "rbf", "max_train_rows": 15000},
    "mlp": {
        "hidden_layer_sizes": [32, 16],
        "max_iter": 200,
        "early_stopping": True,
        "solver": "adam",  # también "sgd" | "lbfgs" (con lbfgs early_stopping no aplica)
    },
    "knn": {"n_neighbors": 5, "weights": "distance"},
    "gaussian_nb": {"var_smoothing": 1e-9},
    "extra_trees": {"n_estimators": 50, "max_depth": 10, "min_samples_leaf": 3},
    "gradient_boosting": {"max_iter": 100, "max_depth": 4, "learning_rate": 0.1},
    "lda": {},
    "linear_svm": {"C": 1.0, "max_iter": 2000},
    "svr": {"C": 1.0, "kernel": "rbf", "max_train_rows": 15000},
    "lasso": {"alpha": 0.1},
    "elasticnet": {"alpha": 0.1, "l1_ratio": 0.5},
}


def meta() -> dict[str, dict[str, Any]]:
    """Ficha por modelo para la interfaz: familia, si exporta a C y si conviene escalar."""

    def m(family, exportable_c, needs_scaling, es, en):
        return {
            "family": family,
            "exportable_c": exportable_c,
            "needs_scaling": needs_scaling,
            "note": tr(es, en),
        }

    return {
        "decision_tree": m(
            "tree",
            True,
            False,
            "Árbol único: muy pequeño e interpretable.",
            "Single tree: very small and interpretable.",
        ),
        "random_forest": m(
            "ensemble",
            True,
            False,
            "Muchos árboles: robusto pero ocupa más flash.",
            "Many trees: robust but uses more flash.",
        ),
        "extra_trees": m(
            "ensemble",
            True,
            False,
            "Como el bosque, con cortes aleatorios: entrena más rápido.",
            "Like a forest with random splits: trains faster.",
        ),
        "gradient_boosting": m(
            "ensemble",
            False,
            False,
            "Muy preciso en tablas; sin exportación a C todavía.",
            "Very accurate on tabular data; no C export yet.",
        ),
        "logreg": m(
            "linear",
            True,
            True,
            "Lineal: diminuto y rápido en el micro.",
            "Linear: tiny and fast on the MCU.",
        ),
        "linear": m(
            "linear",
            True,
            True,
            "Regresión lineal (Ridge): la base más simple.",
            "Linear regression (Ridge): the simplest baseline.",
        ),
        "lasso": m(
            "linear",
            True,
            True,
            "Lineal que anula variables poco útiles.",
            "Linear model that zeroes out weak features.",
        ),
        "elasticnet": m(
            "linear",
            True,
            True,
            "Mezcla de Lasso y Ridge.",
            "Mix of Lasso and Ridge.",
        ),
        "lda": m(
            "linear",
            True,
            False,
            "Análisis discriminante lineal: rápido, supone clases gaussianas.",
            "Linear discriminant analysis: fast, assumes Gaussian classes.",
        ),
        "linear_svm": m(
            "linear",
            True,
            True,
            "SVM lineal: frontera plana, muy ligera.",
            "Linear SVM: flat boundary, very light.",
        ),
        "svm": m(
            "kernel",
            True,
            True,
            "Kernel RBF: la flash crece con los vectores de soporte.",
            "RBF kernel: flash grows with the support vectors.",
        ),
        "svr": m(
            "kernel",
            False,
            True,
            "SVM de regresión (RBF); sin exportación a C todavía.",
            "Regression SVM (RBF); no C export yet.",
        ),
        "knn": m(
            "neighbors",
            False,
            True,
            "Guarda todo el entrenamiento: no cabe en un micro.",
            "Stores the whole training set: does not fit on an MCU.",
        ),
        "gaussian_nb": m(
            "bayes",
            True,
            False,
            "Naive Bayes gaussiano: diminuto, supone variables independientes.",
            "Gaussian naive Bayes: tiny, assumes independent features.",
        ),
        "mlp": m(
            "neural",
            True,
            True,
            "Red neuronal pequeña (scikit-learn).",
            "Small neural network (scikit-learn).",
        ),
    }


def available(task: str) -> tuple[str, ...]:
    return CLASSIFIERS if task == "classification" else REGRESSORS


def build(name: str, task: str, seed: int, params: dict | None = None):
    """Devuelve (estimador, parámetros efectivos). ``class_weight='balanced'`` si aplica."""
    if name not in available(task):
        raise ValueError(
            tr(
                "modelo '{name}' no disponible para {task}: {av}",
                "model '{name}' not available for {task}: {av}",
                name=name,
                task=task,
                av=available(task),
            )
        )
    p = {**DEFAULTS[name], **(params or {})}
    clf = task == "classification"
    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        kw = {k: v for k, v in p.items()}
        est = (
            DecisionTreeClassifier(class_weight="balanced", random_state=seed, **kw)
            if clf
            else DecisionTreeRegressor(random_state=seed, **kw)
        )
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        est = (
            RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, **p)
            if clf
            else RandomForestRegressor(random_state=seed, n_jobs=-1, **p)
        )
    elif name == "logreg":
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression(class_weight="balanced", random_state=seed, **p)
    elif name == "linear":
        from sklearn.linear_model import Ridge

        est = Ridge(random_state=seed, **p)
    elif name == "svm":
        from sklearn.svm import SVC

        kw = {k: v for k, v in p.items() if k != "max_train_rows"}
        est = SVC(class_weight="balanced", random_state=seed, **kw)
    elif name == "knn":
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        est = KNeighborsClassifier(**p) if clf else KNeighborsRegressor(**p)
    elif name == "gaussian_nb":
        from sklearn.naive_bayes import GaussianNB

        est = GaussianNB(**p)
    elif name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        est = (
            ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, **p)
            if clf
            else ExtraTreesRegressor(random_state=seed, n_jobs=-1, **p)
        )
    elif name == "gradient_boosting":  # histograma: mucho más barato que GradientBoosting clásico
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        est = (
            HistGradientBoostingClassifier(
                class_weight="balanced", early_stopping=False, random_state=seed, **p
            )
            if clf
            else HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **p)
        )
    elif name == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        est = LinearDiscriminantAnalysis(**p)
    elif name == "linear_svm":
        from sklearn.svm import LinearSVC

        est = LinearSVC(class_weight="balanced", dual="auto", random_state=seed, **p)
    elif name == "svr":
        from sklearn.svm import SVR

        est = SVR(**{k: v for k, v in p.items() if k != "max_train_rows"})
    elif name == "lasso":
        from sklearn.linear_model import Lasso

        est = Lasso(random_state=seed, **p)
    elif name == "elasticnet":
        from sklearn.linear_model import ElasticNet

        est = ElasticNet(random_state=seed, **p)
    else:  # mlp
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        if p.get("solver", "adam") not in MLP_SOLVERS:
            raise ValueError(
                tr(
                    "solver '{s}' no soportado para el MLP: {ok}",
                    "solver '{s}' not supported for the MLP: {ok}",
                    s=p["solver"],
                    ok=MLP_SOLVERS,
                )
            )
        kw = {**p, "hidden_layer_sizes": tuple(p["hidden_layer_sizes"])}
        est = (
            MLPClassifier(random_state=seed, **kw) if clf else MLPRegressor(random_state=seed, **kw)
        )
    return est, p


def complexity(name: str, est) -> dict[str, int]:
    """Tamaño en unidades propias del modelo (NO es memoria de flash: lo estima el exportador)."""
    if name == "decision_tree":
        return {"nodes": int(est.tree_.node_count), "depth": int(est.get_depth())}
    if name in ("random_forest", "extra_trees"):
        return {
            "trees": len(est.estimators_),
            "nodes": int(sum(t.tree_.node_count for t in est.estimators_)),
        }
    if name in LINEAR_FAMILY:
        return {"parameters": int(np.size(est.coef_) + np.size(est.intercept_))}
    if name == "svm":
        return {"support_vectors": int(est.n_support_.sum())}
    if name == "svr":
        return {"support_vectors": int(len(est.support_))}
    if name == "knn":
        return {"stored_rows": int(est.n_samples_fit_)}
    if name == "gaussian_nb":
        return {"parameters": int(est.theta_.size + est.var_.size + est.class_prior_.size)}
    if name == "gradient_boosting":
        preds = [p for stage in est._predictors for p in stage]
        return {"trees": len(preds), "nodes": int(sum(len(p.nodes) for p in preds))}
    return {
        "parameters": int(sum(w.size for w in est.coefs_) + sum(b.size for b in est.intercepts_))
    }


def importances(name: str, est, names: list[str]) -> list[dict]:
    if hasattr(est, "feature_importances_"):
        v = np.asarray(est.feature_importances_, dtype=float)
    elif name in LINEAR_FAMILY:
        v = np.abs(np.atleast_2d(est.coef_)).mean(0)
        v = v / v.sum() if v.sum() else v
    elif name == "gaussian_nb":  # separación de medias entre clases / desviación típica media
        v = est.theta_.std(0) / np.sqrt(est.var_.mean(0))
        v = v / v.sum() if v.sum() else v
    else:
        return []
    order = np.argsort(-v)[:8]
    return [{"feature": names[i], "importance": round(float(v[i]), 4)} for i in order]
