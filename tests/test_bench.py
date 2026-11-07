"""Benchmark de modelos: tabla, regla 1-SE, McNemar, bootstrap, CV por grupos y API."""

import json
import math
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tinyml_studio.api import create_app
from tinyml_studio.dataset import inspect_files
from tinyml_studio.train import bench as B
from tinyml_studio.train.curves import (
    bootstrap_f1_macro_ci,
    classification_quality,
    matthews_corrcoef,
)

RNG = np.random.default_rng(11)
FAST = ["decision_tree", "logreg"]


def _csv(path, n=450, noise=1.2, groups=False):
    k = n // 3
    base = np.repeat([0.0, 3.0, 6.0], k)
    d = {
        "f1": base + RNG.normal(0, noise, n),
        "f2": RNG.normal(0, 1, n) + base / 2,
        "label": np.repeat(["a", "b", "c"], k),
    }
    if groups:
        d["dev"] = np.tile([f"d{i}" for i in range(9)], n // 9 + 1)[:n]
    pd.DataFrame(d).to_csv(path, index=False)


@pytest.fixture(scope="module")
def recipe(tmp_path_factory):
    p = tmp_path_factory.mktemp("bench")
    _csv(p / "d.csv")
    return inspect_files([p / "d.csv"])["recipe"]


@pytest.fixture(scope="module")
def result(recipe, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    return B.run_benchmark(
        recipe, models=FAST + ["random_forest"], seed=0, n_boot=100, out_dir=out, cv_folds=3
    )


# ------------------------------------------------------------------ statistics


def test_mcc_matches_sklearn():
    from sklearn.metrics import matthews_corrcoef as sk

    r = np.random.default_rng(0)
    for k in (2, 3, 6):
        y = r.integers(0, k, 400)
        p = np.where(r.random(400) < 0.6, y, r.integers(0, k, 400))
        cm = np.zeros((k, k), int)
        np.add.at(cm, (y, p), 1)
        assert matthews_corrcoef(cm) == pytest.approx(sk(y, p))
        q = classification_quality(y, p, None, [str(i) for i in range(k)])
        assert q["mcc"] == pytest.approx(sk(y, p))
    assert matthews_corrcoef(np.array([[5, 0], [5, 0]])) == 0.0


def test_exact_mcnemar_hand_calc():
    # b=2, c=10: n=12, p = 2 * P(X<=2) = 2*(1+12+66)/4096
    assert B.exact_mcnemar(2, 10) == pytest.approx(2 * 79 / 4096)
    assert B.exact_mcnemar(5, 5) == 1.0
    assert B.exact_mcnemar(0, 0) == 1.0
    assert B.exact_mcnemar(0, 8) == pytest.approx(2 / 256)
    assert B.exact_mcnemar(3, 7) == B.exact_mcnemar(7, 3)
    adj = B.holm_adjust([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_is_deterministic_and_contains_point():
    r = np.random.default_rng(1)
    y = r.integers(0, 3, 200)
    p = np.where(r.random(200) < 0.8, y, r.integers(0, 3, 200))
    a = bootstrap_f1_macro_ci(y, p, 3, n_boot=150, seed=5)
    b = bootstrap_f1_macro_ci(y, p, 3, n_boot=150, seed=5)
    c = bootstrap_f1_macro_ci(y, p, 3, n_boot=150, seed=6)
    assert a == b and a != c
    assert a["lo"] <= a["point"] <= a["hi"]
    assert a["hi"] - a["lo"] < 0.3


def test_one_se_rule_prefers_smaller_exportable_of_tied_models():
    rows = [
        {"name": "big", "score": 0.90, "se": 0.03, "fits_flash": None, "exportable": True,
         "size_bytes": 90_000, "predict_us": 5.0},
        {"name": "small", "score": 0.88, "se": 0.03, "fits_flash": None, "exportable": True,
         "size_bytes": 2_000, "predict_us": 9.0},
        {"name": "worse", "score": 0.70, "se": 0.03, "fits_flash": None, "exportable": True,
         "size_bytes": 100, "predict_us": 1.0},
        {"name": "noexp", "score": 0.89, "se": 0.03, "fits_flash": None, "exportable": False,
         "size_bytes": 10, "predict_us": 1.0},
    ]  # fmt: skip
    s = B.select_model(rows)
    assert s["best"] == "big" and s["selected"] == "small"
    assert set(s["within_1se"]) == {"big", "small", "noexp"}
    # sin error estándar: gana el mejor puntaje
    assert B.select_model([{**r, "se": 0.0} for r in rows])["selected"] == "big"
    # presupuesto de flash: lo que no cabe se marca y no compite (pero best sigue siendo global)
    rows[1]["fits_flash"] = False
    rows[0]["fits_flash"] = False
    s = B.select_model(rows)
    assert s["selected"] in ("noexp", "worse") and s["best"] == "big"
    # lower-is-better
    low = [{**r, "score": 1 - r["score"]} for r in rows]
    for r in low:
        r["fits_flash"] = None
    assert B.select_model(low, higher_better=False)["selected"] == "small"


def test_pareto_front():
    pts = {"a": (0.9, 100), "b": (0.9, 50), "c": (0.7, 10), "d": (0.6, 60)}
    assert set(B.pareto_front(pts)) == {"b", "c"}


# ------------------------------------------------------------------ CV folds


def test_group_cv_does_not_leak_groups():
    n = 300
    df = pd.DataFrame({"dev": np.repeat([f"g{i}" for i in range(10)], 30)})
    y = np.tile([0, 1, 2], 100)
    f = B.make_cv_folds({"strategy": "group", "group_column": "dev"}, df, y, 5, 0, "classification")
    assert f["scheme"] == "group" and f["k"] == 5
    seen = []
    for tr_i, va_i in f["folds"]:
        gt, gv = set(df["dev"][tr_i]), set(df["dev"][va_i])
        assert not gt & gv and len(tr_i) + len(va_i) == n
        seen += list(gv)
    assert sorted(seen) == sorted(set(df["dev"]))  # cada grupo valida exactamente una vez


def test_temporal_cv_blocks_are_contiguous_and_stratified_is_balanced():
    df = pd.DataFrame({"t": np.arange(100), "dev": np.repeat(["a", "b"], 50)})
    y = np.tile([0, 1], 50)
    spec = {"strategy": "temporal_blocks", "group_column": "dev", "time_column": "t"}
    f = B.make_cv_folds(spec, df, y, 4, 0, "classification")
    assert f["scheme"] == "temporal_blocks"
    for _, va in f["folds"]:
        for g in ("a", "b"):
            t = np.sort(df["t"][va][df["dev"][va] == g].to_numpy())
            assert (np.diff(t) == 1).all()  # tramo contiguo
    g = B.make_cv_folds({"strategy": "stratified"}, df, y, 5, 0, "classification")
    assert g["scheme"] == "stratified"
    assert all(abs(y[va].mean() - 0.5) < 0.05 for _, va in g["folds"])


# ------------------------------------------------------------------ end to end


def test_leaderboard_shape_ranks_and_selection(result):
    r = result
    assert r["task"] == "classification"
    names = [row["name"] for row in r["leaderboard"]]
    assert set(names) == set(FAST + ["random_forest"])
    assert [row["rank"] for row in r["leaderboard"]] == [1, 2, 3]
    sel = r["selection"]
    assert sel["selected"] in names and sel["metric"] == "f1_macro"
    assert sel["source"] == "cv" and sel["rule"] == "one_standard_error"
    assert sel["selected"] in sel["within_1se"]
    for row in r["leaderboard"]:
        for s in ("train", "val", "test"):
            assert set(row["metrics"][s]) >= {"accuracy", "f1_macro", "mcc", "mAP", "roc_auc_macro"}
        assert row["metrics"]["test"]["mcc"] > 0.5
        assert row["cv"]["folds"] == 3 and len(row["cv"]["scores"]) == 3
        assert row["test_ci"]["lo"] <= row["metrics"]["test"]["f1_macro"] <= row["test_ci"]["hi"]
        assert row["run_subdir"] == "classical" and row["exportable"] is True
        assert isinstance(row["statistically_tied_with_best"], bool)
        assert row["ranks"]["f1_macro"] >= 1
    assert r["rank_by_metric"]["f1_macro"][0] in names
    assert set(r["pareto"]) <= set(names) and r["pareto"]
    # el test no ordena ni elige
    assert r["test_comparison"]["used_for_selection"] is False
    ref = r["test_comparison"]["reference"]
    assert ref == sel["selected"]
    ref_row = next(x for x in r["leaderboard"] if x["name"] == ref)
    assert ref_row["vs_selected_test"] is None
    other = next(x for x in r["leaderboard"] if x["name"] != ref)
    assert 0.0 <= other["vs_selected_test"]["mcnemar_p"] <= 1.0
    assert any("Cawley" in w for w in r["warnings"])
    assert r["summary"] and r["summary_es"] == r["summary"]


def test_mcnemar_in_result_matches_hand_calc(recipe):
    r = B.run_benchmark(recipe, models=FAST, seed=0, n_boot=50)
    ref = r["selection"]["selected"]
    other = next(x for x in r["leaderboard"] if x["name"] != ref)
    # recalcula b y c a mano desde los contadores y el p exacto binomial
    v = other["vs_selected_test"]
    b, c = v["reference_only_correct"], v["other_only_correct"]
    n, kk = b + c, min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(kk + 1)) / 2**n) if n else 1.0
    assert v["mcnemar_p"] == pytest.approx(p, abs=1e-6)


def test_reproducible_and_deterministic_bootstrap(recipe):
    a = B.run_benchmark(recipe, models=FAST, seed=3, n_boot=60)
    b = B.run_benchmark(recipe, models=FAST, seed=3, n_boot=60)
    for ra, rb in zip(a["leaderboard"], b["leaderboard"], strict=True):
        assert ra["test_ci"] == rb["test_ci"] and ra["metrics"] == rb["metrics"]
    assert a["selection"]["selected"] == b["selection"]["selected"]


def test_json_size_and_serializable(result):
    s = json.dumps(result)
    assert len(s) < 500_000


def test_flash_budget_flags_not_drops(recipe):
    r = B.run_benchmark(recipe, models=["decision_tree", "random_forest"], max_flash_kb=0.001)
    assert len(r["leaderboard"]) == 2
    assert all(x["fits_flash"] is False for x in r["leaderboard"])
    assert r["selection"]["none_fit_flash"] is True
    r2 = B.run_benchmark(recipe, models=["decision_tree", "random_forest"], max_flash_kb=100000)
    assert all(x["fits_flash"] is True for x in r2["leaderboard"])


def test_invalid_args(recipe):
    with pytest.raises(ValueError):
        B.run_benchmark(recipe, models=FAST, selection="r2")
    with pytest.raises(ValueError):
        B.run_benchmark(recipe, models=FAST, cv_folds=1)
    with pytest.raises(ValueError):
        B.run_benchmark(recipe, models=FAST, keras=[{"bogus": 1}])
    with pytest.raises(ValueError):
        B.run_benchmark(recipe, models=[])


def test_group_split_benchmark_cv_uses_groups(tmp_path):
    _csv(tmp_path / "g.csv", groups=True)
    rep = inspect_files([tmp_path / "g.csv"])
    recipe = rep["recipe"]
    recipe["roles"]["dev"] = "group"
    recipe["split"] = {"strategy": "group", "group_column": "dev", "ratios": [0.8, 0.1, 0.1]}
    r = B.run_benchmark(recipe, models=["decision_tree"], cv_folds=3, n_boot=30)
    assert r["cv"]["scheme"] == "group"


def test_regression_benchmark(tmp_path):
    n = 300
    x = RNG.normal(size=n)
    pd.DataFrame({"x1": x, "x2": RNG.normal(size=n), "y": 3 * x + RNG.normal(0, 0.3, n)}).to_csv(
        tmp_path / "h.csv", index=False
    )
    recipe = inspect_files([tmp_path / "h.csv"])["recipe"]
    r = B.run_benchmark(recipe, models=["linear", "decision_tree"], n_boot=40, cv_folds=3)
    assert r["selection"]["metric"] == "r2" and r["selection"]["selected"]
    assert r["leaderboard"][0]["metrics"]["test"]["r2"] > 0.8
    assert "mcc" not in r["leaderboard"][0]["metrics"]["test"]


def test_keras_preset_naming():
    ps = B.normalize_keras_presets([{"hidden": [8, 4]}, {"hidden": [8, 4]}, {"name": "big"}])
    assert [p["name"] for p in ps] == ["keras_mlp:h8-4", "keras_mlp:h8-4-2", "keras_mlp:big"]
    assert all(B.SUBDIR_RE.fullmatch(p["dir"]) for p in ps)


def test_keras_skipped_without_tensorflow(recipe, monkeypatch):
    monkeypatch.setattr(B, "_tf_available", lambda: False)
    r = B.run_benchmark(recipe, models=["decision_tree"], keras=[{"hidden": [4]}], n_boot=30)
    assert [s["name"] for s in r["models_skipped"]] == ["keras_mlp:h4"]
    assert len(r["leaderboard"]) == 1


def test_keras_in_benchmark(recipe, tmp_path):
    pytest.importorskip("tensorflow")
    r = B.run_benchmark(
        recipe,
        models=["decision_tree"],
        keras=[{"hidden": [8], "epochs": 5, "patience": 2}],
        cv_folds=3,
        n_boot=30,
        out_dir=tmp_path / "o",
    )
    row = next(x for x in r["leaderboard"] if x["name"].startswith("keras_mlp:"))
    assert row["family"] == "keras_mlp" and row["export_job"] == "export_tflite"
    assert row["run_subdir"] == "keras_mlp_h8" and (tmp_path / "o" / "keras_mlp_h8").is_dir()
    assert r["selection"]["source"] == "val"  # CV no cubre Keras: se explica en warnings
    assert any("CV" in w for w in r["warnings"])


# ------------------------------------------------------------------ API


TOKEN = "b" * 32


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def _wait(client, jid, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{jid}", headers=auth()).json()
        if j["state"] in ("succeeded", "failed", "cancelled", "interrupted"):
            return j
        time.sleep(0.3)
    raise AssertionError("timeout")


def test_benchmark_job_and_export_with_run_subdir(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _csv(data / "d.csv")
    recipe = inspect_files([data / "d.csv"])["recipe"]
    client = TestClient(
        create_app(TOKEN, [data], tmp_path / "ws"), base_url="http://127.0.0.1:8000"
    )

    def submit(kind, params, expect=202):
        r = client.post("/jobs", headers=auth(), json={"kind": kind, "params": params})
        assert r.status_code == expect, r.text
        return r.json()

    # validación estricta
    submit("benchmark", {"recipe": recipe, "cv_folds": 1}, 422)
    submit("benchmark", {"recipe": recipe, "cv_folds": 11}, 422)
    submit("benchmark", {"recipe": recipe, "bogus": 1}, 422)
    submit("benchmark", {"recipe": recipe, "keras": [{"epochs": 0}]}, 422)
    submit("benchmark", {"recipe": recipe, "selection": "nope"}, 422)

    job = submit(
        "benchmark",
        {"recipe": recipe, "models": ["decision_tree", "logreg"], "seed": 1, "n_boot": 40},
    )
    j = _wait(client, job["id"])
    assert j["state"] == "succeeded", j
    res = client.get(f"/jobs/{job['id']}/result", headers=auth()).json()["results"]
    assert res["leaderboard"] and res["out_dir"] is None
    assert len(json.dumps(res)) < 500_000
    chosen = res["selection"]["selected"]

    ok = submit(
        "export_c",
        {"run_job": job["id"], "run_subdir": "classical", "model": chosen},
    )
    e = _wait(client, ok["id"])
    assert e["state"] == "succeeded", e
    # trucos de ruta: rechazados al validar (patrón) o por no existir
    for bad in ("../classical", "a/b", "..", "x.y", "/etc"):
        submit("export_c", {"run_job": job["id"], "run_subdir": bad}, 422)
    ghost = submit("export_c", {"run_job": job["id"], "run_subdir": "nope"})
    assert _wait(client, ghost["id"])["state"] == "failed"
    # sin run_subdir en la raíz del benchmark no hay results.json: falla con claridad
    root = submit("export_c", {"run_job": job["id"]})
    assert _wait(client, root["id"])["state"] == "failed"


def test_benchmark_keras_preset_accepts_optimizer_keys():
    from tinyml_studio.train.bench import normalize_keras_presets

    out = normalize_keras_presets([{"hidden": [8], "optimizer": "adamw", "lr_schedule": "cosine"}])
    assert out[0]["cfg"]["optimizer"] == "adamw" and out[0]["cfg"]["lr_schedule"] == "cosine"
