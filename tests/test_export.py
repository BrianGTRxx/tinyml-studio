import json
import shutil

import numpy as np
import pandas as pd
import pytest

from tinyml_studio.dataset import inspect_files
from tinyml_studio.export import export_run
from tinyml_studio.export.cgen import cf, thr_f32
from tinyml_studio.export.classic import _c_str, budget, verify_c
from tinyml_studio.train import train_models

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="sin compilador C")
RNG = np.random.default_rng(11)


def test_float_literals_and_thresholds():
    assert cf(1) == "1.0f" and cf(float("nan")) == "NAN" and cf(float("-inf")) == "(-INFINITY)"
    assert np.float32(float(cf(0.1)[:-1])) == np.float32(0.1)  # ida y vuelta exacta
    # x <= thr_f32(t) debe equivaler a x <= t para cualquier x float32 (incluso pegado al umbral)
    for _ in range(500):
        a = np.float32(RNG.normal() * 10)
        b = np.nextafter(a, np.float32(np.inf))
        t = (float(a) + float(b)) / 2  # punto medio, como sklearn
        tf = thr_f32(t)
        for x in (a, b):
            assert (float(x) <= t) == (x <= tf)


def test_c_string_escaping():
    s = _c_str('a"b\\c?ñ≈0')
    # comilla, barra, '?' (trigrafos) y UTF-8 en octal de 3 dígitos; el '0' final no se funde
    assert s == '"a\\042b\\134c\\077\\303\\261\\342\\211\\2100"'
    assert "\\x" not in s


def test_budget_verdicts():
    b = budget(500, 100, flash_kb=1.0, ram_kb=None)
    assert b["flash_data"]["verdict"] == "cabe"
    assert b["ram_working"]["verdict"] == "sin presupuesto"
    assert budget(900, 1, 1.0, 1.0)["flash_data"]["verdict"].startswith("justo")
    assert budget(2000, 1, 1.0, 1.0)["flash_data"]["verdict"] == "no cabe"


def _run(tmp_path, csv_builder, models, **kw):
    csv_builder(tmp_path / "d.csv")
    recipe = inspect_files([tmp_path / "d.csv"])["recipe"]
    train_models(recipe, models, seed=0, out_dir=tmp_path / "run", **kw)
    return tmp_path / "run"


def _clf_csv(path, n=600):
    k = n // 3
    base = np.repeat([0.0, 5.0, 10.0], k)
    f1 = base + RNG.normal(0, 1.0, n)
    f1[7] = -999  # centinela -> se limpia en el dispositivo
    f2 = RNG.normal(0, 1, n) + base / 2
    f2[20:30] = np.nan
    pd.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "tipo": RNG.choice(["u", "v", "w"], n),
            "label": np.repeat(["a", "b", "c"], k),
        }
    ).to_csv(path, index=False)


@pytest.mark.parametrize(
    "model,exact",
    [
        ("decision_tree", True),
        ("random_forest", True),
        ("logreg", False),
        ("svm", False),
        ("mlp", False),
    ],
)
def test_export_classifiers_match_sklearn(tmp_path, model, exact):
    run = _run(tmp_path, _clf_csv, [model])
    rep = export_run(run, tmp_path / "c", flash_kb=512, ram_kb=64)
    v = rep["verification"]
    assert v["compile_ok"] and v["compile_warnings"] == [], v
    assert v["compiles_as_cpp"] is True
    assert v["vectors"] >= 30
    assert v["features_max_rel_diff"] < 1e-5
    assert v["model_match"] >= (1.0 if exact else 0.995)
    assert v["raw_to_class_match"] >= 0.99
    assert rep["budget"]["flash_data"]["verdict"] == "cabe"
    assert not (tmp_path / "c" / "verify_host").exists()  # no se entrega el binario de PC
    meta = json.loads((tmp_path / "c" / "model_meta.json").read_text())
    assert meta["inputs"]["num"] == ["f1", "f2"] and "tipo" in meta["inputs"]["cat"]
    assert meta["inputs"]["cat"]["tipo"]["codes"] == {"u": 0, "v": 1, "w": 2}
    assert meta["classes"] == ["a", "b", "c"]
    src = (tmp_path / "c" / "tm_model.c").read_text()
    assert "-999.0f" in src  # el centinela viaja al firmware


def test_verification_detects_a_corrupted_model(tmp_path):
    run = _run(tmp_path, _clf_csv, ["decision_tree"])
    export_run(run, tmp_path / "c")
    src = (tmp_path / "c" / "tm_model.c").read_text()
    i = src.index("tm_thr[")
    j = src.index("{", i)
    k = src.index(",", j)
    # el primer umbral del árbol (raíz) se desplaza mucho: el modelo C deja de coincidir
    bad = src[: j + 1] + "\n  1000.0f" + src[k:]
    (tmp_path / "c" / "tm_model.c").write_text(bad)
    v = verify_c(tmp_path / "c")
    assert v["compile_ok"] and v["model_match"] < 0.9


def test_export_regression(tmp_path):
    def csv(path, n=500):
        x = RNG.normal(size=n)
        pd.DataFrame(
            {"x1": x, "x2": RNG.normal(size=n), "SalePrice": 3 * x + 100 + RNG.normal(0, 0.1, n)}
        ).to_csv(path, index=False)

    for model in ("decision_tree", "random_forest", "linear", "mlp"):
        sub = tmp_path / model
        sub.mkdir()
        run = _run(sub, csv, [model])
        rep = export_run(run, sub / "c")
        v = rep["verification"]
        assert v["compile_ok"] and v["compile_warnings"] == [], (model, v)
        assert v["model_match"] >= 0.99, (model, v)
        assert v["vs_full_python_pipeline"] >= 0.95, (model, v)


def test_unknown_model_and_missing_file(tmp_path):
    run = _run(tmp_path, _clf_csv, ["decision_tree"])
    with pytest.raises(FileNotFoundError):
        export_run(run, tmp_path / "c", "random_forest")




def test_c_reports_inputs_outside_the_training_range(tmp_path):
    import subprocess

    run = _run(tmp_path, _clf_csv, ["decision_tree"])
    out = tmp_path / "c"
    rep = export_run(run, out, "decision_tree")
    assert rep["verification"]["model_match"] == 1.0  # el C sigue siendo idéntico al modelo
    meta = json.loads((out / "model_meta.json").read_text(encoding="utf-8"))
    r = meta["input_ranges"]["f1"]
    assert "tm_out_of_range" in (out / "tm_model.h").read_text(encoding="utf-8")
    driver = out / "oor.c"
    driver.write_text(
        '#include <math.h>\n#include <stdio.h>\n#include "tm_model.h"\n'
        "int main(void) {\n"
        f"  float ok[TM_N_NUM] = {{{(r['min'] + r['max']) / 2:.6f}f, 0.0f}};\n"
        f"  float bad[TM_N_NUM] = {{{r['max'] + 1000:.6f}f, 0.0f}};\n"
        "  float miss[TM_N_NUM] = {NAN, NAN};\n"
        '  printf("%d %d %d\\n", tm_out_of_range(ok),'
        " tm_out_of_range(bad), tm_out_of_range(miss));\n"
        "  return 0;\n}\n",
        encoding="utf-8",
    )
    exe = out / "oor.exe"
    subprocess.run(["cc", "-O2", "tm_model.c", "oor.c", "-lm", "-o", str(exe)], cwd=out, check=True)
    got = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()
    assert got == ["0", "1", "0"]
