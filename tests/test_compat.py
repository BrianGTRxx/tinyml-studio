import pandas as pd

from tinyml_studio.dataset import inspect_files


def _csv(p, cols, n=30):
    pd.DataFrame({c: range(n) for c in cols}).to_csv(p, index=False)
    return p


def test_same_structure_merges(tmp_path):
    a = _csv(tmp_path / "a.csv", ["v", "i", "p"])
    b = _csv(tmp_path / "b.csv", ["v", "i", "p"])
    c = inspect_files([a, b])["compatibility"]
    assert c["verdict"] == "same" and len(c["sets"]) == 1




def test_single_file(tmp_path):
    a = _csv(tmp_path / "a.csv", ["v", "i"])
    assert inspect_files([a])["compatibility"]["verdict"] == "single"
