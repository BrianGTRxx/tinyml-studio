import pytest

from tinyml_studio.io import read_influx_csv


def test_plain_csv(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
    r = read_influx_csv(f)
    assert r.annotation_rows == 0 and r.separator == ";"
    assert r.df.shape == (2, 2)


def test_only_annotations_fails(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("#group,false\n#datatype,double\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_influx_csv(f)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_influx_csv(tmp_path / "no.csv")
