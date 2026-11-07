"""Consejo de PyTorch según la GPU NVIDIA y lectura de temperatura de CPU (sin hardware real)."""

from tinyml_studio.hw.advice import (
    family_from_cc,
    gpu_family,
    torch_gpu_advice,
    wheel_tag,
)
from tinyml_studio.hw.cputemp import acpi_to_celsius


def test_family_by_compute_capability():
    assert family_from_cc(12.0) == "blackwell"
    assert family_from_cc(10.0) == "blackwell"
    assert family_from_cc(9.0) == "hopper"
    assert family_from_cc(8.9) == "ada"
    assert family_from_cc(8.6) == "ampere"
    assert family_from_cc(7.5) == "turing"
    assert family_from_cc(7.0) == "volta"
    assert family_from_cc(6.1) == "pascal"
    assert family_from_cc(5.2) == "legacy"


def test_family_by_name_fallback():
    assert gpu_family("NVIDIA GeForce RTX 5090") == "blackwell"
    assert gpu_family("NVIDIA GeForce RTX 4050 Laptop GPU") == "ada"
    assert gpu_family("NVIDIA GeForce RTX 3060") == "ampere"
    assert gpu_family("NVIDIA GeForce RTX 2070") == "turing"
    assert gpu_family("NVIDIA GeForce GTX 1060") == "pascal"
    assert gpu_family("Tesla V100") == "volta"
    assert gpu_family("NVIDIA H100") == "hopper"
    assert gpu_family("whatever", cc=8.9) == "ada"  # la capacidad manda sobre el nombre


def test_wheel_tag():
    assert wheel_tag("blackwell") == "cu128"
    for fam in ("ada", "ampere", "turing", "volta", "pascal", "hopper"):
        assert wheel_tag(fam) == "cu126"


def test_cpu_only_torch_with_supported_gpu_gets_advice():
    gpus = [{"name": "NVIDIA GeForce RTX 4050 Laptop GPU", "driver": "581.15", "cc": 8.9}]
    msgs = torch_gpu_advice(gpus, "2.9.0+cpu", "nt")
    assert len(msgs) == 1 and "cu126" in msgs[0] and "RTX 4050" in msgs[0]


def test_blackwell_with_old_cuda_wheel_is_flagged():
    gpus = [{"name": "NVIDIA GeForce RTX 5090", "driver": "575.51", "cc": 12.0}]
    msgs = torch_gpu_advice(gpus, "2.6.0+cu126", "nt")
    assert len(msgs) >= 1 and all("cu128" in m for m in msgs[:1]) and "cu126" in msgs[0]


def test_no_advice_when_cuda_wheel_matches():
    gpus = [{"name": "NVIDIA GeForce RTX 3060", "driver": "560.94", "cc": 8.6}]
    assert torch_gpu_advice(gpus, "2.9.0+cu126", "nt") == []


def test_old_driver_warned_and_legacy_gpu_has_no_wheel():
    gpus = [{"name": "NVIDIA GeForce RTX 5080", "driver": "552.12", "cc": 12.0}]
    msgs = torch_gpu_advice(gpus, "2.9.0+cpu", "nt")
    assert len(msgs) == 2 and "552.12" in msgs[1] and "570.65" in msgs[1]
    old = [{"name": "GeForce GTX 780", "driver": "470.0", "cc": 3.5}]
    msgs = torch_gpu_advice(old, "2.9.0+cpu", "nt")
    assert len(msgs) == 1 and "GTX 780" in msgs[0] and "pip install" not in msgs[0]


def test_cpu_temperature_helpers():
    assert acpi_to_celsius(3232) == 50.1  # 323.2 K
    assert acpi_to_celsius([3000, 3232]) == 50.1
    assert acpi_to_celsius(0) is None
