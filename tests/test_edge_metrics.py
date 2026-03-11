# SCRIPT tests/test_edge_metrics.py Mars 4th, 2026. 

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Dict

import pytest


def test_cpu_only_force_cpu_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_CPU", "1")
    monkeypatch.delenv("OFFLINE_MODE", raising=False)

    from medgem_poc.edge_metrics import EdgeMetricsCollector

    collector = EdgeMetricsCollector(backend="unit", offline_mode=True)
    with collector.run(degradation_mode="NONE"):
        with collector.stage("pre"):
            _ = sum(range(10_000))
        with collector.stage("infer"):
            _ = sum(range(20_000))
        with collector.stage("post"):
            _ = sum(range(5_000))

    d = collector.report.to_dict()
    assert d["schema_version"]
    assert d["runtime"]["device"] == "cpu"
    assert d["resilience"]["fallback_used"] is True
    assert d["resilience"]["fallback_reason"] == "FORCE_CPU=1"
    assert d["latency_ms"]["total"] is not None

    s = collector.report.to_json()
    json.loads(s)


def test_mocked_cuda_path_collects_vram_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORCE_CPU", raising=False)

    from medgem_poc import edge_metrics as em

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def reset_peak_memory_stats(self) -> None:
            return None

        def memory_allocated(self) -> int:
            return 128 * 1024 * 1024

        def memory_reserved(self) -> int:
            return 256 * 1024 * 1024

        def max_memory_allocated(self) -> int:
            return 512 * 1024 * 1024

    fake_torch = SimpleNamespace(
        __version__="0.0.fake",
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="fake-cuda"),
    )

    monkeypatch.setattr(em, "_TORCH", fake_torch, raising=True)

    collector = em.EdgeMetricsCollector(backend="unit", offline_mode=True)
    with collector.run(degradation_mode="NONE"):
        with collector.stage("infer"):
            _ = sum(range(1000))

    mem = collector.report.to_dict()["memory"]
    assert mem["vram_allocated_mb"] is not None
    assert mem["vram_reserved_mb"] is not None
    assert mem["vram_peak_allocated_mb"] is not None
    assert collector.report.to_dict()["runtime"]["device"].startswith("cuda")

def test_set_degradation_mode_effective_updates_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORCE_CPU", raising=False)

    from medgem_poc.edge_metrics import EdgeMetricsCollector

    c = EdgeMetricsCollector(backend="unit", offline_mode=True)
    with c.run(degradation_mode="NONE"):
        with c.stage("infer"):
            _ = sum(range(1000))

    # effective degrade decided after run
    c.set_degradation_mode_effective("QC_ONLY")

    d = c.report.to_dict()
    res = d["resilience"]

    assert res.get("degradation_mode_effective") == "QC_ONLY"
    assert d["meta"].get("degradation_mode_effective") == "QC_ONLY"
    assert res.get("edge_readiness_status") in {"DEGRADED", "FALLBACK+DEGRADED"}

def test_set_degradation_mode_effective_with_force_cpu_yields_fallback_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORCE_CPU", "1")

    from medgem_poc.edge_metrics import EdgeMetricsCollector

    c = EdgeMetricsCollector(backend="unit", offline_mode=True)
    with c.run(degradation_mode="NONE"):
        with c.stage("infer"):
            _ = sum(range(1000))

    c.set_degradation_mode_effective("QC_ONLY")

    d = c.report.to_dict()
    res = d["resilience"]

    assert d["runtime"]["device"] == "cpu"
    assert res.get("fallback_used") is True
    assert res.get("fallback_reason") == "FORCE_CPU=1"
    assert res.get("degradation_mode_effective") == "QC_ONLY"
    assert d["meta"].get("degradation_mode_effective") == "QC_ONLY"
    assert res.get("edge_readiness_status") == "FALLBACK+DEGRADED"

def test_pretty_includes_degradation_mode_effective_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORCE_CPU", raising=False)

    from medgem_poc.edge_metrics import EdgeMetricsCollector

    c = EdgeMetricsCollector(backend="unit", offline_mode=True)
    with c.run(degradation_mode="NONE"):
        with c.stage("infer"):
            _ = sum(range(1000))

    c.set_degradation_mode_effective("QC_ONLY")

    s = c.pretty()
    assert "degradation_mode_effective:" in s
    assert "QC_ONLY" in s

import pytest  # si pas déjà importé

@pytest.mark.parametrize(
    "force_cpu,expected",
    [
        (False, "DEGRADED"),
        (True, "FALLBACK+DEGRADED"),
    ],
)
def test_pretty_includes_effective_mode_and_readiness_is_consistent(
    monkeypatch: pytest.MonkeyPatch,
    force_cpu: bool,
    expected: str,
) -> None:
    if force_cpu:
        monkeypatch.setenv("FORCE_CPU", "1")
    else:
        monkeypatch.delenv("FORCE_CPU", raising=False)

    from medgem_poc.edge_metrics import EdgeMetricsCollector

    c = EdgeMetricsCollector(backend="unit", offline_mode=True)
    with c.run(degradation_mode="NONE"):
        with c.stage("infer"):
            _ = sum(range(1000))

    c.set_degradation_mode_effective("QC_ONLY")

    s = c.pretty()
    assert "degradation_mode_effective:" in s
    assert "QC_ONLY" in s
    assert f"edge_readiness_status: {expected}" in s

    d = c.report.to_dict()
    assert d["resilience"]["edge_readiness_status"] == expected

def test_vram_is_none_when_device_is_cpu_even_if_cuda_available(monkeypatch):
    monkeypatch.setenv("FORCE_CPU", "1")

    from medgem_poc import edge_metrics as emod
    from medgem_poc.edge_metrics import EdgeMetricsCollector

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def memory_allocated():
            return 123 * 1024 * 1024

        @staticmethod
        def memory_reserved():
            return 456 * 1024 * 1024

        @staticmethod
        def max_memory_allocated():
            return 789 * 1024 * 1024

        @staticmethod
        def reset_peak_memory_stats():
            return None

    class _FakeTorch:
        __version__ = "fake"
        cuda = _FakeCuda()
        class version:
            cuda = "fake-cuda"

    monkeypatch.setattr(emod, "_TORCH", _FakeTorch, raising=True)

    c = EdgeMetricsCollector(backend="unit", offline_mode=True)
    with c.run(degradation_mode="NONE"):
        with c.stage("infer"):
            _ = sum(range(1000))

    d = c.report.to_dict()
    assert d["runtime"]["device"] == "cpu"
    mem = d["memory"]
    assert mem["vram_allocated_mb"] is None
    assert mem["vram_reserved_mb"] is None
    assert mem["vram_peak_allocated_mb"] is None

# TERMINUS DU SCRIPT tests/test_edge_metrics.py
