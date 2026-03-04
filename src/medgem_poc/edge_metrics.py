### --- file: src/medgem_poc/edge_metrics.py --- ### FEB 27th, 2026 11H37 VERSION 1.000
### --- file: src/medgem_poc/edge_metrics.py --- ### MARS 4th, 2026 16H27 VERSION 1.001 PATCH2 DEF EFF PATCH MINIMAL

from __future__ import annotations

import json
import os
import platform
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


SCHEMA_VERSION = "1.0.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float_ms(ns_delta: int) -> float:
    return float(ns_delta) / 1_000_000.0


def _try_import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def _try_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


_PSUTIL = _try_import_psutil()
_TORCH = _try_import_torch()


def _get_ram_rss_mb() -> Optional[float]:
    if _PSUTIL is None:
        return None
    try:
        proc = _PSUTIL.Process(os.getpid())
        rss = float(proc.memory_info().rss)
        return rss / (1024.0 * 1024.0)
    except Exception:
        return None


def _runtime_info(device: str, backend: str) -> Dict[str, Any]:
    torch_info: Dict[str, Any] = {"available": False}
    if _TORCH is not None:
        torch_info = {
            "available": True,
            "version": getattr(_TORCH, "__version__", None),
            "cuda_available": bool(
                getattr(getattr(_TORCH, "cuda", None), "is_available", lambda: False)()
            )
            if device.startswith("cuda")
            else bool(getattr(getattr(_TORCH, "cuda", None), "is_available", lambda: False)()),
        }
        try:
            torch_info["cuda_version"] = getattr(getattr(_TORCH, "version", None), "cuda", None)
        except Exception:
            torch_info["cuda_version"] = None

    return {
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "backend": backend,
        "device": device,
        "torch": torch_info,
    }


def _cuda_mem_snapshot_mb() -> Dict[str, Optional[float]]:
    if _TORCH is None:
        return {
            "vram_allocated_mb": None,
            "vram_reserved_mb": None,
            "vram_peak_allocated_mb": None,
        }
    cuda = getattr(_TORCH, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)) or not cuda.is_available():
        return {
            "vram_allocated_mb": None,
            "vram_reserved_mb": None,
            "vram_peak_allocated_mb": None,
        }
    try:
        allocated = float(cuda.memory_allocated()) / (1024.0 * 1024.0)
        reserved = float(cuda.memory_reserved()) / (1024.0 * 1024.0)
        peak = float(cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        return {
            "vram_allocated_mb": allocated,
            "vram_reserved_mb": reserved,
            "vram_peak_allocated_mb": peak,
        }
    except Exception:
        return {
            "vram_allocated_mb": None,
            "vram_reserved_mb": None,
            "vram_peak_allocated_mb": None,
        }


def _maybe_reset_cuda_peaks() -> None:
    if _TORCH is None:
        return
    cuda = getattr(_TORCH, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)) or not cuda.is_available():
        return
    try:
        cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _pick_device(force_cpu: bool) -> Tuple[str, bool, Optional[str]]:
    if force_cpu:
        return "cpu", True, "FORCE_CPU=1"

    if _TORCH is None:
        return "cpu", False, None

    cuda = getattr(_TORCH, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        return "cuda:0", False, None

    return "cpu", False, None


def _edge_readiness_status(fallback_used: bool, degradation_mode: str, offline_mode: bool) -> str:
    deg = (degradation_mode or "NONE").upper().strip()
    if fallback_used and deg != "NONE":
        return "FALLBACK+DEGRADED"
    if fallback_used:
        return "FALLBACK"
    if deg != "NONE":
        return "DEGRADED"
    if offline_mode:
        return "OFFLINE_OK"
    return "OK"


@dataclass
class EdgeMetricsReport:
    schema_version: str = SCHEMA_VERSION
    timestamp_utc: str = field(default_factory=_utc_now_iso)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    runtime: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Dict[str, Optional[float]] = field(default_factory=dict)
    memory: Dict[str, Optional[float]] = field(default_factory=dict)
    resilience: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp_utc": self.timestamp_utc,
            "run_id": self.run_id,
            "runtime": self.runtime,
            "latency_ms": self.latency_ms,
            "memory": self.memory,
            "resilience": self.resilience,
            "meta": self.meta,
            "errors": self.errors,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)


class EdgeMetricsCollector:
    """
    Independent edge metrics collector (no dependency on qc.py).

    Usage:
        collector = EdgeMetricsCollector(backend="torch", offline_mode=True)
        with collector.run(degradation_mode="NONE"):
            with collector.stage("pre"): ...
            with collector.stage("infer"): ...
            with collector.stage("post"): ...
        report = collector.report
    """

    def __init__(
        self,
        *,
        backend: str = "unknown",
        offline_mode: Optional[bool] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.backend = backend
        self.offline_mode = _env_flag("OFFLINE_MODE", False) if offline_mode is None else bool(offline_mode)
        self.force_cpu = _env_flag("FORCE_CPU", False)

        device, fb_used, fb_reason = _pick_device(self.force_cpu)
        self.device = device
        self.fallback_used = fb_used
        self.fallback_reason = fb_reason

        self._run_start_ns: Optional[int] = None
        self._run_end_ns: Optional[int] = None
        self._stage_ns: Dict[str, int] = {}
        self._stage_start_ns: Dict[str, int] = {}
        self._degradation_mode: str = "NONE"

        self.report = EdgeMetricsReport()
        self.report.meta = meta or {}

    @contextmanager
    def run(
        self,
        *,
        degradation_mode: str = "NONE",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Iterator["EdgeMetricsCollector"]:
        self._degradation_mode = degradation_mode or "NONE"
        self._run_start_ns = time.perf_counter_ns()
        _maybe_reset_cuda_peaks()

        try:
            yield self
        except Exception as e:
            self.report.errors.append({"stage": "run", "type": type(e).__name__, "message": str(e)})
            raise
        finally:
            self._run_end_ns = time.perf_counter_ns()
            self._finalize(extra_meta=extra_meta)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        n = (name or "").strip().lower()
        if not n:
            raise ValueError("stage name must be non-empty")
        start = time.perf_counter_ns()
        self._stage_start_ns[n] = start
        try:
            yield
        except Exception as e:
            self.report.errors.append({"stage": n, "type": type(e).__name__, "message": str(e)})
            raise
        finally:
            end = time.perf_counter_ns()
            self._stage_ns[n] = self._stage_ns.get(n, 0) + (end - start)

    def _finalize(self, *, extra_meta: Optional[Dict[str, Any]]) -> None:
        if extra_meta:
            self.report.meta.update(extra_meta)

        total_ms = None
        if self._run_start_ns is not None and self._run_end_ns is not None:
            total_ms = _safe_float_ms(self._run_end_ns - self._run_start_ns)

        self.report.runtime = _runtime_info(self.device, self.backend)

        self.report.latency_ms = {
            "pre": _safe_float_ms(self._stage_ns["pre"]) if "pre" in self._stage_ns else None,
            "infer": _safe_float_ms(self._stage_ns["infer"]) if "infer" in self._stage_ns else None,
            "post": _safe_float_ms(self._stage_ns["post"]) if "post" in self._stage_ns else None,
            "total": total_ms,
        }

        mem: Dict[str, Optional[float]] = {"ram_rss_mb": _get_ram_rss_mb()}
        mem.update(_cuda_mem_snapshot_mb())
        self.report.memory = mem

        deg = self._degradation_mode or "NONE"
        self.report.resilience = {
            "offline_mode": self.offline_mode,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "degradation_mode": deg,
            "edge_readiness_status": _edge_readiness_status(self.fallback_used, deg, self.offline_mode),
        }

    def set_degradation_mode_effective(self, mode: Optional[str]) -> None:
        """
        Set the effective degradation mode decided by the pipeline after the run
        (e.g., MODEL_NOT_AVAILABLE -> QC_ONLY) and synchronize resilience fields.

        This does not modify QC outputs; it only updates EdgeMetrics report fields.
        """
        m = (mode or "NONE").strip() or "NONE"
        self.report.meta["degradation_mode_effective"] = m
        self.report.resilience["degradation_mode_effective"] = m
        self.report.resilience["edge_readiness_status"] = _edge_readiness_status(
            bool(self.report.resilience.get("fallback_used")),
            m,
            bool(self.report.resilience.get("offline_mode")),
        )

    def pretty(self) -> str:
        r = self.report.to_dict()
        lat = r.get("latency_ms", {}) or {}
        mem = r.get("memory", {}) or {}
        res = r.get("resilience", {}) or {}
        run = r.get("runtime", {}) or {}
        dev = run.get("device", "unknown")
        backend = run.get("backend", "unknown")

        deg_eff = res.get("degradation_mode_effective")
        if deg_eff is None:
            deg_eff = (r.get("meta", {}) or {}).get("degradation_mode_effective")
        if deg_eff is None:
            deg_eff = res.get("degradation_mode")

        ready_eff = _edge_readiness_status(
            bool(res.get("fallback_used")),
            str(deg_eff or "NONE"),
            bool(res.get("offline_mode")),
        )

        def _fmt(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{v:.3f}"

        lines = [
            "--- EDGE METRICS REPORT ---",
            f"schema_version: {r.get('schema_version')}",
            f"timestamp_utc:  {r.get('timestamp_utc')}",
            f"run_id:         {r.get('run_id')}",
            f"backend/device: {backend} / {dev}",
            "",
            "LATENCY (ms):",
            f"  pre:   {_fmt(lat.get('pre'))}",
            f"  infer: {_fmt(lat.get('infer'))}",
            f"  post:  {_fmt(lat.get('post'))}",
            f"  total: {_fmt(lat.get('total'))}",
            "",
            "MEMORY:",
            f"  ram_rss_mb:           {_fmt(mem.get('ram_rss_mb'))}",
            f"  vram_allocated_mb:    {_fmt(mem.get('vram_allocated_mb'))}",
            f"  vram_reserved_mb:     {_fmt(mem.get('vram_reserved_mb'))}",
            f"  vram_peak_allocated_mb:{_fmt(mem.get('vram_peak_allocated_mb'))}",
            "",
            "RESILIENCE:",
            f"  offline_mode:          {res.get('offline_mode')}",
            f"  fallback_used:         {res.get('fallback_used')}",
            f"  fallback_reason:       {res.get('fallback_reason')}",
            f"  degradation_mode:      {res.get('degradation_mode')}",
            f"  degradation_mode_effective: {deg_eff}",
            f"  edge_readiness_status: {ready_eff}",
        ]
        if self.report.errors:
            lines.append("")
            lines.append("ERRORS:")
            for e in self.report.errors:
                lines.append(f"  - [{e.get('stage')}] {e.get('type')}: {e.get('message')}")
        return "\n".join(lines)


def collect_with_stages(
    *,
    pre: Optional[Callable[[], Any]] = None,
    infer: Optional[Callable[[], Any]] = None,
    post: Optional[Callable[[], Any]] = None,
    backend: str = "unknown",
    offline_mode: Optional[bool] = None,
    degradation_mode: str = "NONE",
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[EdgeMetricsReport, Dict[str, Any]]:
    """
    Helper for quick collection in notebooks/scripts.
    Returns (report, outputs) where outputs contains optional stage results.
    """
    outputs: Dict[str, Any] = {}
    collector = EdgeMetricsCollector(backend=backend, offline_mode=offline_mode, meta=meta)
    with collector.run(degradation_mode=degradation_mode):
        if pre is not None:
            with collector.stage("pre"):
                outputs["pre"] = pre()
        if infer is not None:
            with collector.stage("infer"):
                outputs["infer"] = infer()
        if post is not None:
            with collector.stage("post"):
                outputs["post"] = post()
    return collector.report, outputs


# TERMINUS
# --- file: tests/test_edge_metrics.py ---
