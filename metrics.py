"""Prometheus metrics for BehaviorIQ ML service (scraped via /metrics)."""

from __future__ import annotations

import contextlib
import os
import time
from typing import Iterator

import psutil
from prometheus_client import Counter, Gauge, Histogram, generate_latest

# ── HTTP / traffic ───────────────────────────────────────────────────────────
HTTP_REQUESTS = Counter(
    "behavioriq_http_requests_total",
    "HTTP requests (traffic)",
    ["method", "path", "status_family"],
)

HTTP_REQUEST_DURATION = Histogram(
    "behavioriq_http_request_duration_seconds",
    "End-to-end HTTP request duration",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ── Pipeline steps (latency per step) ────────────────────────────────────────
STEP_DURATION = Histogram(
    "behavioriq_step_duration_seconds",
    "Latency of internal pipeline steps",
    ["endpoint", "step"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ── Vector store / local embedding index ─────────────────────────────────────
VECTOR_LOOKUP_FAILURES = Counter(
    "behavioriq_vector_lookup_failures_total",
    "Product IDs missing from the in-memory vector index (vector store miss)",
    ["endpoint"],
)

# ── Malformed / bad outputs ───────────────────────────────────────────────────
MALFORMED_OUTPUTS = Counter(
    "behavioriq_malformed_outputs_total",
    "Malformed responses, validation issues, or model-shape fallbacks",
    ["reason"],
)

BUSINESS_ERRORS = Counter(
    "behavioriq_business_errors_total",
    "Successful HTTP responses that carry an application-level error payload",
    ["endpoint"],
)

# ── Local cache (embedder product_vectors dict) ──────────────────────────────
LOCAL_CACHE_HITS = Counter(
    "behavioriq_local_cache_hits_total",
    "Hits on the in-process product vector cache",
)

LOCAL_CACHE_MISSES = Counter(
    "behavioriq_local_cache_misses_total",
    "Misses on the in-process product vector cache",
)

# ── Host / process resource view ─────────────────────────────────────────────
PROCESS_CPU_PERCENT = Gauge(
    "behavioriq_process_cpu_percent",
    "CPU percent for this process (see psutil; not normalized across cores)",
)

SYSTEM_CPU_PERCENT = Gauge(
    "behavioriq_system_cpu_percent",
    "System-wide CPU utilization percent",
)

PROCESS_THREAD_COUNT = Gauge(
    "behavioriq_process_threads",
    "Number of threads in this process",
)

PROCESS_OPEN_FDS = Gauge(
    "behavioriq_process_open_fds",
    "Open file descriptors for this process (Unix)",
)

GPU_UTILIZATION = Gauge(
    "behavioriq_gpu_utilization_percent",
    "Average GPU utilization across visible devices (0 if unavailable)",
)

_gpu_nvml_initialized = False


def record_local_cache_lookup(hit: bool) -> None:
    if hit:
        LOCAL_CACHE_HITS.inc()
    else:
        LOCAL_CACHE_MISSES.inc()


def _status_family(code: int) -> str:
    if 200 <= code < 300:
        return "2xx"
    if 300 <= code < 400:
        return "3xx"
    if 400 <= code < 500:
        return "4xx"
    if 500 <= code < 600:
        return "5xx"
    return "other"


def observe_http_request(method: str, path: str, status_code: int, duration_s: float) -> None:
    fam = _status_family(status_code)
    HTTP_REQUESTS.labels(method=method, path=path, status_family=fam).inc()
    HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(duration_s)
    if status_code == 422:
        MALFORMED_OUTPUTS.labels(reason="validation").inc()


@contextlib.contextmanager
def observe_step(endpoint: str, step: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        STEP_DURATION.labels(endpoint=endpoint, step=step).observe(time.perf_counter() - start)


def refresh_resource_gauges() -> None:
    """Update process/system gauges; safe to call from the request hot path occasionally."""
    try:
        proc = psutil.Process(os.getpid())
        PROCESS_CPU_PERCENT.set(proc.cpu_percent(interval=None))
        PROCESS_THREAD_COUNT.set(proc.num_threads())
        if hasattr(proc, "num_fds"):
            PROCESS_OPEN_FDS.set(proc.num_fds())
    except (psutil.Error, OSError):
        pass
    try:
        SYSTEM_CPU_PERCENT.set(psutil.cpu_percent(interval=None))
    except (psutil.Error, OSError):
        pass
    _refresh_gpu_gauge()


def _refresh_gpu_gauge() -> None:
    global _gpu_nvml_initialized
    try:
        import pynvml  # type: ignore[import-untyped]

        if not _gpu_nvml_initialized:
            pynvml.nvmlInit()
            _gpu_nvml_initialized = True
        n = pynvml.nvmlDeviceGetCount()
        if n == 0:
            GPU_UTILIZATION.set(0.0)
            return
        utils = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            utils.append(float(u.gpu))
        GPU_UTILIZATION.set(sum(utils) / len(utils))
    except Exception:
        GPU_UTILIZATION.set(0.0)


def metrics_payload() -> tuple[bytes, str]:
    """Latest exposition format for GET /metrics."""
    data = generate_latest()
    return data, "text/plain; version=0.0.4; charset=utf-8"
