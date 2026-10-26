"""Synthetic telemetry mapping tests; not board measurements."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "dashboard_perf", Path(__file__).resolve().parents[1] / "scripts/demo_dashboard.py"
)
DASH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DASH)


def telemetry(query=False):
    value = {
        "version": 1, "uptime_us": 1000000,
        "internal_heap_total_bytes": 100000,
        "internal_heap_free_bytes": 40000,
        "internal_heap_min_free_bytes": 39000,
        "internal_heap_largest_free_block_bytes": 30000,
    }
    if query:
        value.update(query_heap_free_before_bytes=40000,
                     query_heap_free_after_bytes=40000,
                     query_heap_delta_bytes=0, recognition_us=53600)
    return value


def job(status="COMPLETED", when="2026-10-20T09:00:00+00:00"):
    metrics = {"telemetry": telemetry(True), "encoder_invoke_us": 53500,
               "usb_roundtrip_us": 250000, "request_id": 12}
    return {"id": "query-1", "status": status, "finished_at": when,
            "result": {"kind": "live_mac_query", "top1": "An", "performance": metrics}}


def test_no_memory_inference_from_arena_or_chip_capacity():
    result = DASH._performance_summary({"arena_used_bytes": 38140, "arena_capacity_bytes": 131072}, None, [])
    assert result == {"memory_snapshot": None, "sampled_at": None, "last_query": None}
    assert DASH._query_performance({"invoke_us": 53500}, 80000)["telemetry"] is None


def test_last_query_is_successful_only_not_error_or_active_result():
    value = DASH._performance_summary(None, None, [job("ERROR"), job("RUNNING"), job()])
    assert value["last_query"]["metrics"]["request_id"] == 12
    assert value["memory_snapshot"] == telemetry(True)
    assert DASH._performance_summary(None, None, [job("ERROR")])["last_query"] is None


def test_latest_timestamp_controls_memory_snapshot_and_does_not_replace_query():
    board = {"telemetry": telemetry()}
    later = "2026-10-20T09:10:00+00:00"
    result = DASH._performance_summary(board, later, [job()])
    assert result["sampled_at"] == later
    assert result["memory_snapshot"] == telemetry()
    assert result["last_query"]["sampled_at"] != later
    result["memory_snapshot"]["uptime_us"] = 3
    assert board["telemetry"]["uptime_us"] == 1000000


def test_frozen_query_maps_last_item_not_whole_job_average():
    first, last = {"file": "first", "performance": {"request_id": 1}}, {"file": "last", "performance": {"request_id": 2}}
    result = DASH._performance_summary(None, None, [{"id": "frozen", "status": "COMPLETED", "finished_at": "now", "result": {"queries": [first, last]}}])
    assert result["last_query"]["source"] == "last"
    assert result["last_query"]["metrics"]["request_id"] == 2


def test_query_metrics_keep_host_and_board_times_distinct():
    reply = {"request_id": 12, "invoke_us": 53500, "input_bytes": 11920,
             "output_bytes": 192, "arena_used_bytes": 38140,
             "arena_capacity_bytes": 131072, "telemetry": telemetry(True)}
    result = DASH._query_performance(reply, 250000)
    assert result["usb_roundtrip_us"] == 250000
    assert result["encoder_invoke_us"] == 53500
    assert result["telemetry"]["recognition_us"] == 53600
    assert result["telemetry"]["query_heap_delta_bytes"] == 0


def test_performance_page_is_an_explicit_static_allowlist():
    assert DASH.STATIC_FILES["/performance"][0] == "performance.html"
    for url in ("/performance", "/performance.js", "/performance.css"):
        assert (DASH.WEB_ROOT / DASH.STATIC_FILES[url][0]).is_file()
    js = (DASH.WEB_ROOT / "performance.js").read_text()
    assert 'JSON.stringify({action: "check_board"})' in js
    assert "run_live" not in js and "PROVISION" not in js
    assert "innerHTML" not in js
    assert "memory_snapshot" in js
