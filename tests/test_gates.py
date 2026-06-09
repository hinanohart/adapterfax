"""Run the pre-registered sensitivity gate suite once and assert the verdict.

Uses ``fast=True`` (small, deterministic config).  The hard gates must all pass;
G6 may degrade to P-LITE (that is still ``all_pass=True`` with active_mode=lite).
"""

from __future__ import annotations

from typing import Any

import pytest

from adapterfax.gates import run_gates


@pytest.fixture(scope="module")
def gate_result() -> dict[str, Any]:
    return run_gates(fast=True)


def test_gates_overall_pass(gate_result: dict[str, Any]) -> None:
    assert gate_result["all_pass"] is True
    assert gate_result["active_mode"] in ("full", "lite")


def test_g1_tpr_recovers_supercritical(gate_result: dict[str, Any]) -> None:
    g = gate_result["gates"]["G1_tpr"]
    assert g["passed"] is True
    assert g["detail"]["tpr"] >= 0.95


def test_g2_fpr_calibrated(gate_result: dict[str, Any]) -> None:
    g = gate_result["gates"]["G2_fpr"]
    assert g["passed"] is True
    assert g["detail"]["fpr"] <= 0.05


def test_g3_calibration(gate_result: dict[str, Any]) -> None:
    assert gate_result["gates"]["G3_calibration"]["passed"] is True


def test_g5_determinism(gate_result: dict[str, Any]) -> None:
    assert gate_result["gates"]["G5_determinism"]["passed"] is True


def test_g6_census_or_lite(gate_result: dict[str, Any]) -> None:
    # full => G6 passed; lite => G6 RED but honest-DONE
    g6 = gate_result["gates"]["G6_census_nontrivial"]["passed"]
    assert (gate_result["active_mode"] == "full") == bool(g6)


def test_g8_second_shape_fpr(gate_result: dict[str, Any]) -> None:
    assert gate_result["gates"]["G8_fpr_second_shape"]["passed"] is True


def test_g9_determinism(gate_result: dict[str, Any]) -> None:
    assert gate_result["gates"]["G9_end_to_end_determinism"]["passed"] is True
