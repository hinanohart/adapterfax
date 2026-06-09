"""Report serialization determinism and the CLI surface."""

from __future__ import annotations

import json

from adapterfax import cli, core, synthetic


def _adapters() -> object:
    ad, _ = synthetic.plant_redundant_adapters(
        400, 4, 0.5, shared_groups=((0, 1, 2),), n_adapters=6, seed=4
    )
    return ad


def test_report_json_roundtrip() -> None:
    rep = core.audit(_adapters())  # type: ignore[arg-type]
    s = rep.to_json()
    d = json.loads(s)
    assert d["schema_version"] == 1
    assert d["active_mode"] == "full"
    assert "no downstream accuracy" in " ".join(d["non_claims"])
    assert isinstance(d["effective_capacity_used"], int)


def test_report_determinism_byte_identical() -> None:
    j1 = core.audit(_adapters()).to_json()  # type: ignore[arg-type]
    j2 = core.audit(_adapters()).to_json()  # type: ignore[arg-type]
    assert j1 == j2


def test_report_non_claims_present() -> None:
    rep = core.audit(_adapters())  # type: ignore[arg-type]
    joined = " ".join(rep.non_claims)
    assert "approximate" in joined
    assert "no downstream accuracy claim" in joined


def test_cli_gate_runs(capsys: object) -> None:
    rc = cli.main(["gate", "--json"])
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    data = json.loads(out)
    assert "gates" in data
    assert data["all_pass"] is True
    assert rc == 0  # gate must pass on the synthetic suite


def test_cli_version() -> None:
    import pytest

    with pytest.raises(SystemExit):
        cli.main(["--version"])
