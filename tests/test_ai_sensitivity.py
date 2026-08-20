from __future__ import annotations

from watermet2_repro.sensitivity import morris_sensitivity, sobol_sensitivity


def response(values: dict[str, float]) -> float:
    return 10 * values["capacity"] + values["coc"] + values["capacity"] * values["temperature"]


def test_morris_identifies_capacity_as_dominant() -> None:
    result = morris_sensitivity(
        response, {"capacity": (0, 2), "coc": (3, 8), "temperature": (0, 1)}, trajectories=40
    )
    assert result.iloc[0].parameter == "capacity"
    assert {"mu", "mu_star", "sigma"} <= set(result.columns)


def test_sobol_returns_first_and_total_order_indices() -> None:
    result = sobol_sensitivity(
        response, {"capacity": (0, 2), "coc": (3, 8), "temperature": (0, 1)}, samples=256
    )
    assert result.iloc[0].parameter == "capacity"
    assert (result.ST >= 0).all()
