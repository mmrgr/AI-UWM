from __future__ import annotations

from pathlib import Path

import numpy as np

from watermet2_repro.model import generate_proxy_inputs, load_parameters, run_scenario


PARAMETERS = Path("data/oslo_parameters.json")


def test_proxy_inputs_preserve_published_mean_inflows() -> None:
    params = load_parameters(PARAMETERS)
    inputs = generate_proxy_inputs(params)
    for resource in params["water_supply"]["resources"]:
        expected = resource["mean_inflow_mcm_year"] * 1000 / 365.25
        assert np.isclose(inputs[f"{resource['name'].lower()}_inflow_ml"].mean(), expected)


def test_water_demand_balance_and_non_negative_flows() -> None:
    params = load_parameters(PARAMETERS)
    inputs = generate_proxy_inputs(params).iloc[:370]
    result = run_scenario(inputs, params, "bau")
    np.testing.assert_allclose(
        result["water_demand_ml"],
        result["delivered_total_ml"] + result["unmet_ml"],
        rtol=0,
        atol=1e-8,
    )
    flow_columns = [column for column in result if column.endswith("_ml")]
    assert (result[flow_columns] >= -1e-9).all().all()


def test_interventions_improve_final_period_delivery() -> None:
    params = load_parameters(PARAMETERS)
    inputs = generate_proxy_inputs(params)
    bau = run_scenario(inputs, params, "bau")
    added = run_scenario(inputs, params, "added_resource")
    recycling = run_scenario(inputs, params, "recycling")
    final = inputs["date"].dt.year >= 2038
    bau_ratio = bau.loc[final, "delivered_total_ml"].sum() / bau.loc[final, "water_demand_ml"].sum()
    added_ratio = added.loc[final, "delivered_total_ml"].sum() / added.loc[final, "water_demand_ml"].sum()
    recycling_ratio = recycling.loc[final, "delivered_total_ml"].sum() / recycling.loc[final, "water_demand_ml"].sum()
    assert added_ratio > recycling_ratio > bau_ratio
