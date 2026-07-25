from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from watermet2_repro.model import (
    aggregate_results,
    generate_proxy_inputs,
    load_parameters,
    run_scenario,
)


def build_figures(annual_all: pd.DataFrame, monthly_all: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    labels = {"bau": "BAU", "added_resource": "Added resource", "recycling": "Water recycling"}
    colors = {"bau": "#555555", "added_resource": "#3B82F6", "recycling": "#10B981"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for scenario, group in monthly_all.groupby("scenario"):
        delivered_percent = 100 * group["delivered_total_ml"] / group["water_demand_ml"]
        ax.plot(group["date"], delivered_percent, label=labels[scenario], color=colors[scenario], linewidth=1.2)
    ax.set(xlabel="Year", ylabel="Delivered water demand (%)", ylim=(65, 101))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "fig7_delivered_water.png", dpi=180)
    plt.close(fig)

    final = annual_all.groupby("scenario").agg(
        ghg=("ghg_total_kg_co2e", "mean"),
        acid=("acidification_kg_so2e", "mean"),
        eutro=("eutrophication_total_kg_po4e", "mean"),
        cost=("operational_cost_eur", "mean"),
    )
    final = final.loc[["bau", "added_resource", "recycling"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, column, title, scale in (
        (axes[0, 0], "ghg", "Annual average GHG", 1e6),
        (axes[0, 1], "acid", "Annual average acidification", 1000),
        (axes[1, 0], "eutro", "Annual average eutrophication", 1000),
        (axes[1, 1], "cost", "Annual average O&M cost", 1e6),
    ):
        ax.bar([labels[i] for i in final.index], final[column] / scale, color=[colors[i] for i in final.index])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("10^6 kg CO2-eq")
    axes[0, 1].set_ylabel("tonne SO2-eq")
    axes[1, 0].set_ylabel("tonne PO4-eq")
    axes[1, 1].set_ylabel("million EUR")
    fig.tight_layout()
    fig.savefig(output / "fig8_scenario_impacts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for scenario, group in annual_all.groupby("scenario"):
        ax.plot(group["date"].dt.year, group["ghg_total_kg_co2e"] / 1e6, label=labels[scenario], color=colors[scenario])
    ax.set(xlabel="Year", ylabel="GHG emissions (10^6 kg CO2-eq)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "fig9_annual_ghg.png", dpi=180)
    plt.close(fig)

    selected = monthly_all[monthly_all["date"].dt.year == 2038]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fields = (
        ("runoff_ml", "Runoff (ML/month)"),
        ("ghg_total_kg_co2e", "GHG (kg CO2-eq/month)"),
        ("acidification_kg_so2e", "Acidification (kg SO2-eq/month)"),
        ("eutrophication_total_kg_po4e", "Eutrophication (kg PO4-eq/month)"),
    )
    for ax, (field, title) in zip(axes.flat, fields):
        for scenario, group in selected.groupby("scenario"):
            ax.plot(group["date"].dt.month, group[field], label=labels[scenario], color=colors[scenario])
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[1, 0].set_xlabel("Month in 2038")
    axes[1, 1].set_xlabel("Month in 2038")
    fig.tight_layout()
    fig.savefig(output / "fig10_daily_driver_snapshot.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", default="data/oslo_parameters.json")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    params = load_parameters(args.parameters)
    inputs = generate_proxy_inputs(params)
    inputs.to_csv(output / "proxy_daily_inputs.csv", index=False)

    summaries = []
    monthly_frames = []
    annual_frames = []
    for scenario in ("bau", "added_resource", "recycling"):
        daily = run_scenario(inputs, params, scenario)
        monthly, annual, summary = aggregate_results(daily)
        monthly["scenario"] = scenario
        annual["scenario"] = scenario
        daily.to_csv(output / f"{scenario}_daily.csv", index=False)
        monthly.to_csv(output / f"{scenario}_monthly.csv", index=False)
        annual.to_csv(output / f"{scenario}_annual.csv", index=False)
        summaries.append(summary)
        monthly_frames.append(monthly)
        annual_frames.append(annual)
    summary_all = pd.concat(summaries, ignore_index=True)
    monthly_all = pd.concat(monthly_frames, ignore_index=True)
    annual_all = pd.concat(annual_frames, ignore_index=True)
    summary_all.to_csv(output / "scenario_summary.csv", index=False)
    build_figures(annual_all, monthly_all, output / "figures")
    print(summary_all.to_string(index=False))


if __name__ == "__main__":
    main()
