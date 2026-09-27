"""Build the R2 (next-round) CAWCC scenario package for the AI-UWM framework.

Everything written here is *synthetic but anchored*: the daily drivers are
generated from documented climate/water-endowment anchors with a fixed seed,
and every engineering parameter carries a source tag in `parameter_sources.csv`.
Nothing in this file is presented as measured city data.

Outputs (under examples/cawcc_r2/):
  <domain>/project.json        runnable AI-UWM project (one per domain)
  <domain>/timeseries.csv      daily drivers incl. AI load-factor columns
  <domain>/tech_<tech>.json    per-cooling-technology project variants
  <domain>/policy_<p>.json     per-allocation-policy project variants
  specs/*.json                 constraint / sensitivity / intervention specs
  parameter_sources.csv        parameter, value, unit, source, evidence type
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "cawcc_r2"

# --------------------------------------------------------------------------
# 1. Parameter-domain endpoints (开题报告 3.7 外部校核: 两端点 + 能源富集补充)
# --------------------------------------------------------------------------
DOMAINS: dict[str, dict] = {
    # 缺水—高冷负荷端: 水源余量低、夏季湿球温度高、再生水配置比例高
    "ha": {
        "label": "D-HA 缺水—高冷负荷端",
        "t_annual_c": 13.5,
        "t_amplitude_c": 15.0,
        "t_phase_day": 200.0,          # 峰值出现在 7 月中旬
        "rh_annual": 58.0,
        "rh_summer_bonus": 14.0,
        "rainfall_annual_mm": 550.0,
        "rainy_months": (6, 7, 8),
        # 水源入流必须与取水规模同量级: 原设定 380 ML/d 对 ~240 ML/d 的取水
        # 意味着 1.6 倍余量, 实测水库常年蓄满(95% 库容)并弃水 88 ML/d(占入流
        # 23%), 干旱情景几乎不产生压力。经率定改为 265 ML/d + 加大季节振幅,
        # 使库容在 20%-90% 之间真实年内循环、弃水率降到 10% 以下。
        "inflow_mean_ml": 265.0,
        "inflow_season_amp": 0.34,
        "inflow_noise": 0.12,
        "initial_storage_ml": 65000.0,
        "reservoir_capacity_ml": 120000.0,
        "abstraction_capacity_ml_day": 380.0,
        # 处理能力按"峰值日利用率 ~0.78-0.82"反算: 城市峰值日需水 ~289 ML/d,
        # 取 370 ML/d -> 峰值日 0.78、年均 0.68。原设定 330 使峰值日已达
        # 0.877, 城市本身几乎没有余量, 承载容量分析的结论会被这个"零余量"
        # 而不是被数据中心的需求决定。
        "wtw_capacity_ml_day": 370.0,
        "wwtw_capacity_ml_day": 270.0,
        # 再生水厂产能必须留出日峰值裕度: 产水(分流比例 x 污水量) ~32 ML/d,
        # 产能取 40 ML/d -> 年均利用率 0.80、日峰值 ~0.88。若产能贴着产水量
        # (34 ML/d), 日峰值利用率恒等于 1.0, 规划裕度判据退化为恒失效。
        "reuse_treatment_ml_day": 40.0,
        "central_reuse_fraction": 0.16,
        "reclaimed_target": 0.70,
        "reuse_substitution_target": 0.65,
        "grid_connection_capacity_mw": 1500.0,
        "offsite_water_intensity_l_kwh": 0.40,
        "population": 1_000_000,
        "city_electricity_kwh_per_capita_year": 6000.0,
    },
    # 气候凉爽端: 湿球温度低、自然冷却时数长、再生水配置比例中
    "co": {
        "label": "D-CO 气候凉爽端",
        "t_annual_c": 9.0,
        "t_amplitude_c": 11.0,
        "t_phase_day": 205.0,
        "rh_annual": 66.0,
        "rh_summer_bonus": 6.0,
        "rainfall_annual_mm": 780.0,
        "rainy_months": (6, 7, 8, 9),
        # 同 HA: 原 470 ML/d 使水库弃水 190 ML/d(占入流 40%), 完全不构成约束。
        # 丰水端仍应高于 HA, 故取 290 ML/d (取水 ~230 ML/d 的 1.26 倍)。
        "inflow_mean_ml": 290.0,
        "inflow_season_amp": 0.28,
        "inflow_noise": 0.10,
        "initial_storage_ml": 65000.0,
        "reservoir_capacity_ml": 120000.0,
        "abstraction_capacity_ml_day": 400.0,
        "wtw_capacity_ml_day": 380.0,
        "wwtw_capacity_ml_day": 270.0,
        "reuse_treatment_ml_day": 36.0,
        "central_reuse_fraction": 0.15,
        "reclaimed_target": 0.45,
        "reuse_substitution_target": 0.40,
        "grid_connection_capacity_mw": 1500.0,
        "offsite_water_intensity_l_kwh": 0.18,
        "population": 1_000_000,
        "city_electricity_kwh_per_capita_year": 6000.0,
    },
}

# 能源富集补充情景 (D-HA 叠加): 电网水强度低、接网容量充裕
ENERGY_RICH_OVERLAY = {
    "grid_connection_capacity_mw": 2600.0,
    "offsite_water_intensity_l_kwh": 0.08,
}

# --------------------------------------------------------------------------
# 2. 冷却技术: 逐技术标定 PUE 温度敏感性 (开题 3.3: 干式逼近度随气温恶化)
#    说明: ai_scenarios.apply_ai_scenario 只替换 technology, 不替换
#    temperature_coefficient, 因此冷却技术对照必须"一技术一工程"。
# --------------------------------------------------------------------------
TECHNOLOGIES: dict[str, dict] = {
    "evaporative": {
        "technology_pue_adjustment": 0.00,
        "base_pue": 1.20,
        "temperature_coefficient": 0.0030,
        "cycles_of_concentration": 5.0,
        "hybrid_wet_bulb_start_c": None,
        "hybrid_full_wet_bulb_c": None,
    },
    "efficient_evaporative": {
        "technology_pue_adjustment": -0.02,
        "base_pue": 1.17,
        "temperature_coefficient": 0.0025,
        "cycles_of_concentration": 7.0,
        "hybrid_wet_bulb_start_c": None,
        "hybrid_full_wet_bulb_c": None,
    },
    "hybrid": {
        "technology_pue_adjustment": 0.04,
        "base_pue": 1.24,
        "temperature_coefficient": 0.0055,
        "cycles_of_concentration": 5.0,
        "hybrid_wet_bulb_start_c": 14.0,
        "hybrid_full_wet_bulb_c": 24.0,
    },
    "dry_first": {   # 干式优先 + 高温蒸发辅助: 用 hybrid 阈值切换机制实现
        # 切换带经率定 (scripts/calibrate_dry_first.py, HA 端 1000 MW):
        #   (22, 30) -> 日峰值系数 15.3  (高出文献 6.5-10x)
        #   (19, 28) -> 9.22
        #   (18, 27) -> 7.50  <- 取此值: 落在文献区间中部, 留有余量
        # 全年湿冷份额 0.142 (约 184 天开蒸发辅助), 仍保持"干式为主"语义。
        "technology": "hybrid",
        "technology_pue_adjustment": 0.10,
        "base_pue": 1.30,
        "temperature_coefficient": 0.0110,
        "cycles_of_concentration": 6.0,
        "hybrid_wet_bulb_start_c": 18.0,
        "hybrid_full_wet_bulb_c": 27.0,
    },
    "dry": {
        "technology_pue_adjustment": 0.12,
        "base_pue": 1.32,
        "temperature_coefficient": 0.0130,
        "cycles_of_concentration": 5.0,
        "hybrid_wet_bulb_start_c": None,
        "hybrid_full_wet_bulb_c": None,
    },
    "liquid_to_air": {
        "technology_pue_adjustment": 0.05,
        "base_pue": 1.25,
        "temperature_coefficient": 0.0070,
        "cycles_of_concentration": 5.0,
        "hybrid_wet_bulb_start_c": None,
        "hybrid_full_wet_bulb_c": None,
        "heat_to_cooling_fraction": 0.90,
    },
    "liquid_to_water": {
        "technology_pue_adjustment": -0.02,
        "base_pue": 1.18,
        "temperature_coefficient": 0.0035,
        "cycles_of_concentration": 6.0,
        "hybrid_wet_bulb_start_c": None,
        "hybrid_full_wet_bulb_c": None,
        "heat_to_cooling_fraction": 0.90,
    },
}

ALLOCATION_POLICIES = {
    "policy_a": "resident_first",
    "policy_b": "proportional",
    "policy_c": "ai_first",
}

# --------------------------------------------------------------------------
# 3. 参数证据表 (来源与证据类型; 合成驱动明确标注)
# --------------------------------------------------------------------------
PARAMETER_SOURCES = [
    ("population", 1_000_000, "cap", "参数域设定: 中等规模城市", "assumption"),
    ("domestic_demand", 165, "l_capita_day", "GB/T 50331 城市综合用水量标准量级", "literature"),
    ("industrial_demand", 60, "ml_day", "参数域设定: 一般工业占比", "assumption"),
    ("irrigation_demand", 12, "ml_day", "参数域设定: 季节灌溉(5-9月)", "assumption"),
    ("municipal_misc_demand", 20, "ml_day",
     "参数域设定: 市政杂用+景观环境补水(再生水主要用户之一)", "assumption"),
    ("built_up_area", 12000, "ha", "中国百万人口城市建成区 60-120 m2/人", "literature"),
    ("distribution_leakage", 0.12, "fraction", "CJJ 92 漏损控制区间中位", "literature"),
    # 以下能力值经标定: 使基线(0 MW)下各设施利用率落在 0.55-0.80 的规划区间,
    # 使瓶颈在 400-1600 MW 之间逐次迁移, 而不是在 0 MW 就已饱和。
    ("wtw_capacity", "370 / 380", "ml_day",
     "率定: 按峰值日利用率 ~0.78-0.82 反算(城市峰值日需水 ~289 ML/d)", "calibrated"),
    ("wwtw_capacity", 270, "ml_day",
     "率定: 按峰值日利用率 ~0.82 反算(峰值日污水 ~222 ML/d)", "calibrated"),
    ("reuse_treatment_capacity", "40 / 36", "ml_day",
     "参数域设定: HA 高回用 / CO 中回用; 取产水量的 1.2 倍作日峰值裕度", "assumption"),
    ("central_reuse_fraction", "0.16 / 0.15", "fraction",
     "按再生水产能反算: 产能/污水量, 使产能利用率落在 90-95%", "calibrated"),
    ("reservoir_inflow_mean", "265 / 290", "ml_day",
     "率定: 使弃水率 < 10%、库容年内真实循环(取水量的 1.1-1.26 倍)", "calibrated"),
    ("reservoir_capacity", 120000, "ml",
     "率定: 年取水量的 1.3-1.4 倍(现实多年调节水库 1-2 倍)", "calibrated"),
    ("reservoir_initial_storage", 65000, "ml", "参数域设定: 起调库容 54%", "assumption"),
    ("abstraction_capacity", "380 / 400", "ml_day",
     "率定: 使 0 MW 基线日峰值占用率落在 0.73-0.79", "calibrated"),
    ("local_connection_capacity", 70, "ml_day", "参数域设定: 园区局部接入能力", "assumption"),
    ("cooling_storage", 24, "ml", "园区冷却水调蓄: 1000 MW 设计补水量的 ~15 小时(现实 12-72 h)", "assumption"),
    ("grid_connection_capacity", "1500 / 2600", "MW", "参数域设定: 常规 / 能源富集(HA 补充情景)", "assumption"),
    ("ai_load_factor_mean", 0.75, "fraction", "LBNL 2024 利用率区间中位", "literature"),
    ("training_fraction", 0.65, "fraction", "开题报告 3.3 训练/推理占比设定", "assumption"),
    ("inference_peak_factor", 1.35, "fraction", "开题报告 3.3 推理峰荷系数设定", "assumption"),
    ("coc_quality_limits_tds", 2200, "mg/L", "ASHRAE 冷却水处理: 结垢/腐蚀控制量级", "literature"),
    ("coc_quality_limits_chloride", 500, "mg/L", "ASHRAE 冷却水处理: 不锈钢腐蚀控制量级", "literature"),
    ("reclaimed_tds", 650, "mg/L", "再生水典型 TDS 量级", "assumption"),
    ("reclaimed_chloride", 130, "mg/L", "再生水典型氯化物量级", "assumption"),
    ("potable_tds", 180, "mg/L", "饮用水典型 TDS 量级", "assumption"),
    ("potable_chloride", 35, "mg/L", "饮用水典型氯化物量级", "assumption"),
    ("offsite_water_intensity", "0.40 / 0.18 / 0.08", "L/kWh", "LBNL 间接水足迹情景先验", "assumption"),
    ("driver_series", "synthetic", "-", "合成分成: 正弦季节 + AR(1) 噪声, 固定种子", "synthetic"),
]


def wet_bulb(temperature_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Stull (2011) approximation - identical to aiuwm.data_center."""
    rh_c = np.clip(rh, 0.0, 100.0)
    twb = (
        temperature_c * np.arctan(0.151977 * np.sqrt(rh_c + 8.313659))
        + np.arctan(temperature_c + rh_c)
        - np.arctan(rh_c - 1.676331)
        + 0.00391838 * rh_c**1.5 * np.arctan(0.023101 * rh_c)
        - 4.686035
    )
    return np.minimum(temperature_c, twb)


def build_drivers(domain: dict, start: str, years: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=365 * years, freq="D")
    doy = dates.dayofyear.to_numpy()
    n = len(dates)

    # --- 干球温度: 正弦季节 + AR(1) 天气噪声 -------------------------------
    seasonal = domain["t_annual_c"] + domain["t_amplitude_c"] * np.sin(
        2 * math.pi * (doy - domain["t_phase_day"] + 91.25) / 365.0
    )
    noise = np.zeros(n)
    innov = rng.normal(0.0, 2.4, n)
    for i in range(1, n):
        noise[i] = 0.62 * noise[i - 1] + innov[i]
    temperature = seasonal + noise

    # --- 相对湿度: 季节加成 + 与当日温度距平反向耦合 + 噪声 ----------------
    # 高温日空气更干是该参数域的一阶特征; 若不加入反向耦合, 湿球温度会出现
    # 超过 30 ℃ 的非物理极端值。
    rh_season = domain["rh_annual"] + domain["rh_summer_bonus"] * np.cos(
        2 * math.pi * (doy - 202.0) / 365.0
    )
    rh = np.clip(
        rh_season - 1.8 * (temperature - seasonal) + rng.normal(0.0, 5.0, n), 12.0, 98.0
    )

    # --- 降雨: 泊松发生 x Gamma 强度, 雨季集中, 事后按年总量归一 -----------
    month = dates.month.to_numpy()
    rainy = np.isin(month, domain["rainy_months"])
    lam = np.where(rainy, 0.42, 0.14)
    occur = rng.random(n) < lam
    raw = rng.gamma(shape=0.85, scale=1.0, size=n)
    # 雨季日强度更高
    raw = raw * np.where(rainy, 1.8, 1.0)
    rainfall = np.where(occur, raw, 0.0)
    total = rainfall.sum()
    if total > 0:
        rainfall = rainfall * (domain["rainfall_annual_mm"] * years) / total

    # --- 水源入流: 季节 + 噪声 --------------------------------------------
    inflow_season = 1.0 + domain["inflow_season_amp"] * np.sin(
        2 * math.pi * (doy - 180.0) / 365.0
    )
    inflow = domain["inflow_mean_ml"] * inflow_season * rng.lognormal(
        0.0, domain["inflow_noise"], n
    )

    twb = wet_bulb(temperature, rh)

    # --- 城市用电负荷指数 (用于构造 AI 负荷的城市耦合强度) ------------------
    # 夏季空调峰 + 周末回落 + 冬季小幅峰, 归一化到均值 1.0
    electric = (
        1.0
        + 0.22 * np.cos(2 * math.pi * (doy - 200.0) / 365.0)
        + 0.06 * np.cos(2 * math.pi * (doy - 15.0) / 365.0)
    )
    weekend = dates.dayofweek.to_numpy() >= 5
    electric = np.where(weekend, electric * 0.90, electric)
    electric = electric / electric.mean()
    city_base_mw = (
        float(domain.get("population", 1_000_000))
        * float(domain.get("city_electricity_kwh_per_capita_year", 6000.0))
        / (365.0 * 24.0 * 1000.0)
    )

    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "temperature_c": np.round(temperature, 2),
            "relative_humidity_pct": np.round(rh, 1),
            "rainfall_mm": np.round(rainfall, 2),
            "reservoir_inflow_ml": np.round(inflow, 2),
            "wet_bulb_c": np.round(twb, 2),
            "city_electric_index": np.round(electric, 4),
            # Absolute non-AI city load is an explicit denominator for
            # AI-share and node-余量 diagnostics; the engine still keeps the
            # AI connection constraint separate from the city baseline.
            "city_non_ai_electricity_mw": np.round(city_base_mw * electric, 2),
        }
    )
    # --- AI 日负荷系数: 训练基荷 + 推理跟随城市负荷, 耦合强度 alpha_city ----
    for alpha in (0.0, 0.3, 0.6, 0.9):
        base = 0.75 * (1.0 - alpha) + 0.75 * alpha * electric
        frame[f"ai_load_factor_a{int(alpha * 100):02d}"] = np.round(
            np.clip(base, 0.10, 1.00), 4
        )
    # 恒定满负荷对照 (开题 3.3: alpha(t) 取常数)
    frame["ai_load_factor_flat"] = 0.75
    return frame


def base_project(domain_key: str, domain: dict, years: int = 2) -> dict:
    start, end = "2029-01-01", f"{2029 + years - 1}-12-31"
    return {
        "name": f"R2 CAWCC parameter-domain endpoint - {domain['label']}",
        "ai_database_file": "../../data/ai_data_center_database.json",
        "timeseries_file": "timeseries.csv",
        "simulation": {"start": start, "end": end},
        "city_context": {
            "population": int(domain.get("population", 1_000_000)),
            "electricity_kwh_per_capita_year": float(
                domain.get("city_electricity_kwh_per_capita_year", 6000.0)
            ),
            "non_ai_load_column": "city_non_ai_electricity_mw",
            "load_basis": "synthetic_city_index_scaled_to_annual_per_capita_electricity",
        },
        "finance": {"base_date": start, "discount_rate": 0.03},
        "energy_sources": {
            "electricity": {
                "ghg_kg_co2e_unit": 0.35,
                "acid_kg_so2e_unit": 0.0002,
                "eutro_kg_po4e_unit": 0.00003,
                "cost_eur_unit": 0.11,
            }
        },
        "demand_allocation_policy": "policy_b",
        "components": {
            "RES1": {
                "kind": "water_resource",
                # 库容必须与年取水规模同量级: 年取水 ~84,000-93,000 ML, 现实城市
                # 供水水库库容约为年取水量的 1-2 倍(多年调节)。原设定 40,000 ML
                # 仅 0.48 倍, 在 1000 MW 情景下 730 天内被抽干并产生 13,641 ML
                # 缺水量 —— 那是库容设定过小造成的伪约束, 不是研究对象的约束。
                "capacity_ml": domain["reservoir_capacity_ml"],
                "initial_ml": domain["initial_storage_ml"],
                "inflow_column": "reservoir_inflow_ml",
                "abstraction_capacity_ml_day": domain["abstraction_capacity_ml_day"],
            },
            "SC1": {"kind": "supply_conduit", "daily_capacity_ml": 620, "leakage_fraction": 0.01},
            "WTW1": {
                "kind": "wtw",
                "daily_capacity_ml": domain["wtw_capacity_ml_day"],
                "loss_fraction": 0.02,
                "capacity_ml": domain["wtw_capacity_ml_day"],
                "electricity_kwh_m3": 0.34,
            },
            "TM1": {"kind": "trunk_main", "daily_capacity_ml": 620, "leakage_fraction": 0.005},
            "SR1": {"kind": "service_reservoir", "capacity_ml": 1400, "initial_ml": 800, "daily_capacity_ml": 620, "loss_fraction": 0.001},
            "DM1": {"kind": "distribution_main", "daily_capacity_ml": 460, "leakage_fraction": 0.12, "electricity_kwh_m3": 0.18},
            "CENTRAL_REUSE": {
                "kind": "reuse",
                "reuse_type": "central",
                "capacity_ml": 300,
                "initial_ml": 90,
                "treatment_capacity_ml_day": domain["reuse_treatment_ml_day"],
                "target_areas": ["LA1"],
                # 现实的再生水用户结构(北京 2024: 工业 32.6% / 市政杂用+绿化 /
                # 景观环境补水): 只写 irrigation 会使 34 ML/d 的产能只送出
                # 6.4 ML/d(利用率 19%), 一座常年闲置八成的再生水厂在现实中不
                # 存在, 也会把"再生水产能"虚假地全部腾给数据中心。
                # 数据中心在引擎中始终排在最前(见 full_engine._process_reuse),
                # 因此这里列出的是"被数据中心挤占后仍可承接再生水"的用户。
                "eligible_demands": ["industrial", "municipal_misc", "irrigation"],
                "priority": 5,
                "electricity_kwh_m3": 0.25,
            },
            "SEWER1": {
                "kind": "sewer",
                "sewer_type": "sanitary",
                "capacity_mode": "transmission",
                "capacity_ml": 600,
                "daily_capacity_ml": 600,
                "overflow_to": "RIVER1",
            },
            # 雨水必须与污水分流: 若把 storm_sewer 指向污水管, 降雨径流会
            # 使 WWTW 入流在暴雨日放大数倍, 利用率被虚假抬高。
            "SEWER_S": {
                "kind": "sewer",
                "sewer_type": "storm",
                "capacity_mode": "transmission",
                "capacity_ml": 4000,
                "daily_capacity_ml": 4000,
                "overflow_to": "RIVER1",
            },
            "WWTW1": {
                "kind": "wwtw",
                "capacity_ml": 220,
                "initial_ml": 0,
                "daily_capacity_ml": domain["wwtw_capacity_ml_day"],
                "overflow_to": "RIVER1",
                "pollutant_removal_fraction": {"BOD": 0.90, "TSS": 0.92, "TDS": 0.05},
                "central_reuse_component": "CENTRAL_REUSE",
                "central_reuse_fraction": domain["central_reuse_fraction"],
                "electricity_kwh_m3": 0.30,
            },
            "RIVER1": {"kind": "receiving_water"},
            "AI_DC1": {
                "kind": "data_center",
                "local_area": "LA1",
                "installed_it_capacity_mw": 0.0,
                "load_mode": "timeseries",
                # full_engine 通过 component["load"]["timeseries_column"] 取值,
                # data_center 通过 load_factor_column 取值, 两者必须同时给出。
                "load_factor_column": "ai_load_factor_a60",
                "load": {"mode": "timeseries", "timeseries_column": "ai_load_factor_a60"},
                "pue_mode": "dynamic",
                "base_pue": 1.20,
                "temperature_coefficient": 0.003,
                "reference_temperature_c": 20.0,
                "min_pue": 1.03,
                "max_pue": 1.95,
                # 园区内冷却水调蓄: 1000 MW 设计补水量 ~40 ML/d, 12 ML 只够
                # 7 小时; 现实超大规模园区按 12-72 小时设计(取 ~15 小时)。
                # 该储水只在补水短缺口时被动用, 常态下不改变结果。
                "cooling_storage_ml": 24,
                "initial_cooling_storage_ml": 12,
                "local_connection_capacity_ml_day": 70.0,
                "grid_connection_capacity_mw": domain["grid_connection_capacity_mw"],
                "cooling": {
                    "technology": "evaporative",
                    "model": "tower_balance",
                    # 以设施电量为排热基数: 否则 PUE 与气温无法传导到蒸发量,
                    # H2(高温敏感冷却)在纯湿冷情形下不可检验。
                    "heat_basis": "facility",
                    "heat_rejection_temperature_coefficient": 0.0,
                    "heat_to_cooling_fraction": 1.0,
                    "cycles_of_concentration": 5.0,
                    "coc_mode": "quality_limited",
                    "water_quality_limits": {"TDS": 2200, "chloride": 500},
                    "drift_fraction": 0.0002,
                    "blowdown_return_fraction": 1.0,
                    "internal_recovery_fraction": 0.10,
                    "technology_pue_adjustment": {"evaporative": 0.0},
                    "blowdown_quality_mg_l": {"TDS": 1800},
                },
                "water_sources": {
                    "reclaimed": {
                        "component_id": "CENTRAL_REUSE",
                        "target_fraction": domain["reclaimed_target"],
                        "priority": 1,
                        "quality": {"TDS": 650, "chloride": 130},
                    },
                    "potable": {
                        "target_fraction": 1.0 - domain["reclaimed_target"],
                        "priority": 2,
                        "quality": {"TDS": 180, "chloride": 35},
                    },
                },
                "water_fallback": True,
                "priority_class": "ai_data_center",
                "offsite_electricity_water_intensity_l_kwh": domain["offsite_water_intensity_l_kwh"],
            },
        },
        "local_areas": {
            "LA1": {
                "subcatchment": "SUB1",
                "base_population": 1_000_000,
                # 32000 ha = 320 m2/人, 属低密度郊区蔓延尺度; 中国百万人口城市
                # 建成区约 60-120 m2/人, 取 12000 ha (120 m2/人)。该值只影响
                # 降雨径流/雨水系统规模, 不影响配水与污水主干。
                "area_ha": 12000,
                "weather_columns": {
                    "precipitation": "rainfall_mm",
                    "temperature": "temperature_c",
                    "relative_humidity": "relative_humidity_pct",
                },
                "surfaces": {
                    "roof": {"fraction": 0.10, "runoff_coefficient": 0.90},
                    "road_pavement": {"fraction": 0.12, "runoff_coefficient": 0.85},
                    "pervious": {"fraction": 0.78, "runoff_coefficient": 0.14},
                },
                "demand_profiles": [
                    {"name": "domestic", "base_value": 165, "unit": "l_capita_day",
                     "temperature_sensitivity": 0.006, "return_fraction": 0.92},
                    {"name": "industrial", "base_value": 60, "unit": "ml_day", "return_fraction": 0.80},
                    {"name": "irrigation", "base_value": 12, "unit": "ml_day",
                     "start_mmdd": "05-01", "end_mmdd": "09-30",
                     "temperature_sensitivity": 0.030, "return_fraction": 0.0},
                    # 市政杂用/景观环境(道路清扫、绿化、冲厕、河道景观补水):
                    # 再生水的第二大现实去处, 全年存在、夏季升温抬升。回用比例
                    # 低(15%): 绝大部分被蒸散或进入景观水体, 不回污水管。
                    {"name": "municipal_misc", "base_value": 20, "unit": "ml_day",
                     "temperature_sensitivity": 0.010, "return_fraction": 0.15},
                ],
                "sanitary_sewer": "SEWER1",
                "storm_sewer": "SEWER_S",
                "sanitary_pollutant_load_kg_capita_day": {"BOD": 0.06, "TSS": 0.07},
            }
        },
        "subcatchments": {"SUB1": {"local_areas": ["LA1"]}},
        "supply_paths": [
            {
                "id": "PATH1",
                "local_area": "LA1",
                "chain": ["RES1", "SC1", "WTW1", "TM1", "SR1", "DM1"],
                "allocation": 1.0,
                "priority": 1,
                "demand_priority": [
                    "domestic", "industrial", "data_center::AI_DC1",
                    "irrigation", "municipal_misc"
                ],
            }
        ],
        "wastewater_connections": [
            {"from": "SEWER1", "to": "WWTW1", "fraction": 1.0},
            {"from": "WWTW1", "to": "RIVER1", "fraction": 1.0},
        ],
    }


def apply_technology(project: dict, tech_key: str) -> dict:
    spec = dict(TECHNOLOGIES[tech_key])
    trial = copy.deepcopy(project)
    dc = trial["components"]["AI_DC1"]
    cooling = dc["cooling"]
    cooling["technology"] = spec.get("technology", tech_key)
    cooling["technology_pue_adjustment"] = {
        spec.get("technology", tech_key): spec["technology_pue_adjustment"]
    }
    dc["base_pue"] = spec["base_pue"]
    dc["temperature_coefficient"] = spec["temperature_coefficient"]
    cooling["cycles_of_concentration"] = spec["cycles_of_concentration"]
    if spec.get("heat_to_cooling_fraction") is not None:
        cooling["heat_to_cooling_fraction"] = spec["heat_to_cooling_fraction"]
    if spec.get("hybrid_wet_bulb_start_c") is not None:
        cooling["hybrid_wet_bulb_start_c"] = spec["hybrid_wet_bulb_start_c"]
        cooling["hybrid_full_wet_bulb_c"] = spec["hybrid_full_wet_bulb_c"]
    else:
        cooling.pop("hybrid_wet_bulb_start_c", None)
        cooling.pop("hybrid_full_wet_bulb_c", None)
    dc["research_technology_label"] = tech_key
    return trial


def write_specs(out: Path) -> None:
    specs = out / "specs"
    specs.mkdir(parents=True, exist_ok=True)

    # --- 承载约束 (对齐开题报告表 3-2 前八类 + 能源电网扩展约束) -----------
    # 关键方法细节: 引擎对 WTW/WWTW/reuse/取水能力做硬截断, 利用率不可能 > 1,
    # 因此 "利用率 <= 1" 恒真、无法失效。能力类约束必须写成规划裕度阈值
    # (<= 0.90~0.95); 局部接入与接网容量为软指标(可 >1), 直接用 <= 1。
    for key, domain in DOMAINS.items():
        constraints = {
            "min_mw": 0,
            "max_mw": 2000,
            "step_mw": 50,
            "constraints": {
                "system_reliability_fraction": {"operator": ">=", "value": 0.99},
                "domestic_unmet_ml": {"operator": "<=", "value": 0.0},
                "unmet_cooling_water_ml": {"operator": "<=", "value": 0.0},
                "max_wtw_utilization": {"operator": "<=", "value": 0.95},
                "max_wwtw_utilization": {"operator": "<=", "value": 0.95},
                "max_reuse_utilization": {"operator": "<=", "value": 0.95},
                "max_source_abstraction_ratio": {"operator": "<=", "value": 0.95},
                "reclaimed_water_substitution_ratio": {
                    "operator": ">=", "value": domain["reuse_substitution_target"]
                },
                "peak_capacity_ratio": {"operator": "<=", "value": 1.00},
                "max_local_connection_ratio": {"operator": "<=", "value": 1.00},
                "max_daily_average_grid_connection_ratio": {"operator": "<=", "value": 1.00},
            },
        }
        (specs / f"constraints_{key}.json").write_text(
            json.dumps(constraints, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 概率/稳健性/小时边界三套规格复用同一组"可失效"约束: 容量类必须
        # 用规划裕度 0.95 而不是 1.00 (引擎硬截断使 <= 1 恒真)。
        # Physical carrying-capacity constraints and policy compliance are
        # reported as separate estimands.  The reclaimed-water substitution
        # target is not a capacity signal because it is nearly constant over
        # the capacity ladder; keeping it in the physical set would conflate
        # policy feasibility with resource failure.
        core = {
            name: value for name, value in constraints["constraints"].items()
            # `max_reuse_utilization` 必须排除: 再生水厂配 300 ML 贮池, 日处理
            # 速率达到铭牌属正常调蓄, 该指标在全容量区间恒为 1.0, 用它做判据
            # 会让承载容量恒为 0。再生水的系统级约束由
            # `reclaimed_water_substitution_ratio` 承担。
            if name in {
                "system_reliability_fraction", "domestic_unmet_ml",
                "unmet_cooling_water_ml", "max_wtw_utilization",
                "max_wwtw_utilization",
                "max_source_abstraction_ratio",
                "peak_capacity_ratio",
                "max_local_connection_ratio",
                "max_daily_average_grid_connection_ratio",
            }
        }
        policy = {
            "reclaimed_water_substitution_ratio": constraints["constraints"][
                "reclaimed_water_substitution_ratio"
            ]
        }
        for template in ("probabilistic_threshold.json", "robustness.json", "hourly_boundary.json"):
            path = specs / template
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["constraints"] = dict(core)
            if template == "probabilistic_threshold.json":
                payload["policy_constraints"] = dict(policy)
                # The domain copy is created before the template is rewritten
                # below on a fresh build.  Set the production grid explicitly
                # here so a stale 250 MW/9-point template can never leak into
                # the current per-domain probability specification.
                payload["capacities_mw"] = list(range(0, 2001, 50))
                payload["grid_step_mw"] = 50
            (specs / f"{template[:-5]}_{key}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # 注意: 不能 unlink 模板, 否则第二次运行 build 时模板缺失,
            # 对模板的修改无法传播到分域文件。

    # --- 干预: 连续能力干预 + 离散技术情景 --------------------------------
    # 干预名里的百分比必须与写入的数值一致, 且必须按各域自己的基线计算:
    # 此前用一组固定数值同时作用于 HA/CO 两个基线(330 / 360), 导致
    # "wtw_plus_10pct" 在 CO 实际只有 +1%, 名实不符会把边际增益错误归因。
    # 因此这里按域生成 interventions_<key>.json。
    for key, domain in DOMAINS.items():
        wtw = float(domain["wtw_capacity_ml_day"])
        wwtw = float(domain["wwtw_capacity_ml_day"])
        reuse = float(domain["reuse_treatment_ml_day"])
        local = 70.0
        grid = float(domain["grid_connection_capacity_mw"])
        interventions = {
            # 100 MW 步长(原 200 MW 会把 <200 MW 的增益全部抹平成 0, 也无法
            # 区分"无效干预"与"网格分辨率不足")
            "capacities_mw": list(range(0, 2001, 100)),
            "note": "干预扫描网格(步长 100 MW); 增益必须与该网格下的 baseline "
                    "条目比较; 干预幅度按该域自身基线计算",
            "baseline": {
                "wtw_daily_capacity_ml": wtw,
                "wwtw_daily_capacity_ml": wwtw,
                "reuse_treatment_ml_day": reuse,
                "local_connection_ml_day": local,
                "grid_connection_mw": grid,
            },
            "interventions": [
                {"name": "baseline", "set": []},
                {"name": "wtw_plus_10pct",
                 "set": [{"path": "components.WTW1.daily_capacity_ml", "value": round(wtw * 1.1, 1)},
                         {"path": "components.WTW1.capacity_ml", "value": round(wtw * 1.1, 1)}]},
                {"name": "wtw_plus_20pct",
                 "set": [{"path": "components.WTW1.daily_capacity_ml", "value": round(wtw * 1.2, 1)},
                         {"path": "components.WTW1.capacity_ml", "value": round(wtw * 1.2, 1)}]},
                {"name": "reuse_plus_50pct",
                 "set": [{"path": "components.CENTRAL_REUSE.treatment_capacity_ml_day",
                          "value": round(reuse * 1.5, 1)}]},
                {"name": "reuse_plus_100pct",
                 "set": [{"path": "components.CENTRAL_REUSE.treatment_capacity_ml_day",
                          "value": round(reuse * 2.0, 1)}]},
                {"name": "wwtw_plus_20pct",
                 "set": [{"path": "components.WWTW1.daily_capacity_ml", "value": round(wwtw * 1.2, 1)}]},
                {"name": "local_connection_plus_50pct",
                 "set": [{"path": "components.AI_DC1.local_connection_capacity_ml_day",
                          "value": round(local * 1.5, 1)}]},
                {"name": "grid_plus_50pct",
                 "set": [{"path": "components.AI_DC1.grid_connection_capacity_mw",
                          "value": round(grid * 1.5, 1)}]},
                {"name": "leakage_minus_30pct",
                 "set": [{"path": "components.DM1.leakage_fraction", "value": 0.084}]},
                {"name": "combined_upgrade",
                 "set": [
                     {"path": "components.WTW1.daily_capacity_ml", "value": round(wtw * 1.1, 1)},
                     {"path": "components.WTW1.capacity_ml", "value": round(wtw * 1.1, 1)},
                     {"path": "components.CENTRAL_REUSE.treatment_capacity_ml_day",
                      "value": round(reuse * 1.5, 1)},
                     {"path": "components.DM1.leakage_fraction", "value": 0.084},
                 ]},
            ],
        }
        (specs / f"interventions_{key}.json").write_text(
            json.dumps(interventions, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- 敏感性 (Morris / Sobol) ------------------------------------------
    # 参数必须是"对指标真的有作用"的: 工程用 load_mode=timeseries, 负荷因子
    # 取自定义时间序列, components.AI_DC1.load_factor 是死参数(实测 mu*=0);
    # 指标是 AI 取水量, DM1.leakage_fraction 只影响城市侧, 同样是死参数
    # (实测 mu*~1e-11)。两者都换成对 AI 水量有直接作用的参数。
    sensitivity = {
        "method": "morris",
        "metric": "total_withdrawal_ml",
        "trajectories": 24,
        "seed": 20260918,
        "parameters": {
            "components.AI_DC1.base_pue": [1.12, 1.36],
            "components.AI_DC1.temperature_coefficient": [0.0005, 0.0080],
            "components.AI_DC1.cooling.cycles_of_concentration": [3.0, 9.0],
            "components.AI_DC1.cooling.drift_fraction": [0.00005, 0.0005],
            "components.AI_DC1.cooling.internal_recovery_fraction": [0.0, 0.30],
            "components.AI_DC1.cooling.heat_to_cooling_fraction": [0.85, 1.00],
        },
    }
    (specs / "sensitivity_morris.json").write_text(
        json.dumps(sensitivity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sobol = dict(sensitivity)
    sobol["method"] = "sobol"
    # Sobol 的 samples 是基样本数 N, 总评估次数 = N x (2D + 2); D = 6 时
    # N = 512 需 7168 次全引擎调用 (单域约 80 分钟), 收敛性收益远小于成本。
    # N = 32 (448 次调用) 足以给出一阶/总效应的排序, 正式论文阶段再提高。
    sobol["samples"] = 32
    sobol["metric"] = "total_withdrawal_ml"
    (specs / "sensitivity_sobol.json").write_text(
        json.dumps(sobol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 概率承载边界 ------------------------------------------------------
    probabilistic = {
        "parameter_ranges": {
            "components.AI_DC1.base_pue": [1.12, 1.36],
            "components.AI_DC1.cooling.cycles_of_concentration": [3.0, 9.0],
            # load_factor 在 timeseries 负荷模式下是死参数, 换成温度系数
            "components.AI_DC1.temperature_coefficient": [0.0005, 0.0080],
        },
        # Match the deterministic 50 MW grid.  A coarser grid biases the
        # threshold downward and can make simultaneous failures look like a
        # dictionary-order effect.
        "capacities_mw": list(range(0, 2001, 50)),
        "constraints": {
            "system_reliability_fraction": {"operator": ">=", "value": 0.99},
            "domestic_unmet_ml": {"operator": "<=", "value": 0.0},
            "max_wtw_utilization": {"operator": "<=", "value": 1.00},
            "max_reuse_utilization": {"operator": "<=", "value": 1.00},
        },
        "samples": 40,
        "seed": 20260918,
    }
    (specs / "probabilistic_threshold.json").write_text(
        json.dumps(probabilistic, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 小时压力边界 (训练/推理日内曲线 + 相位) ---------------------------
    hourly = {
        "capacities_mw": [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        "training_fraction": 0.65,
        "inference_fraction": 0.35,
        "inference_peak_factor": 1.35,
        "peak_hours": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "city_phase_hours": 0,
        "constraints": {
            "hourly_peak_capacity_ratio": {"operator": "<=", "value": 1.0},
            "system_reliability_fraction": {"operator": ">=", "value": 0.99},
        },
    }
    (specs / "hourly_boundary.json").write_text(
        json.dumps(hourly, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 稳健性: 关键参数组合 ---------------------------------------------
    robustness = {
        "capacities_mw": [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        "parameter_sets": [
            {"components.AI_DC1.load_factor": 0.65},
            {"components.AI_DC1.load_factor": 0.75},
            {"components.AI_DC1.load_factor": 0.85},
            {"components.AI_DC1.cooling.cycles_of_concentration": 3.0},
            {"components.AI_DC1.cooling.cycles_of_concentration": 9.0},
            {"components.AI_DC1.base_pue": 1.12},
            {"components.AI_DC1.base_pue": 1.36},
            {"components.DM1.leakage_fraction": 0.06},
            {"components.DM1.leakage_fraction": 0.20},
        ],
        "constraints": {
            "system_reliability_fraction": {"operator": ">=", "value": 0.99},
            "domestic_unmet_ml": {"operator": "<=", "value": 0.0},
            "max_wtw_utilization": {"operator": "<=", "value": 1.00},
            "max_reuse_utilization": {"operator": "<=", "value": 1.00},
        },
    }
    (specs / "robustness.json").write_text(
        json.dumps(robustness, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict] = []

    for key, domain in DOMAINS.items():
        folder = OUT / key
        folder.mkdir(parents=True, exist_ok=True)
        # 种子必须确定性: Python 的 hash(str) 受 PYTHONHASHSEED 随机化影响,
        # 用域名在字典中的序号作为偏移。
        drivers = build_drivers(
            domain, "2029-01-01", args.years, args.seed + 101 * list(DOMAINS).index(key)
        )
        drivers.to_csv(folder / "timeseries.csv", index=False)

        project = base_project(key, domain, years=args.years)
        (folder / "project.json").write_text(
            json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        for tech in TECHNOLOGIES:
            (folder / f"tech_{tech}.json").write_text(
                json.dumps(apply_technology(project, tech), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        for policy_name, policy_value in ALLOCATION_POLICIES.items():
            variant = copy.deepcopy(project)
            variant["demand_allocation_policy"] = policy_value
            (folder / f"{policy_name}.json").write_text(
                json.dumps(variant, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        # 能源富集补充情景 (仅 HA 端叠加)
        if key == "ha":
            rich = copy.deepcopy(project)
            dc = rich["components"]["AI_DC1"]
            dc["grid_connection_capacity_mw"] = ENERGY_RICH_OVERLAY["grid_connection_capacity_mw"]
            dc["offsite_electricity_water_intensity_l_kwh"] = ENERGY_RICH_OVERLAY[
                "offsite_water_intensity_l_kwh"
            ]
            (folder / "energy_rich.json").write_text(
                json.dumps(rich, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        summer = drivers[pd.to_datetime(drivers["date"]).dt.month.isin((6, 7, 8))]
        diagnostics.append({
            "domain": key,
            "label": domain["label"],
            "days": len(drivers),
            "t_mean": round(float(drivers["temperature_c"].mean()), 2),
            "t_max": round(float(drivers["temperature_c"].max()), 2),
            "t_min": round(float(drivers["temperature_c"].min()), 2),
            "twb_summer_mean": round(float(summer["wet_bulb_c"].mean()), 2),
            "twb_max": round(float(drivers["wet_bulb_c"].max()), 2),
            "free_cooling_hours_equiv_days_twb_lt_10": int((drivers["wet_bulb_c"] < 10).sum()),
            "rain_annual_mm": round(float(drivers["rainfall_mm"].sum()) / args.years, 1),
            "inflow_mean_ml": round(float(drivers["reservoir_inflow_ml"].mean()), 1),
        })

    write_specs(OUT)

    with (OUT / "parameter_sources.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", "unit", "source", "evidence_type"])
        writer.writerows(PARAMETER_SOURCES)

    (OUT / "domain_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
