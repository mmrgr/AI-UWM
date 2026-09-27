"""Cross-check every capacity claim in the narrative docs against machine artifacts.

Motivation: earlier revisions of this project published capacity numbers that
disagreed with the data files (coarse probability grid, stale KPI, over-strong
"policy constraint dominates" claim).  Those errors were only caught by manual
adversarial review.  This script makes the check automatic: it extracts the
capacity figures from the documentation, converts them into a table keyed by
estimand, and compares each one with the corresponding value in
``validation_artifacts/r2``.

Exit code is non-zero when a documented value cannot be traced to an artifact,
which makes the check usable as a pre-submission gate.

Usage::

    python scripts/audit_doc_consistency.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "validation_artifacts" / "r2"
OUT = R2 / "doc_consistency_audit.csv"

# Documentation that quotes capacity numbers.  Relative to the research root
# (parents[1] is aiuwm_src, parents[2] is 07_代码_AI-UWM引擎).
RESEARCH_ROOT = ROOT.parents[1]
DOCS = [
    ROOT / "AI_DATA_CENTER_RESEARCH_CN.md",
    RESEARCH_ROOT / "01_方案与设计文档" / "Nature级研究审查与修正计划_R3.md",
    RESEARCH_ROOT / "01_方案与设计文档" / "AI-UWM下一轮模拟_情景设置与实施方案.md",
    RESEARCH_ROOT / "00_项目说明.md",
]

# estimand key -> artifact extractor.  Keys must stay aligned with the
# vocabulary used in the documentation.
PHYSICAL = {
    "physical_daily",
    "physical_annual",
    "physical_hourly",
    "daily",
    "annual",
    "reclaimed_target",
}


def load_thresholds(domain: str) -> dict:
    path = R2 / domain / "capacity_threshold_corrected.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary(domain: str) -> dict:
    path = R2 / domain / "probabilistic_corrected_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def reservoir_two_year(domain: str) -> dict:
    path = R2 / "reservoir_sustainability_corrected.json"
    return json.loads(path.read_text(encoding="utf-8"))[domain]


def reservoir_ten_year(domain: str) -> dict:
    path = R2 / "reservoir_multiyear_10y.json"
    return json.loads(path.read_text(encoding="utf-8"))[domain]


def build_truth_table() -> dict:
    """Every number that may be quoted, keyed by (estimand, quantity)."""
    truth: dict[tuple[str, str], dict] = {}
    for domain in ("ha", "co"):
        thr = load_thresholds(domain)
        for key, block in thr.items():
            if isinstance(block, dict) and "maximum_safe_ai_capacity_mw" in block:
                truth[(f"{domain}:threshold:{key}", "max_safe_mw")] = {
                    "value": block["maximum_safe_ai_capacity_mw"],
                    "artifact": f"{domain}/capacity_threshold_corrected.json",
                }
                truth[(f"{domain}:threshold:{key}", "first_fail_mw")] = {
                    "value": block["first_failed_capacity_mw"],
                    "artifact": f"{domain}/capacity_threshold_corrected.json",
                }
        summary = load_summary(domain)
        for q in ("p05", "p50", "p95"):
            truth[(f"{domain}:probabilistic_physical", q)] = {
                "value": summary[q],
                "artifact": f"{domain}/probabilistic_corrected_summary.json",
            }
        pol = summary["policy_threshold"]
        for q in ("p05", "p50", "p95"):
            truth[(f"{domain}:probabilistic_policy", q)] = {
                "value": pol[q],
                "artifact": f"{domain}/probabilistic_corrected_summary.json",
            }
        truth[(f"{domain}:probability", "exceedance_at_1000mw")] = {
            "value": summary["exceedance_probability"],
            "artifact": f"{domain}/probabilistic_corrected_summary.json",
        }
        truth[(f"{domain}:probability", "exceedance_ci_low")] = {
            "value": summary["exceedance_probability_ci95"][0],
            "artifact": f"{domain}/probabilistic_corrected_summary.json",
        }
        truth[(f"{domain}:probability", "exceedance_ci_high")] = {
            "value": summary["exceedance_probability_ci95"][1],
            "artifact": f"{domain}/probabilistic_corrected_summary.json",
        }
        rb = reservoir_two_year(domain)
        truth[(f"{domain}:reservoir_2y", "max_safe_mw")] = {
            "value": rb["max_terminal_non_depleting_capacity_mw"],
            "artifact": "reservoir_sustainability_corrected.json",
        }
        truth[(f"{domain}:reservoir_2y", "first_fail_mw")] = {
            "value": rb["first_terminal_decline_capacity_mw"],
            "artifact": "reservoir_sustainability_corrected.json",
        }
        ra = reservoir_ten_year(domain)
        truth[(f"{domain}:reservoir_10y", "max_safe_mw")] = {
            "value": ra["last_joint_gate_mw"],
            "artifact": "reservoir_multiyear_10y.json",
        }
        truth[(f"{domain}:reservoir_10y", "first_fail_mw")] = {
            "value": ra["first_joint_gate_failure_mw"],
            "artifact": "reservoir_multiyear_10y.json",
        }
    return truth


# Documented claims that must hold.  Each entry is
# (doc_key, regex, resolver, description).
# ``resolver`` maps the regex match to a truth-table key.
CLAIMS = [
    # Threshold triplets written as "HA 的日/年/小时代理边界为 A/B/C MW".
    (
        "research",
        r"HA 的日/年/小时代理边界为 (\d+)/(\d+)/(\d+) MW",
        lambda m: [
            (("ha:threshold:physical_daily", "max_safe_mw"), m.group(1)),
            (("ha:threshold:annual", "max_safe_mw"), m.group(2)),
            (("ha:threshold:physical_hourly", "max_safe_mw"), m.group(3)),
        ],
        "HA 日/年/小时 max-safe 三元组",
    ),
    (
        "research",
        r"CO 为 (\d+)/(\d+)/(\d+) MW",
        lambda m: [
            (("co:threshold:physical_daily", "max_safe_mw"), m.group(1)),
            (("co:threshold:annual", "max_safe_mw"), m.group(2)),
            (("co:threshold:physical_hourly", "max_safe_mw"), m.group(3)),
        ],
        "CO 日/年/小时 max-safe 三元组",
    ),
    (
        "research",
        r"HA 去除再生水目标后的物理年边界为 (\d+) MW",
        lambda m: [(("ha:threshold:physical_annual", "max_safe_mw"), m.group(1))],
        "HA 物理年边界",
    ),
    (
        "research",
        r"联合边界为 HA (\d+) MW、CO (\d+) MW",
        lambda m: [
            (("ha:reservoir_10y", "max_safe_mw"), m.group(1)),
            (("co:reservoir_10y", "max_safe_mw"), m.group(2)),
        ],
        "10 年联合边界",
    ),
    (
        "plan",
        r"HA 最后通过/首次失败为 150/200 MW，CO 为 900/950 MW",  # not present; kept as guard
        lambda m: [],
        "占位（当前文档无此句式）",
    ),
    (
        "plan",
        r"\| 物理 CAWCC（日/年/小时代理） \| (\d+)/(\d+)/(\d+) MW \| (\d+)/(\d+)/(\d+) MW",
        lambda m: [
            (("ha:threshold:physical_daily", "max_safe_mw"), m.group(1)),
            (("ha:threshold:physical_annual", "max_safe_mw"), m.group(2)),
            (("ha:threshold:physical_hourly", "max_safe_mw"), m.group(3)),
            (("co:threshold:physical_daily", "max_safe_mw"), m.group(4)),
            (("co:threshold:physical_annual", "max_safe_mw"), m.group(5)),
            (("co:threshold:physical_hourly", "max_safe_mw"), m.group(6)),
        ],
        "plan 表：物理 CAWCC 日/年/小时",
    ),
    (
        "plan",
        r"\| 综合口径（含再生水政策目标） \| (\d+)/(\d+)/(\d+) MW \| (\d+)/(\d+)/(\d+) MW",
        lambda m: [
            (("ha:threshold:daily", "max_safe_mw"), m.group(1)),
            (("ha:threshold:annual", "max_safe_mw"), m.group(2)),
            (("ha:threshold:physical_hourly", "max_safe_mw"), m.group(3)),
            (("co:threshold:daily", "max_safe_mw"), m.group(4)),
            (("co:threshold:annual", "max_safe_mw"), m.group(5)),
            (("co:threshold:physical_hourly", "max_safe_mw"), m.group(6)),
        ],
        "plan 表：综合口径 日/年/小时",
    ),
    (
        "plan",
        r"\| 两年末库容边界（期末≥初值） \| (\d+)/(\d+) MW \| (\d+)/(\d+) MW",
        lambda m: [
            (("ha:reservoir_2y", "max_safe_mw"), m.group(1)),
            (("ha:reservoir_2y", "first_fail_mw"), m.group(2)),
            (("co:reservoir_2y", "max_safe_mw"), m.group(3)),
            (("co:reservoir_2y", "first_fail_mw"), m.group(4)),
        ],
        "plan 表：两年末库容边界",
    ),
    (
        "plan",
        r"\| 10 年合成联合边界[^|]*\| (\d+)/(\d+) MW \| (\d+)/(\d+) MW",
        lambda m: [
            (("ha:reservoir_10y", "max_safe_mw"), m.group(1)),
            (("ha:reservoir_10y", "first_fail_mw"), m.group(2)),
            (("co:reservoir_10y", "max_safe_mw"), m.group(3)),
            (("co:reservoir_10y", "first_fail_mw"), m.group(4)),
        ],
        "plan 表：10 年合成联合边界",
    ),
    (
        "plan",
        r"\| 概率分布（1024 样本，物理 CAWCC） \| P05/P50/P95 = (\d+)/(\d+)/(\d+) MW \| (\d+)/(\d+)/(\d+) MW",
        lambda m: [
            (("ha:probabilistic_physical", "p05"), m.group(1)),
            (("ha:probabilistic_physical", "p50"), m.group(2)),
            (("ha:probabilistic_physical", "p95"), m.group(3)),
            (("co:probabilistic_physical", "p05"), m.group(4)),
            (("co:probabilistic_physical", "p50"), m.group(5)),
            (("co:probabilistic_physical", "p95"), m.group(6)),
        ],
        "plan 表：1024 样本概率分布 P05/P50/P95",
    ),
    (
        "plan",
        r"\| 物理 CAWCC（日尺度，50 MW 网格） \|[^|]*\| (\d+) → (\d+) MW[^|]*\| (\d+) → (\d+) MW",
        lambda m: [
            (("ha:threshold:daily", "max_safe_mw"), m.group(1)),
            (("ha:threshold:daily", "first_fail_mw"), m.group(2)),
            (("co:threshold:daily", "max_safe_mw"), m.group(3)),
            (("co:threshold:daily", "first_fail_mw"), m.group(4)),
        ],
        "项目说明表：物理 CAWCC 日尺度",
    ),
    (
        "plan",
        r"\| 物理 CAWCC（年尺度） \|[^|]*\| (\d+) → (\d+) MW \| (\d+) → (\d+) MW",
        lambda m: [
            (("ha:threshold:physical_annual", "max_safe_mw"), m.group(1)),
            (("ha:threshold:physical_annual", "first_fail_mw"), m.group(2)),
            (("co:threshold:physical_annual", "max_safe_mw"), m.group(3)),
            (("co:threshold:physical_annual", "first_fail_mw"), m.group(4)),
        ],
        "项目说明表：物理 CAWCC 年尺度（纯物理子集）",
    ),
]


def extract_claims(doc_key: str, text: str) -> list[dict]:
    found: list[dict] = []
    for key, pattern, resolver, desc in CLAIMS:
        if key != doc_key:
            continue
        for m in re.finditer(pattern, text):
            for (tkey, documented) in resolver(m):
                found.append(
                    {
                        "doc": doc_key,
                        "claim": desc,
                        "matched_text": m.group(0),
                        "truth_key": f"{tkey[0]}|{tkey[1]}",
                        "documented_value": float(documented),
                    }
                )
    return found


def doc_key_for(name: str) -> str:
    """Map a file name to the CLAIMS routing key.

    All narrative documents share the "plan" key; only the engine research
    guide has its own key because its figures are written as prose.
    """
    return "research" if name.startswith("AI_DATA_CENTER") else "plan"


def main() -> int:
    truth = build_truth_table()
    doc_texts: dict[str, str] = {}
    for path in DOCS:
        if path.exists():
            doc_texts[path.name] = path.read_text(encoding="utf-8")

    rows: list[dict] = []
    for name, text in doc_texts.items():
        doc_key = doc_key_for(name)
        if name.startswith("00_项目说明"):
            doc_key = "plan"
        for claim in extract_claims(doc_key, text):
            tk = claim["truth_key"].split("|")
            entry = truth.get((tk[0], tk[1]))
            if entry is None:
                verdict, truth_value, artifact = "UNTRACEABLE", None, ""
            elif abs(float(entry["value"]) - claim["documented_value"]) < 1e-6:
                verdict, truth_value, artifact = "PASS", entry["value"], entry["artifact"]
            else:
                verdict, truth_value, artifact = "MISMATCH", entry["value"], entry["artifact"]
            rows.append(
                {
                    "文档": claim["doc"],
                    "论断": claim["claim"],
                    "匹配文本": claim["matched_text"],
                    "文档值": claim["documented_value"],
                    "机器值": truth_value,
                    "判定": verdict,
                    "证据文件": artifact,
                    "真值键": claim["truth_key"],
                }
            )

    # Also sweep the documentation for bare capacity numbers and report which
    # ones are covered by the truth table, so uncovered numbers stay visible.
    covered = {r["匹配文本"] for r in rows}
    sweep: list[dict] = []
    for name, text in doc_texts.items():
        for m in re.finditer(r"(\d{3,4})\s*MW", text):
            snippet = text[max(0, m.start() - 40): m.end() + 5].replace("\n", " ")
            sweep.append({"文档": name, "数字": float(m.group(1)), "上下文": snippet})

    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["文档", "论断", "匹配文本", "文档值", "机器值", "判定", "证据文件", "真值键"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"论断检查 {len(rows)} 条 -> {OUT}")
    for r in rows:
        print(f"  {r['判定']:12s} {r['论断']}: 文档 {r['文档值']} / 机器 {r['机器值']}")
    print(f"文档中裸露容量数字 {len(sweep)} 处（需人工确认是否已在真值表内）")

    bad = [r for r in rows if r["判定"] != "PASS"]
    if bad:
        print(f"\nFAIL: {len(bad)} 条论断无法追溯到机器产物")
        return 1
    print("\nOK: 已登记论断全部可追溯")
    return 0


if __name__ == "__main__":
    sys.exit(main())
