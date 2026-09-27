"""R2 外部有效性：模型输出 vs 文献/行业披露值 的定量对标（任务 E）。

模型可信度不能只在内部自检（V1–V12）成立，还要放在真实世界的量级里看。
本脚本把模型产出的 WUE、PUE 与公开披露的数据中心运行值做定量对标。

！！！重要的方法学观察（本轮发现，值得写进论文）！！！
同一家公司的 WUE 在不同来源之间可相差近一倍（例如 Google FY24:
Goldman Sachs 表给 1.2 L/kWh，另一行业来源给 fleet-wide 0.64 L/kWh）。
差异来自 **口径**：取水量 vs 耗水量、是否含电网间接水、地域范围。
因此跨文献比较 WUE 之前必须先统一口径——这与本方案里已经强调的
"毛替代率 vs 净替代率"是同一类问题的两个实例。

来源性质标注（诚实原则）：
  DISCLOSURE 公司自行披露的可持续发展数据（经第三方报告摘录转述）
  INDUSTRY   行业协会/行业网站给出的基准区间
  GREY       券商研究报告或二手摘录页
除 charity 外，所有值在论文正式引用前**必须回溯到公司原始年报/ESG 报告**。

用法:
    PYTHONPATH="src;scripts" python scripts/build_literature_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = ROOT / "validation_artifacts" / "r2"

# ---------------------------------------------------------------------------
# 文献/披露侧：填写真实来源，不给值的地方留空
# ---------------------------------------------------------------------------
BENCHMARKS = [
    # --- 公司披露 (FY2023-2025) ---
    ("Google", 2024, "PUE", 1.09, 1.09, "DISCLOSURE",
     "FY24 PUE；转述自券商研究报告摘录（高盛 Exhibit 9）", "待回溯至 Google 可持续发展报告"),
    ("Google", 2024, "WUE_L_per_kWh", 1.2, 1.2, "DISCLOSURE",
     "FY24 WUE 1.2 L/kWh；转述自券商研究报告摘录", "待回溯"),
    ("Google", 2024, "WUE_L_per_kWh", 0.64, 0.64, "DISCLOSURE",
     "同一年另一来源报 fleet-wide 0.64 L/kWh——口径差异示例",
     "与上一行同一公司同一年相差近 1 倍，说明 WUE 口径不统一"),
    ("Meta", 2024, "PUE", 1.07, 1.07, "DISCLOSURE", "FY24 PUE；券商摘录", "待回溯"),
    ("Meta", 2024, "WUE_L_per_kWh", 0.2, 0.2, "DISCLOSURE", "FY24 WUE；券商摘录", "待回溯"),
    ("Meta", 2024, "WUE_L_per_kWh", 0.26, 0.26, "INDUSTRY",
     "行业来源报 fleet-wide 0.26 L/kWh（2024，空气侧自然冷却占比高）", "同一年不同来源小差异"),
    ("Amazon/AWS", 2024, "PUE", 1.14, 1.14, "DISCLOSURE", "FY24 PUE；券商摘录", "待回溯"),
    ("Amazon/AWS", 2023, "WUE_L_per_kWh", 0.18, 0.18, "DISCLOSURE", "2023 WUE；券商摘录", "待回溯"),
    ("Microsoft", 2024, "PUE", 1.16, 1.18, "DISCLOSURE", "FY24 PUE；两来源分别给 1.16 / 1.18", "待回溯"),
    ("Microsoft", 2024, "WUE_L_per_kWh", 0.30, 0.49, "DISCLOSURE",
     "FY24 WUE：券商摘录 0.30；行业来源报 fleet-wide 0.49", "口径差异示例"),
    # --- 中国厂商 ---
    ("腾讯", 2024, "PUE", 1.269, 1.269, "DISCLOSURE", "2024 PUE；券商摘录表", "待回溯至腾讯 ESG 报告"),
    ("阿里巴巴", 2025, "PUE", 1.19, 1.253, "DISCLOSURE",
     "2025 PUE 1.19（自建）/ 1.253（租赁 IDC）", "待回溯"),
    ("阿里巴巴", 2025, "WUE_L_per_kWh", 1.144, 1.8, "DISCLOSURE",
     "2025 WUE 1.144 全站均值；蒸发冷却站点 1.8，风冷站点 0.329",
     "该来源同时给出分技术值，最适合与本文技术对照比"),
    ("百度", 2024, "PUE", 1.2, 1.2, "DISCLOSURE", "2024 PUE", "待回溯"),
    ("百度", 2024, "WUE_L_per_kWh", 1.61, 1.61, "DISCLOSURE", "2024 WUE", "待回溯"),
    ("Equinix", 2024, "PUE", 1.39, 1.39, "DISCLOSURE", "2024 PUE（托管型）", "待回溯"),
    ("Equinix", 2024, "WUE_L_per_kWh", 0.95, 1.55, "DISCLOSURE",
     "2024 WUE 0.95 全站；蒸发冷却站点 1.55", "待回溯"),
    # --- 行业基准区间 ---
    ("行业基准-无水冷却", 2025, "WUE_L_per_kWh", 0.0, 0.0, "INDUSTRY",
     "风冷/自然冷却：蒸发量为零时 WUE = 0", "理论下界"),
    ("行业基准-领先超大规模", 2025, "WUE_L_per_kWh", 0.2, 0.5, "INDUSTRY",
     "领先超大规模运营商区间", "行业网站基准"),
    ("行业基准-平均托管型", 2025, "WUE_L_per_kWh", 1.0, 1.8, "INDUSTRY",
     "一般 colocation 设施区间", "行业网站基准"),
    ("行业基准-老旧低效", 2025, "WUE_L_per_kWh", 2.0, 4.0, "INDUSTRY",
     "老旧或低效设施区间", "行业网站基准"),
    ("行业基准-蒸发冷却消耗率", 2025, "WUE_L_per_kWh", 1.8, 4.0, "INDUSTRY",
     "蒸发冷却的出现实消耗率（依气候与系统效率）", "行业网站基准"),
]

# ---------------------------------------------------------------------------
# 城市侧对标锚点（均为已核实到官方原文或权威转述的标准/文件）
# ---------------------------------------------------------------------------
# 字段: (指标, 对标主体, 值_低, 值_高, 来源性质, 年份, 来源说明, 口径与风险备注,
#        比较方式, 对标口径)
#   比较方式: 区间(值落在区间内为佳) / 上限(值 ≤ 阈值达标) / 下限(值 ≥ 阈值达标) / 参考点(只报偏差)
#   对标口径: 平均日 / 最高日 / 任意 —— 只与模型侧同口径的值比对，避免平均日 vs 最高日混比
CITY_BENCHMARKS = [
    ("综合漏损率_pct", "CJJ92-2016 评定标准 一级", 10.0, 10.0, "OFFICIAL", 2018,
     "《城镇供水管网漏损控制及评定标准》CJJ 92-2016（住建部 2018-12-27 局部修订）"
     "5.1.2：漏损率应按两级评定，一级为 10%、二级为 12%",
     "2018 局部修订已废止原修正条款，评定标准即 10/12 两档；"
     "转引自四川省发改委答复(2022-10-31)与国标电子书库原文",
     "上限", "任意"),

    ("综合漏损率_pct", "CJJ92-2016 评定标准 二级", 12.0, 12.0, "OFFICIAL", 2018,
     "同上，二级为 12%", "同上", "上限", "任意"),

    ("综合漏损率_pct", "全国实际水平(2023 年鉴口径)", 12.76, 12.76, "GREY", 2023,
     "《2023 年城乡建设统计年鉴》：城市供水总量约 650.61 亿 m³、漏损水量约 82.99 亿 m³，"
     "漏损率约 12.76%",
     "转引自期刊论文《基于卷积神经网络支持向量机的城市供水管网漏损检测》；"
     "正式引用前必须回溯年鉴原文核对",
     "参考点", "任意"),

    ("综合漏损率_pct", "政策目标 2025", 9.0, 9.0, "OFFICIAL", 2022,
     "住建部、发改委《关于加强公共供水管网漏损控制的通知》：到 2025 年"
     "全国城市公共供水管网漏损率力争控制在 9% 以内",
     "文件名在两处转述中不一致（另一处记作《关于进一步加强水资源节约集约利用的意见》"
     "2023-09），9% 目标本身两来源一致；文件名待核定",
     "上限", "任意"),

    ("人均综合生活用水_L_per_cap_d",
     "GB50013-2018 最高日 Ⅱ型大城市 一区", 220.0, 400.0, "OFFICIAL", 2018,
     "《室外给水设计标准》GB 50013-2018 表 4.0.3-3 最高日综合生活用水定额 "
     "[L/(人·d)]：Ⅱ型大城市 一区 220~400",
     "Ⅱ型大城市 = 城区常住人口 100 万以上 300 万以下；模型人口 100 万，取下界",
     "区间", "最高日"),

    ("人均综合生活用水_L_per_cap_d",
     "GB50013-2018 最高日 Ⅱ型大城市 二区", 150.0, 260.0, "OFFICIAL", 2018,
     "GB 50013-2018 表 4.0.3-3：Ⅱ型大城市 二区 150~260",
     "二区含北京、天津、河北、山东、河南、陕西等；模型未声明分区",
     "区间", "最高日"),

    ("人均综合生活用水_L_per_cap_d",
     "GB50013-2018 平均日 Ⅱ型大城市 一区", 140.0, 300.0, "OFFICIAL", 2018,
     "GB 50013-2018 表 4.0.3-4 平均日综合生活用水定额：Ⅱ型大城市 一区 140~300",
     "与模型输出的日均口径一致",
     "区间", "平均日"),

    ("人均综合生活用水_L_per_cap_d",
     "GB50013-2018 平均日 Ⅱ型大城市 二区", 90.0, 170.0, "OFFICIAL", 2018,
     "GB 50013-2018 表 4.0.3-4：Ⅱ型大城市 二区 90~170", "同上",
     "区间", "平均日"),

    ("再生水利用率_pct", "十四五目标 地级及以上缺水城市", 25.0, 25.0, "OFFICIAL", 2021,
     "《十四五城镇污水处理及资源化利用发展规划》(发改环资〔2021〕827 号)："
     "到 2025 年全国地级及以上缺水城市再生水利用率达到 25% 以上",
     "口径 = 再生水利用量 / 污水处理量；与模型的『再生水替代率』分母不同，见口径警示表",
     "下限", "任意"),

    ("再生水利用率_pct", "十四五基准年水平", 20.0, 20.0, "OFFICIAL", 2021,
     "国家发改委《十四五规划纲要》章节指标解读：我国地级及以上缺水城市"
     "污水资源化利用率为 20% 左右", "2020 前后水平，用于判断模型值是否合理",
     "参考点", "任意"),

    ("再生水利用率_pct", "京津冀目标", 35.0, 35.0, "OFFICIAL", 2021,
     "发改环资〔2021〕827 号：京津冀地区达到 35% 以上", "缺水程度更高的分区目标",
     "下限", "任意"),
]

# ---------------------------------------------------------------------------
# 口径冲突警示：同一指标在不同来源/不同定义下不可直接比较的实例
# 这是本轮最值得写进方法论的一张表——与"毛替代率 vs 净替代率"同类。
# ---------------------------------------------------------------------------
CALIBER_WARNINGS = [
    ("WUE", "同一公司同一年份相差近 1 倍", "1.2 L/kWh", "0.64 L/kWh",
     "Google FY2024",
     "取水量(withdrawal) vs 耗水量(consumption)、是否计入电网间接用水、"
     "地域范围(单站 vs fleet-wide)三者任一不同，WUE 即可相差近一倍",
     "跨文献比较 WUE 前必须统一为同一口径；本文一律使用取水量口径并显式标注"),

    ("城市供水管网漏损率", "同一年份官方口径相差 >2.8 个百分点",
     "12.76%", "10% 以内",
     "2023 年中国城市",
     "《城乡建设统计年鉴》口径 = 漏损水量/供水总量；"
     "水利部对外口径为『城市公共供水管网漏损率』，统计范围与修正方式不同",
     "引用时必须写明是哪一年鉴/哪一部门口径，不能混用"),

    ("再生水利用指标", "政策文件与模型的分母不同",
     "再生水利用率 = 再生水利用量/污水处理量",
     "再生水替代率 = 再生水替代量/总供水量（模型约束）",
     "国家政策 vs AI-UWM 模型",
     "两个指标分子相近但分母差一个量级，数值不可直接互相检验",
     "本文不把模型的『替代率』阈值与政策『利用率』目标做数值比对，"
     "只做方向性说明"),

    ("替代率（论文内部）", "毛替代率 vs 净替代率",
     "毛替代率 = 再生水供应量/总供水",
     "净替代率 = 扣除再生水生产自身耗水后的有效替代",
     "AI-UWM 内部两个口径",
     "毛替代率高估真实节水效果，因为再生水生产本身也消耗水与能",
     "已在 reclaimed_net_substitution.csv 中双口径并列，正文不得只用毛值"),

    ("市政单位造价", "同一指标在不同来源下相差 5 倍",
     "1100 元/(m³·d)（中央预算内投资估算标准）",
     "5595 元/(m³·d)（实际项目决算）；5000–7000（行业造价库）",
     "中水回用工程单位投资",
     "中央预算内投资绩效目标表给的是**补助估算标准**，很可能只覆盖部分工程内容"
     "或仅指提标改造增量；项目决算给的是完整工程含摊销的实际支出",
     "本文不采用 1100 这一值；采用 5000–7000 时须标注"
     "『可能偏高（案例规模比本文情景小两个数量级）』"),

    ("市政单位造价", "官方指标带**价格基准年**，不能与当期价直比",
     "给水：210–278 元/(m³·d)（北京市 1990 年价）",
     "污水：1865–3825 元/(m³·d)（武汉市 2016 年 12 月价）",
     "建标 120-2009 表 7 / 建标 198-2022 表 5",
     "两本标准的表注都写明按指定年份、指定城市的预算价格计算，"
     "使用前须按当时当地价格指数调整；且都是**新建**口径，不含征地拆迁",
     "本文的造价是**当期价的扩容增量**，与官方锚点**口径与价格基准年都不同**，"
     "只作量级参照并显式标注差异方向，不做直接数值互检"),
]

MODEL_SOURCE_NOTE = (
    "模型值为本轮 AI-UWM 引擎输出：现场 WUE = 取水量(ML)×1000 / IT 能耗(MWh)，"
    "即 L/kWh；过程 WUE 另以 external_makeup_ml 计算；"
    "驱动为合成气候序列，不是任何真实城市或真实数据中心。"
)


def load_model_values() -> pd.DataFrame:
    """取模型侧可与文献对标的指标：WUE / PUE。"""
    rows: list[dict] = []

    # 1) 冷却技术对照（最能和"蒸发/风冷站点"的分技术披露值比）
    for domain, label in (("ha", "D-HA"), ("co", "D-CO")):
        path = OUT / domain / "technology_comparison.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig")
        for _, record in frame.iterrows():
            rows.append({
                "来源": f"模型-{label}",
                "对象": record["technology"],
                "指标": "WUE_L_per_kWh",
                "值": float(record["wue_l_kwh"]),
                "附-Pue": float(record.get("mean_pue", float("nan"))),
            })

    # 2) 时间序列情景清单里的常规情景（不同 AI 容量）
    manifest = OUT.parent / "timeseries" / "manifest.csv"
    if manifest.exists():
        frame = pd.read_csv(manifest, encoding="utf-8-sig")
        for _, record in frame.iterrows():
            if str(record.get("情景类型")) != "容量扫描":
                continue
            rows.append({
                "来源": f"模型-{record['域标签'].split()[0]}",
                "对象": f"容量 {record['AI容量_MW']:.0f} MW",
                "指标": "WUE_L_per_kWh",
                "值": float(record.get("WUE_L_per_kWh", float("nan"))),
                "附-Pue": float(record.get("PUE", float("nan"))),
            })
    return pd.DataFrame(rows)


def load_model_city_values() -> pd.DataFrame:
    """从逐日过程 dump（cap_0000 = 无 AI 负荷的现状城市基线）算城市侧指标。

    每个指标都尽量给出**两个口径**，因为本方案的核心方法论结论之一就是
    "口径不同则数值不可比"，与其在正文里解释，不如把两个口径都算出来。
    """
    ts_root = OUT.parent / "timeseries"
    rows: list[dict] = []

    for domain, label in (("ha", "D-HA"), ("co", "D-CO")):
        base = ts_root / domain / "cap_0000"
        if not (base / "system_daily.csv").exists():
            continue
        sysd = pd.read_csv(base / "system_daily.csv", encoding="utf-8-sig")
        comp = pd.read_csv(base / "component_annual.csv", encoding="utf-8-sig")
        area = pd.read_csv(base / "area_daily.csv", encoding="utf-8-sig")
        project = json.loads(
            (ROOT / "examples" / "cawcc_r2" / domain / "project.json").read_text(encoding="utf-8")
        )

        days = float(len(sysd))
        pop = float(area["population"].iloc[-1])
        agg = comp.groupby("component_id")[["inflow_ml", "leakage_ml", "treated_ml"]].sum()

        def loss_pct(cid: str) -> float | None:
            if cid not in agg.index or agg.loc[cid, "inflow_ml"] <= 0:
                return None
            return float(agg.loc[cid, "leakage_ml"] / agg.loc[cid, "inflow_ml"] * 100.0)

        # --- 漏损率：三个口径 ---
        dm, sc, tm = loss_pct("DM1"), loss_pct("SC1"), loss_pct("TM1")
        survive = 1.0
        for value in (sc, tm, dm):
            if value is not None:
                survive *= (1.0 - value / 100.0)
        chain_loss = (1.0 - survive) * 100.0
        sys_loss = float(sysd["leakage_ml"].sum())
        sys_delivered = float(sysd["delivered_total_ml"].sum())
        system_loss = sys_loss / (sys_loss + sys_delivered) * 100.0 if (sys_loss + sys_delivered) else None

        # --- 人均用水：居民生活口径 / 综合生活代理口径；平均日与最高日分别算 ---
        daily_domestic = area.groupby("date")["demand_domestic_ml"].sum()
        daily_misc = area.groupby("date")["demand_municipal_misc_ml"].sum()
        per_liter = 1.0e6  # 1 ML = 1e6 L
        cap_d_domestic = float(daily_domestic.mean()) * per_liter / pop
        cap_d_comprehensive = float((daily_domestic + daily_misc).mean()) * per_liter / pop
        peak_domestic = float(daily_domestic.max()) * per_liter / pop
        peak_comprehensive = float((daily_domestic + daily_misc).max()) * per_liter / pop

        # --- 再生水：政策口径（利用率）与模型口径（替代率）分别算 ---
        wwtw_treated = float(agg.loc["WWTW1", "treated_ml"]) if "WWTW1" in agg.index else 0.0
        reuse_treated = float(agg.loc["CENTRAL_REUSE", "treated_ml"]) if "CENTRAL_REUSE" in agg.index else 0.0
        reuse_delivered = float(sysd["reuse_delivered_ml"].sum())
        policy_ratio = reuse_treated / wwtw_treated * 100.0 if wwtw_treated else None
        model_ratio = reuse_delivered / sys_delivered * 100.0 if sys_delivered else None

        # --- 厂负荷率 ---
        wtw_cap = float(project["components"]["WTW1"].get("daily_capacity_ml", 0.0))
        wwtw_cap = float(project["components"]["WWTW1"].get("daily_capacity_ml", 0.0))
        wtw_load = (float(agg.loc["WTW1", "treated_ml"]) / (wtw_cap * days) * 100.0
                    if wtw_cap else None)
        wwtw_load = (wwtw_treated / (wwtw_cap * days) * 100.0 if wwtw_cap else None)

        rows.extend([
            {"来源": f"模型-{label}", "指标": "综合漏损率_pct", "口径": "配水干管(DM1)",
             "对比口径": "任意",
             "值": dm, "备注": "仅配水干管一段的漏损，对应 CJJ92 管网漏损的直接含义"},
            {"来源": f"模型-{label}", "指标": "综合漏损率_pct", "口径": "输配全链(SC1+TM1+DM1 串联)",
             "对比口径": "任意",
             "值": chain_loss, "备注": "从水源到用户的全程漏损，口径最宽"},
            {"来源": f"模型-{label}", "指标": "综合漏损率_pct", "口径": "系统口径(漏损/(漏损+供水))",
             "对比口径": "任意",
             "值": system_loss, "备注": "与年鉴口径(漏损水量/供水总量)最接近"},
            {"来源": f"模型-{label}", "指标": "人均综合生活用水_L_per_cap_d",
             "口径": "居民生活(domestic) 平均日", "对比口径": "平均日",
             "值": cap_d_domestic, "备注": "仅居民生活，不含公共建筑，应低于 GB50013 综合生活定额"},
            {"来源": f"模型-{label}", "指标": "人均综合生活用水_L_per_cap_d",
             "口径": "居民生活(domestic) 最高日", "对比口径": "最高日",
             "值": peak_domestic, "备注": "730 天内单日最大值，对应 GB50013 最高日定额"},
            {"来源": f"模型-{label}", "指标": "人均综合生活用水_L_per_cap_d",
             "口径": "居民生活+市政杂用(综合生活代理) 平均日", "对比口径": "平均日",
             "值": cap_d_comprehensive, "备注": "用作 GB50013『综合生活』的代理口径"},
            {"来源": f"模型-{label}", "指标": "人均综合生活用水_L_per_cap_d",
             "口径": "居民生活+市政杂用(综合生活代理) 最高日", "对比口径": "最高日",
             "值": peak_comprehensive, "备注": "同上，最高日"},
            {"来源": f"模型-{label}", "指标": "再生水利用率_pct", "口径": "政策口径(再生水/污水处理量)",
             "对比口径": "任意",
             "值": policy_ratio, "备注": "分母为污水处理量，与发改环资〔2021〕827号一致"},
            {"来源": f"模型-{label}", "指标": "再生水利用率_pct", "口径": "模型口径(再生水/总供水量)",
             "对比口径": "任意",
             "值": model_ratio, "备注": "分母为总供水量，即模型约束中的『替代率』；与上者分母不同，不可互换"},
            {"来源": f"模型-{label}", "指标": "水厂负荷率_pct", "口径": "产水/设计日能力",
             "对比口径": "任意",
             "值": wtw_load, "备注": "对照《城镇供水价格管理办法》成本监审中的产能利用口径"},
            {"来源": f"模型-{label}", "指标": "污水厂负荷率_pct", "口径": "处理量/设计日能力",
             "对比口径": "任意",
             "值": wwtw_load, "备注": "同上"},
            {"来源": f"模型-{label}", "指标": "供水保证率_pct", "口径": "delivered_percent 全期均值",
             "对比口径": "任意",
             "值": float(sysd["delivered_percent"].mean()), "备注": "无 AI 负荷下的城市基线保证率"},
        ])
    return pd.DataFrame(rows)


def main() -> None:
    benchmark_frame = pd.DataFrame(
        BENCHMARKS,
        columns=["主体", "年份", "指标", "值_低", "值_高", "来源性质", "来源说明", "备注"],
    )
    city_benchmark_frame = pd.DataFrame(
        CITY_BENCHMARKS,
        columns=["指标", "对标主体", "值_低", "值_高", "来源性质", "年份", "来源说明",
                 "口径与风险备注", "比较方式", "对标口径"],
    )
    model_frame = load_model_values()
    city_model_frame = load_model_city_values()

    # ---- 对标：模型值 vs 每段文献区间的位置判定 -----------------------------
    comparison_rows = []
    for _, model_row in model_frame.iterrows():
        value = model_row["值"]
        if not pd.notna(value):
            continue
        for _, bench in benchmark_frame.iterrows():
            if bench["指标"] != model_row["指标"]:
                continue
            low, high = float(bench["值_低"]), float(bench["值_高"])
            if bench["主体"].startswith("行业基准"):
                status = "在区间内" if low <= value <= high else "在区间外"
                deviation = 0.0 if status == "在区间内" else min(abs(value - low), abs(value - high))
            else:
                status = "可比"
                deviation = float(value) - (low + high) / 2.0
            comparison_rows.append({
                "模型对象": model_row["对象"],
                "模型来源": model_row["来源"],
                "指标": model_row["指标"],
                "模型值": round(float(value), 4),
                "对标主体": bench["主体"],
                "对标年份": bench["年份"],
                "对标区间低": low,
                "对标区间高": high,
                "判定": status,
                "偏差": round(deviation, 4),
                "相对偏差": (round(deviation / ((low + high) / 2.0), 4)
                             if (low + high) != 0 else None),
                "对标来源性质": bench["来源性质"],
                "对标来源说明": bench["来源说明"],
            })

    comparison = pd.DataFrame(comparison_rows)

    # ---- 城市侧对标：模型值 vs 官方标准/政策锚点 -----------------------------
    city_rows = []
    for _, mrow in city_model_frame.iterrows():
        value = mrow["值"]
        if value is None or not pd.notna(value):
            continue
        for _, bench in city_benchmark_frame.iterrows():
            if bench["指标"] != mrow["指标"]:
                continue
            # 同口径匹配：平均日模型值只与平均日定额比，最高日只与最高日比
            if bench["对标口径"] != "任意" and bench["对标口径"] != mrow.get("对比口径"):
                continue
            low, high = float(bench["值_低"]), float(bench["值_高"])
            mode = bench["比较方式"]
            if mode == "上限":
                status = "达标（≤限值）" if value <= high else "超标（>限值）"
                deviation = value - high
            elif mode == "下限":
                status = "达标（≥目标）" if value >= low else "未达标（<目标）"
                deviation = value - low
            elif mode == "参考点":
                status = "高于参考" if value > high else "低于参考"
                deviation = value - high
            else:  # 区间
                status = "在区间内" if low <= value <= high else "在区间外"
                deviation = 0.0 if status == "在区间内" else min(abs(value - low), abs(value - high))
            city_rows.append({
                "模型来源": mrow["来源"],
                "指标": mrow["指标"],
                "模型口径": mrow["口径"],
                "模型值": round(float(value), 2),
                "对标主体": bench["对标主体"],
                "对标年份": bench["年份"],
                "对标区间低": low,
                "对标区间高": high,
                "判定": status,
                "偏差": round(float(deviation), 2),
                "对标来源性质": bench["来源性质"],
                "对标来源说明": bench["来源说明"],
                "口径与风险备注": bench["口径与风险备注"],
                "模型口径备注": mrow["备注"],
            })
    city_comparison = pd.DataFrame(city_rows)

    # ---- 口径冲突警示表 -----------------------------------------------------
    caliber_frame = pd.DataFrame(
        CALIBER_WARNINGS,
        columns=["指标", "冲突类型", "口径A", "口径B", "涉及对象", "差异成因", "本文的处理原则"],
    )

    OUT.mkdir(parents=True, exist_ok=True)
    benchmark_frame.to_csv(OUT / "literature_benchmark_sources.csv", index=False, encoding="utf-8-sig")
    model_frame.to_csv(OUT / "literature_benchmark_model.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "literature_benchmark.csv", index=False, encoding="utf-8-sig")
    city_benchmark_frame.to_csv(
        OUT / "literature_benchmark_city_sources.csv", index=False, encoding="utf-8-sig")
    city_model_frame.to_csv(
        OUT / "literature_benchmark_city_model.csv", index=False, encoding="utf-8-sig")
    city_comparison.to_csv(
        OUT / "literature_benchmark_city.csv", index=False, encoding="utf-8-sig")
    caliber_frame.to_csv(
        OUT / "literature_benchmark_caliber.csv", index=False, encoding="utf-8-sig")

    print("=== 模型侧对标值 ===")
    print(model_frame.to_string(index=False))
    print(f"\n=== 与行业基准区间的位置判定（{len(comparison)} 组比对）===")
    summary = (
        comparison[comparison["对标主体"].str.startswith("行业基准")]
        .groupby(["对标主体", "判定"])
        .size()
        .unstack(fill_value=0)
    )
    print(summary.to_string())

    print("\n=== 城市侧：模型值 vs 官方标准/政策锚点（cap_0000 无 AI 基线）===")
    print(city_model_frame.to_string(index=False))
    print("\n--- 关键判定 ---")
    key = city_comparison[city_comparison["指标"].isin(
        ["综合漏损率_pct", "人均综合生活用水_L_per_cap_d", "再生水利用率_pct"])]
    print(key[["模型来源", "指标", "模型口径", "模型值", "对标主体", "判定"]
             ].to_string(index=False))

    print("\n=== 口径冲突警示（4 例，跨文献/跨定义比较前必读）===")
    print(caliber_frame[["指标", "冲突类型", "口径A", "口径B", "涉及对象"]].to_string(index=False))

    print(f"\n-> {OUT / 'literature_benchmark.csv'}")
    print(f"-> {OUT / 'literature_benchmark_sources.csv'}")
    print(f"-> {OUT / 'literature_benchmark_model.csv'}")
    print(f"-> {OUT / 'literature_benchmark_city.csv'}")
    print(f"-> {OUT / 'literature_benchmark_city_sources.csv'}")
    print(f"-> {OUT / 'literature_benchmark_city_model.csv'}")
    print(f"-> {OUT / 'literature_benchmark_caliber.csv'}")
    print(f"\n{MODEL_SOURCE_NOTE}")


if __name__ == "__main__":
    main()
