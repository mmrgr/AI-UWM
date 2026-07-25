# WaterMet² 论文精读与复现报告

> 当前工程包含两层实现：`scripts/run_reproduction.py` 用于 Oslo 论文结果的量级与趋势复现；`src/watermet2_repro/full_engine.py` 是覆盖官方手册功能的通用配置驱动引擎。完整能力及边界见 `FUNCTIONAL_COVERAGE_CN.md`，自己的数据接入步骤见中文 `README.md`。

## 1. 论文问题与贡献

论文要解决的问题不是管网水力瞬时过程，而是城市水系统在 20-30 年规划期内的整体“代谢”表现：水、能源、材料、化学品、污染物、温室气体、酸化、富营养化和成本如何在供水、用水、雨洪、污水与回用环节之间传递。

WaterMet² 的关键创新有三点：

1. 将供水、需求、污水/雨洪、循环水与资源回收放入同一日尺度质量平衡；
2. 用 System → Subcatchment → Local → Indoor 四个尺度表达任意复杂度的城市；
3. 水量求解后，用单位水量/物质量影响系数计算能源、GHG、酸化、富营养化和成本。

## 2. 原模型执行顺序

每天分两遍计算：

1. 从下游需求点向上游汇总需求，并逐层加入输水/配水漏损；
2. 从水源向下游释放，逐层受水源储量、输水、WTW 和配水能力限制。

核心储量方程为：

`S(i,t+1) = S(i,t) + I(i,t) - O(i,t)`

水源需求汇总为：

`RD(i,t) = Σ CF(i,j) × WD(j,t) × (1 + CL(i,j)/100)`

降雨径流以 Rational Method/SWMM 思路计算；污水与污染物采用完全混合、无弥散、无衰减的源-汇追踪。水量确定后，各组件的影响量按 `处理或输送水量 × 单位强度` 求得。

## 3. 案例系统

公开论文中的城市模型为单一 Subcatchment/Local area 的聚合模型：

- 两个地表水源、两座 WTW、两个服务水池和两条配水主线；
- WR1/WR2 容量 120/13.8 MCM，平均入流 287/12 MCM/year；
- WTW 能力 370/43.2 ML/day；
- 配水漏损 22%；
- 一套混合概念管网，37% combined、30% sanitary、33% storm；
- 两座 WWTW 分担 63%/37%，能力 770/320 ML/day；
- 2011-2040 日步长。

需求由 320,000 户、2.35 人/户构成。室内需求 180 L/(capita·day)，其中 dishwasher 3.2%、hand basin 12%、kitchen sink 12.8%、washing machine 16%、shower 25%、toilet 30%。工业、灌溉、防冻和未登记需求的参数已录入 `data/oslo_parameters.json`。

## 4. 三种状态

- BAU：不增加资本性措施。
- Added resource：2020 年启用新水源与处理能力。本复现将公开报告的 2,000 ML/day 无限入流概念和论文结果约束结合，使用 130 ML/day 的新增可用处理能力，使 2040 年最差月仍有约 4% 缺水；论文报告约 3%。
- Water recycling：2015 年起 50% 家庭采用 RWH + GWR。RWH 收集屋面、道路和铺装径流，供应厕所、工业与灌溉；GWR 收集洗浴、洗衣、洗碗和洗手盆灰水，再供应厕所、工业与灌溉。

## 5. 结果验证

本复现的关键行为与论文一致：

| 指标 | 论文描述 | 本复现 |
|---|---:|---:|
| BAU 2040 最差月未满足需求 | 约 27% | 约 27% |
| 新增水源 2040 最差月未满足需求 | 约 3% | 约 4% |
| 回用 2040 最差月未满足需求 | 约 14% | 约 15% |
| 新增水源对 GHG | 略升 | 平均年 GHG 较 BAU 上升约 4% |
| 回用对 GHG | 下降 | 平均年 GHG 较 BAU 下降约 9% |
| 回用对富营养化 | 明显下降 | 平均年 PO4-eq 下降约 12% |

这里比较的是趋势、量级与约束行为。由于原始逐日序列和闭源实现缺失，不应将代理序列产生的逐日点值解释为原作者结果。

## 6. 校准与论文质量判断

原文手动 trial-and-error 校准。供水部分用 2011 校准、2012 验证；污水部分用 2010 校准、2011 验证。报告指标：

- 供水：NSE 0.25/0.22，RSR 0.86/0.89，PBIAS -0.50%/-0.30%；
- 污水：NSE 0.51/0.56，RSR 0.70/0.67，PBIAS 6.02%/2.45%。

这说明模型适合战略比较和量级评估，但供水逐日预测能力有限。论文自己也强调：它不是详细水力模型，且案例因数据不足采用聚合表示。

## 7. 无法公开恢复的内容

已核查 ScienceDirect、欧盟 CORDIS、University of Exeter/Figshare、UWL Repository 及 TRUST 官网历史存档。可确认：

- 原始日尺度 SCADA、水源入流、气象、WTW 产水和 WWTW 入流没有随论文公开；
- D33.2 说明 WaterMet² 是 C#/.NET 闭源程序，核心封装为 `WaterMet2.dll`；
- 软件当时“通过向开发者申请或 TRUST 网站获取”，当前 TRUST 下载链接失效；
- Internet Archive/Common Crawl 对相关二进制仅保存了截断对象，无法可靠恢复完整安装包；
- CORDIS 所列 `iddesc=140` 的历史文件实际为 DSS Strategic Planning 软件，不是 WaterMet² 源码。

因此本工程没有伪造“官方原始数据”或冒充原 DLL，而是把全部代理和标定参数显式化。

## 8. 工程映射

- 模型与方程：`src/watermet2_repro/model.py`
- 论文/手册参数：`data/oslo_parameters.json`
- 代理数据生成、三场景运行和作图：`scripts/run_reproduction.py`
- 守恒与干预排序测试：`tests/test_model.py`
- 完整日/月/年结果：`artifacts/`
- 官方论文、D33.2、案例报告：`references/`

## 9. 官方来源

- ScienceDirect article: https://www.sciencedirect.com/science/article/pii/S0921344915000798
- DOI: https://doi.org/10.1016/j.resconrec.2015.03.015
- CORDIS WaterMet² page: https://cordis.europa.eu/article/id/121974-the-watermet-tool-
- Exeter/Figshare D33.2: https://uoe.figshare.com/articles/Quantitative_UWS_performance_model_WaterMet2/29706515
- Exeter/Figshare Oslo report: https://uoe.figshare.com/articles/Oslo_Case_Study_Report/29701109
- UWL accepted manuscript: https://repository.uwl.ac.uk/id/eprint/1824/
