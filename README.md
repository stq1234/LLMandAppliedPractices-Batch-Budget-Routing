# 有限批次预算的仓储运价决策实验

本目录包含一个合成的仓储运价报价场景，用于比较批次调度策略怎样在请求价值、等待期限、信息检索、可信报价执行、带噪核验和局部修订之间分配有限计算资源。基准请求不含个人信息，也不对应真实运营数据。

## 数据与隔离

- `data/dev_requests.jsonl`：24 个开发请求，用于估计首轮质量和路由增益。
- `data/test_requests.jsonl`：48 个基准测试请求。
- `data/shift_requests.jsonl`：48 个分布变化请求；重量更大、时限更紧、快递服务比例更高，且较少直接提供费率卡。
- `data/rate_catalog.jsonl`：合成承运商费率、容量、到达时间和服务范围。运行时可按车道查询。
- `data/oracle/*_gold.jsonl`：仅供开发校准、事后评价和独立审计的标签。运行时模块不读取此目录。
- `data/model_outputs/*_first_round_api.jsonl`：显式采集的固定模型首轮输出和供应商返回的 token 用量、实际 API 响应时间。文件中不含 API 密钥。

数据生成使用固定种子。`src/build_data.py` 与 `src/audit_data.py` 的 ground truth 实现彼此独立；审计程序逐条重算标签、检查划分隔离及输入中是否混入标签。顺序反转与价值交换试验复用同一批 48 个测试事实，不视作额外独立样本。

## 决策规则

每个操作消耗一个策略资源单位（RU）。RU 是本实验的离线调度自变量，不是用户 API 安全限额，也不代表真实货币、token 或延迟。运行时状态只包含当前请求、可见费率卡、查询返回、报价执行结果、噪声核验信号和预算账本。

| 状态与动作 | 可用信号 | 操作 | 可记录的停止或回退原因 |
|---|---|---|---|
| 首轮待生成 | 请求任务字段、可见费率卡（若有） | 固定 `deepseek-v4.1-flash` 首轮生成；离线无 API 回放时使用确定性 stub | 批预算耗尽、等待时限不足 |
| 费率卡缺失 | 车道标识、任务重量、服务和时限 | 查询合成承运商目录；返回来源行数及内容 | 查询无结果则保留未确认并转局部修订/停止 |
| 候选路线可执行 | 当前候选、已可见或查询到的目录行 | 可信执行器计算候选路线是否可报价及准确金额 | 约束拒绝则修订或停止 |
| 执行结果待核验 | 候选字段、报价回执 | 带噪核验器输出 `pass` 或 `flag`；内部真值只用于开发估计和事后评价 | `pass` 后可停止；`flag` 后进入修订；达到修订次数上限则记录核验冲突 |
| 核验失败或信息不足 | 可见目录行与执行/核验输出 | 本地证据修订器选最低价可行路线，或在无可行路线时给出 `hold`，再执行报价 | 修订后再次拒绝则保留拒绝回执，不循环重试 |
| 任意阶段 | 全局 RU、请求价值、截止时间、当前可见信息 | FIFO、价值优先或开发集校准的自适应策略选择下一个动作 | `batch_budget_exhausted`、`deadline_exceeded_before_next_action`、`low_expected_marginal_value`、`validated_pass` 等 |

执行器输出只作为证据，不会悄悄覆盖模型候选。只有显式执行局部修订后，修订后的回答才取用该证据。核验器只核对所选路线与回执的一致性，不证明它是全局最低价；事后指标分别报告路线可行性、最低价精确率与价格遗憾，用于揭示这种信号边界。运行时路由不读取测试 gold，也不使用模型自报置信度作正确性标签。

对照策略包括 FIFO、请求价值优先、自适应以及关闭噪声核验器的自适应消融。自适应策略只读取开发集聚合质量增益，动作按“请求价值 × 开发集估计增益 / RU”排序；保留 8% 预算作可见费率/需查询请求组的确定性探索；最低边际价值阈值与每请求一次局部修订上限保存在 `results/dev_calibration.json`。修订路线被执行器拒绝时，轨迹记录回退到修订前候选并停止。请求的外部价值和截止时间不会发送给模型。原始批次回放将 `service_time_limit_ms` 解释为单个请求累计服务时长上限，不含排队时间；其停止率只表示服务时长触顶，不能解读为端到端等待超时。

## 复现

从本目录执行以下命令。常规离线回放不会访问网络，也不会读取或要求 API 密钥。

```powershell
python src/build_data.py
python src/audit_data.py
python src/calibrate.py
python src/run_experiment.py
python src/plot_results.py
python src/run_queue_experiment.py
python src/plot_queue_results.py
```

如需实际采集 DeepSeek 首轮输出，先在启动 Python 的进程环境中设置 `CASDAO_API_KEY`，再显式运行：

```powershell
python src/collect_api.py --splits dev,test,shift
python src/calibrate.py
python src/run_experiment.py
python src/plot_results.py
python src/run_queue_experiment.py
python src/plot_queue_results.py
python src/analyze_model_outputs.py
python src/make_review_packet.py
```

API 请求使用 `https://maasapi.casdao.com/v1/chat/completions` 与模型 `deepseek-v4.1-flash`，请求参数关闭 thinking 并为实际 JSON 输出保留 250 个 completion tokens。客户端禁用环境代理并拒绝 HTTP 重定向。采集器逐条写入请求模型、服务端实际返回型号、`finish_reason`（包括 `length` 截断状态）、原始响应、解析后的候选、API 返回 token 用量和实测 API 响应时间；不写入密钥，不设置总调用限额。若已有部分采集文件，默认按 request ID 续采缺失条目；`--overwrite` 会重新采集并覆盖。完成 API 采集后，回放策略复用同一首轮响应来做离线反事实比较；指标中的 `api_output_capture_calls_for_split` 是数据采集调用，`api_first_round_replay_calls_imputed` 和 token 数是被该策略选中的首轮调用估算，不能解释为本次回放又真实发起的请求。

## 结果文件

- `results/dev_calibration.json`：开发集校准统计和自适应策略参数。
- `results/summary.csv`、`results/summary.json`：基准测试、分布变化、顺序反转和价值交换的指标。
- `results/traces/*.jsonl`：逐请求动作轨迹、每步预算与模拟等待时间、可见信号、停止原因及独立事后评价。
- `results/allocation_diagnostics.csv`：高价值请求获得的资源占比。
- `results/queue_summary.csv`、`results/queue_summary.json`：真实排队敏感性结果；每个数据划分分成 6 个同期到达的 8 单小批，每批有 4 个并行服务槽；比较标准响应期限和按 0.40 缩放的紧期限。
- `results/queue_traces/*.jsonl`：逐请求排队等待、服务时间、响应时间、超时/未调度状态、按时答案及独立事后评价。
- `results/model_output_summary.csv`、`results/model_output_summary.json`：真实首轮回答、实测 token 和 API 耗时的分组汇总。
- `results/api_collection_manifest.json`：120 条真实首轮调用的服务端型号、finish reason、token 汇总和单次 API 实测耗时汇总。
- `results/manual_review_packet.jsonl`：供 GPT-6.2 Astra 主代理模型复核的未标注样例及对应可见上下文、目录、gold 与服务端原文。
- `results/astra_review.json`：GPT-6.2 Astra 主代理模型复核 5 条首轮输出的记录。
- `results/figures/*.png`：预算质量曲线、价值/顺序干预及分布变化下资源分配图。

`run_queue_experiment.py` 以同期到达的 8 个请求为一批，按现有动作依赖在 4 个并行服务槽中调度。每个请求从到达时开始计 `response_deadline_ms`，端到端响应时间包含每个动作的模拟服务时长与进入槽位前的等待时间。`standard` 使用数据生成时的响应期限；`tight` 使用其 0.40 倍。每批 RU 上限分别为 8、16、24、40。独立的 `first_round_only` 参照只生成首轮答案即停止，记录首轮质量、RU、缓存首答对应的真实 API tokens/耗时以及模拟排队结果。

真实 API token 用量与响应耗时来自供应商响应；策略资源单位是预先设定的离线预算；单请求服务时长以及共享队列服务/等待时长均为确定性模拟值。三者分别记录，不相加、不互相替代。队列服务时长的 2300 ms 首轮生成参数与实测 API 耗时独立设定；真实 API elapsed 不用于模拟截止判定。若没有采集 API 输出，结果会明确标记 `deterministic_offline_stub`，不得表述为大模型实测质量。

依赖：Python 3.10+、Matplotlib（绘图）。数据生成、双 oracle 审计、采集和调度回放仅使用 Python 标准库。
