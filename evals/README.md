# 精简验收集

这里的任务用于验证完整的 Coding Agent 闭环，而不是单独评价模型生成代码的能力。每个任务
固定包含：

```text
cases/<case-id>/
├── case.json       # 稳定元数据与 verifier 超时
├── instruction.md  # 原样发给 Agent 的任务说明
├── workspace/      # 每次复制得到的干净初始仓库
└── verify.py       # Agent 结束后由外部运行的确定性验证器
```

第一版刻意不提供自动调用模型、自动批准权限或排行榜。使用同一个模型配置人工运行任务，可以先
观察工具轨迹、失败恢复和最终代码质量；验证结果只由可执行测试决定，不使用 LLM judge。

`python_project_instructions` 的 workspace 根目录包含 `AGENTS.md`。任务说明不重复其中的项目
约束，用于验证 Agent 是否接收并遵守创建 session 时加载的项目指令；verifier 仍保持在
workspace 外。

## 使用

检查并列出任务：

```bash
python3 evals/run.py check
python3 evals/run.py list
```

准备一个全新的 workspace。目标目录必须不存在，防止旧修改污染结果：

```bash
python3 evals/run.py prepare <case-id> /tmp/coding-agent-eval/<case-id>
```

`prepare` 会同时打印任务说明。随后用该目录启动 `coding-agent`，把任务说明作为一条用户消息
发送。Agent 完成后，在项目根目录运行外部验证器：

```bash
python3 evals/run.py verify <case-id> /tmp/coding-agent-eval/<case-id>
```

不要在 Agent 执行期间主动提供 `verify.py`。当前机器尚未使用容器隔离，因此这些测试只能算
人工监督的本地验收，而不是防作弊 benchmark；未来自动批准 Shell 时，应把 Agent 放入一次性
容器，并在 Agent 退出后才向 verifier 提供 workspace。
