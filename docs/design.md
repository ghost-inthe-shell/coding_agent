# 设计与运行机制

本项目从零实现一个精简 coding agent。允许使用模型厂商
客户端与原生 tool calling，但不使用 Agent 框架、Agent SDK 或服务端托管的代码与文件执行
能力。第一版采用同步 Runtime，以 Linux 为第一支持平台。

设计目标不是复制成熟产品的全部功能，而是用尽量少的协议形成可靠闭环：模型读取代码、提出
工具调用、在本地执行、观察结果、继续推理，直到给出答案或触发明确的停止条件。

## 系统全景

```text
REPL ──用户输入──> Runtime ──标准请求──> Provider ──HTTP──> Model API
 │                    │                      │
 │                    │                      └─ 标准化 AssistantMessage
 │                    │
 │                    ├─ ToolExecutor ──> 权限与路径校验 ──> 本地文件 / Shell
 │                    ├─ ContextBudget / Compaction
 │                    └─ 强类型 RuntimeEvent ──> TerminalRenderer
 │
 └─ 稳定轮次结束──> SessionStore ──> session.json + tool-results/
```

边界职责如下：

- REPL 只负责输入、命令分派、展示和稳定边界保存，不实现 Agent 推理。
- Runtime 是唯一的 Agent Loop，负责消息顺序、预算、工具执行和停止条件。
- Provider 只做标准协议与厂商 API 之间的转换，不保存会话。
- ToolExecutor 严格校验参数并执行一个工具，不决定对话下一步。
- `SessionState` 是会话事实来源；Renderer 收到的流式 delta 只用于展示。

## 会话生命周期

### 创建与恢复

新会话先解析 workspace，组合基础 system prompt、根目录 `AGENTS.md` 和项目技能元数据目录，
生成一次不可变的 `SessionState.system_prompt` 快照。初始 checkpoint 在进入 REPL 前保存。

恢复会话只接受显式 `--resume SESSION_ID`：从 checkpoint 恢复原 workspace、system prompt、完整
消息、compact checkpoint、文件版本表和累计 usage。Provider、模型、reasoning、context window
与调用预算属于本次进程配置，不写入 `SessionState`，恢复时需要重新传入。恢复不会重新加载
`AGENTS.md` 或自动技能目录，避免同一 session 的高优先级提示静默变化。

### 三种不同的数据视图

系统刻意区分以下状态：

1. `SessionState.messages`：完整、可审计的标准消息历史，compact 不删除它。
2. active context：实际发给 Provider 的摘要加近期消息，是完整历史的投影视图。
3. text/thinking delta：Provider 流式返回、Renderer 即时展示的临时事件，不进入会话。

只有 Provider 成功组装出的完整 `AssistantMessage` 才能进入历史。因此网络流中断时，终端可能
已经显示部分文本，但 checkpoint 不会把该片段误当成可靠回答。

### 稳定 checkpoint

会话只在创建完成、`Runtime.run_turn` 正常返回或手动 compact 正常返回后保存。`running` 状态
或末尾存在未配对 tool call 的状态不允许持久化；保存失败立即终止 REPL，避免后续对话建立在
只存在于内存的历史上。

checkpoint 按会话创建时的本地日期存放在：

```text
${XDG_STATE_HOME:-~/.local/state}/coding-agent/sessions/YYYY/MM/DD/<session-id>/session.json
```

保存采用同目录临时文件、文件 `fsync`、原子替换和目录 `fsync`。目录权限为 `0700`，JSON 为
`0600`。旧版平铺 checkpoint 可以继续恢复并原地保存，但不会被隐式迁移；同一 ID 出现多个
checkpoint 时拒绝猜测。

## 消息协议

核心消息只有三类：`UserMessage`、`AssistantMessage` 和 `ToolResultMessage`。Assistant 内容由
`TextBlock`、`ThinkingBlock` 与 `ToolCall` 组成；Provider 特有响应对象永远不进入会话状态。

每个 assistant tool call 必须按声明顺序紧邻一个具有相同 ID 和工具名的 result，且 call ID 在
会话中唯一。正常稳定状态不能以 pending tool call 结尾。这个约束同时服务于：

- 向不同 Provider 重放一致的历史；
- compact 时按完整消息组切分；
- 中断、超限和失败后仍能保存合法 checkpoint。

核心消息和会话状态使用 dataclass。Pydantic v2 只承担工具入参与技能 frontmatter 的严格校验
（`strict=True`、禁止额外字段），工具 JSON Schema 也由 Pydantic 生成；技能 YAML 使用
`yaml.safe_load`。

## Agent Loop

`Runtime.run_turn(state, user_input)` 对一次用户输入执行一个同步循环：

1. 验证已有消息协议，将 session 标记为 `running`，追加 `UserMessage`。
2. 组装本轮 artifact store、工具执行器、workspace、权限处理器和文件版本表。
3. 在每次模型调用前估算完整请求；达到阈值时先自动 compact。
4. 用 `system_prompt + active_messages + tool schemas` 调用 Provider。
5. 将标准化 `AssistantMessage` 追加到完整历史并累计 usage。
6. 如果没有 tool call，结束本轮；否则按模型给出的顺序同步执行每个工具。
7. 每个调用无论成功、拒绝、超时或参数错误，都生成配对的 `ToolResultMessage`。
8. 将工具结果加入历史，再回到第 3 步，让模型观察结果并决定下一步。
9. 结束时设置稳定状态、验证完整消息序列、发出 `TurnFinished`，由 REPL 保存 checkpoint。

第一版故意顺序执行同一 assistant 消息中的多个工具。这样权限询问、文件版本变化、输出顺序和
中断语义都是确定的；暂不为工具并行引入资源冲突和结果乱序处理。

### 预算和最后一次调用

单个用户 turn 默认最多执行 32 次 Agent 模型调用和 32 个实际工具调用。最后一次模型调用不
提供任何工具，并追加 Runtime limit 指令，要求模型如实说明已完成内容、未完成内容和下一步，
从而避免在没有后续执行机会时继续承诺工具操作。

工具预算耗尽时，同一批尚未执行的调用仍会收到合成错误结果；若模型预算允许，再进行一次无
工具的最终调用。由于任务是被预算强制停止，即使最终文本有用，`RunResult` 仍标记为
`limit_reached`。

### 特殊停止条件

- 正常文本且无 tool call：返回 `completed`。
- Provider 错误：返回 `provider_error`，不伪造成工具结果。
- 用户中断：正在执行和等待执行的 tool call 都补成 `cancelled` result，随后返回
  `interrupted`；正在执行的操作可能已有部分副作用，结果会明确提示。
- 模型达到输出 token 上限且只有文本：保留部分文本并返回 `limit_reached`。
- 模型达到输出 token 上限且包含 tool call：不执行可能被截断的参数，为每个 call 生成错误
  result；还有模型预算时允许模型重新发出完整调用。
- 未预期的协议或程序错误：把 session 标记为 `error` 并向上抛出，不伪装成可恢复业务错误。

## 上下文预算与 Compact

### 触发阈值

Runtime 不依赖特定模型 tokenizer，而是对 system prompt、active messages 和 tool schemas 做
保守估算：ASCII 约按 4 字符/token，非 ASCII 约按 3 UTF-8 字节/token。自动 compact 在以下
阈值中较早到达者触发：

```text
min(
  context_window × 80%,
  context_window - max_output_tokens - safety_margin
)
```

其中 `safety_margin = max(1024, context_window × 2%)`。第二个阈值为下一次完整输出预留空间，
避免输入未到 80% 却已经挤占输出预算。

### Rolling summary

compact 不重写或删除完整历史，只保存：

```text
CompactionCheckpoint(
  summary,
  first_kept_message_index,
  tokens_before,
  created_at,
)
```

之后 Provider 看到的 active context 为：

```text
[Previous conversation summary]
<summary>
[End previous conversation summary]
+ SessionState.messages[first_kept_message_index:]
```

自动 compact 从旧 checkpoint 继续向前滚动，输入是“已有 summary + 本次移出的完整消息组”，
同时保留近期后缀；近期目标为 `min(20,000 tokens, context_window × 25%)`。切点只能位于消息组
边界，绝不拆开 assistant tool calls 及其 results。system prompt 始终单独发送，不进入摘要。

摘要调用复用当前 Provider 和模型，但使用独立 compaction system prompt、关闭所有工具、最多
输出 2,048 tokens，并请求 Provider 支持的最小推理强度。摘要调用计入 session 和本 turn 的
token usage，但不占 Agent 模型调用预算。

### 候选摘要的提交规则

摘要必须满足以下条件才会替换旧 checkpoint：

- 返回完整、非空、纯文本内容；
- 没有 tool call，且未因输出 token 上限截断；
- 使用相同估算器和当前 tools 重新计算后，`tokens_after < tokens_before`。

因此手动 `/compact` 后 token 反而增加时，候选摘要不会生效；完整历史和旧 checkpoint 保持
不变，但已经发生的摘要调用 usage 仍会记录。自动 compact 无法安全产生更小上下文，或没有
合法消息组可继续压缩且请求仍放不下时，本轮明确失败，而不是发送已知会溢出的请求。

手动 `/compact` 只能在稳定 turn 边界运行，尝试总结当前全部 active history；自动 compact
通常保留近期后缀。第一版不实现 tool result 的 microcompact/snip，也不接受自定义 compact
指令。

## Provider、流式事件与 Renderer

Provider 接收标准 `CompletionRequest`，直接返回标准化 `AssistantMessage`。OpenAI-compatible
使用同步 Chat Completions；Anthropic 使用同步 Messages API，并把连续 tool results 合并成
厂商要求的 user 消息。不同 Provider 的原始对象与异常不会泄漏进 `SessionState`。

OpenAI-compatible 传输与 thinking 扩展分离。启动时显式选择 `generic`、`deepseek`、
`dashscope` 或 `moonshot` dialect，不根据模型名称或 URL 猜测。Runtime 表达统一的 reasoning
意图，由 Provider 映射成厂商字段；不支持的用户配置直接拒绝，不静默换档。

同步流式输出直接消费厂商 SDK 的同步 iterator，不引入 asyncio。Runtime 只暴露八种强类型
事件：turn started、model requested、model text delta、model thinking delta、model responded、
tool started、tool finished、turn finished；不实现 EventBus 或通用 hook 框架。

Renderer 默认以 `brief` 隐藏长 thinking，只展示活动提示和字符数；`full` 与 `hidden` 只改变
终端显示，不影响 Provider、持久化或 token usage。工具轨迹使用有界摘要，Shell 最多展示前
两行、160 字符；权限确认仍显示 JSON 转义后的完整命令，避免用户批准未展示的操作。

## 项目指令与技能

创建新会话时只加载 workspace 根目录的 `AGENTS.md`。文件必须是不超过 50,000 字符的 UTF-8
普通文本；workspace 内符号链接允许，越界链接拒绝。缺失或空白文件视为没有项目指令，其他
读取错误终止创建。第一版不搜索父目录、子目录或用户级全局指令。

项目技能只从 `.agents/skills` 的直接子目录发现，最多 64 个。每个 `SKILL.md` 不超过 50,000
字符，YAML frontmatter 只允许与目录一致的 kebab-case `name` 和非空 `description`。新 session
只快照经过 XML 转义且总长不超过 50,000 字符的技能元数据目录；自动选择后由模型用
`read_file` 读取正文，实现 progressive disclosure。

`/skill:name [request]` 在调用时严格读取当前正文，将其与请求展开为实际 UserMessage 后交给
Runtime，所以旧 session 也能显式使用后来新增或更新的技能。第一版不提供全局技能、嵌套发现、
Skill tool、MCP 或子 Agent。项目指令和技能都不能放宽工具权限或 Runtime 安全边界。

## 本地工具、权限与文件一致性

只读工具为 `read_file`、`glob_files` 和 `grep_search`。grep 优先使用 ripgrep，不可用时回退
系统 grep；glob 和 grep 不设置匹配条数上限。`read_file` 保留 offset/limit，因为分页属于读取
语义。

路径先解析真实位置，再执行权限策略：

- workspace 和当前 session artifact 内读取自动允许；workspace 外读取逐次询问且不记忆。
- workspace 外写入、artifact 写入和符号链接逃逸硬拒绝。
- workspace 内有效写入逐次询问；无效请求在询问前拒绝。
- 操作系统权限始终是最后一层约束；当前策略不是 Shell 沙箱。

workspace 内成功的 `read_file` 会把规范相对路径、`mtime_ns`、文件大小和全文件 SHA-256 登记
到 `SessionState.read_file_versions`，但不保存正文。`write_file` 只排他创建新 UTF-8 文件并登记
版本；`edit_file` 要求目标已有可信版本、当前版本仍一致、`old_text` 非空且精确出现一次。确认
后再重复校验，以同目录临时文件原子替换并更新版本，缩小 read-before-edit 的竞态窗口。

`run_shell` 只接受 command 和可选 timeout，固定通过 `/bin/bash -c` 从 workspace 根目录同步
执行，不向模型暴露 cwd。每次调用都询问且不做命令分类；stdin 连接 `/dev/null`，默认超时 120
秒、最大 600 秒。超时或原始输出达到硬上限时终止整个进程组。子进程继承普通环境，但删除
`OPENAI_API_KEY` 与 `ANTHROPIC_API_KEY`。

## 工具结果与 Artifact

所有工具结果统一经过 `ToolResultProcessor`。模型可见文本最多 50,000 字符；更长内容返回头尾
预览，并把完整内容保存在当前 session 的 `tool-results/`。`read_file` 和 `grep_search` 可以读取
这些 artifact，让模型继续定位原始输出。

单次原始捕获与单个 artifact 使用 10 MiB 硬上限；达到上限即停止捕获并标记结果不完整，避免
先无限占用内存或磁盘再做展示截断。artifact 跟随 checkpoint 的实际目录，因此恢复旧版平铺
session 时也继续写入原目录。

## 评测边界

真实任务验收使用干净 workspace 副本和 Agent 结束后运行的确定性外部 verifier，不比较参考
diff，也不使用 LLM judge。无人值守评测必须在一次性容器内运行；容器隔离完成前只做逐次确认
权限的人工监督验收。

`evals/run.py report` 通过 case ID 和 session ID 重新运行 verifier，并从严格 checkpoint 派生
只读指标。Agent 模型调用数只统计 assistant 消息，compact 内部调用只反映在累计 usage；
conversation span 包含用户确认等待。报告不估算不稳定的模型价格，也不修改 session 或
workspace。PASS、验收失败和配置错误分别使用退出码 0、1、2。
