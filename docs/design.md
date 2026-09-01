# 设计决策

本项目以学习为主、作品质量为验收标准，从零实现 coding agent。允许使用模型厂商客户端与
原生 tool calling，但不使用 agent 框架、agent SDK 或服务端托管的代码/文件执行能力。
第一版采用同步 Runtime，并以 Linux 为首要运行环境。

真实任务验收使用干净 workspace 副本和 Agent 结束后运行的确定性外部 verifier，不比较参考
diff，也不使用 LLM judge。无人值守评测必须在一次性容器内运行；容器隔离完成前只做逐次确认
权限的人工监督验收。

交互客户端直接采用同步 REPL，并在同一个 `SessionState` 上执行多轮对话。第一版提供 help、
compact 和 exit 命令，Linux 交互 TTY 使用 Python 标准库 GNU readline 编辑，并显式启用
bracketed paste。普通 Enter 提交；奇数个行尾反斜杠表示移除最后一个反斜杠并继续收集下一行，
从而在同一条用户消息中插入换行。终端展示使用标准库 ANSI Renderer 和同步流式输出；不实现
one-shot 模式、TUI 或 REPL 内会话切换。

## 核心边界

- `SessionState` 是会话状态的唯一事实来源。`Runtime.run_turn` 修改传入的状态，不在内部
  维护第二份消息历史。
- 会话只在创建完成及 `Runtime.run_turn` 正常返回后的稳定边界持久化；`running` 状态或存在
  未配对 tool call 的状态不得写入。checkpoint 失败立即终止 REPL，避免继续产生仅存在于
  内存中的历史。
- `--resume SESSION_ID` 只恢复显式指定的会话，使用其中保存的 workspace，并与
  `--workspace` 互斥。不提供 latest、会话列表或 REPL 内切换。
- checkpoint 保存到
  `${XDG_STATE_HOME:-~/.local/state}/coding-agent/sessions/<session-id>/session.json`，采用同目录
  临时文件、`fsync` 和原子替换；会话目录权限为 `0700`，JSON 文件为 `0600`。
- workspace 内成功的 `read_file` 将规范相对路径及 `mtime_ns`、大小和全文件 SHA-256
  登记到 `SessionState.read_file_versions`。不保存文件正文；artifact 与 workspace 外读取不登记。
- `prompts/system.md` 是可编辑的 system prompt 源文件。创建会话时把解析后的文本保存到
  `SessionState.system_prompt`；恢复旧会话时继续使用原快照。
- Provider 接收标准消息与工具 schema，直接返回标准化 `AssistantMessage`。同步请求可以额外
  发出 text/thinking delta 供即时展示，但 delta 不进入 `SessionState`；厂商特有响应对象也
  不得进入会话状态。
- CLI 默认启用同步流式输出，并提供 `--no-stream` 回退。Provider 直接消费厂商 SDK 的同步
  iterator，不引入 asyncio；Renderer 只展示 text/thinking delta，tool call 必须完整组装后才
  能进入 Runtime。流中断时已经展示的 delta 不持久化，成功返回的完整 `AssistantMessage` 才是
  会话事实。
- thinking 展示与模型 reasoning 配置相互独立。Renderer 默认使用 `brief`，只显示活动提示和
  本次隐藏字符数；`full` 显示完整增量，`hidden` 不显示。三种模式都不改变标准消息、持久化或
  token usage。
- Renderer 的工具轨迹使用按工具定制的有界摘要；Shell 摘要最多显示前两行、160 字符。展示
  截断不改变工具参数，Shell 权限确认始终显示 JSON 转义后的完整命令。
- OpenAI-compatible 传输与厂商 thinking 扩展分离。启动时显式选择 `generic`、`deepseek`、
  `dashscope` 或 `moonshot` API dialect；不根据模型名或 URL 猜测。Runtime 只表达
  `default/off/low/medium/high/max/minimal` 推理意图，由 Provider 映射为厂商字段。不支持的
  用户配置明确拒绝，不静默换档；这些启动配置不进入 `SessionState`。
- `AssistantMessage` 可包含标准化 `ThinkingBlock`；它不并入用户可见文本，但会随会话
  持久化。可选 `replay_field` 只记录 OpenAI-compatible 接口回传推理内容所需的字段名。
- OpenAI-compatible Provider 使用同步 Chat Completions，将响应中第一个非空的
  `reasoning_content`/`reasoning`/`reasoning_text` 标准化为 `ThinkingBlock`，并在后续请求
  中按原字段回传。
- Anthropic Provider 使用同步 Messages API，并把连续 tool results 合并为一个 user 消息。
- 核心消息和会话状态使用 dataclass。Pydantic v2 只用于工具入参的严格校验
  （`strict=True`、禁止额外字段）和 JSON Schema 生成。

## Runtime

`Runtime.run_turn(state, user_input)` 执行一个同步的模型/工具循环。Assistant tool call 与
tool result 始终按原顺序完整配对。单轮默认最多执行 32 次 Agent 模型调用、实际执行工具
32 次；最后一次 Agent 调用固定不提供工具，用于说明已完成内容、未完成内容和后续步骤。
若同一批调用超过工具预算，所有跳过的调用都会收到合成错误结果，并在剩余模型预算内发出
不带工具的最终请求。因任一预算进入最终请求时，即使模型返回了有用文本，`RunResult` 仍标记
为 `limit_reached`。

两个 Provider 的单次默认输出上限统一为 16,384 tokens。`RunResult.max_output_tokens`
记录单次请求上限，而 `RunResult.usage.output_tokens` 是整个 turn 多次模型调用的累计值。

模型 context window 是启动配置，默认 128,000 tokens，不进入 `SessionState`。Runtime 使用
本地保守估算，并在以下两个阈值中较早到达者触发自动压缩：context window 的 80%，或
`context_window - max_output_tokens - safety_margin`；安全余量为 `max(1024, context window 的
2%)`。第二项为下一次响应保留完整输出空间，避免输入虽未达到 80%，却已挤占模型输出预算。

压缩不删除 `SessionState.messages` 中的原始历史，只保存一个向前移动的 compaction
checkpoint；Provider 看到的是 checkpoint 摘要加未压缩后缀。自动压缩总结“已有 rolling
summary + 本次新移出的历史”并保留近期后缀；稳定边界的手动 `/compact` 尝试总结全部 active
history。system prompt 不进入摘要。摘要调用复用当前 Provider 和模型、关闭 tools、最多输出
2,048 tokens，并请求该 dialect 可提供的最小推理强度。候选 checkpoint 只有在相同估算器和
tools 下满足 `tokens_after < tokens_before` 才能生效；否则保留旧 checkpoint，但仍累计已经
发生的摘要 usage。自动摘要属于 Runtime 内部维护调用：计入本 turn 和 session 的 token usage，
但不占 Agent 模型调用预算；事件中的 Provider 调用序号仍包含它。手动摘要位于 turn 外，也不受
单 turn 调用次数限制。截断、失败、空文本或包含 tool call 的摘要不得替换旧 checkpoint。
第一版不做 tool result 的 microcompact/snip；裸命令 `/compact` 不接收自定义压缩指令。

若模型因输出 token 限制停止，纯文本作为部分结果以 `limit_reached` 结束；若响应包含 tool
calls，Runtime 不执行整批调用，而是逐个生成配对错误结果，并在模型调用预算允许时继续循环。

Runtime 只提供一个可选的同步事件接收函数，并且只有八种强类型事件：turn started、model
requested、model text delta、model thinking delta、model responded、tool started、tool
finished、turn finished。不实现 EventBus 或通用 hook 框架。

预期内的 Provider 失败、非法工具入参、路径拒绝和普通工具 I/O 失败转换为明确结果。未预期
的程序或协议错误会停止 Runtime，不得伪装成普通工具输出。

## 本地工具与权限

第一批工具为 `read_file`、`glob_files` 和 `grep_search`。适合时优先调用 ripgrep；
`grep_search` 在 ripgrep 不可用时使用系统 grep。glob 和 grep 不设置各自的匹配条数上限。
`read_file` 仍使用 offset/limit，因为分页属于读取语义，不是另一套结果限制策略。

解析真实路径后，只自动允许读取 workspace 根目录和当前 session 的 agent artifact 根目录，
从而阻止符号链接越界。REPL 对 workspace 外的每次只读调用同步询问且不记忆授权；没有权限
处理器、用户拒绝、中断或输入结束时均拒绝。批准只允许工具尝试本次读取，操作系统权限始终
是最后一层约束。workspace 外写入仍拒绝。

写路径策略分为两个阶段：先解析真实路径并硬拒绝 workspace 外、agent artifact 和符号链接
逃逸；写工具完成自身校验后，再对 workspace 内目标逐次同步确认且不记忆授权。这样无效的
写入请求不会提前打扰用户。

`write_file` 只创建新的 UTF-8 文件，保留模型给出的确切内容，不覆盖已有路径，也不自动创建
父目录。排他创建用于防止确认后出现的文件被覆盖；成功后把新文件登记为可信版本。

`edit_file` 每次只接受一组 `path`、`old_text`、`new_text`，不做模糊匹配、换行归一化或批量
替换。目标必须已有可信版本，当前 `mtime_ns`、大小和 SHA-256 必须仍一致，且 `old_text` 必须
非空并精确出现一次。确认后重复校验，再以同目录临时文件原子替换并更新可信版本。

`run_shell` 只接受 `command` 和可选的 `timeout_seconds`，固定通过 `/bin/bash -c` 在 workspace
根目录同步执行，不向模型暴露 cwd。每次有效调用都逐次确认且不做安全命令分类；这不是 Shell
沙箱，命令仍可访问操作系统允许的 workspace 外资源。stdin 连接 `/dev/null`，默认超时 120
秒、最大 600 秒；超时或达到原始输出硬上限时终止整个进程组。子进程继承普通环境，但删除
`OPENAI_API_KEY` 和 `ANTHROPIC_API_KEY`。

## 工具输出

所有结果统一经过 `ToolResultProcessor`。模型最多看到 50,000 字符；更长的文本以头尾预览
替代，完整输出保存到：

```text
${XDG_STATE_HOME:-~/.local/state}/coding-agent/sessions/<session-id>/tool-results/
```

`read_file` 和 `grep_search` 可以自动读取当前 session 的 artifact。单次捕获与单个 artifact
采用全局 10 MiB 硬上限；达到上限后停止继续捕获，并把 artifact 标记为不完整。
