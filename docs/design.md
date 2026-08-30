# 设计决策

本项目以学习为主、作品质量为验收标准，从零实现 coding agent。允许使用模型厂商客户端与
原生 tool calling，但不使用 agent 框架、agent SDK 或服务端托管的代码/文件执行能力。
第一版采用同步 Runtime，并以 Linux 为首要运行环境。

交互客户端直接采用单行同步 REPL，并在同一个 `SessionState` 上执行多轮对话。第一版只有
help/exit 命令，不实现 one-shot 模式、TUI、流式输出、多行编辑或会话恢复。

## 核心边界

- `SessionState` 是会话状态的唯一事实来源。`Runtime.run_turn` 修改传入的状态，不在内部
  维护第二份消息历史。
- workspace 内成功的 `read_file` 将规范相对路径及 `mtime_ns`、大小和全文件 SHA-256
  登记到 `SessionState.read_file_versions`。不保存文件正文；artifact 与 workspace 外读取不登记。
- `prompts/system.md` 是可编辑的 system prompt 源文件。创建会话时把解析后的文本保存到
  `SessionState.system_prompt`；恢复旧会话时继续使用原快照。
- Provider 接收标准消息与工具 schema，直接返回标准化 `AssistantMessage`。厂商特有响应
  对象不得进入 SessionState。
- OpenAI-compatible Provider 使用同步 Chat Completions，以兼容常见的 OpenAI 网关。
- Anthropic Provider 使用同步 Messages API，并把连续 tool results 合并为一个 user 消息。
- 核心消息和会话状态使用 dataclass。Pydantic v2 只用于工具入参的严格校验
  （`strict=True`、禁止额外字段）和 JSON Schema 生成。

## Runtime

`Runtime.run_turn(state, user_input)` 执行一个同步的模型/工具循环。Assistant tool call 与
tool result 始终按原顺序完整配对。单轮默认最多调用模型 8 次、实际执行工具 32 次。若同一
批调用超过工具预算，所有跳过的调用都会收到合成错误结果；如果还剩一次模型预算，Runtime
会发出不带工具的最终请求。即使该请求返回了有用文本，`RunResult` 仍标记为
`limit_reached`。

若模型因输出 token 限制停止，纯文本作为部分结果以 `limit_reached` 结束；若响应包含 tool
calls，Runtime 不执行整批调用，而是逐个生成配对错误结果，并在模型调用预算允许时继续循环。

Runtime 只提供一个可选的同步事件接收函数，并且只有六种强类型事件：turn started、model
requested、model responded、tool started、tool finished、turn finished。不实现 EventBus 或
通用 hook 框架。

预期内的 Provider 失败、非法工具入参、路径拒绝和普通工具 I/O 失败转换为明确结果。未预期
的程序或协议错误会停止 Runtime，不得伪装成普通工具输出。

## 只读工具与路径

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

## 工具输出

所有结果统一经过 `ToolResultProcessor`。模型最多看到 50,000 字符；更长的文本以头尾预览
替代，完整输出保存到：

```text
${XDG_STATE_HOME:-~/.local/state}/coding-agent/sessions/<session-id>/tool-results/
```

`read_file` 和 `grep_search` 可以自动读取当前 session 的 artifact。单次捕获与单个 artifact
采用全局 10 MiB 硬上限；达到上限后停止继续捕获，并把 artifact 标记为不完整。
