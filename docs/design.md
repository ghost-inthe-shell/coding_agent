# 设计决策

本项目以学习为主、作品质量为验收标准，从零实现 coding agent。允许使用模型厂商客户端与
原生 tool calling，但不使用 agent 框架、agent SDK 或服务端托管的代码/文件执行能力。
第一版采用同步 Runtime，并以 Linux 为首要运行环境。

## 核心边界

- `SessionState` 是会话状态的唯一事实来源。`Runtime.run_turn` 修改传入的状态，不在内部
  维护第二份消息历史。
- `prompts/system.md` 是可编辑的 system prompt 源文件。创建会话时把解析后的文本保存到
  `SessionState.system_prompt`；恢复旧会话时继续使用原快照。
- Provider 接收标准消息与工具 schema，直接返回标准化 `AssistantMessage`。厂商特有响应
  对象不得进入 SessionState。
- OpenAI-compatible Provider 使用同步 Chat Completions，以兼容常见的 OpenAI 网关。
- 核心消息和会话状态使用 dataclass。Pydantic v2 只用于工具入参的严格校验
  （`strict=True`、禁止额外字段）和 JSON Schema 生成。

## Runtime

`Runtime.run_turn(state, user_input)` 执行一个同步的模型/工具循环。Assistant tool call 与
tool result 始终按原顺序完整配对。单轮默认最多调用模型 8 次、实际执行工具 32 次。若同一
批调用超过工具预算，所有跳过的调用都会收到合成错误结果；如果还剩一次模型预算，Runtime
会发出不带工具的最终请求。即使该请求返回了有用文本，`RunResult` 仍标记为
`limit_reached`。

Runtime 只提供一个可选的同步事件接收函数，并且只有六种强类型事件：turn started、model
requested、model responded、tool started、tool finished、turn finished。不实现 EventBus 或
通用 hook 框架。

预期内的 Provider 失败、非法工具入参、路径拒绝和普通工具 I/O 失败转换为明确结果。未预期
的程序或协议错误会停止 Runtime，不得伪装成普通工具输出。

## 只读工具与路径

第一批工具为 `read_file`、`glob_files` 和 `grep_search`。适合时优先调用 ripgrep，glob 和
grep 不设置各自的匹配条数上限。`read_file` 仍使用 offset/limit，因为分页属于读取语义，
不是另一套结果限制策略。

解析真实路径后，只自动允许读取 workspace 根目录和当前 session 的 agent artifact 根目录，
从而阻止符号链接越界。第一版直接拒绝其他路径；未来交互客户端可把拒绝替换成 ask/confirm，
操作系统权限始终是最后一层约束。

## 工具输出

所有结果统一经过 `ToolResultProcessor`。模型最多看到 50,000 字符；更长的文本以头尾预览
替代，完整输出保存到：

```text
${XDG_STATE_HOME:-~/.local/state}/coding-agent/sessions/<session-id>/tool-results/
```

`read_file` 和 `grep_search` 可以自动读取当前 session 的 artifact。单次捕获与单个 artifact
采用全局 10 MiB 硬上限；达到上限后停止继续捕获，并把 artifact 标记为不完整。
