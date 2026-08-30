# Coding Agent

从零实现的 Python 编程智能体。目前已经完成同步 Runtime、核心协议、两个模型 Provider、
只读工具、最小文件写入/精确编辑工具和同步 REPL；命令执行工具尚未实现。

## 结构

```text
src/coding_agent/
├── core/           # 消息、会话状态、事件和同步 Runtime
├── providers/      # 模型 provider 边界
├── tools/          # 工具、执行器、结果处理和 artifact
├── permissions/    # workspace/artifact 读取边界
└── prompts/        # system prompt 源文件与加载器
```

核心协议遵循三个原则：

- `SessionState` 是唯一的会话历史，system prompt 以会话快照保存。
- Provider 特有对象不会进入会话状态。
- assistant tool call 与 tool result 的配对关系可以在运行时验证。
- 工具输入由 Pydantic v2 严格校验；模型可见工具输出统一限制为 50,000 字符。

长期设计决策见 [`docs/design.md`](docs/design.md)。

## 运行

OpenAI-compatible API：

```bash
python3 -m pip install -e '.[openai]'
export OPENAI_API_KEY='...'
coding-agent --model <model> --workspace <path>
```

自定义兼容网关可增加 `--base-url`。使用 Anthropic Messages API 时：

```bash
python3 -m pip install -e '.[anthropic]'
export ANTHROPIC_API_KEY='...'
coding-agent --provider anthropic --model <model> --workspace <path>
```

REPL 在同一个 `SessionState` 上逐轮调用 Runtime。输入 `/help` 查看命令，输入 `/exit`
或按 Ctrl-D 退出。workspace 外的只读工具调用会显示规范化路径并逐次请求确认。第一版只
接收单行输入，不提供 TUI、流式输出或会话恢复。

## 验证

安装并运行测试：

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
