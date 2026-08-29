# Coding Agent

从零实现的 Python 编程智能体。目前已经完成同步 Runtime、核心协议和第一批只读工具，
尚未接入真实模型 Provider 与写入/命令工具。

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

## 验证

安装并运行测试：

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
