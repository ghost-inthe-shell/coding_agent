# Coding Agent

从零实现的 Python 编程智能体。目前完成了项目骨架与第一版核心协议，尚未接入真实模型和工具执行循环。

## 结构

```text
src/coding_agent/
├── core/           # 消息、usage、运行结果和会话状态
├── providers/      # 模型 provider 边界
├── tools/          # 工具声明与执行结果
├── permissions/    # 权限策略（后续实现）
├── context/        # 系统上下文和压缩（后续实现）
├── observability/  # 可持久化事件协议
└── prompts/        # 稳定 prompt 资源
```

核心协议遵循三个原则：

- 模型消息、工具执行结果和运行事件相互独立。
- Provider 特有对象不会进入会话状态。
- assistant tool call 与 tool result 的配对关系可以在运行时验证。

## 验证

项目当前只依赖 Python 标准库：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
