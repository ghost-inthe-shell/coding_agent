# Coding Agent

从零实现的 Python 编程智能体。目前已经完成同步 Runtime、核心协议、两个模型 Provider、
只读工具、最小文件写入/精确编辑工具、受确认保护的 Shell 工具，以及支持会话恢复的同步
REPL。

## 结构

```text
src/coding_agent/
├── core/           # 消息、会话状态、事件和同步 Runtime
├── providers/      # 模型 provider 边界
├── tools/          # 工具、执行器、结果处理和 artifact
├── permissions/    # 文件与命令权限边界
├── prompts/        # system prompt 源文件与加载器
└── ui/             # 同步终端 Renderer
```

核心协议遵循三个原则：

- `SessionState` 是唯一的会话历史，system prompt 以会话快照保存。
- Provider 特有对象不会进入会话状态。
- assistant tool call 与 tool result 的配对关系可以在运行时验证。
- 工具输入由 Pydantic v2 严格校验；模型可见工具输出统一限制为 50,000 字符。

长期设计决策见 [`docs/design.md`](docs/design.md)。

## 运行

### 首次安装

`coding-agent` 是安装项目时生成的命令行入口，不是系统自带命令。在项目根目录
创建虚拟环境，并以可编辑模式安装项目及两个 Provider 依赖：

```bash
cd /home/lmz/coding_agent/coding_agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[openai,anthropic]'
```

只需安装一个 Provider 时，可分别使用 `.[openai]` 或 `.[anthropic]`。

### 在新 Bash 中启动

虚拟环境不会自动跨 Bash 进程生效。每次打开新 Bash 后，首先激活已创建的
虚拟环境：

```bash
cd /home/lmz/coding_agent/coding_agent
source .venv/bin/activate
```

然后加载 OpenAI-compatible 接口配置。`set -a` 会使 `.env` 中没有显式写
`export` 的变量也能传递给 Python 进程：

```bash
set -a
source /home/lmz/.config/coding-agent/test.env
set +a

mkdir -p /home/lmz/test_agent
coding-agent \
  --provider openai-compatible \
  --workspace /home/lmz/test_agent \
  --model deepseek-v4-flash \
  --base-url "$OPENAI_BASE_URL" \
  --api-dialect deepseek \
  --reasoning low \
  --max-tokens 32768 \
  --max-turns 32
```

`OPENAI_API_KEY` 由 Provider 从环境变量读取；兼容网关的地址通过
`--base-url` 显式传入。如果使用 OpenAI 官方地址，可省略 `--base-url`。

OpenAI-compatible 只统一基础消息格式，不统一 thinking 扩展。`--api-dialect` 应按照实际
API 端点选择，而不只是看模型名称：DeepSeek 官方端点使用 `deepseek`，阿里云百炼使用
`dashscope`，Moonshot 官方端点使用 `moonshot`，其他端点保持 `generic`。`generic` 不会发送
任何厂商 thinking 参数。

主请求可通过 `--reasoning default|off|low|medium|high|max` 设置推理意图。`default` 不发送
控制字段；其他值必须被所选 dialect 明确支持，否则启动失败。例如：

```bash
# Qwen 经 DashScope 调用；具体模型仍可能只支持其中部分档位
coding-agent \
  --provider openai-compatible \
  --workspace /home/lmz/test_agent \
  --model <qwen-model> \
  --base-url <dashscope-compatible-base-url> \
  --api-dialect dashscope \
  --reasoning low

# Kimi 经 Moonshot 官方端点调用；当前支持 default/off
coding-agent \
  --provider openai-compatible \
  --workspace /home/lmz/test_agent \
  --model <kimi-model> \
  --base-url <moonshot-compatible-base-url> \
  --api-dialect moonshot \
  --reasoning off
```

同一模型若通过另一家兼容网关调用，应选择网关实际实现的 dialect。摘要请求自动使用最小
推理策略：DeepSeek/Moonshot 关闭 thinking，DashScope 使用 low，generic 不发送扩展字段。

使用 Anthropic Messages API 时：

```bash
set -a
source /home/lmz/.config/coding-agent/test_anthropic.env
set +a

coding-agent \
  --provider anthropic \
  --workspace /home/lmz/test_agent \
  --model "$CODING_AGENT_TEST_MODEL"
```

Anthropic Provider 直接从环境变量读取 `ANTHROPIC_API_KEY` 和可选的
`ANTHROPIC_BASE_URL`，因此不接收 `--base-url`。

程序启动后会显示当前 session ID。之后可使用同样的 Provider 配置显式恢复该会话；workspace
从 checkpoint 中读取，因此不能同时传入 `--workspace`：

```bash
coding-agent \
  --provider openai-compatible \
  --model deepseek-v4-flash \
  --base-url "$OPENAI_BASE_URL" \
  --api-dialect deepseek \
  --reasoning low \
  --max-tokens 32768 \
  --max-turns 32 \
  --resume <session-id>
```

会话在创建时及每轮结束后保存到
`${XDG_STATE_HOME:-~/.local/state}/coding-agent/sessions/<session-id>/session.json`。保存失败会立即
终止 REPL，防止后续消息建立在未持久化的历史上。

也可不激活虚拟环境，直接使用其中的可执行文件：

```bash
/home/lmz/coding_agent/coding_agent/.venv/bin/coding-agent \
  --provider openai-compatible \
  --workspace /home/lmz/test_agent \
  --model deepseek-v4-flash \
  --base-url "$OPENAI_BASE_URL" \
  --api-dialect deepseek \
  --reasoning low \
  --max-tokens 32768 \
  --max-turns 32
```

### `coding-agent: command not found`

该错误表示当前 Bash 的 `PATH` 中没有项目命令，与 `--workspace` 或 `--model`
参数无关。按顺序检查：

```bash
cd /home/lmz/coding_agent/coding_agent
source .venv/bin/activate
command -v coding-agent
coding-agent --help
```

正常情况下，`command -v` 应输出
`/home/lmz/coding_agent/coding_agent/.venv/bin/coding-agent`。如果 `.venv` 不存在，执行上面的
“首次安装”；如果它存在但命令仍不存在，在激活后重新执行：

```bash
python -m pip install -e '.[openai,anthropic]'
```

REPL 在同一个 `SessionState` 上逐轮调用 Runtime。输入 `/help` 查看命令，输入 `/compact`
可立即将较早历史更新为 rolling summary，输入 `/exit` 或按 Ctrl-D 退出。原始消息仍完整保存
在 session checkpoint 中；压缩只改变后续发送给模型的活动上下文。workspace 外的只读工具
调用会显示规范化路径并逐次请求确认。Linux 交互式终端使用 GNU readline 提供光标移动、
退格和当前进程内历史，并显式启用 bracketed paste，使终端支持时多行粘贴保持为一条消息。
普通 Enter 提交；行尾输入奇数个反斜杠后按 Enter 会移除最后一个反斜杠、插入真实换行，并以
`... ` 提示继续输入；组合后的消息保留原始缩进和内部空行。不提供 TUI、latest 会话选择或
REPL 内会话切换。文件写入和每条 Shell 命令也会逐次请求确认；Shell 固定在 workspace 根目录
启动，并使用 120 秒默认超时。

模型回答默认同步流式显示：`thinking>` 使用 dim 样式，`assistant>` 显示回答，`tool>` 显示
工具开始和结果状态。Renderer 仅使用标准库 ANSI；非 TTY 或设置 `NO_COLOR` 时自动退化为纯
文本，也可通过 `--color auto|always|never` 控制。若兼容网关的流实现存在问题，可用
`--no-stream` 回退到完整响应；`--stream` 可显式开启默认行为。流中断时已经显示的片段不会写入
session，只有 Provider 成功组装的完整 `AssistantMessage` 才会持久化。

模型每次调用默认最多生成 16,384 tokens，可通过 `--max-tokens` 调整；context window 默认
按 128,000 tokens 估算，可用 `--context-window` 设置为实际模型值。历史接近安全阈值时会
自动压缩，摘要调用复用当前 Provider/模型且不开放工具。

每次用户输入默认最多进行 32 次 Agent 模型调用，可通过 `--max-turns` 调整，例如复杂任务
可使用 `--max-turns 64`。最后一次调用不提供工具，只用于如实说明已完成内容和未完成内容；
自动 compact 的内部摘要请求累计 token usage，但不占用该回合预算。`--max-turns`、
`--max-tokens` 等 Provider/Runtime 启动配置不保存在 session 中，恢复会话时需要重新传入。

## 验证

安装并运行测试：

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

### 真实任务验收

`evals/` 提供三个不调用 LLM judge 的确定性任务：

- `cpp_binary_search`：读取题面，修复 C++ 二分边界并编译验证。
- `python_ttl_cache`：根据失败测试修复 Python 到期边界。
- `python_task_priority`：跨模型、存储和 CLI 实现向后兼容的新功能。

先复制一个干净 workspace，再把 `prepare` 打印的任务说明原样发给 Agent：

```bash
python3 evals/run.py list
python3 evals/run.py prepare cpp_binary_search /tmp/coding-agent-eval/cpp_binary_search

coding-agent \
  --provider openai-compatible \
  --workspace /tmp/coding-agent-eval/cpp_binary_search \
  --model <model> \
  --base-url <base-url>
```

Agent 结束后从项目根目录运行外部 verifier：

```bash
python3 evals/run.py verify \
  cpp_binary_search \
  /tmp/coding-agent-eval/cpp_binary_search
```

不同版本之间比较时，应固定模型、Provider、reasoning、`--max-tokens`、`--max-turns` 和任务
初始版本，并保留 session ID。完整协议、安全边界和其他 case 用法见
[`evals/README.md`](evals/README.md)。
