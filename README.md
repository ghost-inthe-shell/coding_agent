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
└── prompts/        # system prompt 源文件与加载器
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
  --base-url "$OPENAI_BASE_URL"
```

`OPENAI_API_KEY` 由 Provider 从环境变量读取；兼容网关的地址通过
`--base-url` 显式传入。如果使用 OpenAI 官方地址，可省略 `--base-url`。

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
  --base-url "$OPENAI_BASE_URL"
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

REPL 在同一个 `SessionState` 上逐轮调用 Runtime。输入 `/help` 查看命令，输入 `/exit`
或按 Ctrl-D 退出。workspace 外的只读工具调用会显示规范化路径并逐次请求确认。第一版只
接收单行输入；Linux 交互式终端使用 GNU readline 提供光标移动、退格和当前进程内
历史。不提供 TUI、流式输出、latest 会话选择或 REPL 内会话切换。文件写入和每条 Shell
命令也会逐次请求确认；Shell 固定在 workspace 根目录启动，并使用 120 秒默认超时。模型
每次调用默认最多生成 16,384 tokens，可通过 `--max-tokens` 调整。

## 验证

安装并运行测试：

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
