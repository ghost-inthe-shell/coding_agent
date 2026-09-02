# Coding Agent

一个从零实现的精简 Python coding agent。它使用模型原生 tool calling，在本地完成代码读取、
搜索、创建、精确编辑、命令执行和测试，并支持流式 REPL、权限确认、会话恢复、上下文压缩、
`AGENTS.md` 与项目技能。

项目不依赖 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK 等 Agent 框架，也不
使用服务端托管的代码执行或文件工具。架构和运行机制见 [docs/design.md](docs/design.md)。

## 安装

Linux 是第一支持平台。进入包含 `setup.cfg` 的项目根目录后执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[openai,anthropic]'
```

只使用一个 Provider 时，可以安装 `.[openai]` 或 `.[anthropic]`。`coding-agent` 是安装项目时
生成的命令，因此每个新 Bash 都需要先在项目根目录激活虚拟环境：

```bash
source .venv/bin/activate
coding-agent --help
```

如果不想激活虚拟环境，也可以从项目根目录直接运行：

```bash
.venv/bin/coding-agent --help
```

若仍显示 `coding-agent: command not found`，确认当前目录是项目根目录，然后重新执行可编辑安装：

```bash
source .venv/bin/activate
python -m pip install -e '.[openai,anthropic]'
command -v coding-agent
```

## 启动

### OpenAI-compatible

至少设置 API key，并把模型、workspace 和兼容网关地址换成自己的配置：

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://example.com/v1'
export MODEL_NAME='your-model'

mkdir -p ./workspace
coding-agent \
  --provider openai-compatible \
  --workspace ./workspace \
  --model "$MODEL_NAME" \
  --base-url "$OPENAI_BASE_URL"
```

使用 OpenAI 官方地址时可以省略 `--base-url`。OpenAI-compatible 只统一基础消息格式；厂商
thinking 扩展需要显式选择实际 API 端点支持的 dialect：

```text
--api-dialect generic|deepseek|dashscope|moonshot
--reasoning default|off|low|medium|high|max
```

`generic` 不发送厂商 thinking 参数。不支持的 reasoning 档位会在启动时明确拒绝，而不会静默
降级。

### Anthropic

```bash
export ANTHROPIC_API_KEY='...'
export MODEL_NAME='your-model'

mkdir -p ./workspace
coding-agent \
  --provider anthropic \
  --workspace ./workspace \
  --model "$MODEL_NAME"
```

Anthropic Provider 还会读取可选的 `ANTHROPIC_BASE_URL`。它不接受 `--base-url` 和
OpenAI-compatible 专用的 dialect/reasoning 参数。

如果配置保存在环境文件中，可以在启动前加载；仓库不要求固定文件名或位置：

```bash
set -a
source ./model.env
set +a
```

### 常用参数

```text
--max-tokens 16384          单次模型调用的最大输出 token
--max-turns 32              一次用户输入最多进行的 Agent 模型调用
--context-window 128000     本地上下文预算
--thinking-display brief    brief、full 或 hidden
--no-stream                 禁用默认的同步流式输出
--color auto                auto、always 或 never
```

模型的真实 context window 和输出能力由所用 API 决定，应据此调整参数。

## REPL 与会话

启动后直接输入任务。常用命令：

```text
/help                       显示帮助
/compact                    手动压缩活动上下文
/skill:name [request]       显式调用项目技能
/exit                       退出（已完成轮次此前已保存）
```

普通 Enter 提交消息。行尾输入奇数个反斜杠再按 Enter，可以插入真实换行并继续输入；支持
bracketed paste 的终端会把多行粘贴作为一条消息。文件写入和每条 Shell 命令都会逐次询问，
workspace 外读取也会逐次询问。自动上下文压缩开始时会显示
`context> Compacting conversation...`；手动 `/compact` 成功时还会报告压缩前后的估算 token。

程序启动时显示 session ID。恢复时使用原 Provider 配置并传入该 ID；已保存的 workspace 会随
session 恢复，因此不能同时使用 `--workspace`：

```bash
export SESSION_ID='your-session-id'

coding-agent \
  --provider openai-compatible \
  --model "$MODEL_NAME" \
  --base-url "$OPENAI_BASE_URL" \
  --resume "$SESSION_ID"
```

checkpoint 默认保存到：

```text
${XDG_STATE_HOME:-~/.local/state}/coding-agent/sessions/YYYY/MM/DD/<session-id>/session.json
```

## 项目指令与技能

新 session 会读取 workspace 根目录的 `AGENTS.md`，并将其与 system prompt 一起保存为会话
快照。只读取这一个项目文件，不搜索父目录、子目录或用户级全局配置。修改后需要新建 session
才能影响自动行为。

项目技能位于 `.agents/skills/<name>/SKILL.md`：

```markdown
---
name: review
description: Review an implementation and its tests.
---
# Review workflow

Read the implementation and tests, then report correctness risks.
```

技能名必须是与目录一致的 kebab-case；frontmatter 只允许 `name` 和 `description`。新 session
只把名称、描述和路径放入 prompt，模型匹配后再用 `read_file` 读取正文。显式执行
`/skill:review check the parser` 时会加载当前正文，因此 resume 后也能调用后来新增或更新的技能。

第一版只支持项目级、依赖现有工具的技能，不扫描 `~/.codex/skills`，也不提供 Skill tool、MCP、
插件或子 Agent。`AGENTS.md` 和技能都不能放宽代码强制执行的权限与安全边界。

## 本地工具与安全边界

内置工具包括：

- `read_file`、`glob_files`、`grep_search`
- `write_file`、`edit_file`
- `run_shell`

workspace 内读取自动允许；workspace 外读取逐次确认；写入只能位于 workspace 内且逐次确认。
`edit_file` 要求先读取目标文件，再进行唯一精确替换。Shell 固定从 workspace 根目录启动，每次
调用都询问，默认超时 120 秒、最大 600 秒，并从子进程环境移除两个模型 API key。

模型可见的单个工具结果最多 50,000 字符。更长内容返回头尾预览，完整结果保存在当前 session
的 `tool-results/` 目录，供后续用读取工具定位。

这些限制不是操作系统沙箱；无人值守运行不可信代码时仍应使用一次性容器或其他外部隔离。

## 验证

运行完整测试：

```bash
python -m unittest discover -s tests -v
```

`evals/` 还提供不依赖 LLM judge 的真实任务和确定性 verifier：

```bash
python evals/run.py list
python evals/run.py prepare cpp_binary_search ./.eval-workspaces/cpp_binary_search
python evals/run.py verify cpp_binary_search ./.eval-workspaces/cpp_binary_search
python evals/run.py report cpp_binary_search "$SESSION_ID"
```

`prepare` 后，从生成的 workspace 根目录启动 Agent，并把打印的任务说明原样发送给它。Agent
结束后再运行 `verify`；不同版本比较时应固定模型、Provider、reasoning 和各项预算。详细说明见
[evals/README.md](evals/README.md)。
