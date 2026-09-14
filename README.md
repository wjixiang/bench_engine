# Bench Engine

Bench Engine 是一个独立的本地 benchmark 测试驱动器。核心编排器不绑定任何具体 benchmark；
HLE 作为第一个 adapter 注册，提供数据加载、prompt 构造、exact scoring 和可选 model grading。

## 安装与验证

```bash
uv sync
uv run bench-engine --help
uv run bench-engine datasets
```

## 运行

预览 prompt：

```bash
uv run bench-engine evaluate --benchmark hle-biomedical --limit 2 --dry-run
```

使用 Autonomics TUI：

```bash
uv run bench-engine evaluate \
  --benchmark hle-biomedical \
  --limit 5 \
  --model provider:model-name \
  --output runs/biomedical.jsonl
```

默认 TUI 是
`/mnt/projects/autonomics_projects/autonomics/target/release/tui`，
可用 `--tui PATH` 或 `BENCH_ENGINE_TUI=PATH` 覆盖。

使用外部 solver：

```bash
uv run bench-engine evaluate --benchmark hle-all --limit 10 \
  --solver-command 'python my_solver.py' \
  --output runs/custom.jsonl
```

外部 solver 收到 UTF-8 JSON stdin；若 stdout 是 JSON object，则依次读取
`answer`、`response`、`final_message` 或 `output`，否则把整个 stdout 当作响应。
输入 schema 是：

```json
{
  "id": "question-id",
  "question": "question text",
  "image": "data URI or empty string",
  "answer_type": "multipleChoice or exactMatch",
  "category": "source category"
}
```

TUI transport 目前只发送文本。默认不会把图像 data URI 放进 prompt；确要如此测试时，
使用 `--include-image-uri`。图像必需题目建议改用支持多模态输入的 `--solver-command`。

自由答案建议使用二次模型评分：

```bash
uv run bench-engine evaluate \
  --benchmark hle-biomedical \
  --grader model \
  --grader-model provider:grader-model \
  --output runs/model-graded.jsonl
```

中断后可用 `--resume --output runs/biomedical.jsonl` 续跑。结果为 JSONL，
旁边生成 `.summary.json`。日志只包含 ID、耗时、返回码和元数据，不包含题目、图像、
答案或模型响应。

## 架构

```text
bench-engine evaluate
  -> cli.py
  -> benchmarks.hle.HLEBenchmark            # prompt + scoring adapter
  -> core.runner.evaluate_examples          # concurrency, artifacts, metrics
  -> solvers.custom.CustomCommandSolver     # --solver-command
  -> solvers.tui.AutonomicsTuiSolver        # default TUI backend
```

新增 benchmark 时，在 `src/bench_engine/benchmarks` 中实现 `Benchmark` adapter，
把数据放在 `datasets/<adapter>/`，再在 CLI 注册名称。HLE 数据不得公开发布或用于训练。
