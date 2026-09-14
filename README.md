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

运行 LAB-Bench public 子集：

```bash
uv run bench-engine evaluate --benchmark lab-dbqa --limit 2 --dry-run
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
  "category": "source category",
  "data": [
    {
      "name": "asset name",
      "path": "absolute input file path",
      "media_type": "image/png",
      "role": "image"
    }
  ]
}
```

`data` 仅在 task package 含外部文件时出现。Bench Engine 支持两种 dataset 存储：
Parquet 表格和 task package 路径组织。Parquet 适合 HLE 这类无外部文件的数据，
统一放在 `datasets/<adapter>/<dataset>.parquet`，canonical columns 如下：

| column | type | constraint |
| --- | --- | --- |
| `id` | `String` | required, unique, non-blank |
| `question` | `String` | required, non-blank |
| `image` | `String` | empty string or `data:image/...` URI |
| `answer` | `String` | required, non-blank |
| `answer_type` | `String` | `exactMatch` or `multipleChoice` |
| `category` | `String` | required, non-blank |

Dataset 可以保留其他列，例如 HLE 的源数据元信息；加载时会校验上述 canonical
schema，再归一化为内部 `Example`。

需要外部文件时使用 task package，一个 task 一个文件夹。`instruction` 合并在
`task.json` 内，不使用单独的 `instruction.md`：

```text
datasets/<benchmark>/<dataset>/<task-id>/
  task.json
  data/
```

`task.json` 使用 task schema v1：

```json
{
  "schema_version": 1,
  "id": "task-id",
  "instruction": "complete agent instruction",
  "category": "Dataset",
  "subtask": "optional source subtask",
  "grading": {
    "method": "exact",
    "answer_type": "multipleChoice",
    "answer": "B"
  },
  "metadata": {},
  "data": [
    {
      "name": "image",
      "path": "data/image.png",
      "media_type": "image/png",
      "role": "image"
    }
  ]
}
```

`data[].path` 必须是相对 task 文件夹且位于 `data/` 下的文件路径。外部 solver
收到的是解析后的绝对路径；答案和评分信息不会进入 solver payload。

HLE 也已迁移为 task package：

```bash
uv run python scripts/convert_hle.py \
  /path/to/hle-full.parquet \
  datasets/hle
```

`datasets/hle/full` 是 canonical 存储，`biomedical` 和 `biomedical_visual`
通过 task 链接定义子集，不重复复制数据或图片。

LAB-Bench 数据由转换脚本生成，不直接提交源 JSONL 和图片：

```bash
uv run --with pillow python scripts/convert_lab_bench.py \
  /mnt/base/benchmark/lab_bench/LAB-Bench \
  datasets/lab_bench
```

生成结果按原始大类拆分为 8 个 task package collection，共包含 LAB-Bench public
release 的 1,967 道多选题：`cloning_scenarios`、`dbqa`、`figqa`、`litqa2`、
`protocolqa`、`seqqa`、`suppqa`、`tableqa`。CLI 名称分别为
`lab-cloning-scenarios`、`lab-dbqa`、`lab-figqa`、`lab-litqa2`、`lab-protocolqa`、
`lab-seqqa`、`lab-suppqa`、`lab-tableqa`；`lab-bench` 会合并全部 8 个数据集。
每个 task 的完整指令、选项、评分和源元数据在 `task.json` 中；FigQA/TableQA 图片
保存在各自 `data/` 下，TableQA 多表题会垂直合成一张图。LAB-Bench 源数据包含
canary，不得用于训练。

TUI transport 目前只发送文本。默认不会把图像 data URI 放进 prompt；确要如此测试时，
使用 `--include-image-uri`。图像必需题目建议改用支持多模态输入的 `--solver-command`。

自由答案建议使用二次模型评分：

```bash
export OPENAI_API_KEY=...
uv run bench-engine evaluate \
  --benchmark hle-biomedical \
  --grader model \
  --grader-model <openai-model> \
  --output runs/model-graded.jsonl
```

模型评分使用内置 OpenAI SDK 客户端，直接调用 OpenAI Responses API，不会再复用
当前被测 solver。模型名也可通过 `OPENAI_GRADER_MODEL` 配置；API 凭证使用
`OPENAI_API_KEY`。

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
  -> solvers.autonomics_solver.AutonomicsTuiSolver
```

新增 benchmark 时，在 `src/bench_engine/benchmarks` 中实现 `Benchmark` adapter，
把数据放在 `datasets/<adapter>/`，再在 CLI 注册名称。HLE 数据不得公开发布或用于训练。
