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

使用 Autonomics headless：

```bash
uv run bench-engine evaluate \
  --benchmark hle-biomedical \
  --limit 5 \
  --model provider:model-name \
  --autonomics-gateway \
  --data-mount-path /mnt/base/agent_workspace/smoke_vascular/runs/bench-engine/biomedical-demo \
  --output /mnt/base/agent_workspace/smoke_vascular/runs/biomedical.jsonl
```

默认 executable 是
`/mnt/projects/autonomics_projects/autonomics/target/release/autonomics`，
可用 `--autonomics PATH`（兼容 `--tui PATH`）、`BENCH_ENGINE_AUTONOMICS` 或兼容的
`BENCH_ENGINE_TUI` 覆盖。

Autonomics CLI 现在总是连接/启动 resident gateway，并通过 `--name` 为每个 task
创建一个全新的 agent（例如 `be_da_1_3_0f1e2d3c`）。Agent name 只包含小写字母、
数字和下划线，且不超过 Autonomics 的 32 字符限制。

传入 `--data-mount-path` 时，必须对使用 Autonomics 且带外部文件的 benchmark 追加
`--autonomics-gateway`。Bench Engine 会把 task 数据和工作目录映射到 gateway
VFS 中的 `/data` 与 `/app` 语义，并把对应虚拟路径写入 prompt。Host 侧仍从
mount root 收集 `answer.txt` 和 `trace.md`；`answer.txt` 会优先作为最终响应。
由于 run CLI 不再提供 per-process mount，`--data-mount-path` 必须位于 gateway
VFS 根挂载之下（本机是
`/mnt/base/agent_workspace/smoke_vascular`）。每个 task 使用独立 agent，因此可
配合 `--jobs N` 并发执行，同时以 `--no-memory` 关闭 memory。

使用外部 solver：

```bash
uv run bench-engine evaluate --benchmark hle-all --limit 10 \
  --solver-command 'python my_solver.py' \
  --data-mount-path runs/solver-data \
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

`data[].path` 必须是相对 task 文件夹且位于 `data/` 下的文件或目录路径。答案和
评分信息不会进入 solver payload。传入 `--data-mount-path` 后，runner 会把该路径
传给 solver；solver 内部为每个 task 创建 `<mount-root>/<task-id>/`，把 assets 按
相对路径符号链接到 `data/`，并创建可写 `work/`。外部 solver payload 中的
`data[].path` 会映射到挂载后的路径；未指定挂载路径时保留源路径。

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

OmicOS-BiomniBench 是开放的数据分析任务集合；每个 task 都是 task package，
`grading` 含 `method: "rubric"`、`answer_type: "rubric"`、`answer: null`、
`rubric` 文本和 `pass_threshold`（0-1）。Task loader 接受 `pass_threshold` 或
`threshold` 任一字段，并把 rubric / 阈值填充到 `Example.rubric` 与
`Example.grade_threshold`：

```bash
uv run bench-engine datasets
uv run bench-engine evaluate --benchmark omicos-biomnibench --limit 2 --dry-run
```

OmicOS 任务要求 solver 同时写 `/app/answer.txt` 与 `/app/trace.md`。Autonomics
solver 自动收集这两个文件并放入 `solver_artifacts`；`answer.txt` 替换
`solver_result.response`，`trace.md` 路径作为 `solver_artifacts["trace"]`。
OmicOS 任务允许 `task.json` 把 `data` 声明为指向其他位置的目录符号链接，task
loader 只在 `data/` 是符号链接且目标为目录时跳过 `data/` 子树约束，方便把大
体积输入数据存放在共享目录里。

OmicOS rubric 评分可通过 `--grader omicos` 使用 OpenAI-compatible API，也可通过
`--grader codex-local` 调用本机已登录的 headless Codex CLI。后者会读取
`BENCH_ENGINE_CODEX` 指定的可执行文件（默认 `codex`），使用与 `OmicOSGrader`
相同的 rubric prompt、分数范围和 pass threshold：

```bash
export OPENAI_API_KEY=...
uv run bench-engine evaluate \
  --benchmark omicos-biomnibench \
  --limit 5 \
  --grader omicos \
  --grader-model <openai-model> \
  --autonomics-gateway \
  --data-mount-path /mnt/base/agent_workspace/smoke_vascular/runs/bench-engine/omicos-demo \
  --output /mnt/base/agent_workspace/smoke_vascular/runs/omicos.jsonl
```

复用常驻 gateway 的 `da-1-3` 烟测：

```bash
uv run bench-engine evaluate \
  --benchmark omicos-biomnibench \
  --question-id da-1-3 \
  --grader omicos \
  --autonomics-gateway \
  --data-mount-path /mnt/base/agent_workspace/smoke_vascular/runs/bench-engine \
  --jobs 5 \
  --output /mnt/base/agent_workspace/smoke_vascular/runs/da-1-3-gateway.jsonl
```

`OmicOSGrader` 把 rubric 文本、threshold、`/app/answer.txt` 内容以及
`/app/trace.md` 内容拼成一个评分 prompt，要求模型以 `Score: <0-100>` 一行
结尾；得分 ≥ `pass_threshold * 100` 即视为通过，结果的 `grade_method` 为
`"rubric"`，`grade_detail` 形如 `score=85/100 threshold=70/100 verdict=PASS`。
`grade_method` 等于 `"exact"` 且无 rubric 时，CLI 会提示
`rubric tasks require --grader omicos or --grader model`，避免误把空答案
当作通过。

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
`OPENAI_API_KEY`。自定义 OpenAI-compatible endpoint 可用 `--grader-base-url`
或 `OPENAI_BASE_URL` 配置。这些变量会自动从当前目录的 `.env` 读取，已导出的
环境变量优先。

中断后可用 `--resume --output runs/biomedical.jsonl` 续跑。结果为 JSONL，
旁边生成 `.summary.json`。日志只包含 ID、耗时、返回码和元数据，不包含题目、图像、
答案或模型响应。

大规模 OmicOS 批量测试建议使用：

```bash
scripts/start_omicos_mass_gateway.sh
journalctl --user -u bench-engine-omicos.service -f
```

该 launcher 会创建可自动重启的用户级 systemd service，限制整批内存，并让
gateway、solver frontend 和工具进程位于同一个 `KillMode=control-group` cgroup。
若某个 task 被内存回收终止，runner 会把 returncode `-9` 或明确 OOM 诊断写入
`<output>.oom-skipped.jsonl`；下次 `--resume` 会跳过这些 task，且不会把它们计入
评分结果。主结果中的 `solver_agent_name` 记录实际传给 Autonomics 的
`--name`，便于把 OOM sidecar 和具体 agent 对齐。

## 架构

```text
bench-engine evaluate
  -> cli.py
  -> benchmarks.hle.HLEBenchmark            # prompt + scoring adapter
  -> benchmarks.omicos.OMICOS              # OmicOS-BiomniBench adapter
  -> core.runner.evaluate_examples          # concurrency, artifacts, metrics
  -> grading.omicos.OmicOSGrader            # rubric scoring for --grader omicos
  -> core.runner.OpenAIGrader              # generic --grader model
  -> solvers.custom.CustomCommandSolver     # --solver-command
  -> solvers.autonomics_solver.AutonomicsTuiSolver
```

新增 benchmark 时，在 `src/bench_engine/benchmarks` 中实现 `Benchmark` adapter，
把数据放在 `datasets/<adapter>/`，再在 CLI 注册名称。HLE 数据不得公开发布或用于训练。
