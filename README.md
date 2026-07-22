# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![demo](https://img.shields.io/badge/GitHub-Demo-orange.svg)](https://swivid.github.io/F5-TTS/)
[![hfspace](https://img.shields.io/badge/🤗-HF%20Space-yellow)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
[![msspace](https://img.shields.io/badge/🤖-MS%20Space-blue)](https://modelscope.cn/studios/AI-ModelScope/E2-F5-TTS)
[![lab](https://img.shields.io/badge/🏫-X--LANCE-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)
[![lab](https://img.shields.io/badge/🏫-SII-grey?labelColor=lightgrey)](https://www.sii.edu.cn/)
[![lab](https://img.shields.io/badge/🏫-PCL-grey?labelColor=lightgrey)](https://www.pcl.ac.cn)
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->

**F5-TTS**: Diffusion Transformer with ConvNeXt V2, faster trained and inference.

**E2 TTS**: Flat-UNet Transformer, closest reproduction from [paper](https://arxiv.org/abs/2406.18009).

**Sway Sampling**: Inference-time flow step sampling strategy, greatly improves performance

### Thanks to all the contributors !

## News
- **2025/03/12**: 🔥 F5-TTS v1 base model with better training and inference performance. [Few demo](https://swivid.github.io/F5-TTS_updates).
- **2024/10/08**: F5-TTS & E2 TTS base models on [🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS), [🤖 Model Scope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN), [🟣 Wisemodel](https://wisemodel.cn/models/SJTU_X-LANCE/F5-TTS_Emilia-ZH-EN).

## Installation

### Create a separate environment if needed

```bash
# Create a conda env with python_version>=3.10  (you could also use virtualenv)
conda create -n f5-tts python=3.11
conda activate f5-tts

# Install FFmpeg if you haven't yet
conda install ffmpeg
```

### Install PyTorch with matched device

<details>
<summary>NVIDIA GPU</summary>

> ```bash
> # Install pytorch with your CUDA version, e.g.
> pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
> 
> # And also possible previous versions, e.g.
> pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
> # etc.
> ```

</details>

<details>
<summary>AMD GPU</summary>

> ```bash
> # Install pytorch with your ROCm version (Linux only), e.g.
> pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --extra-index-url https://download.pytorch.org/whl/rocm6.2
> ```

</details>

<details>
<summary>Intel GPU</summary>

> ```bash
> # Install pytorch with your XPU version, e.g.
> # Intel® Deep Learning Essentials or Intel® oneAPI Base Toolkit must be installed
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/test/xpu
> 
> # Intel GPU support is also available through IPEX (Intel® Extension for PyTorch)
> # IPEX does not require the Intel® Deep Learning Essentials or Intel® oneAPI Base Toolkit
> # See: https://pytorch-extension.intel.com/installation?request=platform
> ```

</details>

<details>
<summary>Apple Silicon</summary>

> ```bash
> # Install the stable pytorch, e.g.
> pip install torch torchaudio
> ```

</details>

### Then you can choose one from below:

> ### 1. As a pip package (if just for inference)
> 
> ```bash
> pip install f5-tts
> ```
> 
> ### 2. Local editable (if also do training, finetuning)
> 
> ```bash
> git clone https://github.com/SWivid/F5-TTS.git
> cd F5-TTS
> # git submodule update --init --recursive  # (optional, if use bigvgan as vocoder)
> pip install -e .
> ```

### Docker usage also available
```bash
# Build from Dockerfile
docker build -t f5tts:v1 .

# Run from GitHub Container Registry
docker container run --rm -it --gpus=all --mount 'type=volume,source=f5-tts,target=/root/.cache/huggingface/hub/' -p 7860:7860 ghcr.io/swivid/f5-tts:main

# Quickstart if you want to just run the web interface (not CLI)
docker container run --rm -it --gpus=all --mount 'type=volume,source=f5-tts,target=/root/.cache/huggingface/hub/' -p 7860:7860 ghcr.io/swivid/f5-tts:main f5-tts_infer-gradio --host 0.0.0.0
```

### Runtime

Deployment solution with Triton and TensorRT-LLM.

#### Benchmark Results
Decoding on a single L20 GPU, using 26 different prompt_audio & target_text pairs, 16 NFE.

| Model               | Concurrency    | Avg Latency | RTF    | Mode            |
|---------------------|----------------|-------------|--------|-----------------|
| F5-TTS Base (Vocos) | 2              | 253 ms      | 0.0394 | Client-Server   |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.0402 | Offline TRT-LLM |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.1467 | Offline Pytorch |

See [detailed instructions](src/f5_tts/runtime/triton_trtllm/README.md) for more information.


## Inference

- In order to achieve desired performance, take a moment to read [detailed guidance](src/f5_tts/infer).
- By properly searching the keywords of problem encountered, [issues](https://github.com/SWivid/F5-TTS/issues?q=is%3Aissue) are very helpful.

### 1. Gradio App

Currently supported features:

- Basic TTS with Chunk Inference
- Multi-Style / Multi-Speaker Generation
- Voice Chat powered by Qwen2.5-3B-Instruct
- [Custom inference with more language support](src/f5_tts/infer/SHARED.md)

```bash
# Launch a Gradio app (web interface)
f5-tts_infer-gradio

# Specify the port/host
f5-tts_infer-gradio --port 7860 --host 0.0.0.0

# Launch a share link
f5-tts_infer-gradio --share
```

<details>
<summary>NVIDIA device docker compose file example</summary>

```yaml
services:
  f5-tts:
    image: ghcr.io/swivid/f5-tts:main
    ports:
      - "7860:7860"
    environment:
      GRADIO_SERVER_PORT: 7860
    entrypoint: ["f5-tts_infer-gradio", "--port", "7860", "--host", "0.0.0.0"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  f5-tts:
    driver: local
```

</details>

### 2. CLI Inference

```bash
# Run with flags
# Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
f5-tts_infer-cli --model F5TTS_v1_Base \
--ref_audio "asset/actor03_neutral-normal_to_angry-strong.wav" \
--ref_text "Kids are talking by the door. Kids are talking by the door." \
--gen_text "I received the gift from another country. It's really delicated and beautiful."

f5-tts_infer-cli --model F5TTS_v1_Base \
--gen_text "I received the gift from another country."

# Run with default setting. src/f5_tts/infer/examples/basic/basic.toml
f5-tts_infer-cli
# Or with your own .toml file
f5-tts_infer-cli -c custom.toml

# Multi voice. See src/f5_tts/infer/README.md
f5-tts_infer-cli -c src/f5_tts/infer/examples/multi/story.toml
```


## Training

### 1. With Hugging Face Accelerate

Refer to [training & finetuning guidance](src/f5_tts/train) for best practice.

### 2. With Gradio App

```bash
# Quick start with Gradio web interface
f5-tts_finetune-gradio
```

Read [training & finetuning guidance](src/f5_tts/train) for more instructions.


## [Evaluation](src/f5_tts/eval)


## Development

Use pre-commit to ensure code quality (will run linters and formatters automatically):

```bash
pip install pre-commit
pre-commit install
```

When making a pull request, before each commit, run: 

```bash
pre-commit run --all-files
```

Note: Some model components have linting exceptions for E722 to accommodate tensor notation.


## Acknowledgements

- [E2-TTS](https://arxiv.org/abs/2406.18009) brilliant work, simple and effective
- [Emilia](https://arxiv.org/abs/2407.05361), [WenetSpeech4TTS](https://arxiv.org/abs/2406.05763), [LibriTTS](https://arxiv.org/abs/1904.02882), [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) valuable datasets
- [lucidrains](https://github.com/lucidrains) initial CFM structure with also [bfs18](https://github.com/bfs18) for discussion
- [SD3](https://arxiv.org/abs/2403.03206) & [Hugging Face diffusers](https://github.com/huggingface/diffusers) DiT and MMDiT code structure
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as ODE solver, [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) and [BigVGAN](https://github.com/NVIDIA/BigVGAN) as vocoder
- [FunASR](https://github.com/modelscope/FunASR), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [UniSpeech](https://github.com/microsoft/UniSpeech), [SpeechMOS](https://github.com/tarepan/SpeechMOS) for evaluation tools
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) for speech edit test
- [mrfakename](https://x.com/realmrfakename) huggingface space demo ~
- [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx/tree/main) Implementation with MLX framework by [Lucas Newman](https://github.com/lucasnewman)
- [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) ONNX Runtime version by [DakeQQ](https://github.com/DakeQQ)
- [Yuekai Zhang](https://github.com/yuekaizhang) Triton and TensorRT-LLM support ~

## Citation
If our work and codebase is useful for you, please cite as:
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.

# RAVDESS Dataset

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)

- 来源: https://zenodo.org/records/1188976
- 24 位专业演员 (12 男 / 12 女)，使用标准北美口音录制
- 当前下载内容: Audio Speech (语音音频，1440 个文件)

## 文件命名规则

文件名由 7 个数字编码组成，以 `-` 分隔，格式为:

```
{模态}-{声道}-{情感}-{情感强度}-{语句}-{重复次数}-{演员编号}.wav
```

例如: `03-01-06-01-02-01-12.wav`

### 各位置含义

| 位置 | 含义                  | 编码值                                                       |
| ---- | --------------------- | ------------------------------------------------------------ |
| 1    | 模态 (Modality)       | 01 = 完整音视频, 02 = 仅视频, 03 = 仅音频                    |
| 2    | 声道 (Vocal Channel)  | 01 = 语音 (Speech), 02 = 歌曲 (Song)                         |
| 3    | 情感 (Emotion)        | 01 = 中性, 02 = 平静, 03 = 快乐, 04 = 悲伤, 05 = 愤怒, 06 = 恐惧, 07 = 厌恶, 08 = 惊讶 |
| 4    | 情感强度 (Intensity)  | 01 = 正常, 02 = 强烈 (中性情感无强烈版本)                    |
| 5    | 语句 (Statement)      | 01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door" |
| 6    | 重复次数 (Repetition) | 01 = 第1次, 02 = 第2次                                       |
| 7    | 演员编号 (Actor)      | 01-24，奇数 = 男性，偶数 = 女性                              |

### 示例

`03-01-05-02-01-01-03.wav` 表示:

- 03 = 仅音频
- 01 = 语音
- 05 = 愤怒
- 02 = 强烈
- 01 = "Kids are talking by the door"
- 01 = 第1次重复
- 03 = 演员03 (男性)

## 目录结构

```
RAVDESS/
  Actor_01/    # 演员01 (男)
  Actor_02/    # 演员02 (女)
  ...
  Actor_24/    # 演员24 (女)
```

每位演员 60 个语音音频文件，共 1440 个文件。

## 情感拼接
python emotion_concat.py \
  --actor 06 \
  --emotion1 03 --intensity1 02 \
  --emotion2 07 --intensity2 02 \
  --output_dir asset

python emotion_concat.py \
  --actor 03,\
  --emotion1 03,05,06 --intensity1 02 \
  --emotion2 07,08,04 --intensity2 02 \
  --skip_existing \
  --output_dir asset

# TTO (Test-Time Optimization)

`src/f5_tts/infer/tto.py` 在 F5-TTS 采样的若干 ODE step 上对 latent `x_t` 做 Adam 优化，让 vocoder 解出的 wav 在 wav2vec2 [VAD (valence/arousal/dominance)](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) 空间里逼近参考音频。CFM transformer / vocoder / VAD encoder 全程冻结。

默认只优化 VAD loss；可通过 `--spk-loss-weight λ` 附加 **SIM-O speaker similarity loss**（WavLM-Large + ECAPA-TDNN，与 `eval_metric.py` 的 `spk_sim` 完全同源），按 `L = L_vad + λ·L_spk` 加权求和。UTMOS loss 已移除；`eval_metric.py` 里 `compute_utmos` 仍作为独立评测指标保留。

## CLI：三种输入模式

### 1. 单条推理

```bash
# 不做 TTO 的 baseline（等价于 CFM.sample）
python src/f5_tts/infer/tto.py \
  --ref-audio path/to/ref.wav \
  --ref-text "the text spoken in ref.wav" \
  --gen-text "the text you want to generate" \
  --opt-at "" \
  --output baseline.wav

# 开 TTO：在 ODE step 2/4/6/8/10/12/14 各跑 50 步 Adam (lr=1e-2)
python src/f5_tts/infer/tto.py \
  --ref-audio path/to/ref.wav \
  --ref-text "..." \
  --gen-text "..." \
  --loss-mode embedding --vad-level both \
  --opt-at 2,4,6,8,10,12,14 --opt-steps 50 --opt-lr 1e-2 \
  --window-size 1.0 --hop-size 0.25 \
  --output tto_demo.wav

python src/f5_tts/infer/tto.py \
  --ref-audio /home/yanzhang/F5-TTS/tto_eval/esd_emochange/refs/0012_happy2sad_012.wav \
  --ref-text "He was still in the forest! That I owe my thanks to you." \
  --gen-text "The fisherman and his wife see George every day. I believe you are one of them!" \
  --loss-mode embedding --vad-level frame \
  --opt-at 2,4,6,8,10,12,14 --opt-steps 50 --opt-lr 1e-2 \
  --window-size 0.5 --hop-size 0.25 \
  --output e_f_2,4,6,8,10,12,14_50_1e-2_0.5_0.25.wav
```

主要旋钮：

| 参数 | 取值 | 含义 |
|------|------|------|
| `--loss-mode` | `value` / `embedding` | VAD 头的 3-维 (V/A/D) 输出，还是 hidden embedding (1024-维) |
| `--vad-level` | `frame` / `utter` / `both` | frame-wise MSE / utterance-level MSE / 两者求和 |
| `--opt-at` | CSV 索引 (e.g. `"2,4,6,8"`) | 哪些 ODE step 触发 TTO；`""` = 完全关掉 |
| `--opt-steps` | int | 每个 step 的 Adam 内迭代步数 |
| `--opt-lr` | float | Adam 学习率 |
| `--spk-loss-weight` | float (default `0.0`) | SIM-O loss 权重 λ；`0` 关闭（不加载 WavLM）；典型 `0.01–1.0`，**见下方量级提示** |
| `--window-size` / `--hop-size` | sec | frame VAD 提取的滑窗 |
| `--steps` | int (default 32) | ODE 总步数 |
| `--cfg-strength` | float (default 2.0) | 推理时 CFG 强度 |

> **λ 量级提示**：冷启动 generation 的 SIM-O 距离 `1 - cos` ≈ 0.6–0.7，而 `--loss-mode embedding` 下的 VAD MSE ≈ 0.005–0.02，两者**相差约 50–100×**。所以 `λ=1.0` 会让 SIM-O 完全主导、压过 VAD 信号；想让两路平衡，从 `λ=0.05` 起步扫。`--loss-mode value` 下 VAD MSE 大一个量级（~0.1），`λ≈1.0` 才接近平衡。启用后会额外加载 WavLM-Large (316 M) 并 backprop 通过 24 层 transformer，**VRAM 涨 6–10 GB**——OOM 时调小 `--opt-at` / `--opt-steps`。

### 2. 批量模式 (legacy `--ref-dir`)

从一个目录随机采 N 条 ref 跑同一对 `--ref-text` / `--gen-text`：

```bash
python src/f5_tts/infer/tto.py \
  --ref-dir asset --batch-size 120 \
  --ref-text "Kids are talking by the door. Kids are talking by the door." \
  --gen-text "Dogs are walking on the floor. Dogs are walking on the floor." \
  --loss-mode embedding --vad-level frame \
  --opt-at 2,4,6,8,10,12,14 --opt-steps 50 --opt-lr 1e-2 \
  --output tto_outputs/exp_batch
```

### 3. Manifest 模式 (推荐，每行独立 ref/gen 文本)

每行 JSONL 自带 `stem / ref_wav / ref_text / gen_text`，专为 ESD ([Emotional Speech Dataset](https://github.com/HLTSingapore/Emotional-Speech-Data)) 0011–0020 的 emo-change 评测设计：

```bash
# (a) 一次性构建评测集（默认 spk=0011+0012，300 stem，~52 MB refs/）
python src/f5_tts/eval/build_emochange_eval.py \
  --esd-root /mnt/disk1/datasets/ESD \
  --out-dir tto_eval/esd_emochange

# (b) 跑生成
python src/f5_tts/infer/tto.py \
  --manifest tto_eval/esd_emochange/manifest_disjoint.jsonl \
  --output tto_outputs/esd_disjoint_emo0.0001 \
  --loss-mode embedding --vad-level frame \
  --opt-at 2,4,6,8,10,12,14 --opt-steps 50 --opt-lr 1e-2 \
  --emo-loss-weight 0.0001

```

`build_emochange_eval.py` 默认产两个 manifest，共享同一份 `refs/`：

- **`manifest_disjoint.jsonl`** — `gen_text` 与 `ref_text` 完全不重叠，测「情感是否真转移到了新内容」
- **`manifest_sameText.jsonl`** — `gen_text == ref_text`，上限对照（spk/e2v sim 期望接近 1.0）

`--manifest` 与 `--batch-size` 互斥。

## 评测

### 单对：`eval_metric.py`

6 个指标（WER/CER、spk_sim、EMO-sim_utt/frame、av_sim_utt/chunk、pcp_score）+ UTMOS naturalness：

```bash
python src/f5_tts/eval/eval_metric.py \
  --ref ref.wav --gen gen.wav --text "the gen text"

python src/f5_tts/eval/eval_metric.py \
  --ref /home/yanzhang/F5-TTS/tto_eval/esd_emochange/refs/0012_happy2sad_012.wav --gen v_f_2,4,6,8,10,12,14_50_1e-2_1.0_0.25.wav --text "The fisherman and his wife see George every day. I believe you are one of them!"
```

### 批量：`batch_eval.py`

**Flat 模式**（对应 `--ref-dir` 生成；全 batch 共用一个 gen_text，stem-by-stem 配对 ref）：

```bash
python src/f5_tts/eval/batch_eval.py \
  --gen-dir tto_outputs/exp_batch \
  --ref-dir asset \
  --gen-text "Dogs are walking on the floor. Dogs are walking on the floor." \
  --out-csv tto_outputs/exp_batch/metrics.csv
```

**Manifest 模式**（对应 `--manifest` 生成；每行独立 ref/gen_text）：

```bash
python src/f5_tts/eval/batch_eval.py \
  --gen-dir tto_outputs/esd_disjoint_spk0.01_asr0.0001 \
  --manifest tto_eval/esd_emochange/manifest_disjoint.jsonl \
  --out-csv tto_outputs/esd_disjoint_spk0.01_asr0.0001/metrics.csv
```

两种模式都产 `metrics.csv` + `metrics.summary.txt`。

## Shell wrappers

### `run_tto.sh` —— 单次跑（生成 + 自动配对评测）

```bash
./run_tto.sh                                          # 默认 value+frame，BATCH_SIZE=8
./run_tto.sh --loss-mode embedding --vad-level both --opt-steps 30
./run_tto.sh --skip-eval                              # 只生成，不评测
```

TAG 形如 `<loss>-<vad>_w<ws>_h<hs>_at<oa>_s<steps>_lr<lr>`，输出落 `tto_outputs/<TAG>/`。
可配置项：`--window-size / --hop-size / --opt-at / --opt-steps / --opt-lr / --loss-mode / --vad-level / --batch-size / --ref-dir`。

> 注：当前 `run_tto.sh` 仍走 legacy `--ref-dir` + 共用 `REF_TEXT/GEN_TEXT` 的 flat 模式。如需 manifest 评测，直接调上文 `tto.py --manifest` + `batch_eval.py --manifest` 两条命令。

### `sweep_tto.sh` —— 多 config × 多 GPU 并行

config 行格式：`ws|hs|opt_at|opt_steps|lr|loss_mode|vad_level`（7 字段），每行调一次 `run_tto.sh`。

```bash
./sweep_tto.sh                               # 单 GPU 串行
./sweep_tto.sh --gpus 0-3                    # 4 卡并行（工作队列动态分派）
./sweep_tto.sh --gpus 0-3 --defer-eval       # 生成全跑完后再串行评测
```

- 每张 GPU 始终只跑一个 config，跑完自动领下一个；任务时长不均也不空转。
- 并发 stdout 落 `tto_outputs/_logs/<ts>/<idx>_gpu<N>_<TAG>.log`，用 `tail -f` 跟。
- `--defer-eval`：避免在 TTO 采样器之上再叠 ~5 GB 的 eval 模型栈。
- Ctrl-C 会清掉所有 children，不留孤儿。

### `run_exps.sh` —— 固定实验配置

```bash
./run_exps.sh --gpus 0-1                     # exp55 + exp66 × NUM_RUNS
./run_exps.sh --runs 3 --gpus 0-3            # 每 config 跑 3 次（不同 seed）
```

## 跨 config 聚合

```bash
# 快速看每个 TAG 的 summary
for f in tto_outputs/*/metrics.summary.txt; do
  echo "=== $f ==="
  grep -E "^(wer|cer|utmos|spk_sim|EMO-sim|av_sim|pcp_score)" "$f"
done

# 导出 Excel-friendly CSV
python src/f5_tts/infer/aggregate_sweep.py
```

# tto -> text_tto

python src/f5_tts/infer/text_VAD.py \
    --text "I am extremely happy today, but tomorrow will be sad" \
    --n-win 30 --weight-mode chars

单条：
python src/f5_tts/infer/text_tto.py \
    --gen-text "I am very happy today" \
    --opt-at 16,24 --opt-steps 5 --opt-lr 5e-3 \
    --output /tmp/test.wav

python src/f5_tts/infer/text_tto.py \
    --text-vad-scale 0.2,0.9 \
    ...
