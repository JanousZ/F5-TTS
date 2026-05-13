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

# TTO

```bash
python src/f5_tts/infer/tto.py \
  --ref-audio asset/actor04_angry-strong_to_sad-strong.wav \
  --ref-text "Kids are talking by the door. Kids are talking by the door." \
  --gen-text "Kids are talking by the door. Kids are talking by the door." \
  --loss-mode value --opt-at 2,4,6,8,10,12,14,16,18,20,24,28 --opt-steps 50 --opt-lr 1e-2 \
  --vad-level frame \
  --output tto_demo.wav \
  --window-size 1.0 --hop-size 0.25 \
  --viz-path vis

python src/f5_tts/infer/tto.py \
  --ref-audio asset/actor02_sad-strong_to_surprised-strong.wav \
  --ref-text "Kids are talking by the door. Kids are talking by the door." \
  --gen-text "Kids are talking by the door. Kids are talking by the door." \
  --opt-at "" \
  --output plain_demo.wav

python src/f5_tts/infer/tto.py \
  --use-attn-mask \
  --ref-audio "/mnt/disk1/datasets/RAVDESS/Actor_02/03-01-04-02-01-01-02.wav||/mnt/disk1/datasets/RAVDESS/Actor_02/03-01-08-02-01-01-02.wav" \
  --ref-text "Kids are talking by the door.||Kids are talking by the door." \
  --gen-text "Kids are talking by the door.||Kids are talking by the door." \
  --opt-at "" \
  --output mask.wav

#批量处理
python src/f5_tts/infer/tto.py \
  --ref-text "Kids are talking by the door. Kids are talking by the door." \
  --gen-text "Dogs are walking on the floor. Dogs are walking on the floor." \
  --loss-mode value --opt-at 2,4,6,8,10,12,14,16,18,20,24,28 --opt-steps 30 --opt-lr 1e-2 \
  --vad-level both \
  --window-size 1.0 --hop-size 0.5 \
  --batch-size 8 \
  --ref-dir asset \
  --output tto_outputs/1.0_0.5_both \
  --viz-path tto_viz/1.0_0.5_both

python src/f5_tts/infer/run_budget_sweep.py \
  --n-samples 5 \
  --out-dir experiments/budget_sweep

python src/f5_tts/infer/aggregate_budget_sweep.py \
  --run-dir experiments/budget_sweep/20260422_042004

#指标检测
python src/f5_tts/eval/eval_metric.py \
  --ref ./asset/actor01_angry-strong_to_surprised-strong.wav \
  --gen ./asset/actor01_happy-strong_to_sad-strong.wav \
  --text "Some call me nature, others call me mother nature."

TAG=value-frame_w1.0_h0.25_at2-4-6-8-10-12-14_s50_lr1e-2_smhidden
CUDA_VISIBLE_DEVICES=2 python src/f5_tts/eval/batch_eval.py \
  --gen-dir tto_outputs/$TAG \
  --ref-dir asset \
  --gen-text "Dogs are sitting by the door. Dogs are sitting by the door." \
  --out-csv tto_outputs/$TAG/metrics.csv

for d in tto_outputs/*/; do
  [ -f "$d/metrics.csv" ] || continue
  CUDA_VISIBLE_DEVICES=2 python src/f5_tts/eval/batch_eval.py \
    --gen-dir "$d" \
    --ref-dir asset \
    --gen-text "Dogs are sitting by the door. Dogs are sitting by the door." \
    --out-csv "$d/metrics.csv"
done

# 批量生成 + 自动配对评测 (run_tto.sh)

单次跑：从 `--ref-dir` 随机采 `--batch-size` 条 ref 生成，结束后自动对每条 gen/ref 配对调 `batch_eval.py`，CSV+summary 落到 `tto_outputs/<TAG>/`。TAG 形如 `<loss>-<vad>_w<ws>_h<hs>_at<oa>_s<steps>_lr<lr>_sm<slide-mode>`，不同组合互不覆盖。

```bash
./run_tto.sh                               # 默认: value+frame+hidden, w=1.0 h=0.5, 8 条
./run_tto.sh --loss-mode embedding --vad-level both --opt-steps 30
./run_tto.sh --vad-slide-mode audio        # 切回旧的 audio-slide 实现做对照
./run_tto.sh --skip-eval                   # 只生成，不评测

# 查看结果
ls tto_outputs/<TAG>/                      # metrics.csv + metrics.summary.txt + *.wav
cat tto_outputs/*/metrics.summary.txt      # 所有 TAG 聚合统计
```

可配置项：`--window-size / --hop-size / --opt-at / --opt-steps / --opt-lr / --loss-mode / --vad-level / --vad-slide-mode / --batch-size / --ref-dir`。

`--vad-slide-mode` 决定 frame 模式下 VAD 怎么提取：

- **`hidden`** (默认): 整段音频一次过 wav2vec2 → `(T, 1024)` hidden state → 在时间轴上滑窗 mean → 分类头。**1 次 backbone forward**，每帧看到完整 attention 上下文，跟训练分布一致。
- **`audio`** (legacy): 在原始音频上滑窗 → 每窗独立跑整个模型。**N 次 forward**，每帧只看 1 s 局部，OOD 输入。
- 两种模式产出的 TAG 不同（`_smhidden` / `_smaudio`），可同时存放做对照实验。

# 批量扫参 + 多 GPU 并行 (sweep_tto.sh)

config 行格式：`ws|hs|opt_at|opt_steps|lr|loss_mode|vad_level|slide_mode`（8 字段），每行调一次 `run_tto.sh`，多卡时按任务队列动态分派。`slide_mode` 字段缺省时回落 `hidden`，旧的 7 字段 config 仍兼容。

```bash
# 单 GPU 串行（原行为）
./sweep_tto.sh

# 4 卡并行
./sweep_tto.sh --gpus 0-1

# 多卡生成 + 串行评测（显存吃紧时推荐）
./sweep_tto.sh --gpus 0-3 --defer-eval

# 混合 GPU id
./sweep_tto.sh --gpus 0,1,3,5-6
```

- **工作队列**：每张 GPU 始终只跑一个 config，跑完自动领下一个；不做静态切分，任务时长不均也不空转。
- **日志分离**：并发 stdout 会互相覆盖，每个 run 的完整输出到 `tto_outputs/_logs/<ts>/<idx>_gpu<N>_<TAG>.log`，用 `tail -f` 跟进。
- **`--defer-eval`**：生成阶段传 `--skip-eval` 给 `run_tto.sh`，全跑完后再在第一张 GPU 上串行评测——避免在 TTO 采样器之上再叠 ~5 GB 的 eval 模型栈。
- **Ctrl-C**：trap 会杀掉所有 children，不留孤儿进程。

跨 config 对比：

```bash
for f in tto_outputs/*/metrics.summary.txt; do
  echo "=== $f ==="; grep -E "^(spk_sim|e2v_sim_utt|e2v_sim_frame|av_sim_utt|av_sim_chunk|utmos|spk_sim|wer|cer)" "$f"
done

# 导出结果到csv分析
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
