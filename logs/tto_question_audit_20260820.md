# TTO 问题排查记录

日期: 2026-08-20

## 结论

现有 F5-TTS 工作区可以回答你问题中的一部分，但不能完整回答全部问题。

能回答的部分:
- 代理目标已经明确实现为 VAD / speaker sim / ASR / emotion embedding 等可微损失。
- `loss_mode=value|embedding` 和 `vad_level=frame|utter|both` 已经把“用 VAD 值还是 hidden state”这件事编码进实现。
- `spk_loss`、`asr_loss`、`emo_loss` 已经提供了额外约束，可用于观察 WER、音质、说话人相似度、情感相似度的权衡。

不能被当前材料直接回答的部分:
- 没有保留下来的 `tto_outputs/*/metrics.csv` / `metrics.summary.txt`，也没有训练/优化过程日志。
- 因此无法从现有记录中直接比较不同 `opt_at`、`opt_steps`、`opt_lr`、`loss_mode`、`vad_level`、`spk/asr/emo weight` 对 WER/UTMOS/情感指标的真实影响。
- 也无法证明“优化 latent codec vs speech code embedding”哪种更稳，因为仓库里没有对应的对照实验。

## 关键证据

- `src/f5_tts/infer/tto.py`
  - 优化对象是 `x_t`，不是模型参数。
  - 代理损失支持 `loss_mode=("value","embedding")` 和 `vad_level=("frame","utter","both")`。
  - 额外约束支持 `spk_loss_weight` / `asr_loss_weight` / `emo_loss_weight`。
  - 使用 one-step `x1` 估计后再过 vocoder 进入代理模型。

- `src/f5_tts/infer/vad_loss.py`
  - VAD 代理来自 wav2vec2 MSP-DIM。
  - 支持 3 维回归输出，也支持 hidden embedding。
  - 支持 frame-wise 滑窗和 utterance-level 计算。

- `src/f5_tts/eval/eval_metric.py` 和 `src/f5_tts/eval/batch_eval.py`
  - 评测已经覆盖 `wer/cer`、`utmos`、`spk_sim`、`EMO-sim_utt/frame`、`av_sim_utt/chunk`、`pcp_score`。
  - 这说明“是否会伤 WER / 音质 / 情感 / 说话人”是可被统一评测的。

- 现有结果文件
  - 只有 `tto_eval/esd_emochange/*/results.summary.txt` 四份基线摘要。
  - 它们是 `index-tts` / `cosyvoice` 在 sameText / disjoint 上的评测，不是 F5-TTS TTO 网格日志。
  - 当前工作区没有 `tto_outputs/`、没有 `metrics.csv`、没有实验 stdout log。

## 逐项判断

1. 损失函数的选择
   - VAD 作为情感强度代理: 可以部分回答。
   - 代码层面已支持 `value` 和 `embedding` 两种代理，但没有完整实验结果证明哪种更优。
   - 情感分类置信度代理: 代码里没有直接用分类置信度做 TTO loss。

2. 是否需要额外损失防止 WER / 音质下降
   - 可以作为实验框架回答，但现有结果不足以下结论。
   - `ASR` / `speaker` / `EMO` 约束已经实现，`UTMOS` 只保留为评测，不再作为优化项。

3. 计算输入选什么
   - 代码默认是先预测当前 `x1`，过 vocoder，再送入代理模型。
   - 这条链路确实存在，且注释也明确承认链路长。
   - 直接在 `x_t` 上训练代理模型的方案仓库里没有实现和结果。

4. 被优化参数的选择
   - 当前实现只优化 latent `x_t`。
   - 没有看到对 `hidden codec` / `speech code embedding` 的替代实验或对照记录。

5. 时间效率平衡
   - `opt_at` / `opt_steps` / `opt_lr` 已经是可调旋钮。
   - 但目前没有保留下来的 sweep 结果，无法判断哪些时间步最有效、每步几次最合适、是否需要梯度放缩。

## 建议

- 如果你想让这组问题有可回答的实证结论，需要补一轮系统化 sweep，并保存：
  - 运行日志
  - 每条样本的 `metrics.csv`
  - 汇总 `metrics.summary.txt`
  - 配置快照（`loss_mode` / `vad_level` / `opt_at` / `opt_steps` / `opt_lr` / 各 loss weight）
- 最优先的对照是:
  - `loss_mode=value` vs `embedding`
  - `vad_level=frame` vs `utter` vs `both`
  - 是否加 `spk/asr/emo` 辅助损失
  - 不同 `opt_at` 和 `opt_steps`

## 2026-08-20 追加记录: target-span baseline 实现

### 本次改动

- 新增 `src/f5_tts/infer/target_span.py`
  - 提供 `span_to_sample_bounds`、`crop_waveform`、`blend_vad_targets`、`target_latent_mask`
  - 让 target span 统一按生成音频时间轴定义
- 修改 `src/f5_tts/infer/tto.py`
  - 增加 `vad_level="target"` 分支
  - 支持 `--target-audio`
  - 支持 `--target-span-start-sec` / `--target-span-end-sec`
  - 支持 target span 的局部 latent mask
  - 将 prompt 侧与 target 侧的参考音频分离
- 新增 `src/f5_tts/infer/target_span_test.py`
  - 只做轻量单元测试，不依赖模型权重

### 当前 baseline 定义

- prompt 音频: 中性音色参考
- target 音频: 目标情感参考
- target span: `tts_text` 中人工指定的词/短语区间
- 监督目标: target 音频的 pooled VAD value
- 优化对象: 生成侧的 target span 对应 latent 区域
- 默认模式: `loss_mode=value` + `vad_level=target` + one-step denoise

### 验证结果

- `python -m py_compile src/f5_tts/infer/tto.py src/f5_tts/infer/target_span.py src/f5_tts/infer/target_span_test.py`
  - 通过
- `/home/yanzhang/.conda/envs/f5-tts/bin/python -m unittest f5_tts.infer.target_span_test`
  - 4 个测试通过
- 旧的系统 Python 环境缺少 `torch`，所以不适合跑这组验证

### 备注

- 现阶段还没有把 ESD 的自动数据生成和 forced alignment 接进来
- 当前实现默认 target span 由外部提前给定，后续再接词级对齐器

## 2026-08-20 追加记录: target-span smoke test

### 跑通项

- 示例命令使用 `src/f5_tts/infer/examples/basic/basic_ref_en.wav`
- prompt / target audio 已分离
- `vad_level=target` 已生效
- 输出已写入:
  - `/tmp/f5_tto_baseline.wav`
  - `/tmp/f5_tto_target_smoke.wav`

### 可量化对照

- target-span VAD MSE against prompt target:
  - baseline: `0.005827`
  - TTO: `0.004438`
- 结论: 局部 target-span 监督链路有效，优化后目标值更接近 reference

### 备注

- 当前 smoke test 使用的是仓库示例语音，不是正式情绪数据集
- 这一步只证明链路和局部损失可用，不代表最终 local emotion control 已完成

## 2026-08-21 追加记录: baseline -> forced alignment -> local TTO pilot

### 两阶段数据流程

- pilot manifest: `tto_eval/esd_local_target/pilot_manifest.jsonl`，16 条。
- 每条使用 ESD Neutral 音频作为 prompt/音色，使用同 speaker 的 Happy/Sad/Surprise/Angry 音频作为目标情感参考。
- `gen_text` 与 prompt/target 音频文本无关，target word 由 manifest 明确指定。
- baseline 输出: `tto_eval/esd_local_target/baseline/`，16/16 成功。
- forced alignment: `torchaudio.pipelines.MMS_FA`，模型缓存于 `tto_eval/model_cache/torchaudio_mms_fa/model.pt`。
- aligned manifest: `tto_eval/esd_local_target/aligned_manifest.jsonl`。
- target span 全部从 baseline 音频对齐得到，例如 `final=[2.7303,3.0716]`、`quiet=[0.3208,0.7618]`、`remember=[0.9667,1.3493]`。

### 当前局部 TTO 配置

- `loss_mode=value`
- `vad_level=target`
- target context `0.5 sec`
- local latent context `0.25 sec`
- ODE steps `16`，seed `0`
- optimization steps `4,8,12`，每个时间步 `opt_steps=1`，`opt_lr=0.005`
- 未加入 speaker/ASR/emotion 辅助 loss
- TTO 输出: `tto_eval/esd_local_target/tto_value_target/`，16/16 成功。

### 自动指标

- target-span VAD MSE: `0.036519 -> 0.035675`，相对改善 `2.31%`。
- target-span VAD MSE 改善样本数: `10/16`。
- 全句 VAD MSE: `0.038144 -> 0.038085`，基本不变。
- WER: baseline 与 TTO 均为 `0.000000`。
- UTMOS: `3.994796 -> 3.960889`，下降 `0.033907`；7/16 条下降。
- RMS: `0.017909 -> 0.017738`；所有输出 finite。
- 按 target word：`final` 改善 `4.46%`，`quiet` 改善 `6.91%`，`remember` 下降 `2.29%`，`silver` 下降 `6.77%`。
- 按情感：Sad 改善 `5.00%`，Angry 改善 `4.31%`，Surprise 改善 `2.11%`，Happy 下降 `3.46%`。

### 当前判断

- baseline -> forced alignment -> local latent TTO 的工程链路已跑通。
- WER 没有下降，说明这一轮局部优化没有明显破坏内容。
- VAD 改善较弱且依赖 target word/情感类别；UTMOS 有轻微下降。
- 下一步最有信息量的局部消融是增加 `opt_steps`，先固定 target word=`final`、四种情感，避免同时改变 span、loss 或数据分布。

### 结果文件

- `logs/local_target_baseline_20260821.log`
- `logs/local_target_forced_align_20260821.log`
- `logs/local_target_tto_value_20260821.log`
- `logs/local_target_metrics_20260821.log`
- `logs/local_target_quality_20260821.log`
- `tto_eval/esd_local_target/metrics/metrics.csv`
- `tto_eval/esd_local_target/metrics/quality_metrics.csv`

### 第一轮局部消融：opt_steps 与 loss_mode

#### `opt_steps=5`

- 数据：4 条 `target_word=final`，四种情感；其余配置与主实验相同。
- 输出：`tto_eval/esd_local_target/tto_value_target_opt5/`。
- target-span value-MSE: `0.043652 -> 0.045439`，相对变化 `-4.09%`。
- 改善样本数: `0/4`。
- 观察：优化日志中的 inner loss 不单调，较大的 inner steps 在当前 `opt_lr=0.005` 下出现振荡/过优化迹象。

#### `loss_mode=embedding`, `opt_steps=1`

- 数据：同样 4 条 `target_word=final`，只改变 loss mode。
- 输出：`tto_eval/esd_local_target/tto_embedding_target/`。
- 外部 target-span value-MSE: `0.043652 -> 0.042112`，相对改善 `3.53%`。
- 改善样本数: `3/4`。
- WER: `0.000000 -> 0.000000`。
- UTMOS: `3.958841 -> 3.940001`，下降 `0.018840`；1/4 条下降。

#### 当前默认取舍

- 暂定默认：`loss_mode=value` + `opt_steps=1` + `opt_at=4,8,12` + `opt_lr=0.005`。
- 原因：主 16 条 pilot 上 value 已得到 `2.31%` target-span MSE 改善；4 条 `final` 配对中 value 的 `4.46%` 改善略高于 embedding 的 `3.53%`；而 `opt_steps=5` 明显恶化。
- 这不是最终结论：当前样本量和 target word 数量仍不足以决定 loss mode 的普适优劣。

### 当前未完成的评估

- 尚未进行人工听感/情感强度判断。
- 尚未进行 target word 局部情感强度的专门人评协议。
- 尚未系统消融 target context、opt_at、opt_lr、局部 mask 范围。


## 2026-08-21: emotional ref_wav correction and full rerun

### Problem and correction

- Previous pilot incorrectly used Neutral audio as the F5-TTS/DiT ref_wav.
- Corrected manifest semantics:
  - ref_wav/ref_text and prompt_wav/prompt_text: matching Happy/Sad/Surprise/Angry ESD utterance.
  - target_audio: same emotional utterance as ref_wav.
  - speaker_wav/speaker_text: matching Neutral utterance, reserved for optional speaker auxiliary loss.
- ESD ref_text is now looked up by full utterance id, because emotional transcripts can differ from Neutral.
- infer.tto now prioritizes ref_wav/ref_text over legacy prompt fields and loads speaker_wav only for the optional speaker loss.
- VAD CSV dimension labels were corrected from the wrong v/a/d mapping to the model order a/d/v. MSE values were unaffected.
- Old Neutral-conditioned outputs were preserved. Corrected outputs use tto_eval/esd_local_target_emoref/.

### Corrected pipeline outputs

- pilot manifest: tto_eval/esd_local_target_emoref/pilot_manifest.jsonl
- baseline: tto_eval/esd_local_target_emoref/baseline/ (16/16)
- aligned manifest: tto_eval/esd_local_target_emoref/aligned_manifest.jsonl (16/16)
- TTO: tto_eval/esd_local_target_emoref/tto_value_target/ (16/16)
- metrics: tto_eval/esd_local_target_emoref/metrics/
- logs:
  - logs/local_target_emoref_manifest_20260821.log
  - logs/local_target_emoref_baseline_20260821.log
  - logs/local_target_emoref_forced_align_20260821.log
  - logs/local_target_emoref_tto_value_20260821.log
  - logs/local_target_emoref_metrics_20260821.log
  - logs/local_target_emoref_quality_20260821.log

### Corrected automatic results

- target-span VAD MSE: 0.024031 -> 0.025277; relative improvement -5.19%; improved 8/16.
- global VAD MSE: 0.022285 -> 0.022095.
- mean WER/text error: 0.019345 -> 0.037202; one sample worsened.
- mean UTMOS: 3.573739 -> 3.578334; delta +0.004596, although 9/16 samples decreased.
- all outputs finite; baseline/TTO durations matched per sample.
- by emotion target-span MSE: Surprise +0.60% improvement; Angry -3.51%, Sad -6.94%, Happy -25.73%.
- by word target-span MSE: quiet +4.45%; final -8.98%, remember -5.83%, silver -13.90%.

### Interpretation

- Emotional DiT conditioning substantially changes the baseline and target-word alignment; old Neutral-based spans/results are not valid for the intended setup.
- With emotional ref_wav already conditioning F5-TTS, the previous TTO default does not improve mean target-span VAD MSE. The next tuning stage should reduce/update the local optimization schedule instead of treating the old +2.31% result as the baseline.
- Human local-emotion-strength evaluation remains pending.

## 2026-08-21: NRC VAD prototype, denser optimization, and x_t update cap

### Public VAD target basis

The official NRC VAD Lexicon defines V/A/D scores on [0, 1]. The exact anchor-word values used here are:

- happy: V=1.000, A=0.735, D=0.772
- sad: V=0.225, A=0.333, D=0.149
- angry: V=0.122, A=0.830, D=0.604
- surprise: V=0.875, A=0.875, D=0.562

The implementation converts these from V/A/D to the local model order A/D/V. alpha interpolates from the neutral midpoint [0.5, 0.5, 0.5] to the anchor. The current run uses alpha=1.0. These are public lexical anchors, not word-level speech ground truth or category means.

Sources:
- NRC VAD Lexicon: https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip
- NRC VAD paper: https://aclanthology.org/P18-1017.pdf

### Code changes

- Added src/f5_tts/infer/emotion_vad_target.py.
- Added --vad-target-mode {reference_audio,emotion_proto} and --emotion-vad-alpha.
- Added gradient normalization using the local x_t norm before Adam.
- Added post-Adam projection enforcing local update norm ||delta|| / ||x_t|| <= 0.05 per inner optimization step.
- TTO logs now include update_ratio.
- Changed evaluation to use the same emotion prototype target when manifest has vad_target_mode=emotion_proto.

### New run

- Manifest: tto_eval/esd_local_target_emoref/aligned_manifest_proto.jsonl
- TTO output: tto_eval/esd_local_target_emoref/tto_value_proto_moreopt/
- Settings: steps=16, opt_at=2,4,6,8,10,12,14, opt_steps=1, opt_lr=0.005, alpha=1.0, target context=0.5 sec, latent context=0.25 sec.
- All 16 outputs completed.
- Logged update ratios were approximately 0.0023-0.0055, below the 0.05 cap.

### Results

- Prototype target-span VAD MSE: 0.088509 -> 0.083385; relative improvement 5.79%; improved 13/16.
- Prototype global VAD MSE: 0.089547 -> 0.088058.
- WER/text error: 0.019345 -> 0.019345; text-worse samples 0/16.
- UTMOS: 3.573739 -> 3.559457; delta -0.014282; UTMOS-worse samples 10/16.
- All outputs finite.

Remaining uncertainty: NRC anchor values are not speech-level category means. Moderate intensity alpha=0.5 and ESD-derived speech prototype statistics should be compared next, followed by human local-span emotion ratings.

- Synchronized TTO defaults: opt_at=2,4,6,8,10,12,14; opt_steps=1; opt_lr=0.005. Added emotion_vad_target_test.py. Final validation: 10 unit tests passed, compile passed, git diff --check passed.

## 2026-08-21: reaching the 50% prototype VAD MSE target

### Ablation decision basis

Four final-word pilot ablations were tested before the full run:

- 7 points, lr=0.02, 1 inner step: 7.37% improvement.
- 7 points, lr=0.05, 1 inner step: 20.83% improvement.
- 15 points, lr=0.02, 1 inner step: 19.51% improvement.
- 7 points, lr=0.01, 3 inner steps: 3.61% improvement.
- 15 points, lr=0.05, 1 inner step: 48.30% improvement.
- 15 points, lr=0.05, 2 inner steps: 37.29% improvement.

The evidence favors larger per-point updates, dense ODE coverage, and one inner step. More inner steps caused oscillation. The 5% projection was retained throughout.

### Full 16-sample run

- Output: tto_eval/esd_local_target_emoref/tto_value_proto_dense15_lr05/
- Settings: emotion_proto alpha=1.0; opt_at=1..15; opt_steps=1; opt_lr=0.05; target context=0.5 sec; latent context=0.25 sec.
- Prototype target-span VAD MSE: 0.088509 -> 0.038480; relative improvement 56.52%; improved 15/16.
- Global VAD MSE: 0.089547 -> 0.066103.
- WER/text error: 0.019345 -> 0.019345; text-worse 0/16.
- UTMOS: 3.573739 -> 3.583127; delta +0.009389; UTMOS lower in 9/16 paired samples.
- RMS: 0.045810 -> 0.047754; peak: 0.342907 -> 0.368568.
- Update ratios: mean 0.045409, max 0.050000, 166/240 updates projected at the cap.

By emotion: Angry 58.97%, Sad 50.21%, Happy 70.06%, Surprise 42.99%.
By target word: final 63.33%, silver 58.91%, quiet 70.00%, remember 31.38%.

### Decision

Set the CLI candidate defaults to opt_at=1..15, opt_steps=1, opt_lr=0.05. Keep the previous 7-point/0.005 configuration as the comparison baseline. Human listening is still required, especially for Surprise and remember, and for checking local emotion spillover and peak-energy artifacts.
