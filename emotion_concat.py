"""
从 RAVDESS 数据集中选取同一说话人、不同情感/强度的两段音频，拼接成一段"情感变化"的参考音频。

用法示例:
    python emotion_concat.py \
        --actor 03 \
        --emotion1 01 --intensity1 01 \
        --emotion2 05 --intensity2 02 \
        --silence_ms 300 \
        --output_dir /mnt/disk1/datasets/RAVDESS/output
"""

import argparse
import os
import glob

import torch
import torchaudio

# RAVDESS 编码映射
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}


def find_audio(dataset_dir, actor, emotion, intensity, statement="01", repetition="01"):
    """
    根据 RAVDESS 命名规则查找音频文件。
    格式: {modality}-{vocal_channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
    仅音频(03) + 语音(01)
    """
    pattern = f"03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav"
    path = os.path.join(dataset_dir, f"Actor_{actor}", pattern)
    if os.path.exists(path):
        return path

    # 如果指定的 statement/repetition 找不到，尝试其他组合
    search = os.path.join(dataset_dir, f"Actor_{actor}", f"03-01-{emotion}-{intensity}-*-*-{actor}.wav")
    candidates = sorted(glob.glob(search))
    if candidates:
        return candidates[0]

    return None


def trim_silence(wav, threshold_db=-50):
    """
    裁掉音频首尾的静音部分。
    threshold_db: 低于此分贝的视为静音，越小越宽松（保留更多）。
    """
    # 转为能量(dB)
    energy = 20 * torch.log10(wav.abs().clamp(min=1e-10))
    mask = (energy > threshold_db).squeeze(0)

    nonzero = torch.nonzero(mask)
    if len(nonzero) == 0:
        return wav
    start = nonzero[0].item()
    end = nonzero[-1].item() + 1
    return wav[:, start:end]


def concat_audio(path1, path2, target_sr=24000, trim=True, threshold_db=-50):
    """
    加载两段音频，统一采样率，裁掉首尾静音后拼接返回。
    """
    wav1, sr1 = torchaudio.load(path1)
    wav2, sr2 = torchaudio.load(path2)

    # 转单声道
    if wav1.shape[0] > 1:
        wav1 = wav1.mean(dim=0, keepdim=True)
    if wav2.shape[0] > 1:
        wav2 = wav2.mean(dim=0, keepdim=True)

    # 重采样
    if sr1 != target_sr:
        wav1 = torchaudio.transforms.Resample(sr1, target_sr)(wav1)
    if sr2 != target_sr:
        wav2 = torchaudio.transforms.Resample(sr2, target_sr)(wav2)

    # 裁掉首尾静音
    if trim:
        wav1 = trim_silence(wav1, threshold_db)
        wav2 = trim_silence(wav2, threshold_db)

    # 拼接
    result = torch.cat([wav1, wav2], dim=1)
    return result, target_sr


def main():
    parser = argparse.ArgumentParser(description="拼接 RAVDESS 不同情感音频")
    parser.add_argument("--dataset_dir", type=str, default="/mnt/disk1/datasets/RAVDESS",
                        help="RAVDESS 数据集路径")
    parser.add_argument("--actor", type=str, default="03",
                        help="演员编号，如 03")
    parser.add_argument("--emotion1", type=str, default="01",
                        help="第一段情感编码 (01-08)")
    parser.add_argument("--intensity1", type=str, default="01",
                        help="第一段强度编码 (01=normal, 02=strong)")
    parser.add_argument("--emotion2", type=str, default="05",
                        help="第二段情感编码 (01-08)")
    parser.add_argument("--intensity2", type=str, default="02",
                        help="第二段强度编码 (01=normal, 02=strong)")
    parser.add_argument("--output_dir", type=str, default="/mnt/disk1/datasets/RAVDESS/output",
                        help="输出目录")
    args = parser.parse_args()

    # 查找两段音频
    path1 = find_audio(args.dataset_dir, args.actor, args.emotion1, args.intensity1)
    path2 = find_audio(args.dataset_dir, args.actor, args.emotion2, args.intensity2)

    if path1 is None:
        print(f"错误: 找不到音频 — actor={args.actor}, emotion={args.emotion1}, intensity={args.intensity1}")
        return
    if path2 is None:
        print(f"错误: 找不到音频 — actor={args.actor}, emotion={args.emotion2}, intensity={args.intensity2}")
        return

    emo1_name = EMOTION_MAP.get(args.emotion1, args.emotion1)
    int1_name = INTENSITY_MAP.get(args.intensity1, args.intensity1)
    emo2_name = EMOTION_MAP.get(args.emotion2, args.emotion2)
    int2_name = INTENSITY_MAP.get(args.intensity2, args.intensity2)

    print(f"片段1: {path1}")
    print(f"  → {emo1_name} / {int1_name}")
    print(f"片段2: {path2}")
    print(f"  → {emo2_name} / {int2_name}")

    # 拼接
    result, sr = concat_audio(path1, path2)
    duration = result.shape[1] / sr
    print(f"拼接完成: {duration:.2f}s (采样率 {sr}Hz)")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"actor{args.actor}_{emo1_name}-{int1_name}_to_{emo2_name}-{int2_name}.wav"
    output_path = os.path.join(args.output_dir, filename)
    torchaudio.save(output_path, result, sr)
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
