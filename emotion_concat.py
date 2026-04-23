"""
从 RAVDESS 数据集中选取同一说话人、不同情感/强度的两段音频，拼接成一段"情感变化"的参考音频。

单条用法:
    python emotion_concat.py \
        --actor 03 \
        --emotion1 01 --intensity1 01 \
        --emotion2 05 --intensity2 02 \
        --output_dir /mnt/disk1/datasets/RAVDESS/output

批量用法（逗号分隔/或 all，各维度做笛卡尔积）:
    python emotion_concat.py \
        --actor 03,05,07 \
        --emotion1 01 --intensity1 01 \
        --emotion2 03,05,06 --intensity2 01,02 \
        --skip_existing

或者从 JSON 配置批量（每个 job 是 {actor, emotion1, intensity1, emotion2, intensity2}）:
    python emotion_concat.py --config jobs.json
"""

import argparse
import json
import os
import glob
import re
from itertools import product

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


def trim_silence(wav, threshold_db=-70):
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


def concat_audio(path1, path2, target_sr=24000, trim=True, threshold_db=-70):
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


def run_one(dataset_dir, output_dir, actor, emotion1, intensity1,
            emotion2, intensity2, skip_existing=False, verbose=True):
    """处理单个 concat job。返回 'ok' / 'skip' / 'missing'."""
    emo1_name = EMOTION_MAP.get(emotion1, emotion1)
    int1_name = INTENSITY_MAP.get(intensity1, intensity1)
    emo2_name = EMOTION_MAP.get(emotion2, emotion2)
    int2_name = INTENSITY_MAP.get(intensity2, intensity2)
    filename = f"actor{actor}_{emo1_name}-{int1_name}_to_{emo2_name}-{int2_name}.wav"
    output_path = os.path.join(output_dir, filename)

    if skip_existing and os.path.exists(output_path):
        if verbose:
            print(f"  跳过（已存在）: {output_path}")
        return "skip"

    path1 = find_audio(dataset_dir, actor, emotion1, intensity1)
    path2 = find_audio(dataset_dir, actor, emotion2, intensity2)

    if path1 is None or path2 is None:
        if verbose:
            miss = []
            if path1 is None:
                miss.append(f"emotion={emotion1}/intensity={intensity1}")
            if path2 is None:
                miss.append(f"emotion={emotion2}/intensity={intensity2}")
            print(f"  缺失 actor={actor}: {', '.join(miss)}")
        return "missing"

    if verbose:
        print(f"  片段1: {os.path.basename(path1)}  → {emo1_name}/{int1_name}")
        print(f"  片段2: {os.path.basename(path2)}  → {emo2_name}/{int2_name}")

    result, sr = concat_audio(path1, path2)
    duration = result.shape[1] / sr
    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(output_path, result, sr)
    if verbose:
        print(f"  已保存: {output_path}  ({duration:.2f}s)")
    return "ok"


def _expand(raw, *, all_values):
    """把 CLI 字符串展开成代码列表。支持 'all' / 逗号分隔 / 单值。"""
    if raw.strip().lower() == "all":
        return list(all_values)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _scan_actors(dataset_dir):
    """从 dataset_dir 下扫描 Actor_XX 目录，返回排序后的 actor 编号列表。"""
    actors = []
    for d in sorted(os.listdir(dataset_dir)):
        m = re.fullmatch(r"Actor_(\d+)", d)
        if m and os.path.isdir(os.path.join(dataset_dir, d)):
            actors.append(m.group(1))
    return actors


def _load_config(config_path):
    """加载 JSON 配置文件，返回 job dict 列表。"""
    with open(config_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{config_path} 应该是一个 job 列表（JSON array）")
    required = {"actor", "emotion1", "intensity1", "emotion2", "intensity2"}
    for i, job in enumerate(data):
        missing = required - job.keys()
        if missing:
            raise ValueError(f"job #{i} 缺少字段: {missing}")
    return data


def main():
    parser = argparse.ArgumentParser(description="拼接 RAVDESS 不同情感音频（支持批量）")
    parser.add_argument("--dataset_dir", type=str, default="/mnt/disk1/datasets/RAVDESS",
                        help="RAVDESS 数据集路径")
    parser.add_argument("--actor", type=str, default="03",
                        help="演员编号，支持逗号分隔或 all 扫描数据集，例如 03,05,07")
    parser.add_argument("--emotion1", type=str, default="01",
                        help="第一段情感编码，支持逗号分隔或 all (01-08)")
    parser.add_argument("--intensity1", type=str, default="01",
                        help="第一段强度编码，支持逗号分隔或 all (01=normal, 02=strong)")
    parser.add_argument("--emotion2", type=str, default="05",
                        help="第二段情感编码，支持逗号分隔或 all")
    parser.add_argument("--intensity2", type=str, default="02",
                        help="第二段强度编码，支持逗号分隔或 all")
    parser.add_argument("--output_dir", type=str, default="/mnt/disk1/datasets/RAVDESS/output",
                        help="输出目录")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 配置文件路径，指定后忽略上述 actor/emotion/intensity 维度参数")
    parser.add_argument("--skip_existing", action="store_true",
                        help="输出文件已存在时跳过")
    parser.add_argument("--quiet", action="store_true",
                        help="仅输出汇总信息")
    args = parser.parse_args()

    # 构建 job 列表
    if args.config:
        jobs = _load_config(args.config)
    else:
        actors = (_scan_actors(args.dataset_dir)
                  if args.actor.strip().lower() == "all"
                  else _expand(args.actor, all_values=()))
        emos1 = _expand(args.emotion1, all_values=EMOTION_MAP.keys())
        ints1 = _expand(args.intensity1, all_values=INTENSITY_MAP.keys())
        emos2 = _expand(args.emotion2, all_values=EMOTION_MAP.keys())
        ints2 = _expand(args.intensity2, all_values=INTENSITY_MAP.keys())
        jobs = [
            {"actor": a, "emotion1": e1, "intensity1": i1,
             "emotion2": e2, "intensity2": i2}
            for a, e1, i1, e2, i2 in product(actors, emos1, ints1, emos2, ints2)
        ]

    if not jobs:
        print("没有可处理的 job，检查参数")
        return

    total = len(jobs)
    print(f"共 {total} 个 job（output_dir={args.output_dir}）")
    counts = {"ok": 0, "skip": 0, "missing": 0}
    for idx, job in enumerate(jobs, 1):
        if not args.quiet:
            print(f"[{idx}/{total}] actor={job['actor']} "
                  f"{job['emotion1']}-{job['intensity1']} → "
                  f"{job['emotion2']}-{job['intensity2']}")
        status = run_one(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            verbose=not args.quiet,
            **job,
        )
        counts[status] += 1

    print(f"\n汇总: ok={counts['ok']}  skip={counts['skip']}  missing={counts['missing']}  total={total}")


if __name__ == "__main__":
    main()
