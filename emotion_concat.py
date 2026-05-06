"""
从情感语音数据集中选取同一说话人、不同情感的两段音频，拼接成一段"情感变化"的参考音频。
支持 RAVDESS 与 ESD 两个数据集。

=========================
RAVDESS 用法 (默认)
=========================
单条:
    python emotion_concat.py \
        --actor 03 \
        --emotion1 01 --intensity1 01 \
        --emotion2 05 --intensity2 02 \
        --output_dir /mnt/disk1/datasets/RAVDESS/output

批量 (笛卡尔积):
    python emotion_concat.py \
        --actor 03,05,07 \
        --emotion1 01 --intensity1 01 \
        --emotion2 03,05,06 --intensity2 01,02 \
        --skip_existing

=========================
ESD 用法
=========================
ESD 每说话人每情感各 350 句不同文本，不存在 intensity。

单条 (默认 idx=0 取每情感目录的第一句):
    python emotion_concat.py --dataset esd \
        --speaker 0011 \
        --emotion1 Happy --emotion2 Sad \
        --output_dir /mnt/disk1/datasets/ESD/output

按 utt id 精确指定:
    python emotion_concat.py --dataset esd \
        --speaker 0011 \
        --emotion1 Happy --utt1 0011_000704 \
        --emotion2 Sad   --utt2 0011_001045

批量扫描全部 20 个英/中说话人 × 全情感对:
    python emotion_concat.py --dataset esd \
        --speaker all --emotion1 all --emotion2 all --skip_existing

=========================
JSON 配置批量
=========================
RAVDESS: 每个 job = {actor, emotion1, intensity1, emotion2, intensity2}
ESD:     每个 job = {speaker, emotion1, emotion2, [idx1], [idx2], [utt1], [utt2]}

    python emotion_concat.py --dataset esd --config jobs.json
"""

import argparse
import json
import os
import glob
import re
from itertools import product

import torch
import torchaudio

# ============================================================
# RAVDESS 编码映射
# ============================================================
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

# ============================================================
# ESD 常量
# ============================================================
ESD_EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
ESD_DEFAULT_DIR = "/mnt/disk1/datasets/ESD/Emotion Speech Dataset"


def _normalize_esd_emotion(name):
    """大小写不敏感地把用户输入归一化到 ESD 标准情感名。"""
    norm = name.strip().capitalize()
    if norm not in ESD_EMOTIONS:
        raise ValueError(f"ESD emotion 必须属于 {ESD_EMOTIONS}, 收到 {name!r}")
    return norm


# ============================================================
# 文件查找
# ============================================================
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


def find_audio_esd(dataset_dir, speaker, emotion, idx=0, utt_id=None):
    """
    根据 ESD 目录结构查找音频。
    路径: <dataset_dir>/<speaker>/<Emotion>/<speaker>_<6位数字>.wav
    优先用 utt_id（精确文件名，不含扩展名），否则按情感目录内排序后的索引 idx 选取。
    """
    emo = _normalize_esd_emotion(emotion)
    emo_dir = os.path.join(dataset_dir, speaker, emo)
    if not os.path.isdir(emo_dir):
        return None

    if utt_id:
        candidate = os.path.join(emo_dir, f"{utt_id}.wav")
        return candidate if os.path.exists(candidate) else None

    wavs = sorted(glob.glob(os.path.join(emo_dir, "*.wav")))
    if not wavs or idx < 0 or idx >= len(wavs):
        return None
    return wavs[idx]


# ============================================================
# 通用音频处理（数据集无关）
# ============================================================
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


def _save_concat(path1, path2, output_path, label1, label2, skip_existing, verbose):
    """共享的"检查 → 加载 → 拼接 → 保存"流程。返回 'ok' / 'skip' / 'missing'。"""
    if skip_existing and os.path.exists(output_path):
        if verbose:
            print(f"  跳过（已存在）: {output_path}")
        return "skip"

    if path1 is None or path2 is None:
        if verbose:
            miss = []
            if path1 is None:
                miss.append(f"片段1={label1}")
            if path2 is None:
                miss.append(f"片段2={label2}")
            print(f"  缺失: {', '.join(miss)}")
        return "missing"

    if verbose:
        print(f"  片段1: {os.path.basename(path1)}  → {label1}")
        print(f"  片段2: {os.path.basename(path2)}  → {label2}")

    result, sr = concat_audio(path1, path2)
    duration = result.shape[1] / sr
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torchaudio.save(output_path, result, sr)
    if verbose:
        print(f"  已保存: {output_path}  ({duration:.2f}s)")
    return "ok"


# ============================================================
# 单 job 执行：RAVDESS
# ============================================================
def run_one(dataset_dir, output_dir, actor, emotion1, intensity1,
            emotion2, intensity2, skip_existing=False, verbose=True):
    """处理单个 RAVDESS concat job。"""
    emo1_name = EMOTION_MAP.get(emotion1, emotion1)
    int1_name = INTENSITY_MAP.get(intensity1, intensity1)
    emo2_name = EMOTION_MAP.get(emotion2, emotion2)
    int2_name = INTENSITY_MAP.get(intensity2, intensity2)
    filename = f"actor{actor}_{emo1_name}-{int1_name}_to_{emo2_name}-{int2_name}.wav"
    output_path = os.path.join(output_dir, filename)

    path1 = find_audio(dataset_dir, actor, emotion1, intensity1)
    path2 = find_audio(dataset_dir, actor, emotion2, intensity2)

    return _save_concat(
        path1, path2, output_path,
        label1=f"{emo1_name}/{int1_name}",
        label2=f"{emo2_name}/{int2_name}",
        skip_existing=skip_existing, verbose=verbose,
    )


# ============================================================
# 单 job 执行：ESD
# ============================================================
def run_one_esd(dataset_dir, output_dir, speaker, emotion1, emotion2,
                idx1=0, idx2=0, utt1=None, utt2=None,
                skip_existing=False, verbose=True):
    """处理单个 ESD concat job。"""
    emo1 = _normalize_esd_emotion(emotion1)
    emo2 = _normalize_esd_emotion(emotion2)

    # 输出文件名：utt 优先（取后 6 位数字作短标识），否则用 idx
    def _suffix(utt, idx):
        if utt:
            m = re.search(r"(\d{6})$", utt)
            return m.group(1) if m else utt
        return f"i{idx}"

    s1, s2 = _suffix(utt1, idx1), _suffix(utt2, idx2)
    filename = f"spk{speaker}_{emo1.lower()}-{s1}_to_{emo2.lower()}-{s2}.wav"
    output_path = os.path.join(output_dir, filename)

    path1 = find_audio_esd(dataset_dir, speaker, emo1, idx=idx1, utt_id=utt1)
    path2 = find_audio_esd(dataset_dir, speaker, emo2, idx=idx2, utt_id=utt2)

    return _save_concat(
        path1, path2, output_path,
        label1=f"{emo1}/{utt1 or f'idx{idx1}'}",
        label2=f"{emo2}/{utt2 or f'idx{idx2}'}",
        skip_existing=skip_existing, verbose=verbose,
    )


# ============================================================
# CLI 工具
# ============================================================
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


def _scan_speakers_esd(dataset_dir):
    """扫描 ESD 顶层 4 位数字说话人目录。"""
    speakers = []
    for d in sorted(os.listdir(dataset_dir)):
        if re.fullmatch(r"\d{4}", d) and os.path.isdir(os.path.join(dataset_dir, d)):
            speakers.append(d)
    return speakers


def _load_config(config_path, dataset_type):
    """加载 JSON 配置文件，按 dataset_type 校验必需字段。"""
    with open(config_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{config_path} 应该是一个 job 列表（JSON array）")
    if dataset_type == "ravdess":
        required = {"actor", "emotion1", "intensity1", "emotion2", "intensity2"}
    else:  # esd
        required = {"speaker", "emotion1", "emotion2"}
    for i, job in enumerate(data):
        missing = required - job.keys()
        if missing:
            raise ValueError(f"job #{i} ({dataset_type}) 缺少字段: {missing}")
    return data


def main():
    parser = argparse.ArgumentParser(description="拼接同说话人不同情感音频（RAVDESS / ESD）")
    parser.add_argument("--dataset", choices=["ravdess", "esd"], default="ravdess",
                        help="数据集类型，默认 ravdess")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="数据集根目录；不指定时按 --dataset 选默认路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录；不指定时按 --dataset 选默认路径")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 配置文件路径，指定后忽略下方各维度参数")
    parser.add_argument("--skip_existing", action="store_true",
                        help="输出文件已存在时跳过")
    parser.add_argument("--quiet", action="store_true",
                        help="仅输出汇总信息")

    # ----- RAVDESS 维度 -----
    parser.add_argument("--actor", type=str, default="03",
                        help="[ravdess] 演员编号，逗号分隔或 all 扫描")
    parser.add_argument("--intensity1", type=str, default="01",
                        help="[ravdess] 第一段强度 (01=normal, 02=strong)，逗号分隔或 all")
    parser.add_argument("--intensity2", type=str, default="02",
                        help="[ravdess] 第二段强度，逗号分隔或 all")

    # ----- ESD 维度 -----
    parser.add_argument("--speaker", type=str, default="0011",
                        help="[esd] 说话人编号 (0001-0010 中文 / 0011-0020 英文)，逗号分隔或 all")
    parser.add_argument("--idx1", type=str, default="0",
                        help="[esd] 第一段 wav 在情感目录内的索引（0 起），逗号分隔或 all (0..349)")
    parser.add_argument("--idx2", type=str, default="0",
                        help="[esd] 第二段 wav 索引，逗号分隔或 all")
    parser.add_argument("--utt1", type=str, default=None,
                        help="[esd] 第一段精确 utt id (如 0011_000704)，指定后覆盖 idx1")
    parser.add_argument("--utt2", type=str, default=None,
                        help="[esd] 第二段精确 utt id，指定后覆盖 idx2")

    # ----- 共用 -----
    parser.add_argument("--emotion1", type=str, default=None,
                        help="第一段情感（ravdess 用 01-08 编码 / esd 用 Happy 等名称），逗号分隔或 all")
    parser.add_argument("--emotion2", type=str, default=None,
                        help="第二段情感，逗号分隔或 all")
    args = parser.parse_args()

    # 默认路径与情感按 dataset 切换
    if args.dataset == "ravdess":
        dataset_dir = args.dataset_dir or "/mnt/disk1/datasets/RAVDESS"
        output_dir = args.output_dir or "/mnt/disk1/datasets/RAVDESS/output"
        emotion1 = args.emotion1 or "01"
        emotion2 = args.emotion2 or "05"
    else:
        dataset_dir = args.dataset_dir or ESD_DEFAULT_DIR
        output_dir = args.output_dir or "/mnt/disk1/datasets/ESD/output"
        emotion1 = args.emotion1 or "Happy"
        emotion2 = args.emotion2 or "Sad"

    # 构建 job 列表
    if args.config:
        jobs = _load_config(args.config, args.dataset)
    elif args.dataset == "ravdess":
        actors = (_scan_actors(dataset_dir)
                  if args.actor.strip().lower() == "all"
                  else _expand(args.actor, all_values=()))
        emos1 = _expand(emotion1, all_values=EMOTION_MAP.keys())
        ints1 = _expand(args.intensity1, all_values=INTENSITY_MAP.keys())
        emos2 = _expand(emotion2, all_values=EMOTION_MAP.keys())
        ints2 = _expand(args.intensity2, all_values=INTENSITY_MAP.keys())
        jobs = [
            {"actor": a, "emotion1": e1, "intensity1": i1,
             "emotion2": e2, "intensity2": i2}
            for a, e1, i1, e2, i2 in product(actors, emos1, ints1, emos2, ints2)
        ]
    else:  # esd
        speakers = (_scan_speakers_esd(dataset_dir)
                    if args.speaker.strip().lower() == "all"
                    else _expand(args.speaker, all_values=()))
        emos1 = _expand(emotion1, all_values=ESD_EMOTIONS)
        emos2 = _expand(emotion2, all_values=ESD_EMOTIONS)
        # utt 指定则只跑一组；否则按 idx 笛卡尔积
        if args.utt1 or args.utt2:
            jobs = [
                {"speaker": s, "emotion1": e1, "emotion2": e2,
                 "utt1": args.utt1, "utt2": args.utt2}
                for s, e1, e2 in product(speakers, emos1, emos2)
            ]
        else:
            all_idxs = [str(i) for i in range(350)]
            idxs1 = [int(x) for x in _expand(args.idx1, all_values=all_idxs)]
            idxs2 = [int(x) for x in _expand(args.idx2, all_values=all_idxs)]
            jobs = [
                {"speaker": s, "emotion1": e1, "emotion2": e2,
                 "idx1": i1, "idx2": i2}
                for s, e1, e2, i1, i2 in product(speakers, emos1, emos2, idxs1, idxs2)
            ]

    if not jobs:
        print("没有可处理的 job，检查参数")
        return

    total = len(jobs)
    print(f"[{args.dataset}] 共 {total} 个 job（output_dir={output_dir}）")
    counts = {"ok": 0, "skip": 0, "missing": 0}
    for idx, job in enumerate(jobs, 1):
        if not args.quiet:
            if args.dataset == "ravdess":
                desc = (f"actor={job['actor']} "
                        f"{job['emotion1']}-{job['intensity1']} → "
                        f"{job['emotion2']}-{job['intensity2']}")
            else:
                tail = (f"utt={job.get('utt1')}|{job.get('utt2')}"
                        if job.get("utt1") or job.get("utt2")
                        else f"idx={job.get('idx1', 0)}|{job.get('idx2', 0)}")
                desc = (f"spk={job['speaker']} "
                        f"{job['emotion1']} → {job['emotion2']}  {tail}")
            print(f"[{idx}/{total}] {desc}")

        runner = run_one if args.dataset == "ravdess" else run_one_esd
        status = runner(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            skip_existing=args.skip_existing,
            verbose=not args.quiet,
            **job,
        )
        counts[status] += 1

    print(f"\n汇总: ok={counts['ok']}  skip={counts['skip']}  missing={counts['missing']}  total={total}")


if __name__ == "__main__":
    main()
