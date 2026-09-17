import bisect
import hashlib
import io
import json
import math
import os
import time
from array import array
from functools import lru_cache
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from nemo.utils import logging
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def _split_parquet_row_uri(audio_path: str):
    marker = ".parquet#row="
    marker_idx = audio_path.find(marker)
    if marker_idx < 0:
        return None
    parquet_path = audio_path[: marker_idx + len(".parquet")]
    row_text = audio_path[marker_idx + len(marker) :]
    try:
        row_idx = int(row_text)
    except ValueError as exc:
        raise RuntimeError(f"Invalid parquet row audio URI: {audio_path}") from exc
    if row_idx < 0:
        raise RuntimeError(f"Invalid negative parquet row audio URI: {audio_path}")
    return parquet_path, row_idx


@lru_cache(maxsize=128)
def _cached_parquet_file(parquet_path: str):
    import pyarrow.parquet as pq

    return pq.ParquetFile(parquet_path)


@lru_cache(maxsize=128)
def _cached_parquet_row_group_ends(parquet_path: str):
    parquet_file = _cached_parquet_file(parquet_path)
    row_group_ends = []
    end = 0
    for row_group_idx in range(parquet_file.num_row_groups):
        end += parquet_file.metadata.row_group(row_group_idx).num_rows
        row_group_ends.append(end)
    return tuple(row_group_ends)


@lru_cache(maxsize=1)
def _cached_parquet_audio_row_group(parquet_path: str, row_group_idx: int):
    """Keep one audio row group resident in each DataLoader worker."""
    return _cached_parquet_file(parquet_path).read_row_group(
        row_group_idx, columns=["audio"]
    )


def load_parquet_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    """Load embedded audio from a ``path.parquet#row=N`` URI."""
    parsed = _split_parquet_row_uri(audio_path)
    if parsed is None:
        raise ValueError(
            "audio_filepath must use the Parquet row URI format "
            f"'path.parquet#row=N', got: {audio_path!r}"
        )
    parquet_path, row_idx = parsed
    try:
        parquet_file = _cached_parquet_file(parquet_path)
        row_group_ends = _cached_parquet_row_group_ends(parquet_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open Parquet shard: {parquet_path}") from exc
    if row_idx >= parquet_file.metadata.num_rows:
        raise RuntimeError(
            f"Parquet row audio URI row out of range: {audio_path} "
            f"(rows={parquet_file.metadata.num_rows})"
        )
    row_group_idx = bisect.bisect_right(row_group_ends, row_idx)
    row_group_start = 0 if row_group_idx == 0 else row_group_ends[row_group_idx - 1]
    table = _cached_parquet_audio_row_group(parquet_path, row_group_idx)
    audio = table.column("audio")[row_idx - row_group_start].as_py()

    if isinstance(audio, dict):
        audio = audio.get("bytes")
    if not isinstance(audio, (bytes, bytearray, memoryview)):
        raise RuntimeError(
            "Parquet `audio` must contain embedded bytes (binary or struct bytes); "
            f"external paths are unsupported: {audio_path}"
        )
    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(bytes(audio)))
    except Exception as exc:
        raise RuntimeError(
            f"torchaudio failed to decode embedded audio: {audio_path}"
        ) from exc
    if waveform.numel() == 0 or waveform.size(-1) == 0:
        raise RuntimeError(f"Decoded zero audio frames: {audio_path}")
    return waveform, sample_rate


def _manifest_index_ready(paths) -> bool:
    return all(
        os.path.isfile(paths[name])
        for name in ("offsets", "durations", "storage_groups", "meta")
    )


def _build_manifest_index_cache(paths):
    os.makedirs(paths["cache_dir"], exist_ok=True)
    parquet_row_group_ends = {}
    offsets = array("Q")
    durations = array("f")
    storage_groups = array("Q")
    total_rows = 0
    empty_rows = 0
    invalid_json = 0
    skipped_manifest_error = 0
    skipped_missing_duration = 0
    total_time = 0.0

    logging.info(f"Building ASR manifest index cache for {paths['real_path']}")
    started = time.time()
    file_size = max(os.path.getsize(paths["real_path"]), 1)
    progress_interval = float(os.environ.get("ASR_INDEX_PROGRESS_INTERVAL", "30"))
    next_progress = started + progress_interval
    with open(paths["real_path"], "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                empty_rows += 1
                continue
            total_rows += 1
            try:
                item = json.loads(line)
            except Exception:
                invalid_json += 1
                continue
            if item.get("error") == "missing":
                skipped_manifest_error += 1
                continue
            if "duration" not in item:
                skipped_missing_duration += 1
                continue
            try:
                duration = float(item["duration"])
            except Exception:
                skipped_missing_duration += 1
                continue
            if not math.isfinite(duration):
                skipped_missing_duration += 1
                continue
            audio_path = str(item.get("audio_filepath", ""))
            parquet_row = _split_parquet_row_uri(audio_path)
            if parquet_row is None:
                raise ValueError(
                    "Every manifest audio_filepath must use 'path.parquet#row=N'; "
                    f"got {audio_path!r} in {paths['real_path']}"
                )
            parquet_path, parquet_row_idx = parquet_row
            real_parquet_path = os.path.realpath(parquet_path)
            row_group_ends = parquet_row_group_ends.get(real_parquet_path)
            if row_group_ends is None:
                if not os.path.isfile(real_parquet_path):
                    raise FileNotFoundError(
                        f"Parquet shard not found: {real_parquet_path}"
                    )
                row_group_ends = _cached_parquet_row_group_ends(real_parquet_path)
                if not row_group_ends:
                    raise RuntimeError(
                        f"Parquet file has no row groups: {real_parquet_path}"
                    )
                parquet_row_group_ends[real_parquet_path] = row_group_ends
            if parquet_row_idx >= row_group_ends[-1]:
                raise RuntimeError(
                    f"Parquet row URI out of range while indexing: {audio_path} "
                    f"(rows={row_group_ends[-1]})"
                )
            row_group_idx = bisect.bisect_right(row_group_ends, parquet_row_idx)
            group_key = f"parquet:{real_parquet_path}#rg={row_group_idx}"
            storage_group = (
                int.from_bytes(
                    hashlib.blake2b(group_key.encode("utf-8"), digest_size=8).digest(),
                    byteorder="little",
                )
                or 1
            )
            offsets.append(offset)
            durations.append(duration)
            storage_groups.append(storage_group)
            total_time += duration
            now = time.time()
            if progress_interval > 0 and now >= next_progress:
                pos = f.tell()
                elapsed = max(now - started, 1e-6)
                rate = pos / elapsed
                remaining = max(file_size - pos, 0)
                eta = remaining / rate if rate > 0 else float("inf")
                logging.info(
                    f"ASR manifest index progress {paths['real_path']}: "
                    f"{pos / file_size:.2%} rows={total_rows:,} indexed={len(offsets):,} "
                    f"rate={rate / (1024 * 1024):.1f} MiB/s eta={eta / 60:.1f} min"
                )
                next_progress = now + progress_interval

    tmp_offsets = f"{paths['base']}.tmp.{os.getpid()}.offsets.npy"
    tmp_durations = f"{paths['base']}.tmp.{os.getpid()}.durations.npy"
    tmp_storage_groups = f"{paths['base']}.tmp.{os.getpid()}.storage_groups.npy"
    tmp_meta = f"{paths['base']}.tmp.{os.getpid()}.meta.json"
    np.save(tmp_offsets, np.asarray(offsets, dtype=np.uint64))
    np.save(tmp_durations, np.asarray(durations, dtype=np.float32))
    np.save(tmp_storage_groups, np.asarray(storage_groups, dtype=np.uint64))
    meta = {
        "manifest_path": paths["real_path"],
        "rows": total_rows,
        "indexed_rows": len(offsets),
        "empty_rows": empty_rows,
        "invalid_json": invalid_json,
        "skipped_manifest_error": skipped_manifest_error,
        "skipped_missing_duration": skipped_missing_duration,
        "total_time": total_time,
        "elapsed_sec": time.time() - started,
        "exact_parquet_row_groups": True,
        "parquet_files_indexed": len(parquet_row_group_ends),
    }
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(tmp_offsets, paths["offsets"])
    os.replace(tmp_durations, paths["durations"])
    os.replace(tmp_storage_groups, paths["storage_groups"])
    os.replace(tmp_meta, paths["meta"])
    logging.info(
        f"Built ASR manifest index cache for {paths['real_path']}: "
        f"indexed_rows={len(offsets):,} elapsed={meta['elapsed_sec']:.1f}s"
    )


def _load_or_build_manifest_index(manifest_path: str):
    real_path = os.path.realpath(manifest_path)
    stat = os.stat(real_path)
    fingerprint = (
        f"{real_path}|{stat.st_size}|{stat.st_mtime_ns}|asr-index-v6-parquet-only"
    )
    cache_dir = os.environ.get("ASR_MANIFEST_CACHE_DIR") or os.path.join(
        os.path.dirname(real_path), ".asr_index_cache"
    )
    base = os.path.join(
        cache_dir,
        f"{os.path.basename(real_path)}.{hashlib.sha1(fingerprint.encode()).hexdigest()[:20]}",
    )
    paths = {
        "real_path": real_path,
        "cache_dir": cache_dir,
        "base": base,
        "offsets": f"{base}.offsets.npy",
        "durations": f"{base}.durations.npy",
        "storage_groups": f"{base}.storage_groups.npy",
        "meta": f"{base}.meta.json",
        "lock": f"{base}.lock",
    }
    if not _manifest_index_ready(paths):
        os.makedirs(paths["cache_dir"], exist_ok=True)
        lock_acquired = False
        while not lock_acquired:
            try:
                fd = os.open(paths["lock"], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"pid={os.getpid()} time={time.time()}\n")
                lock_acquired = True
            except FileExistsError:
                if _manifest_index_ready(paths):
                    break
                try:
                    lock_age = time.time() - os.path.getmtime(paths["lock"])
                except OSError:
                    lock_age = 0
                if lock_age > 6 * 3600:
                    logging.warning(
                        f"Removing stale ASR manifest index lock: {paths['lock']}"
                    )
                    try:
                        os.remove(paths["lock"])
                    except FileNotFoundError:
                        pass
                    continue
                logging.info(
                    f"Waiting for ASR manifest index cache lock: {paths['lock']}"
                )
                time.sleep(30)
        if lock_acquired:
            try:
                if not _manifest_index_ready(paths):
                    _build_manifest_index_cache(paths)
            finally:
                try:
                    os.remove(paths["lock"])
                except FileNotFoundError:
                    pass
    offsets = np.load(paths["offsets"], mmap_mode="r")
    durations = np.load(paths["durations"], mmap_mode="r")
    storage_groups = np.load(paths["storage_groups"], mmap_mode="r")
    with open(paths["meta"], encoding="utf-8") as f:
        meta = json.load(f)
    return offsets, durations, storage_groups, meta, paths


def _rand_uniform(low: float, high: float) -> float:
    """Uniform sample in [low, high] using torch RNG (seeded per dataloader worker)."""
    if high <= low:
        return low
    return low + (high - low) * torch.rand(1).item()


def _rand_int(low: int, high: int) -> int:
    """Inclusive random integer in [low, high] using torch RNG."""
    if high <= low:
        return low
    return int(torch.randint(low, high + 1, (1,)).item())


class AudioAugmentor:
    """
    On-the-fly waveform augmentation for training, implemented entirely with
    torch / torchaudio (no pydub / librosa).

    Applied per-sample inside the dataloader workers on the mono, un-padded
    waveform. Effects are each gated by their own probability and sample their
    parameters from a configured range. Order mirrors the reference pipeline:
        volume -> blur -> echo -> smoothing -> pitch -> additive noise (last)

    Additive noise draws a random file from the `noise_manifests` pool and mixes
    it at a target SNR (dB) via torchaudio.functional.add_noise, with length matching:
      * noise longer  than speech: random-crop a speech-length window, mix over all.
      * noise shorter than speech: random start in the speech, mix only that span.
    """

    def __init__(self, cfg, sample_rate: int):
        self.sample_rate = sample_rate
        self.enabled = bool(cfg.get("enabled", True))

        noise_cfg = cfg.get("noise", {}) or {}
        self.noise_prob = float(noise_cfg.get("prob", 0.0))
        self.min_snr_db = float(noise_cfg.get("min_snr_db", 10.0))
        self.max_snr_db = float(noise_cfg.get("max_snr_db", 20.0))
        # Noise pool from jsonl manifest(s) with `{"audio_filepath": ...}` lines
        # (same format as the dataset manifests).
        self.noise_files = sorted(
            set(self._read_noise_manifests(noise_cfg.get("noise_manifests", []) or []))
        )

        self.effects = cfg.get("effects", {}) or {}

        if self.enabled:
            logging.info(
                f"AudioAugmentor enabled: {len(self.noise_files)} noise files "
                f"(prob={self.noise_prob}, snr=[{self.min_snr_db},{self.max_snr_db}] dB); "
                f"effects={[k for k, v in self.effects.items() if float((v or {}).get('prob', 0.0)) > 0]}"
            )

    @staticmethod
    def _read_noise_manifests(manifests) -> list[str]:
        """Read Parquet row URIs from noise manifests."""
        files = []
        for fp in manifests:
            if not fp or not os.path.isfile(fp):
                logging.warning(
                    f"AudioAugmentor: noise manifest not found, skipping: {fp}"
                )
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    path = json.loads(line).get("audio_filepath")
                    parsed = _split_parquet_row_uri(path) if path else None
                    if parsed is None:
                        raise ValueError(
                            "Noise audio_filepath must use 'path.parquet#row=N'; "
                            f"got {path!r} in {fp}"
                        )
                    if not os.path.isfile(parsed[0]):
                        raise FileNotFoundError(
                            f"Noise Parquet shard not found: {parsed[0]}"
                        )
                    files.append(path)
        return files

    # ---- individual effects (operate on a 1D float waveform) ----

    def _volume(self, wav, c):
        db = _rand_uniform(float(c.get("min_db", 0.0)), float(c.get("max_db", 0.0)))
        return wav * (10.0 ** (db / 20.0))

    def _blur(self, wav, c):
        k = int(c.get("kernel_size", 10))
        if k <= 1:
            return wav
        kernel = torch.ones(1, 1, k, dtype=wav.dtype) / k
        out = F.conv1d(wav.view(1, 1, -1), kernel, padding=k // 2)
        return out.view(-1)[: wav.size(0)]

    def _echo(self, wav, c):
        delay = int(float(c.get("delay_ms", 20)) * self.sample_rate / 1000.0)
        decay = float(c.get("decay", 0.3))
        if delay <= 0 or delay >= wav.size(0):
            return wav
        out = wav.clone()
        out[delay:] = out[delay:] + decay * wav[:-delay]
        return out

    def _smoothing(self, wav, c):
        seg = int(float(c.get("segment_ms", 10)) * self.sample_rate / 1000.0)
        if seg <= 0:
            return wav
        n_full = wav.size(0) // seg
        if n_full == 0:
            return wav
        out = wav.clone()
        chunks = out[: n_full * seg].view(n_full, seg)
        rms = chunks.pow(2).mean(dim=1, keepdim=True).sqrt()
        # Normalize each chunk to ~unit RMS (full scale); silence guard avoids div-by-0.
        scale = torch.where(rms > 1e-8, 1.0 / (rms + 1e-8), torch.zeros_like(rms))
        out[: n_full * seg] = (chunks * scale).view(-1)
        return out * (10.0 ** (-20.0 / 20.0))  # overall -20 dB, matching reference

    def _pitch(self, wav, c):
        semitones = _rand_uniform(
            float(c.get("min_semitones", 0.0)), float(c.get("max_semitones", 0.0))
        )
        if abs(semitones) < 1e-3:
            return wav
        return AF.pitch_shift(
            wav.unsqueeze(0), self.sample_rate, n_steps=semitones
        ).squeeze(0)

    def _add_noise(self, wav):
        if not self.noise_files:
            return wav
        path = self.noise_files[_rand_int(0, len(self.noise_files) - 1)]
        noise, sr = load_parquet_audio(path)
        if sr != self.sample_rate:
            noise = torchaudio.transforms.Resample(sr, self.sample_rate)(noise)
        if noise.dim() > 1:
            noise = noise.mean(dim=0)
        noise = noise.reshape(-1).to(wav.dtype)

        L, n = wav.size(0), noise.size(0)
        if n == 0 or noise.pow(2).sum() < 1e-8:  # empty / silent noise -> no-op
            return wav
        snr = torch.tensor(
            [_rand_uniform(self.min_snr_db, self.max_snr_db)], dtype=wav.dtype
        )

        if n >= L:
            # Noise longer: random-crop a speech-length window and mix over the whole clip.
            start = _rand_int(0, n - L)
            noise = noise[start : start + L]
            return AF.add_noise(wav.unsqueeze(0), noise.unsqueeze(0), snr).squeeze(0)
        else:
            # Noise shorter: random start in the speech; mix only over that span.
            s = _rand_int(0, L - n)
            mixed_seg = AF.add_noise(
                wav[s : s + n].unsqueeze(0), noise.unsqueeze(0), snr
            ).squeeze(0)
            out = wav.clone()
            out[s : s + n] = mixed_seg
            return out

    def apply(self, wav):
        if not self.enabled:
            return wav

        cfg = self.effects
        if "volume" in cfg and torch.rand(1).item() < float(
            cfg["volume"].get("prob", 0.0)
        ):
            wav = self._volume(wav, cfg["volume"])
        if "blur" in cfg and torch.rand(1).item() < float(cfg["blur"].get("prob", 0.0)):
            wav = self._blur(wav, cfg["blur"])
        if "echo" in cfg and torch.rand(1).item() < float(cfg["echo"].get("prob", 0.0)):
            wav = self._echo(wav, cfg["echo"])
        if "smoothing" in cfg and torch.rand(1).item() < float(
            cfg["smoothing"].get("prob", 0.0)
        ):
            wav = self._smoothing(wav, cfg["smoothing"])
        if "pitch" in cfg and torch.rand(1).item() < float(
            cfg["pitch"].get("prob", 0.0)
        ):
            wav = self._pitch(wav, cfg["pitch"])
        if self.noise_files and torch.rand(1).item() < self.noise_prob:
            wav = self._add_noise(wav)

        # Anti-clipping: only renormalize if augmentation pushed the signal past full scale.
        peak = wav.abs().max()
        if peak > 1.0:
            wav = wav / peak
        return wav


def pad_list_of_tensors(
    tensors: list[torch.Tensor], pad_value: float = 0, max_length: int | None = None
) -> torch.Tensor:
    """Pad 1-D tensors to ``max_length`` or the longest input."""
    if max_length is None:
        max_length = max(t.size(0) for t in tensors)

    padded_tensors = torch.full(
        (len(tensors), max_length),
        pad_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )
    for i, tensor in enumerate(tensors):
        assert tensor.size(0) <= max_length, (
            "Tensor length is greater than the max length"
        )
        padded_tensors[i, : tensor.size(0)] = tensor
    return padded_tensors


class ASRDataset(Dataset):
    """Batched ASR dataset backed by indexed JSONL and embedded Parquet audio.

    Each manifest row must contain ``audio_filepath``, ``text``, and ``duration``.
    Audio paths use ``path.parquet#row=N`` and may include optional ``context`` and
    language metadata. The manifest index is always created or reused at startup.
    """

    def __init__(
        self,
        manifest_filepath: list[str],
        tokenizer,
        sample_rate: int = 16000,
        language_mapping: dict[str, int] | None = None,
        language_drop_rate: float = 0.0,
        never_drop_language: list[str] | None = None,
        batch_size: int = 16,
        max_duration: float | None = None,
        min_duration: float | None = None,
        audio_chunk_size: float | None = None,
        audio_chunk_step: float | None = None,
        bucket_by: Literal["audio", "text", None] = "audio",
        drop_last: bool = False,
        text_bucket_size: int | None = None,
        max_context_tokens: int | None = None,
        augmentation=None,
    ):
        super().__init__()
        if max_context_tokens is not None and max_context_tokens < 0:
            raise ValueError("max_context_tokens must be non-negative or None")
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        # On-the-fly audio augmentation (training only). Built only when an
        # `augmentation` block is configured (val/test omit it -> clean audio).
        self.augmentor = (
            AudioAugmentor(augmentation, sample_rate)
            if augmentation is not None and augmentation.get("enabled", True)
            else None
        )
        self.language_mapping = language_mapping
        self.language_drop_rate = language_drop_rate
        self.never_drop_language = set(never_drop_language or [])
        self.max_duration = max_duration if max_duration is not None else float("inf")
        self.min_duration = min_duration if min_duration is not None else 0
        self.audio_chunk_size = (
            int(audio_chunk_size * sample_rate)
            if audio_chunk_size is not None
            else None
        )
        self.audio_chunk_step = (
            int(audio_chunk_step * sample_rate)
            if audio_chunk_step is not None
            else None
        )
        self.bucket_by = bucket_by
        self.drop_last = drop_last
        self.batch_size = batch_size
        self._manifest_files = []
        self._file_ids = None
        self._offsets = None
        self._order = None
        self._num_items = 0
        self._num_batches = 0
        self._manifest_handles = {}
        self.text_bucket_size = text_bucket_size
        self.max_context_tokens = max_context_tokens
        self._build_batches(manifest_filepath)

    def _build_batches(self, manifest_filepath: list[str]):
        file_id_arrays = []
        offset_arrays = []
        duration_arrays = []
        storage_group_arrays = []
        total_time = 0.0
        filtered_time = 0.0
        indexed_rows = 0
        self._manifest_files = [os.path.realpath(fp) for fp in manifest_filepath]

        for file_idx, fp in enumerate(manifest_filepath):
            offsets, durations, storage_groups, meta, paths = (
                _load_or_build_manifest_index(fp)
            )
            if len(offsets) != len(durations) or len(offsets) != len(storage_groups):
                raise RuntimeError(f"Manifest index cache length mismatch for {fp}")
            if len(offsets) == 0:
                logging.warning(f"ASR manifest index has no rows: {fp}")
                continue
            mask = (durations >= self.min_duration) & (durations <= self.max_duration)
            kept = int(mask.sum())
            filtered = int(len(offsets) - kept)
            total_time += float(durations[mask].sum()) if kept else 0.0
            filtered_time += float(durations[~mask].sum()) if filtered else 0.0
            if kept == 0:
                logging.warning(f"ASR manifest index fully filtered by duration: {fp}")
                continue
            selected_offsets = offsets[mask]
            selected_durations = durations[mask]
            selected_storage_groups = storage_groups[mask]
            file_id_arrays.append(np.full(kept, file_idx, dtype=np.uint16))
            offset_arrays.append(np.asarray(selected_offsets, dtype=np.uint64))
            duration_arrays.append(np.asarray(selected_durations, dtype=np.float32))
            storage_group_arrays.append(
                np.asarray(selected_storage_groups, dtype=np.uint64)
            )
            indexed_rows += kept
            logging.info(
                f"{fp} - Indexed rows: {kept:,}/{len(offsets):,}, "
                f"Filtered rows: {filtered:,}, cache={paths['base']}"
            )
            if (
                meta.get("invalid_json")
                or meta.get("skipped_missing_duration")
                or meta.get("skipped_manifest_error")
            ):
                logging.info(
                    f"{fp} - Index skipped invalid_json={meta.get('invalid_json', 0)}, "
                    f"missing_duration={meta.get('skipped_missing_duration', 0)}, "
                    f"manifest_error_missing={meta.get('skipped_manifest_error', 0)}"
                )

        if indexed_rows == 0:
            raise RuntimeError(
                f"No valid rows after indexing manifests: {manifest_filepath}"
            )

        self._file_ids = np.concatenate(file_id_arrays)
        self._offsets = np.concatenate(offset_arrays)
        durations = np.concatenate(duration_arrays)
        storage_groups = np.concatenate(storage_group_arrays)
        if self.bucket_by == "audio":
            # Form complete batches within a Parquet row group, then combine
            # group remainders by duration to avoid badly padded tail batches.
            group_order = np.lexsort((durations, storage_groups))
            ordered_groups = storage_groups[group_order]
            boundaries = np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]) + 1
            group_starts = np.concatenate(([0], boundaries))
            group_ends = np.concatenate((boundaries, [len(group_order)]))
            full_parts = []
            remainder_parts = []
            for start, end in zip(group_starts, group_ends):
                group_indices = group_order[start:end]
                full_end = (len(group_indices) // self.batch_size) * self.batch_size
                full_parts.append(group_indices[:full_end])
                remainder_parts.append(group_indices[full_end:])

            remainders = np.concatenate(remainder_parts)
            remainders = remainders[np.argsort(durations[remainders], kind="stable")]
            self._order = np.concatenate([*full_parts, remainders]).astype(
                np.int64, copy=False
            )
            logging.info(
                f"Storage-local audio bucketing enabled for {len(group_starts):,} "
                f"row groups; globally sorted remainders={len(remainders):,}"
            )
        elif self.bucket_by == "text":
            logging.warning(
                "Manifest cache does not contain token lengths; preserving manifest order."
            )
            self._order = np.arange(len(durations), dtype=np.int64)
        else:
            self._order = np.arange(len(durations), dtype=np.int64)
        self._num_items = int(len(self._order))
        if self.drop_last:
            self._num_batches = self._num_items // self.batch_size
        else:
            self._num_batches = math.ceil(self._num_items / self.batch_size)
        logging.info(
            f"{manifest_filepath} - Indexed manifests: rows={self._num_items:,}, "
            f"batches={self._num_batches:,}, Total time: {total_time:.2f}s, "
            f"Filtered time: {filtered_time:.2f}s"
        )

    def __len__(self):
        return self._num_batches

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self._num_items)
        return [
            self._load_sample(self._load_manifest_item(pos))
            for pos in range(start, end)
        ]

    def _load_manifest_item(self, pos: int):
        ref_idx = int(self._order[pos])
        file_id = int(self._file_ids[ref_idx])
        offset = int(self._offsets[ref_idx])
        handle = self._manifest_handles.get(file_id)
        if handle is None:
            handle = open(self._manifest_files[file_id], "rb")
            self._manifest_handles[file_id] = handle
        handle.seek(offset)
        line = handle.readline()
        if not line:
            raise RuntimeError(
                f"Empty manifest line at file_id={file_id}, offset={offset}, "
                f"path={self._manifest_files[file_id]}"
            )
        return json.loads(line)

    def _load_sample(self, item):
        audio_path = item["audio_filepath"]
        transcription = item["text"]
        context = item.get("context", "")

        # Load audio
        waveform, sr = load_parquet_audio(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        # Convert to mono if stereo
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        # Truncate to audio_chunk_size to guard against manifests whose stated duration
        # is slightly shorter than the actual file length after resampling.
        if (
            self.audio_chunk_size is not None
            and waveform.size(0) > self.audio_chunk_size
        ):
            waveform = waveform[: self.audio_chunk_size]
        # On-the-fly augmentation (noise / effects); no-op when not configured.
        if self.augmentor is not None:
            waveform = self.augmentor.apply(waveform)
        # Drop language with probability language_drop_rate
        # Historical manifests use ``lang`` while newer rebuilt manifests use
        # ``language`` (some carry both).  Treat them as aliases, but reject an
        # actual disagreement instead of silently conditioning on the wrong ID.
        language_field = item.get("language")
        lang_field = item.get("lang")
        if (
            language_field is not None
            and lang_field is not None
            and language_field != lang_field
        ):
            raise ValueError(
                f"Conflicting language tags: language={language_field!r}, lang={lang_field!r}, "
                f"audio_filepath={audio_path!r}"
            )
        language = lang_field if lang_field is not None else language_field
        if language is None:
            language = "<|NO_LANGUAGE_ID|>"
        if (
            torch.rand(1).item() < self.language_drop_rate
            and language not in self.never_drop_language
        ):
            language = "<|NO_LANGUAGE_ID|>"

        context_tokens = [self.tokenizer.bos_id]
        context_tokens.append(self.tokenizer.token_to_id(language))
        if self.language_mapping is not None:
            if language not in self.language_mapping:
                raise ValueError(
                    f"Unknown language token {language!r} for audio_filepath={audio_path!r}"
                )
            language_id = self.language_mapping[language]

        # Tokenize context and transcription separately to track indices
        if context:
            manifest_context_tokens = self.tokenizer.text_to_ids(context)
            if self.max_context_tokens is not None:
                # Retain the most recent history. BOS and the language token are
                # kept separately above and therefore do not consume this limit.
                if self.max_context_tokens == 0:
                    manifest_context_tokens = []
                else:
                    manifest_context_tokens = manifest_context_tokens[
                        -self.max_context_tokens :
                    ]
            context_tokens = context_tokens + manifest_context_tokens

        transcription_tokens = self.tokenizer.text_to_ids(transcription)

        # Note that the input idices for decoder are 0 ~ n-2, and that for the llm target are 1 ~ n-1
        full_tokens = context_tokens + transcription_tokens

        # Calculate start/end indices of current transcription (excluding BOS, context, and EOS)
        target_start = len(context_tokens)
        target_end = len(full_tokens)

        return {
            "waveform": waveform,
            "context": torch.tensor(full_tokens, dtype=torch.long),
            "target": torch.tensor(transcription_tokens, dtype=torch.long),
            "target_start": target_start,
            "target_end": target_end,
            "language_id": language_id,
        }

    def collate_fn(self, batch):
        """
        Collate function for DataLoader.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Tuple of (context, target, attn_mask, target_starts, target_ends, waveforms, language_ids)
        """
        waveforms = [item["waveform"] for item in batch]
        context_list = [item["context"] for item in batch]
        target_list = [item["target"] for item in batch]
        target_starts = torch.tensor(
            [item["target_start"] for item in batch], dtype=torch.long
        )
        target_ends = torch.tensor(
            [item["target_end"] for item in batch], dtype=torch.long
        )
        language_ids = torch.tensor(
            [item["language_id"] for item in batch], dtype=torch.long
        )
        # Pad to nearest audio_chunk_step boundary (e.g. 5s), capped at audio_chunk_size (e.g. 30s)
        if self.audio_chunk_step is not None:
            max_wave_len = max(w.size(0) for w in waveforms)
            effective_max = (
                (max_wave_len + self.audio_chunk_step - 1) // self.audio_chunk_step
            ) * self.audio_chunk_step
            if self.audio_chunk_size is not None:
                effective_max = min(effective_max, self.audio_chunk_size)
        else:
            effective_max = self.audio_chunk_size
        waveforms = pad_list_of_tensors(
            waveforms, pad_value=0, max_length=effective_max
        )

        def _bucket_len(lengths):
            n = max(lengths)
            b = self.text_bucket_size
            return ((n + b - 1) // b) * b if b else None

        context = pad_list_of_tensors(
            context_list,
            pad_value=self.tokenizer.pad_id,
            max_length=_bucket_len([t.size(0) for t in context_list]),
        )
        target = pad_list_of_tensors(
            target_list,
            pad_value=self.tokenizer.pad_id,
            max_length=_bucket_len([t.size(0) for t in target_list]),
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        attn_mask = (context != self.tokenizer.pad_id).long()
        attn_mask[:, 0] = 1  # To prevent the first token to be masked
        # attn_mask[:, -1] = 0    # To prevent the last token to be masked (we do not predict the eos token, so we mask out the last token)

        return (
            context,
            target,
            attn_mask,
            target_starts,
            target_ends,
            waveforms,
            language_ids,
        )


class ResumableDataloader(DataLoader):
    def __iter__(self):
        # Count consumption-side (per batch actually pulled by the training loop), NOT in
        # the sampler's __iter__: the DataLoader drains the sampler's index stream ahead of
        # training to fill the worker prefetch queue (num_workers * prefetch_factor), so a
        # sampler-side counter over-counts by the prefetch depth and resume would skip
        # batches. Here the generator only advances when the consumer pulls a batch, so
        # `consumed_batches` tracks true progress (off by <=1 vs Lightning's 1-batch
        # fetcher). This is the authoritative count read back in on_save_checkpoint.
        for batch in super().__iter__():
            if hasattr(self.sampler, "consumed_batches"):
                self.sampler.consumed_batches += 1
            yield batch

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state_dict):
        self.sampler.load_state_dict(state_dict)


class ResumableSampler(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consumed_batches = 0

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        # Resume offsets are meaningful for shuffled training epochs only.
        # Validation/test loaders use shuffle=False and may be iterated multiple
        # times within one epoch; they must restart from the beginning each time.
        start_idx = (
            self.consumed_batches - self.num_samples * self.epoch if self.shuffle else 0
        )
        for idx in indices[start_idx:]:
            yield idx

    def state_dict(self):
        return {"epoch": self.epoch, "consumed_batches": self.consumed_batches}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.consumed_batches = state_dict["consumed_batches"]


def get_asr_dataset(
    manifest_filepath: str | list[str] | None,
    tokenizer,
    batch_size: int = 16,
    sample_rate: int = 16000,
    language_file: str = "",
    language_drop_rate: float = 0.0,
    never_drop_language: list[str] | None = None,
    max_duration: float | None = None,
    min_duration: float | None = None,
    audio_chunk_size: float | None = None,
    audio_chunk_step: float | None = None,
    bucket_by: Literal["audio", "text", None] = "audio",
    drop_last: bool = False,
    text_bucket_size=None,
    max_context_tokens: int | None = None,
    augmentation=None,
) -> ASRDataset | None:
    if not manifest_filepath:
        return None

    if isinstance(manifest_filepath, str):
        manifest_filepath = [manifest_filepath]
    if language_file:
        with open(language_file, "r", encoding="utf-8") as f:
            language_mapping = {language.strip(): i for i, language in enumerate(f)}
    else:
        language_mapping = None
    return ASRDataset(
        manifest_filepath=manifest_filepath,
        tokenizer=tokenizer,
        sample_rate=sample_rate,
        language_mapping=language_mapping,
        language_drop_rate=language_drop_rate,
        never_drop_language=never_drop_language,
        batch_size=batch_size,
        max_duration=max_duration,
        min_duration=min_duration,
        audio_chunk_size=audio_chunk_size,
        audio_chunk_step=audio_chunk_step,
        bucket_by=bucket_by,
        drop_last=drop_last,
        text_bucket_size=text_bucket_size,
        max_context_tokens=max_context_tokens,
        augmentation=augmentation,
    )
