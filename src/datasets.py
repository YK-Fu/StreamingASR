import json
import math
import os
import random
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from typing import List, Optional, Union, Literal
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import numpy as np
from nemo.utils import logging


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
        self.enabled = bool(cfg.get('enabled', True))

        noise_cfg = cfg.get('noise', {}) or {}
        self.noise_prob = float(noise_cfg.get('prob', 0.0))
        self.min_snr_db = float(noise_cfg.get('min_snr_db', 10.0))
        self.max_snr_db = float(noise_cfg.get('max_snr_db', 20.0))
        # Noise pool from jsonl manifest(s) with `{"audio_filepath": ...}` lines
        # (same format as the dataset manifests).
        self.noise_files = sorted(set(self._read_noise_manifests(noise_cfg.get('noise_manifests', []) or [])))

        self.effects = cfg.get('effects', {}) or {}

        if self.enabled:
            logging.info(
                f"AudioAugmentor enabled: {len(self.noise_files)} noise files "
                f"(prob={self.noise_prob}, snr=[{self.min_snr_db},{self.max_snr_db}] dB); "
                f"effects={[k for k, v in self.effects.items() if float((v or {}).get('prob', 0.0)) > 0]}"
            )

    @staticmethod
    def _read_noise_manifests(manifests) -> List[str]:
        """Read jsonl manifest(s) listing noise clips via the `audio_filepath` field
        (same format as the training manifests). Missing files are skipped."""
        files = []
        for fp in manifests:
            if not fp or not os.path.isfile(fp):
                logging.warning(f"AudioAugmentor: noise manifest not found, skipping: {fp}")
                continue
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    path = json.loads(line).get('audio_filepath')
                    if path and os.path.exists(path):
                        files.append(path)
        return files

    # ---- individual effects (operate on a 1D float waveform) ----

    def _volume(self, wav, c):
        db = _rand_uniform(float(c.get('min_db', 0.0)), float(c.get('max_db', 0.0)))
        return wav * (10.0 ** (db / 20.0))

    def _blur(self, wav, c):
        k = int(c.get('kernel_size', 10))
        if k <= 1:
            return wav
        kernel = torch.ones(1, 1, k, dtype=wav.dtype) / k
        out = F.conv1d(wav.view(1, 1, -1), kernel, padding=k // 2)
        return out.view(-1)[:wav.size(0)]

    def _echo(self, wav, c):
        delay = int(float(c.get('delay_ms', 20)) * self.sample_rate / 1000.0)
        decay = float(c.get('decay', 0.3))
        if delay <= 0 or delay >= wav.size(0):
            return wav
        out = wav.clone()
        out[delay:] = out[delay:] + decay * wav[:-delay]
        return out

    def _smoothing(self, wav, c):
        seg = int(float(c.get('segment_ms', 10)) * self.sample_rate / 1000.0)
        if seg <= 0:
            return wav
        n_full = wav.size(0) // seg
        if n_full == 0:
            return wav
        out = wav.clone()
        chunks = out[:n_full * seg].view(n_full, seg)
        rms = chunks.pow(2).mean(dim=1, keepdim=True).sqrt()
        # Normalize each chunk to ~unit RMS (full scale); silence guard avoids div-by-0.
        scale = torch.where(rms > 1e-8, 1.0 / (rms + 1e-8), torch.zeros_like(rms))
        out[:n_full * seg] = (chunks * scale).view(-1)
        return out * (10.0 ** (-20.0 / 20.0))  # overall -20 dB, matching reference

    def _pitch(self, wav, c):
        semitones = _rand_uniform(float(c.get('min_semitones', 0.0)), float(c.get('max_semitones', 0.0)))
        if abs(semitones) < 1e-3:
            return wav
        return AF.pitch_shift(wav.unsqueeze(0), self.sample_rate, n_steps=semitones).squeeze(0)

    def _add_noise(self, wav):
        if not self.noise_files:
            return wav
        path = self.noise_files[_rand_int(0, len(self.noise_files) - 1)]
        try:
            noise, sr = torchaudio.load(path)
        except Exception as e:  # noqa: BLE001 - a single bad noise file must not crash training
            logging.warning(f"AudioAugmentor: failed to load noise file {path}: {e}")
            return wav
        if sr != self.sample_rate:
            noise = torchaudio.transforms.Resample(sr, self.sample_rate)(noise)
        if noise.dim() > 1:
            noise = noise.mean(dim=0)
        noise = noise.reshape(-1).to(wav.dtype)

        L, n = wav.size(0), noise.size(0)
        if n == 0 or noise.pow(2).sum() < 1e-8:  # empty / silent noise -> no-op
            return wav
        snr = torch.tensor([_rand_uniform(self.min_snr_db, self.max_snr_db)], dtype=wav.dtype)

        if n >= L:
            # Noise longer: random-crop a speech-length window and mix over the whole clip.
            start = _rand_int(0, n - L)
            noise = noise[start:start + L]
            return AF.add_noise(wav.unsqueeze(0), noise.unsqueeze(0), snr).squeeze(0)
        else:
            # Noise shorter: random start in the speech; mix only over that span.
            s = _rand_int(0, L - n)
            mixed_seg = AF.add_noise(wav[s:s + n].unsqueeze(0), noise.unsqueeze(0), snr).squeeze(0)
            out = wav.clone()
            out[s:s + n] = mixed_seg
            return out

    def apply(self, wav):
        if not self.enabled:
            return wav

        cfg = self.effects
        if 'volume' in cfg and torch.rand(1).item() < float(cfg['volume'].get('prob', 0.0)):
            wav = self._volume(wav, cfg['volume'])
        if 'blur' in cfg and torch.rand(1).item() < float(cfg['blur'].get('prob', 0.0)):
            wav = self._blur(wav, cfg['blur'])
        if 'echo' in cfg and torch.rand(1).item() < float(cfg['echo'].get('prob', 0.0)):
            wav = self._echo(wav, cfg['echo'])
        if 'smoothing' in cfg and torch.rand(1).item() < float(cfg['smoothing'].get('prob', 0.0)):
            wav = self._smoothing(wav, cfg['smoothing'])
        if 'pitch' in cfg and torch.rand(1).item() < float(cfg['pitch'].get('prob', 0.0)):
            wav = self._pitch(wav, cfg['pitch'])
        if self.noise_files and torch.rand(1).item() < self.noise_prob:
            wav = self._add_noise(wav)

        # Anti-clipping: only renormalize if augmentation pushed the signal past full scale.
        peak = wav.abs().max()
        if peak > 1.0:
            wav = wav / peak
        return wav


def pad_list_of_tensors(tensors: List[torch.Tensor], pad_value: float = 0, max_length: Optional[int] = None) -> torch.Tensor:
    """
    Pad tensors to the longest one in the batch, or to a specified max_length.
    
    Args:
        tensors: List of 1D tensors to pad
        pad_value: Value to use for padding
        max_length: If specified, pad to this length instead of the longest in batch
    
    Returns:
        Padded tensor of shape [batch_size, padded_length]
    """
    batch_size = len(tensors)
    if max_length is None:
        max_length = max([t.size(0) for t in tensors])
    
    padded_tensors = torch.full((batch_size, max_length), pad_value, dtype=tensors[0].dtype, device=tensors[0].device)
    for i, tensor in enumerate(tensors):
        assert tensor.size(0) <= max_length, "Tensor length is greater than the max length"
        padded_tensors[i, :tensor.size(0)] = tensor
    return padded_tensors

class ASRDataset(Dataset):
    """
    ASR Dataset that reads JSONL/JSON manifest files with audio path, transcription, and optional context.
    
    Supports NeMo-style manifest format:
    {"audio_filepath": "/path/to/audio.wav", "text": "hello world", "duration": 1.5, "context": "previous sentence"}
    
    The 'context' field is optional. If provided, it will be prepended to the transcription.
    
    Args:
        manifest_filepath: Path to a single JSONL file
        tokenizer: Tokenizer with encode/decode methods
        sample_rate: Expected sample rate of audio files
        max_duration: Maximum audio duration in seconds (filter out longer samples)
        min_duration: Minimum audio duration in seconds (filter out shorter samples)
        audio_chunk_size: If specified, pad audio to this fixed length (in seconds)
    
    Returns (via collate_fn):
        context: [B, Wc + Wt] - combined context + current transcription token IDs
        target: [B, Wt] - current transcription token IDs
        attn_mask: [B, Wc + Wt] - attention mask (1 for real tokens, 0 for padding)
        target_start: [B] - start indices of current transcription (after context)
        target_end: [B] - end indices of transcription
        speech: [B, T] - raw waveform
    """
    
    def __init__(
        self, 
        manifest_filepath: str,
        tokenizer,
        sample_rate: int = 16000,
        language_mapping: dict[str, int] = None,
        language_drop_rate: float = 0.0,
        never_drop_language: List[str] = [],
        batch_size: int = 16,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        audio_chunk_size: Optional[float] = None,
        audio_chunk_step: Optional[float] = None,
        bucket_by: Literal['audio', 'text', None] = 'audio',
        drop_last: bool = False,
        text_bucket_size: int = None,
        augmentation=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        # On-the-fly audio augmentation (training only). Built only when an
        # `augmentation` block is configured (val/test omit it -> clean audio).
        self.augmentor = (
            AudioAugmentor(augmentation, sample_rate)
            if augmentation is not None and augmentation.get('enabled', True)
            else None
        )
        self.language_mapping = language_mapping
        self.language_drop_rate = language_drop_rate
        self.never_drop_language = set(never_drop_language)
        self.max_duration = max_duration if max_duration is not None else float('inf')
        self.min_duration = min_duration if min_duration is not None else 0
        self.audio_chunk_size = int(audio_chunk_size * sample_rate) if audio_chunk_size is not None else None
        self.audio_chunk_step = int(audio_chunk_step * sample_rate) if audio_chunk_step is not None else None
        self.bucket_by = bucket_by
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.batches = self._build_batches(manifest_filepath)
        
    def _build_batches(self, manifest_filepath: list[str]):
        lengths = []
        data = []
        batches = []
        total_time = 0
        filtered_time = 0   
        for fp in manifest_filepath:
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if not os.path.exists(item['audio_filepath']):
                            continue
                        if 'duration' not in item:
                            duration = torchaudio.load(item['audio_filepath'])[0].size(0) / self.sample_rate
                        else:
                            duration = item['duration']
                        if duration > self.max_duration or duration < self.min_duration:
                            filtered_time += duration
                            continue
                        total_time += duration
                        if self.bucket_by == 'text':
                            lengths.append(len(self.tokenizer.text_to_ids(item['text'])))
                        elif self.bucket_by == 'audio':
                            lengths.append(duration)
                        data.append(item)
            logging.info(f"{manifest_filepath} - Total time: {total_time:.2f}s, Filtered time: {filtered_time:.2f}s")
        if len(lengths) > 0:
            sorted_indices = np.argsort(lengths)
        else:
            sorted_indices = list(range(len(data)))
        for start_idx in range(0, len(sorted_indices), self.batch_size):
            batch = [data[i] for i in sorted_indices[start_idx:start_idx + self.batch_size]]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return [self.__get_one_sample__(item) for item in self.batches[idx]]

    def __get_one_sample__(self, item):
        audio_path = item['audio_filepath']
        transcription = item['text']
        context = item.get('context', '')

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        # Convert to mono if stereo
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        # Truncate to audio_chunk_size to guard against manifests whose stated duration
        # is slightly shorter than the actual file length after resampling.
        if self.audio_chunk_size is not None and waveform.size(0) > self.audio_chunk_size:
            waveform = waveform[:self.audio_chunk_size]
        # On-the-fly augmentation (noise / effects); no-op when not configured.
        if self.augmentor is not None:
            waveform = self.augmentor.apply(waveform)
        # Drop language with probability language_drop_rate
        language = item.get('language', '<|NO_LANGUAGE_ID|>')
        if torch.rand(1).item() < self.language_drop_rate and language not in self.never_drop_language:
            language = '<|NO_LANGUAGE_ID|>'
            
        context_tokens = [self.tokenizer.bos_id]
        context_tokens.append(self.tokenizer.token_to_id(language))
        if self.language_mapping is not None:
            language_id = self.language_mapping[language]

        # Tokenize context and transcription separately to track indices
        if context:    
            context_tokens = context_tokens + self.tokenizer.text_to_ids(context)
        
        transcription_tokens = self.tokenizer.text_to_ids(transcription)
        
        # Note that the input idices for decoder are 0 ~ n-2, and that for the llm target are 1 ~ n-1
        full_tokens = context_tokens + transcription_tokens

        # Calculate start/end indices of current transcription (excluding BOS, context, and EOS)
        target_start = len(context_tokens)
        target_end = len(full_tokens)
        
        return {
            'waveform': waveform,
            'context': torch.tensor(full_tokens, dtype=torch.long),
            'target': torch.tensor(transcription_tokens, dtype=torch.long),
            'target_start': target_start,
            'target_end': target_end,
            'language_id': language_id,
        }
    
    def collate_fn(self, batch):
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of samples from __getitem__
        
        Returns:
            Tuple of (context, target, attn_mask, target_starts, target_ends, waveforms, language_ids)
        """
        waveforms = [item['waveform'] for item in batch]
        context_list = [item['context'] for item in batch]
        target_list = [item['target'] for item in batch]
        target_starts = torch.tensor([item['target_start'] for item in batch], dtype=torch.long)
        target_ends = torch.tensor([item['target_end'] for item in batch], dtype=torch.long)
        language_ids = torch.tensor([item['language_id'] for item in batch], dtype=torch.long)
        # Pad to nearest audio_chunk_step boundary (e.g. 5s), capped at audio_chunk_size (e.g. 30s)
        if self.audio_chunk_step is not None:
            max_wave_len = max(w.size(0) for w in waveforms)
            effective_max = ((max_wave_len + self.audio_chunk_step - 1) // self.audio_chunk_step) * self.audio_chunk_step
            if self.audio_chunk_size is not None:
                effective_max = min(effective_max, self.audio_chunk_size)
        else:
            effective_max = self.audio_chunk_size
        waveforms = pad_list_of_tensors(waveforms, pad_value=0, max_length=effective_max)
        def _bucket_len(lengths):
            n = max(lengths)
            b = self.text_bucket_size
            return ((n + b - 1) // b) * b if b else None
        context = pad_list_of_tensors(
            context_list, pad_value=self.tokenizer.pad_id,
            max_length=_bucket_len([t.size(0) for t in context_list]))
        target = pad_list_of_tensors(
            target_list, pad_value=self.tokenizer.pad_id,
            max_length=_bucket_len([t.size(0) for t in target_list]))
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attn_mask = (context != self.tokenizer.pad_id).long()
        attn_mask[:, 0] = 1     # To prevent the first token to be masked
        # attn_mask[:, -1] = 0    # To prevent the last token to be masked (we do not predict the eos token, so we mask out the last token)

        return context, target, attn_mask, target_starts, target_ends, waveforms, language_ids

class ResumableDataloader(DataLoader):
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
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        start_idx = self.consumed_batches - self.num_samples * self.epoch
        for idx in indices[start_idx:]:
            yield idx

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'consumed_batches': self.consumed_batches
        }
    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.consumed_batches = state_dict['consumed_batches']

def get_asr_dataset(
    manifest_filepath: Union[str, List[str]],
    tokenizer,
    batch_size: int = 16,
    sample_rate: int = 16000,
    language_file: str = "",
    language_drop_rate: float = 0.0,
    never_drop_language: List[str] = [],
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
    audio_chunk_size: Optional[float] = None,
    audio_chunk_step: Optional[float] = None,
    bucket_by: Literal['audio', 'text', None] = 'audio',
    drop_last: bool = False,
    text_bucket_size = None,
    augmentation=None,
) -> Dataset:
    # Handle None case (no manifest configured)
    if manifest_filepath is None:
        return None
    
    # Handle single file case
    if isinstance(manifest_filepath, str):
        manifest_filepath = [manifest_filepath]
    
    # Handle empty list
    if len(manifest_filepath) == 0:
        return None
    if language_file:
        with open(language_file, 'r') as f:
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
        augmentation=augmentation,
    )
