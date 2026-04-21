import json
import math
import random
import torch
import torchaudio
from typing import List, Optional, Union, Literal
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import numpy as np
from nemo.utils import logging


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
        position_ids: [B, Wc + Wt] - position IDs for HF models
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
        bucket_by: Literal['audio', 'text', None] = 'audio',
        drop_last: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.language_mapping = language_mapping
        self.language_drop_rate = language_drop_rate
        self.never_drop_language = set(never_drop_language)
        self.max_duration = max_duration if max_duration is not None else float('inf')
        self.min_duration = min_duration if min_duration is not None else 0
        self.audio_chunk_size = int(audio_chunk_size * sample_rate) if audio_chunk_size is not None else None
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
        # Drop language with probability language_drop_rate
        language = item.get('language', '<|NO_LANGUAGE_ID|>')
        if random.random() < self.language_drop_rate and language not in self.never_drop_language:
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
            Tuple of (context, target, attn_mask, position_ids, target_starts, target_ends, waveforms)
        """
        waveforms = [item['waveform'] for item in batch]
        context_list = [item['context'] for item in batch]
        target_list = [item['target'] for item in batch]
        target_starts = torch.tensor([item['target_start'] for item in batch], dtype=torch.long)
        target_ends = torch.tensor([item['target_end'] for item in batch], dtype=torch.long)
        language_ids = torch.tensor([item['language_id'] for item in batch], dtype=torch.long)
        # Pad to longest in batch or to fixed audio_chunk_size
        waveforms = pad_list_of_tensors(waveforms, pad_value=0, max_length=self.audio_chunk_size)
        context = pad_list_of_tensors(context_list, pad_value=self.tokenizer.pad_id)
        target = pad_list_of_tensors(target_list, pad_value=self.tokenizer.pad_id)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attn_mask = (context != self.tokenizer.pad_id).long()
        attn_mask[:, 0] = 1     # To prevent the first token to be masked
        # attn_mask[:, -1] = 0    # To prevent the last token to be masked (we do not predict the eos token, so we mask out the last token)

        # Create position IDs
        batch_size, seq_len = attn_mask.shape
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).clone()

        return context, target, attn_mask, position_ids, target_starts, target_ends, waveforms, language_ids

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
    bucket_by: Literal['audio', 'text', None] = 'audio',
    drop_last: bool = False,
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
        bucket_by=bucket_by,
        drop_last=drop_last,
    )
