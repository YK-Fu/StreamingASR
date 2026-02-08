import torch
from typing import Optional
from transformers import WhisperFeatureExtractor

class WhisperMelExtractor(WhisperFeatureExtractor):
    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: torch.Tensor, length: torch.Tensor = None, padding_value: float = 0.0
    ) -> torch.Tensor:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        normed_input_values = []
        if length is not None:
            for vector, leng in zip(input_values, length):
                normed_slice = (vector - vector[:leng].mean()) / torch.sqrt(vector[:leng].var() + 1e-7)
                if leng < normed_slice.size(0):
                    normed_slice[leng:] = padding_value

                normed_input_values.append(normed_slice)
            normed_input_values = torch.stack(normed_input_values)
        else:
            normed_input_values = (input_values - input_values.mean(-1, keepdim=True)) / torch.sqrt(input_values.var(-1, keepdim=True) + 1e-7)
        return normed_input_values

    def _torch_extract_fbank_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-mel spectrogram of the audio using PyTorch's GPU-accelerated STFT implementation with batching,
        yielding results similar to cpu computing with 1e-5 tolerance.
        """
        window = torch.hann_window(self.n_fft, device=waveform.device)

        if self.dither != 0.0:
            waveform += self.dither * torch.randn(waveform.shape, dtype=waveform.dtype, device=waveform.device)

        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).to(waveform.device, torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec.detach()

    def pad(self, raw_speech: torch.Tensor, padding: Optional[str] = "longest") -> torch.Tensor:
        if padding == "longest":
            return raw_speech
        elif padding == "max_length":
            pad_size = self.n_samples - raw_speech.size(1)
            raw_speech = torch.nn.functional.pad(raw_speech, (0, pad_size))
            return raw_speech
        else:
            raise ValueError(f"Invalid padding: {padding}")

    def __call__(
        self,
        raw_speech: Optional[torch.Tensor] = None,
        length: Optional[bool] = None,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = True,
        padding: Optional[str] = "longest",
        **kwargs,
    ):
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        if len(raw_speech.size()) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = len(raw_speech.size()) > 1

        # always return batch
        if not is_batched:
            raw_speech = raw_speech.unsqueeze(0)
        raw_speech = self.pad(raw_speech, padding)

        # zero-mean and unit-variance normalization
        if do_normalize:
            input_features = self.zero_mean_unit_var_norm(
                raw_speech,
                length=length,
                padding_value=self.padding_value,
            )
        else:
            input_features = raw_speech

        input_features = self._torch_extract_fbank_features(input_features)

        if length is not None:
            # rescale from sample (48000) to feature (3000)
            length = (length // self.hop_length).detach()
            length.clamp_(min=1, max=input_features.size(2) - 1)
        else:
            length = torch.empty(input_features.size(0), device=input_features.device, dtype=torch.long).fill_(input_features.size(2))

        return input_features, length