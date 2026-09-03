# Streaming Hybrid Transformer Transducer CTC ASR

A streaming automatic speech recognition (ASR) system built on NVIDIA NeMo, combining:
- **Whisper encoder** for robust audio feature extraction
- **Qwen LLM decoder** for language modeling
- **RNN-T (Transducer)** for streaming inference
- **CTC auxiliary loss** for improved training stability and flexible decoding strategy
- **ALiBi positional embeddings** for flexible context length

## Features

- **Streaming-capable**: Uses causal attention with ALiBi for real-time transcription
- **Knowledge distillation**: Distill from pretrained Whisper teacher to streaming student
- **Hybrid loss**: Combines RNN-T, CTC, and LM losses for robust training
- **Flexible architecture**: Supports various Whisper and Qwen model sizes


## Environment Setup

### Prerequisites

1. Enroll in [NGC](https://ngc.nvidia.com/) and get an API key
2. Log into NGC
```
echo $NGC_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

### Docker Setup

This project is developed against the **NeMo 26.02** framework container
(`nvcr.io/nvidia/nemo:26.02`), which ships:

| Component | Version |
|-----------|---------|
| PyTorch | `2.10.0a0` (NVIDIA PyTorch 25.11 base) |
| CUDA | 13.0 |
| Python | 3.12 |

> The container's torch is a **custom NGC build**. PyPI `torchaudio` / `torchcodec`
> wheels link against stock PyTorch's C++ ABI and will **not** load against it, so
> those two must be **built from source** against the installed torch (the scripts
> below do this). `bitsandbytes` (the 8-bit optimizer) is installed via pip.

```bash
# Pull and run the NeMo 26.02 container
docker run -it -u=0 --gpus=all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $WORKSPACE:$WORKSPACE \
    nvcr.io/nvidia/nemo:26.02 bash

# Inside the container
cd $WORKSPACE
git clone https://github.com/YK-Fu/StreamingASR
cd StreamingASR

# (one-time) system audio backends needed to build/run torchaudio + torchcodec
apt update && apt install -y ffmpeg sox libavdevice-dev

# One-shot install of everything: k2, torchaudio + torchcodec (built from source
# against the container's torch 2.10), and bitsandbytes (8-bit AdamW optimizer).
bash install.sh

# Or install only selected components, e.g.:
#   bash install.sh k2 bitsandbytes
```

## Data Preparation

### Manifest Format

Prepare your data in JSONL format:

```json
{"audio_filepath": "/path/to/audio1.wav", "duration": 3.35, "text": "hello world", "context": "previous sentence"}
{"audio_filepath": "/path/to/audio2.wav", "duration": 5.64, "text": "how are you"}
```

| Field | Required | Description |
|-------|----------|-------------|
| `audio_filepath` | Yes | Path to audio file |
| `text` | Yes | Transcription |
| `duration` | No | Duration in seconds (loaded from audio if not provided) |
| `context` | No | Previous transcription for context-aware ASR |

### Configuration

Update dataset paths in the config file:

```yaml
model:
  train_ds:
    manifest_filepath:
      - /path/to/train1.json
      - /path/to/train2.json
  validation_ds:
    manifest_filepath:
      - /path/to/val.json
```

## Training Pipeline

### Stage 1: Knowledge Distillation

Distill a pretrained Whisper model into a streaming-capable student encoder.

#### 1.1 Convert HuggingFace Checkpoint

```bash
cd $WORKSPACE/StreamingASR/

python -m ckpt_conversion.convert_hf_to_nemo \
    --whisper openai/whisper-large-v2 \
    --config conf/hybrid_distil_ctc.yaml \
    --output ckpt_conversion/distil.nemo
```

#### 1.2 (Optional) Verify Conversion

```bash
python -m ckpt_conversion.verify_checkpoint \
    --checkpoint ckpt_conversion/distil.nemo \
    --config conf/hybrid_distil_ctc.yaml \
    --whisper openai/whisper-large-v2
```

#### 1.3 Run Distillation Training

```bash
cd $WORKSPACE/StreamingASR/

python distiller_train.py \
    init_from_nemo_model=ckpt_conversion/distil.nemo \
    trainer.max_epochs=10
```

### Stage 2: RNN-T Training

Train the full streaming ASR model with RNN-T loss.

#### 2.1 (Optional) Prune Tokenizer Vocabulary

To reduce VRAM usage, you can prune the Qwen tokenizer vocabulary. See [Multilingual-Qwen-Tokenizer-Pruner](https://github.com/your-repo/Multilingual-Qwen-Tokenizer-Pruner) for details.

#### 2.2 Convert Distilled Model to RNN-T Format

```bash
cd $WORKSPACE/StreamingASR/

python -m ckpt_conversion.convert_distill_to_rnnt \
    --distill-checkpoint /path/to/trained_distil.nemo \
    --qwen Qwen/Qwen2.5-0.5B \
    --config conf/hybrid_transducer_ctc.yaml \
    --output ckpt_conversion/rnnt_model.nemo
```

#### 2.3 Run RNN-T Training

```bash
cd $WORKSPACE/StreamingASR/

python asr_train.py \
    init_from_nemo_model=ckpt_conversion/rnnt_model.nemo \
    trainer.max_epochs=50
```

## Configuration Files

| Config | Description |
|--------|-------------|
| `conf/hybrid_distil_ctc.yaml` | Distillation training (Whisper teacher → ALiBi student) |
| `conf/hybrid_transducer_ctc.yaml` | RNN-T training with Qwen decoder |


## Project Structure

```
StreamingASR/
├── conf/
│   ├── hybrid_distil_ctc.yaml      # Distillation config
│   └── hybrid_transducer_ctc.yaml  # RNN-T training config
├── ckpt_conversion/
│   ├── convert_hf_to_nemo.py       # HF Whisper/Qwen → NeMo
│   ├── convert_distill_to_rnnt.py  # Distill → RNN-T format
│   └── verify_checkpoint.py        # Verify conversion
├── src/
│   ├── models/
│   │   ├── rnnt_model.py           # Hybrid RNN-T CTC model
│   │   └── causal_distill.py       # Distillation model
│   ├── modules/
│   │   ├── transformer_encoder.py  # Whisper encoder with ALiBi
│   │   ├── transformer_decoder.py  # Qwen decoder wrapper
│   │   ├── transformer_layer.py    # Transformer layer with ALiBi
│   │   └── projection.py           # Projection layers
│   ├── datasets.py                 # Dataset classes
│   └── extractor.py                # Mel feature extractor
├── asr_train.py                    # RNN-T training script
├── distiller_train.py              # Distillation training script
└── install.sh                      # one-shot installer: k2, torchaudio, torchcodec, bitsandbytes
```

## References

- [Whisper](https://github.com/openai/whisper) - OpenAI's robust speech recognition
- [Tokenizer Pruner](https://github.com/KaihuaTang/Qwen-Tokenizer-Pruner/) - Qwen tokenizer pruning reference
- [NeMo](https://github.com/NVIDIA/NeMo) - NVIDIA's conversational AI toolkit

## License

Apache License 2.0
