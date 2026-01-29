# 🎙️ Qwen3-TTS Voice Cloning WebUI

A Gradio-based WebUI for Qwen3-TTS with **automatic text chunking** for generating long audio files.

**Optimized for Runpod with RTX 4090** (24GB VRAM)

## ✨ Features

- **Voice Cloning** - Clone any voice with just 3-30 seconds of reference audio
- **Automatic Chunking** - Generate unlimited length audio with intelligent text splitting
- **Multi-Language** - English, German, Chinese, Japanese, Korean, French, Spanish, Italian, Portuguese, Russian
- **GPU Optimized** - Flash Attention 2, bf16 precision, memory management
- **Flexible Output** - Merged WAV or ZIP with individual chunks
- **Progress Tracking** - Real-time progress bar during generation

## 🚀 Quick Start

### Option 1: Docker (Recommended for Runpod)

```bash
# Pull and run
docker pull ghcr.io/nicremo/qwen3-tts-chunked-webui:latest
docker run --gpus all -p 7860:7860 \
    -v ~/.cache/huggingface:/runpod-volume/huggingface \
    ghcr.io/nicremo/qwen3-tts-chunked-webui
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/Nicremo/qwen3-tts-chunked-webui.git
cd qwen3-tts-chunked-webui

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention 2 for faster inference
pip install flash-attn --no-build-isolation

# Run
python app.py
```

Open http://localhost:7860 in your browser.

## 📦 Runpod Deployment

### 1. Create a Pod

- **GPU**: RTX 4090 (24GB) recommended
- **Template**: PyTorch 2.2+ with CUDA 12.1
- **Expose Port**: 7860 (HTTP)

### 2. In the Pod Terminal

```bash
cd /workspace
git clone https://github.com/Nicremo/qwen3-tts-chunked-webui.git
cd qwen3-tts-chunked-webui
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python app.py
```

### 3. Access

Use the Runpod proxy URL for port 7860, or add `--share` for a Gradio public link:

```bash
GRADIO_SHARE=true python app.py
```

## 🎯 Usage

1. **Enter Text** - Paste any length of text you want to convert to speech
2. **Upload Reference** - Upload 3-60 seconds of clear audio in your target voice
3. **Enter Transcript** - Type the exact words spoken in your reference audio
4. **Adjust Settings** (optional):
   - **Chunk Size**: 1000-1500 words works best
   - **Silence**: Pause between chunks (250ms default)
   - **Crossfade**: Smooth transitions (50ms default)
5. **Generate** - Click the button and wait for processing

## ⚙️ Settings Explained

| Setting | Description | Recommended |
|---------|-------------|-------------|
| Chunk Size | Words per chunk | 1000-1500 |
| Silence | Pause between chunks (ms) | 200-300 |
| Crossfade | Smooth transitions (ms) | 50-100 |
| Output Format | Merged WAV or ZIP | Merged WAV |

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GRADIO_SERVER_PORT` | Server port | 7860 |
| `GRADIO_SHARE` | Create public link | false |
| `USE_FLASH_ATTENTION` | Enable Flash Attention 2 | true |
| `HF_HOME` | HuggingFace cache directory | /runpod-volume/huggingface |

## 📊 Technical Details

- **Model**: Qwen3-TTS-12Hz-1.7B-Base
- **Sample Rate**: 24kHz
- **VRAM Usage**: ~6-8 GB
- **Precision**: bf16
- **Attention**: Flash Attention 2 (fallback to eager)

## 🐛 Troubleshooting

### "Flash Attention not available"
This is normal if flash-attn installation failed. The app will use eager attention (slower but works).

### "CUDA out of memory"
Reduce chunk size or restart the pod to clear VRAM.

### "502 Bad Gateway" on Runpod
Use `GRADIO_SHARE=true` to get a direct Gradio link that bypasses the proxy.

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team
- [pySBD](https://github.com/nipunsadvilkar/pySBD) for sentence boundary detection
- [Gradio](https://gradio.app/) for the WebUI framework
