"""
Qwen3-TTS Voice Cloning WebUI with Automatic Text Chunking

Optimized for Runpod with RTX 4090 (24GB VRAM)
"""

import os
import tempfile
from typing import Optional, Tuple

import gradio as gr
import numpy as np

from src.model import TTSModel
from src.chunker import TextChunker
from src.audio_processor import AudioProcessor
from src.utils import detect_language, get_language_code, validate_reference_audio

# Global instances
model: Optional[TTSModel] = None
audio_processor = AudioProcessor(sample_rate=24000)


def load_model():
    """Load the TTS model on startup."""
    global model

    # Check for Flash Attention
    use_flash = os.environ.get("USE_FLASH_ATTENTION", "true").lower() == "true"

    model = TTSModel(
        device="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda",
        use_flash_attention=use_flash,
        compile_model=False,  # Disabled for stability
    )
    model.load()


def preview_chunks(
    text: str,
    chunk_size: int,
    language: str
) -> str:
    """Generate chunk preview."""
    if not text.strip():
        return "⚠️ Please enter text to preview chunks."

    # Determine language
    actual_lang = language if language != "Auto" else detect_language(text)
    lang_code = get_language_code(actual_lang)

    chunker = TextChunker(
        language=actual_lang,
        target_words=chunk_size
    )

    return chunker.preview(text)


def generate_tts(
    text: str,
    ref_audio: Optional[Tuple[int, np.ndarray]],
    ref_text: str,
    language: str,
    chunk_size: int,
    silence_ms: int,
    crossfade_ms: int,
    output_format: str,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Main TTS generation function.

    Returns:
        (merged_audio_path, zip_path, status_message)
    """
    global model

    # Validation
    if not text.strip():
        return None, None, "❌ Please enter text to generate."

    if ref_audio is None:
        return None, None, "❌ Please upload reference audio (3-60 seconds)."

    if not ref_text.strip():
        return None, None, "❌ Please enter the transcript of your reference audio."

    # Validate reference audio
    is_valid, msg = validate_reference_audio(ref_audio)
    if not is_valid:
        return None, None, f"❌ Reference audio error: {msg}"

    try:
        progress(0, desc="Initializing...")

        # Determine language
        actual_language = language if language != "Auto" else detect_language(text)

        # Chunking
        progress(0.05, desc="Analyzing text...")
        lang_code = get_language_code(actual_language)

        chunker = TextChunker(
            language=actual_language,
            target_words=chunk_size
        )
        chunk_result = chunker.chunk_text(text)
        chunks = chunk_result.chunks

        if not chunks:
            return None, None, "❌ No valid text chunks found."

        # Prepare reference audio
        progress(0.1, desc="Creating voice prompt...")
        sr, audio_data = ref_audio

        # Convert to float32
        audio_data = audio_processor.convert_to_float32(audio_data)

        # Handle stereo -> mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Create voice prompt (once, reuse for all chunks)
        voice_prompt = model.create_voice_prompt(
            ref_audio=(audio_data, sr),
            ref_text=ref_text
        )

        # Generate audio for each chunk
        generated_chunks = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            progress_val = 0.1 + (0.8 * (i / total_chunks))
            progress(progress_val, desc=f"Generating chunk {i + 1}/{total_chunks}...")

            # Generate single chunk
            audio, _ = model.generate_single(
                text=chunk.text,
                language=actual_language,
                voice_prompt=voice_prompt
            )

            generated_chunks.append(audio)

            # Memory cleanup every 5 chunks
            if (i + 1) % 5 == 0:
                model.cleanup()

        progress(0.9, desc="Merging audio...")

        # Merge chunks
        merged_audio = audio_processor.merge_chunks(
            chunks=generated_chunks,
            silence_ms=silence_ms,
            crossfade_ms=crossfade_ms,
            normalize=True
        )

        # Calculate duration
        duration = audio_processor.get_duration(merged_audio)
        duration_str = audio_processor.format_duration(duration)

        # Save outputs
        merged_path = None
        zip_path = None

        if output_format in ["Merged WAV", "Both"]:
            merged_path = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ).name
            audio_processor.save_wav(merged_audio, merged_path)

        if output_format in ["ZIP with Chunks", "Both"]:
            zip_bytes = audio_processor.create_chunks_zip(
                generated_chunks,
                prefix="chunk"
            )
            zip_path = tempfile.NamedTemporaryFile(
                suffix=".zip",
                delete=False
            ).name
            with open(zip_path, 'wb') as f:
                f.write(zip_bytes)

        # Final cleanup
        model.cleanup()

        progress(1.0, desc="Done!")

        # Status message
        status = f"✅ Successfully generated!\n"
        status += f"📊 {total_chunks} chunks | {chunk_result.total_words} words\n"
        status += f"⏱️ Duration: {duration_str}\n"
        status += f"🎤 Language: {actual_language}"

        return merged_path, zip_path, status

    except Exception as e:
        if model:
            model.cleanup()
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Qwen3-TTS Voice Cloning",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("""
        # 🎙️ Qwen3-TTS Voice Cloning WebUI

        Clone any voice and generate long texts with automatic chunking.
        Optimized for Runpod with RTX 4090.

        **How to use:**
        1. Enter your text (any length)
        2. Upload 3-60 seconds of reference audio
        3. Enter the exact transcript of your reference
        4. Click Generate!
        """)

        with gr.Row():
            # Left column: Inputs
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text to Generate",
                    placeholder="Enter the text you want to convert to speech...\n\nYou can enter very long texts - they will be automatically split into chunks.",
                    lines=12,
                    max_lines=50
                )

                with gr.Row():
                    ref_audio = gr.Audio(
                        label="Reference Audio (3-60 seconds)",
                        type="numpy",
                        sources=["upload", "microphone"]
                    )

                    ref_text = gr.Textbox(
                        label="Reference Transcript",
                        placeholder="Enter the EXACT text spoken in the reference audio...",
                        lines=4
                    )

            # Right column: Settings
            with gr.Column(scale=1):
                language = gr.Dropdown(
                    choices=["Auto", "English", "German", "Chinese", "Japanese", "Korean", "French", "Spanish", "Italian", "Portuguese", "Russian"],
                    value="Auto",
                    label="Language"
                )

                chunk_size = gr.Slider(
                    minimum=500,
                    maximum=2000,
                    value=1200,
                    step=100,
                    label="Chunk Size (words)",
                    info="Larger = fewer chunks, but may affect quality"
                )

                silence_ms = gr.Slider(
                    minimum=0,
                    maximum=1000,
                    value=250,
                    step=50,
                    label="Pause between Chunks (ms)"
                )

                crossfade_ms = gr.Slider(
                    minimum=0,
                    maximum=200,
                    value=50,
                    step=10,
                    label="Crossfade (ms)",
                    info="Smooth transitions between chunks"
                )

                output_format = gr.Radio(
                    choices=["Merged WAV", "ZIP with Chunks", "Both"],
                    value="Merged WAV",
                    label="Output Format"
                )

        # Preview section
        with gr.Accordion("📋 Chunk Preview", open=False):
            preview_btn = gr.Button("🔍 Preview Chunks", variant="secondary")
            chunk_preview = gr.Textbox(
                label="",
                lines=10,
                interactive=False
            )

        # Generate button
        generate_btn = gr.Button(
            "🎤 Generate Audio",
            variant="primary",
            size="lg"
        )

        # Outputs
        with gr.Row():
            with gr.Column():
                output_audio = gr.Audio(
                    label="Generated Audio",
                    type="filepath"
                )
            with gr.Column():
                output_zip = gr.File(
                    label="ZIP with Individual Chunks"
                )

        status_output = gr.Textbox(
            label="Status",
            interactive=False,
            lines=4
        )

        # Footer
        gr.Markdown("""
        ---
        **Tips:**
        - Reference audio should be clear, without background noise
        - 10-30 seconds of reference usually works best
        - For best results, use audio in the same language as your text
        """)

        # Event handlers
        preview_btn.click(
            fn=preview_chunks,
            inputs=[text_input, chunk_size, language],
            outputs=[chunk_preview]
        )

        generate_btn.click(
            fn=generate_tts,
            inputs=[
                text_input, ref_audio, ref_text, language,
                chunk_size, silence_ms, crossfade_ms, output_format
            ],
            outputs=[output_audio, output_zip, status_output]
        )

    return demo


# Main entry point
if __name__ == "__main__":
    print("=" * 50)
    print("Qwen3-TTS Voice Cloning WebUI")
    print("Optimized for Runpod with RTX 4090")
    print("=" * 50)

    print("\nLoading model...")
    load_model()

    print("\nStarting Gradio interface...")
    demo = create_interface()

    # Queue for handling multiple requests
    demo.queue(
        max_size=10,
        default_concurrency_limit=2
    )

    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        show_error=True,
        share=os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    )
