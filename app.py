"""
Qwen3-TTS Voice Cloning WebUI with Automatic Text Chunking

Optimized for Runpod with RTX 4090 (24GB VRAM)
"""

import os
import sys
import tempfile
import traceback
import logging
from datetime import datetime
from typing import Optional, Tuple

import gradio as gr
import numpy as np

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("=" * 60)
logger.info("STARTING Qwen3-TTS Voice Cloning WebUI")
logger.info("=" * 60)

from src.model import TTSModel
from src.chunker import TextChunker
from src.audio_processor import AudioProcessor
from src.utils import detect_language, get_language_code, validate_reference_audio

logger.info("All imports successful")

# Global instances
model: Optional[TTSModel] = None
audio_processor = AudioProcessor(sample_rate=24000)


def load_model():
    """Load the TTS model on startup."""
    global model

    logger.info("=" * 40)
    logger.info("LOADING MODEL")
    logger.info("=" * 40)

    # Check for Flash Attention
    use_flash = os.environ.get("USE_FLASH_ATTENTION", "true").lower() == "true"
    logger.info(f"USE_FLASH_ATTENTION: {use_flash}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda"
    logger.info(f"Selected device: {device}")

    model = TTSModel(
        device=device,
        use_flash_attention=use_flash,
        compile_model=False,
    )

    logger.info("Calling model.load()...")
    model.load()
    logger.info("Model loading complete!")


def preview_chunks(
    text: str,
    chunk_size: int,
    language: str
) -> str:
    """Generate chunk preview."""
    logger.info(f"preview_chunks called: text_len={len(text)}, chunk_size={chunk_size}, language={language}")

    if not text.strip():
        logger.warning("Empty text provided")
        return "⚠️ Please enter text to preview chunks."

    actual_lang = language if language != "Auto" else detect_language(text)
    logger.info(f"Detected/selected language: {actual_lang}")

    chunker = TextChunker(
        language=actual_lang,
        target_words=chunk_size
    )

    result = chunker.preview(text)
    logger.info(f"Preview generated successfully")
    return result


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
    Main TTS generation function with detailed logging.
    """
    global model

    logger.info("=" * 60)
    logger.info("GENERATE_TTS CALLED")
    logger.info("=" * 60)

    # Log all inputs
    logger.info(f"Input text length: {len(text) if text else 0} chars")
    logger.info(f"Input text preview: {text[:100] if text else 'EMPTY'}...")
    logger.info(f"Reference audio provided: {ref_audio is not None}")
    if ref_audio is not None:
        sr, audio_data = ref_audio
        logger.info(f"  - Sample rate: {sr}")
        logger.info(f"  - Audio shape: {audio_data.shape}")
        logger.info(f"  - Audio dtype: {audio_data.dtype}")
        logger.info(f"  - Audio duration: {len(audio_data)/sr:.2f}s")
    logger.info(f"Reference text length: {len(ref_text) if ref_text else 0} chars")
    logger.info(f"Reference text: {ref_text[:100] if ref_text else 'EMPTY'}...")
    logger.info(f"Language: {language}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Silence ms: {silence_ms}")
    logger.info(f"Crossfade ms: {crossfade_ms}")
    logger.info(f"Output format: {output_format}")

    # Validation
    logger.info("--- VALIDATION PHASE ---")

    if not text.strip():
        logger.error("VALIDATION FAILED: Empty text")
        return None, None, "❌ Please enter text to generate."

    if ref_audio is None:
        logger.error("VALIDATION FAILED: No reference audio")
        return None, None, "❌ Please upload reference audio (3-60 seconds)."

    if not ref_text.strip():
        logger.error("VALIDATION FAILED: No reference text")
        return None, None, "❌ Please enter the transcript of your reference audio."

    is_valid, msg = validate_reference_audio(ref_audio)
    logger.info(f"Reference audio validation: valid={is_valid}, msg={msg}")
    if not is_valid:
        logger.error(f"VALIDATION FAILED: Reference audio - {msg}")
        return None, None, f"❌ Reference audio error: {msg}"

    logger.info("All validations passed!")

    try:
        # Initialize
        logger.info("--- INITIALIZATION PHASE ---")
        progress(0, desc="Initializing...")

        # Determine language
        actual_language = language if language != "Auto" else detect_language(text)
        logger.info(f"Final language: {actual_language}")

        # Chunking
        logger.info("--- CHUNKING PHASE ---")
        progress(0.05, desc="Analyzing text...")

        chunker = TextChunker(
            language=actual_language,
            target_words=chunk_size
        )
        logger.info(f"TextChunker created with language={actual_language}, target_words={chunk_size}")

        chunk_result = chunker.chunk_text(text)
        chunks = chunk_result.chunks

        logger.info(f"Chunking complete: {len(chunks)} chunks, {chunk_result.total_words} total words")
        for i, chunk in enumerate(chunks):
            logger.info(f"  Chunk {i+1}: {chunk.word_count} words, {chunk.char_count} chars")
            logger.debug(f"  Chunk {i+1} text: {chunk.text[:50]}...")

        if not chunks:
            logger.error("No valid text chunks found!")
            return None, None, "❌ No valid text chunks found."

        # Prepare reference audio
        logger.info("--- REFERENCE AUDIO PROCESSING ---")
        progress(0.1, desc="Creating voice prompt...")

        sr, audio_data = ref_audio
        logger.info(f"Original audio: sr={sr}, shape={audio_data.shape}, dtype={audio_data.dtype}")

        # Convert to float32
        logger.info("Converting to float32...")
        audio_data = audio_processor.convert_to_float32(audio_data)
        logger.info(f"After conversion: shape={audio_data.shape}, dtype={audio_data.dtype}")

        # Handle stereo -> mono
        if len(audio_data.shape) > 1:
            logger.info(f"Converting stereo to mono (shape was {audio_data.shape})")
            audio_data = audio_data.mean(axis=1)
            logger.info(f"After mono conversion: shape={audio_data.shape}")

        # Create voice prompt
        logger.info("--- CREATING VOICE PROMPT ---")
        logger.info(f"Calling model.create_voice_prompt with:")
        logger.info(f"  - ref_audio shape: {audio_data.shape}")
        logger.info(f"  - ref_audio sr: {sr}")
        logger.info(f"  - ref_text: {ref_text[:50]}...")

        try:
            voice_prompt = model.create_voice_prompt(
                ref_audio=(audio_data, sr),
                ref_text=ref_text
            )
            logger.info(f"Voice prompt created successfully!")
            logger.info(f"Voice prompt type: {type(voice_prompt)}")
            if isinstance(voice_prompt, dict):
                logger.info(f"Voice prompt keys: {voice_prompt.keys()}")
        except Exception as e:
            logger.error(f"FAILED to create voice prompt: {e}")
            logger.error(traceback.format_exc())
            raise

        # Generate audio for each chunk
        logger.info("--- GENERATION PHASE ---")
        generated_chunks = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            logger.info(f"--- Generating chunk {i + 1}/{total_chunks} ---")
            logger.info(f"Chunk text ({chunk.word_count} words): {chunk.text[:100]}...")

            progress_val = 0.1 + (0.8 * (i / total_chunks))
            progress(progress_val, desc=f"Generating chunk {i + 1}/{total_chunks}...")

            try:
                logger.info(f"Calling model.generate_single...")
                logger.info(f"  - text length: {len(chunk.text)}")
                logger.info(f"  - language: {actual_language}")

                audio, returned_sr = model.generate_single(
                    text=chunk.text,
                    language=actual_language,
                    voice_prompt=voice_prompt
                )

                logger.info(f"Generation successful!")
                logger.info(f"  - Returned audio shape: {audio.shape if hasattr(audio, 'shape') else 'N/A'}")
                logger.info(f"  - Returned audio type: {type(audio)}")
                logger.info(f"  - Returned sample rate: {returned_sr}")

                generated_chunks.append(audio)
                logger.info(f"Chunk {i + 1} added to generated_chunks (total: {len(generated_chunks)})")

            except Exception as e:
                logger.error(f"FAILED to generate chunk {i + 1}: {e}")
                logger.error(traceback.format_exc())
                raise

            # Memory cleanup every 5 chunks
            if (i + 1) % 5 == 0:
                logger.info("Running memory cleanup...")
                model.cleanup()
                logger.info("Memory cleanup complete")

        logger.info(f"All {total_chunks} chunks generated successfully!")

        # Merge
        logger.info("--- MERGING PHASE ---")
        progress(0.9, desc="Merging audio...")

        logger.info(f"Merging {len(generated_chunks)} chunks")
        logger.info(f"  - silence_ms: {silence_ms}")
        logger.info(f"  - crossfade_ms: {crossfade_ms}")

        try:
            merged_audio = audio_processor.merge_chunks(
                chunks=generated_chunks,
                silence_ms=silence_ms,
                crossfade_ms=crossfade_ms,
                normalize=True
            )
            logger.info(f"Merge successful! Merged audio shape: {merged_audio.shape}")
        except Exception as e:
            logger.error(f"FAILED to merge chunks: {e}")
            logger.error(traceback.format_exc())
            raise

        # Calculate duration
        duration = audio_processor.get_duration(merged_audio)
        duration_str = audio_processor.format_duration(duration)
        logger.info(f"Total duration: {duration:.2f}s ({duration_str})")

        # Save outputs
        logger.info("--- SAVING PHASE ---")
        merged_path = None
        zip_path = None

        if output_format in ["Merged WAV", "Both"]:
            logger.info("Saving merged WAV...")
            merged_path = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ).name
            audio_processor.save_wav(merged_audio, merged_path)
            logger.info(f"Merged WAV saved to: {merged_path}")

        if output_format in ["ZIP with Chunks", "Both"]:
            logger.info("Creating ZIP with chunks...")
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
            logger.info(f"ZIP saved to: {zip_path}")

        # Final cleanup
        logger.info("Final memory cleanup...")
        model.cleanup()

        progress(1.0, desc="Done!")

        # Status message
        status = f"✅ Successfully generated!\n"
        status += f"📊 {total_chunks} chunks | {chunk_result.total_words} words\n"
        status += f"⏱️ Duration: {duration_str}\n"
        status += f"🎤 Language: {actual_language}"

        logger.info("=" * 60)
        logger.info("GENERATION COMPLETE - SUCCESS!")
        logger.info(status.replace('\n', ' | '))
        logger.info("=" * 60)

        return merged_path, zip_path, status

    except Exception as e:
        logger.error("=" * 60)
        logger.error("GENERATION FAILED - EXCEPTION!")
        logger.error("=" * 60)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())

        if model:
            logger.info("Running cleanup after error...")
            model.cleanup()

        return None, None, f"❌ Error: {str(e)}\n\nCheck terminal for full traceback."


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    logger.info("Creating Gradio interface...")

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

    logger.info("Gradio interface created successfully")
    return demo


# Main entry point
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MAIN ENTRY POINT")
    logger.info("=" * 60)

    print("=" * 50)
    print("Qwen3-TTS Voice Cloning WebUI")
    print("Optimized for Runpod with RTX 4090")
    print("=" * 50)

    print("\nLoading model...")
    load_model()

    print("\nStarting Gradio interface...")
    demo = create_interface()

    # Queue for handling multiple requests
    logger.info("Setting up queue with max_size=10, concurrency_limit=2")
    demo.queue(
        max_size=10,
        default_concurrency_limit=2
    )

    # Launch
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"

    logger.info(f"Launching server on 0.0.0.0:{server_port}, share={share}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        show_error=True,
        share=share
    )
