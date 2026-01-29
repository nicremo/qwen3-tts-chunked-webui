"""
Qwen3-TTS Model wrapper with optimizations for RTX 4090.
"""

import gc
import os
from typing import List, Tuple, Optional, Union
import numpy as np

# Set environment variables before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,garbage_collection_threshold:0.8")

import torch


class TTSModel:
    """
    Wrapper for Qwen3-TTS with performance optimizations.

    Features:
    - Flash Attention 2 support
    - bf16/fp16 precision
    - torch.compile optimization
    - Voice prompt caching for efficient batch generation
    - Automatic memory cleanup
    """

    MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    SAMPLE_RATE = 24000

    SUPPORTED_LANGUAGES = [
        "Auto",
        "English",
        "German",
        "Chinese",
        "Japanese",
        "Korean",
        "French",
        "Spanish",
        "Italian",
        "Portuguese",
        "Russian",
    ]

    def __init__(
        self,
        model_id: str = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
        compile_model: bool = False,  # Disabled by default, can cause issues
    ):
        """
        Initialize the TTS model wrapper.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to load model on (cuda:0, cpu, auto)
            dtype: Model precision (torch.bfloat16, torch.float16)
            use_flash_attention: Enable Flash Attention 2
            compile_model: Enable torch.compile (experimental)
        """
        self.model_id = model_id or self.MODEL_ID
        self.device = device
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.compile_model = compile_model

        self.model = None
        self._is_loaded = False

        # Setup CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def load(self) -> None:
        """Load the model with optimizations."""
        if self._is_loaded:
            return

        from qwen_tts import Qwen3TTSModel

        # Determine attention implementation
        attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"

        print(f"Loading model: {self.model_id}")
        print(f"Device: {self.device}, Dtype: {self.dtype}, Attention: {attn_impl}")

        try:
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            # Fallback without flash attention
            print(f"Warning: Could not load with flash attention: {e}")
            print("Falling back to eager attention...")
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation="eager",
            )

        # Optional: torch.compile for faster inference
        if self.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")

        self._is_loaded = True
        print("Model loaded successfully!")

        # Warmup
        self._warmup()

    def _warmup(self) -> None:
        """Warmup inference for optimal performance."""
        print("Warming up model...")
        # Note: Actual warmup would require reference audio
        # This is just to ensure CUDA is initialized
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warmup complete!")

    def create_voice_prompt(
        self,
        ref_audio: Union[str, Tuple[np.ndarray, int]],
        ref_text: str
    ) -> dict:
        """
        Create a reusable voice clone prompt.

        This extracts speaker features once and can be reused for
        multiple generations, improving efficiency.

        Args:
            ref_audio: Path to reference audio OR (numpy_array, sample_rate)
            ref_text: Transcript of the reference audio

        Returns:
            Voice prompt dictionary for generate_voice_clone
        """
        if not self._is_loaded:
            self.load()

        return self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False
        )

    def generate_single(
        self,
        text: str,
        language: str,
        voice_prompt: dict,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio for a single text.

        Args:
            text: Text to synthesize
            language: Language (English, German, etc.)
            voice_prompt: Voice prompt from create_voice_prompt()
            **kwargs: Additional generation parameters

        Returns:
            (audio_array, sample_rate)
        """
        if not self._is_loaded:
            self.load()

        with torch.inference_mode():
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_prompt,
                **kwargs
            )

        # Handle both single and list returns
        audio = wavs[0] if isinstance(wavs, list) else wavs

        return audio, sr

    def generate_batch(
        self,
        texts: List[str],
        language: str,
        voice_prompt: dict,
        **kwargs
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate audio for multiple texts (batch).

        Args:
            texts: List of texts to synthesize
            language: Language (applied to all texts)
            voice_prompt: Voice prompt from create_voice_prompt()
            **kwargs: Additional generation parameters

        Returns:
            (list_of_audio_arrays, sample_rate)
        """
        if not self._is_loaded:
            self.load()

        languages = [language] * len(texts)

        with torch.inference_mode():
            wavs, sr = self.model.generate_voice_clone(
                text=texts,
                language=languages,
                voice_clone_prompt=voice_prompt,
                **kwargs
            )

        # Ensure we have a list
        if not isinstance(wavs, list):
            wavs = [wavs]

        return wavs, sr

    def cleanup(self) -> None:
        """Clean up GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.SUPPORTED_LANGUAGES.copy()

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._is_loaded = False
        self.cleanup()
