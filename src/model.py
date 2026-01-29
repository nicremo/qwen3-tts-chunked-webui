"""
Qwen3-TTS Model wrapper with optimizations for RTX 4090.
"""

import gc
import os
import logging
import traceback
from typing import List, Tuple, Optional, Union
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Set environment variables before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,garbage_collection_threshold:0.8")

import torch


class TTSModel:
    """
    Wrapper for Qwen3-TTS with performance optimizations.
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
        compile_model: bool = False,
    ):
        logger.info("TTSModel.__init__ called")
        logger.info(f"  model_id: {model_id or self.MODEL_ID}")
        logger.info(f"  device: {device}")
        logger.info(f"  dtype: {dtype}")
        logger.info(f"  use_flash_attention: {use_flash_attention}")
        logger.info(f"  compile_model: {compile_model}")

        self.model_id = model_id or self.MODEL_ID
        self.device = device
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.compile_model = compile_model

        self.model = None
        self._is_loaded = False

        # Setup CUDA optimizations
        if torch.cuda.is_available():
            logger.info("CUDA is available, setting optimizations...")
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA current device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
        else:
            logger.warning("CUDA is NOT available!")

    def load(self) -> None:
        """Load the model with optimizations."""
        if self._is_loaded:
            logger.info("Model already loaded, skipping")
            return

        logger.info("=" * 40)
        logger.info("TTSModel.load() starting...")
        logger.info("=" * 40)

        logger.info("Importing qwen_tts...")
        try:
            from qwen_tts import Qwen3TTSModel
            logger.info("qwen_tts imported successfully")
        except Exception as e:
            logger.error(f"Failed to import qwen_tts: {e}")
            logger.error(traceback.format_exc())
            raise

        # Determine attention implementation
        attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"

        logger.info(f"Loading model: {self.model_id}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Dtype: {self.dtype}")
        logger.info(f"Attention: {attn_impl}")

        try:
            logger.info("Calling Qwen3TTSModel.from_pretrained...")
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_impl,
            )
            logger.info("Model loaded with flash attention!")
        except Exception as e:
            logger.warning(f"Could not load with flash attention: {e}")
            logger.info("Falling back to eager attention...")
            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    dtype=self.dtype,
                    attn_implementation="eager",
                )
                logger.info("Model loaded with eager attention")
            except Exception as e2:
                logger.error(f"Failed to load model even with eager attention: {e2}")
                logger.error(traceback.format_exc())
                raise

        # Optional: torch.compile
        if self.compile_model and hasattr(torch, 'compile'):
            logger.info("Attempting torch.compile...")
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                logger.info("torch.compile successful")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        self._is_loaded = True
        logger.info("Model loading complete!")

        # Warmup
        self._warmup()

    def _warmup(self) -> None:
        """Warmup inference for optimal performance."""
        logger.info("Warming up model...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        logger.info("Warmup complete!")

    def create_voice_prompt(
        self,
        ref_audio: Union[str, Tuple[np.ndarray, int]],
        ref_text: str
    ) -> dict:
        """Create a reusable voice clone prompt."""
        logger.info("=" * 40)
        logger.info("create_voice_prompt called")
        logger.info("=" * 40)

        if not self._is_loaded:
            logger.info("Model not loaded, loading now...")
            self.load()

        logger.info(f"ref_audio type: {type(ref_audio)}")
        if isinstance(ref_audio, tuple):
            audio_arr, sr = ref_audio
            logger.info(f"ref_audio is tuple: array shape={audio_arr.shape}, sr={sr}")
        else:
            logger.info(f"ref_audio is: {ref_audio}")
        logger.info(f"ref_text: {ref_text[:100] if ref_text else 'EMPTY'}...")

        try:
            logger.info("Calling model.create_voice_clone_prompt...")
            result = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=False
            )
            logger.info(f"Voice prompt created successfully!")
            logger.info(f"Result type: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"create_voice_clone_prompt FAILED: {e}")
            logger.error(traceback.format_exc())
            raise

    def generate_single(
        self,
        text: str,
        language: str,
        voice_prompt: dict,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Generate audio for a single text."""
        logger.info("=" * 40)
        logger.info("generate_single called")
        logger.info("=" * 40)

        if not self._is_loaded:
            logger.info("Model not loaded, loading now...")
            self.load()

        logger.info(f"text: {text[:100]}..." if len(text) > 100 else f"text: {text}")
        logger.info(f"language: {language}")
        logger.info(f"voice_prompt type: {type(voice_prompt)}")
        logger.info(f"additional kwargs: {kwargs}")

        try:
            logger.info("Entering torch.inference_mode()...")
            with torch.inference_mode():
                logger.info("Calling model.generate_voice_clone...")

                if torch.cuda.is_available():
                    logger.info(f"CUDA memory before generation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_prompt,
                    **kwargs
                )

                if torch.cuda.is_available():
                    logger.info(f"CUDA memory after generation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            logger.info(f"generate_voice_clone returned!")
            logger.info(f"wavs type: {type(wavs)}")
            logger.info(f"sr: {sr}")

            # Handle both single and list returns
            if isinstance(wavs, list):
                logger.info(f"wavs is a list with {len(wavs)} elements")
                audio = wavs[0]
            else:
                logger.info("wavs is not a list, using directly")
                audio = wavs

            logger.info(f"Final audio type: {type(audio)}")
            if hasattr(audio, 'shape'):
                logger.info(f"Final audio shape: {audio.shape}")
            if hasattr(audio, 'dtype'):
                logger.info(f"Final audio dtype: {audio.dtype}")

            return audio, sr

        except Exception as e:
            logger.error(f"generate_single FAILED: {e}")
            logger.error(traceback.format_exc())
            raise

    def generate_batch(
        self,
        texts: List[str],
        language: str,
        voice_prompt: dict,
        **kwargs
    ) -> Tuple[List[np.ndarray], int]:
        """Generate audio for multiple texts (batch)."""
        logger.info(f"generate_batch called with {len(texts)} texts")

        if not self._is_loaded:
            self.load()

        languages = [language] * len(texts)

        try:
            with torch.inference_mode():
                wavs, sr = self.model.generate_voice_clone(
                    text=texts,
                    language=languages,
                    voice_clone_prompt=voice_prompt,
                    **kwargs
                )

            if not isinstance(wavs, list):
                wavs = [wavs]

            logger.info(f"Batch generation complete: {len(wavs)} audio files")
            return wavs, sr

        except Exception as e:
            logger.error(f"generate_batch FAILED: {e}")
            logger.error(traceback.format_exc())
            raise

    def cleanup(self) -> None:
        """Clean up GPU memory."""
        logger.info("cleanup() called")
        gc.collect()
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / 1e9
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated() / 1e9
            logger.info(f"CUDA memory: {before:.2f} GB -> {after:.2f} GB")

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.SUPPORTED_LANGUAGES.copy()

    def unload(self) -> None:
        """Unload the model to free memory."""
        logger.info("unload() called")
        if self.model is not None:
            del self.model
            self.model = None
        self._is_loaded = False
        self.cleanup()
