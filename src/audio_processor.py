"""
Audio processing utilities for merging, crossfading, and exporting.
"""

import io
import zipfile
from typing import List, Optional
import numpy as np
from scipy.io import wavfile


class AudioProcessor:
    """
    Audio processing utilities for TTS output.

    Features:
    - Merge multiple audio chunks
    - Add silence between chunks
    - Apply crossfade for smooth transitions
    - Normalize audio levels
    - Export as WAV or ZIP
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the audio processor.

        Args:
            sample_rate: Sample rate for audio processing (default: 24000 for Qwen3-TTS)
        """
        self.sample_rate = sample_rate

    def create_silence(self, duration_ms: int) -> np.ndarray:
        """
        Create a silence array.

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Numpy array of zeros
        """
        num_samples = int(self.sample_rate * duration_ms / 1000)
        return np.zeros(num_samples, dtype=np.float32)

    def normalize_audio(
        self,
        audio: np.ndarray,
        target_db: float = -3.0
    ) -> np.ndarray:
        """
        Normalize audio to target dB level.

        Args:
            audio: Input audio array
            target_db: Target loudness in dB (default: -3.0)

        Returns:
            Normalized audio
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))

        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            audio = audio * (target_rms / rms)

        # Clip to prevent clipping
        return np.clip(audio, -1.0, 1.0)

    def apply_crossfade(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        crossfade_ms: int = 50
    ) -> np.ndarray:
        """
        Apply crossfade between two audio segments.

        Args:
            audio1: First audio segment
            audio2: Second audio segment
            crossfade_ms: Crossfade duration in milliseconds

        Returns:
            Merged audio with crossfade
        """
        crossfade_samples = int(self.sample_rate * crossfade_ms / 1000)

        # Handle edge cases
        if crossfade_samples <= 0:
            return np.concatenate([audio1, audio2])

        if len(audio1) < crossfade_samples or len(audio2) < crossfade_samples:
            return np.concatenate([audio1, audio2])

        # Create fade curves (linear)
        fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)

        # Apply fades
        audio1_end = audio1[-crossfade_samples:] * fade_out
        audio2_start = audio2[:crossfade_samples] * fade_in

        # Merge: audio1 (without end) + crossfade region + audio2 (without start)
        crossfaded_region = audio1_end + audio2_start

        return np.concatenate([
            audio1[:-crossfade_samples],
            crossfaded_region,
            audio2[crossfade_samples:]
        ])

    def merge_chunks(
        self,
        chunks: List[np.ndarray],
        silence_ms: int = 250,
        crossfade_ms: int = 0,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Merge multiple audio chunks into one.

        Args:
            chunks: List of audio arrays
            silence_ms: Silence between chunks in milliseconds
            crossfade_ms: Crossfade duration (0 = no crossfade)
            normalize: Whether to normalize the final audio

        Returns:
            Merged audio array
        """
        if not chunks:
            return np.array([], dtype=np.float32)

        if len(chunks) == 1:
            result = chunks[0].astype(np.float32)
            return self.normalize_audio(result) if normalize else result

        # Ensure all chunks are float32
        chunks = [c.astype(np.float32) for c in chunks]

        silence = self.create_silence(silence_ms) if silence_ms > 0 else None

        if crossfade_ms > 0:
            # Merge with crossfade
            result = chunks[0]
            for chunk in chunks[1:]:
                # Add silence first, then crossfade
                if silence is not None:
                    result = np.concatenate([result, silence])
                result = self.apply_crossfade(result, chunk, crossfade_ms)
        else:
            # Simple concatenation with silence
            segments = []
            for i, chunk in enumerate(chunks):
                segments.append(chunk)
                if i < len(chunks) - 1 and silence is not None:
                    segments.append(silence)
            result = np.concatenate(segments)

        if normalize:
            result = self.normalize_audio(result)

        return result

    def save_wav(
        self,
        audio: np.ndarray,
        filepath: str
    ) -> str:
        """
        Save audio as WAV file.

        Args:
            audio: Audio array (float32, -1 to 1)
            filepath: Output file path

        Returns:
            Saved file path
        """
        # Convert to int16 for WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(filepath, self.sample_rate, audio_int16)
        return filepath

    def to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """
        Convert audio to WAV bytes.

        Args:
            audio: Audio array

        Returns:
            WAV file as bytes
        """
        audio_int16 = (audio * 32767).astype(np.int16)
        buffer = io.BytesIO()
        wavfile.write(buffer, self.sample_rate, audio_int16)
        buffer.seek(0)
        return buffer.read()

    def create_chunks_zip(
        self,
        chunks: List[np.ndarray],
        prefix: str = "chunk"
    ) -> bytes:
        """
        Create a ZIP archive with individual chunk files.

        Args:
            chunks: List of audio arrays
            prefix: Filename prefix for chunks

        Returns:
            ZIP file as bytes
        """
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, chunk in enumerate(chunks):
                wav_bytes = self.to_wav_bytes(chunk)
                filename = f"{prefix}_{i + 1:03d}.wav"
                zf.writestr(filename, wav_bytes)

        buffer.seek(0)
        return buffer.read()

    def get_duration(self, audio: np.ndarray) -> float:
        """
        Get duration in seconds.

        Args:
            audio: Audio array

        Returns:
            Duration in seconds
        """
        return len(audio) / self.sample_rate

    def format_duration(self, seconds: float) -> str:
        """
        Format duration as MM:SS.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def convert_to_float32(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to float32 format (-1 to 1).

        Args:
            audio: Input audio (any dtype)

        Returns:
            Float32 audio array
        """
        if audio.dtype == np.float32:
            return audio
        elif audio.dtype == np.float64:
            return audio.astype(np.float32)
        elif audio.dtype == np.int16:
            return audio.astype(np.float32) / 32767.0
        elif audio.dtype == np.int32:
            return audio.astype(np.float32) / 2147483647.0
        else:
            # Assume it's already normalized
            return audio.astype(np.float32)
