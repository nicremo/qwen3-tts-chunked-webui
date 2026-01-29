"""
Utility functions for the Qwen3-TTS WebUI.
"""

import re
from typing import Optional


def detect_language(text: str) -> str:
    """
    Detect language based on character set.

    Args:
        text: Input text

    Returns:
        Language name (English, German, Chinese, etc.)
    """
    # Chinese characters
    if re.search(r'[\u4e00-\u9fff]', text):
        return "Chinese"

    # Japanese (Hiragana, Katakana)
    if re.search(r'[\u3040-\u30ff]', text):
        return "Japanese"

    # Korean (Hangul)
    if re.search(r'[\uac00-\ud7af\u1100-\u11ff]', text):
        return "Korean"

    # Russian (Cyrillic)
    if re.search(r'[\u0400-\u04ff]', text):
        return "Russian"

    # German indicators (common umlauts and ß)
    german_indicators = ['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü']
    german_count = sum(1 for char in text if char in german_indicators)
    if german_count >= 2:
        return "German"

    # French indicators
    french_indicators = ['é', 'è', 'ê', 'ë', 'à', 'â', 'ù', 'û', 'ç', 'œ']
    french_count = sum(1 for char in text if char.lower() in french_indicators)
    if french_count >= 2:
        return "French"

    # Spanish indicators
    spanish_indicators = ['ñ', '¿', '¡', 'á', 'é', 'í', 'ó', 'ú']
    spanish_count = sum(1 for char in text if char.lower() in spanish_indicators)
    if spanish_count >= 2:
        return "Spanish"

    # Italian indicators (similar to Spanish but without ñ)
    if re.search(r'\b(che|gli|sono|questo|quella)\b', text.lower()):
        return "Italian"

    # Portuguese indicators
    if re.search(r'[ãõ]', text):
        return "Portuguese"

    # Default to English
    return "English"


def get_language_code(language: str) -> str:
    """
    Convert language name to code for chunker.

    Args:
        language: Language name

    Returns:
        Language code (en, de, etc.)
    """
    mapping = {
        "auto": "en",
        "english": "en",
        "german": "de",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "french": "fr",
        "spanish": "es",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
    }
    return mapping.get(language.lower(), "en")


def validate_reference_audio(
    audio_data: Optional[tuple],
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 60.0
) -> tuple[bool, str]:
    """
    Validate reference audio.

    Args:
        audio_data: Tuple of (sample_rate, numpy_array) from Gradio
        min_duration_sec: Minimum duration in seconds
        max_duration_sec: Maximum duration in seconds

    Returns:
        (is_valid, message)
    """
    if audio_data is None:
        return False, "No audio provided"

    try:
        sr, audio = audio_data
        duration = len(audio) / sr

        if duration < min_duration_sec:
            return False, f"Audio too short: {duration:.1f}s (minimum: {min_duration_sec}s)"

        if duration > max_duration_sec:
            return False, f"Audio too long: {duration:.1f}s (maximum: {max_duration_sec}s)"

        return True, f"Audio valid: {duration:.1f}s"

    except Exception as e:
        return False, f"Invalid audio format: {str(e)}"


def estimate_generation_time(
    num_chunks: int,
    avg_words_per_chunk: int = 1000
) -> str:
    """
    Estimate generation time (rough approximation).

    Args:
        num_chunks: Number of chunks
        avg_words_per_chunk: Average words per chunk

    Returns:
        Estimated time string
    """
    # Rough estimate: ~10 seconds per chunk on RTX 4090
    seconds_per_chunk = 10
    total_seconds = num_chunks * seconds_per_chunk

    if total_seconds < 60:
        return f"~{total_seconds} seconds"
    else:
        minutes = total_seconds / 60
        return f"~{minutes:.1f} minutes"


def truncate_text(text: str, max_chars: int = 100) -> str:
    """
    Truncate text for preview.

    Args:
        text: Input text
        max_chars: Maximum characters

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
