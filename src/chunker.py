"""
Intelligent text chunking for TTS with sentence boundary detection.
"""

from dataclasses import dataclass
from typing import List
import re

try:
    import pysbd
    HAS_PYSBD = True
except ImportError:
    HAS_PYSBD = False


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    index: int
    text: str
    word_count: int
    char_count: int


@dataclass
class ChunkingResult:
    """Result of the chunking process."""
    chunks: List[TextChunk]
    total_words: int
    total_chars: int

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


class TextChunker:
    """
    Intelligent text chunking with sentence boundary detection.

    Uses pySBD for robust sentence segmentation that handles:
    - Abbreviations (Mr., Dr., etc.)
    - Numbers and decimals
    - URLs and emails
    - Multiple punctuation marks
    """

    # Language code mapping
    LANGUAGE_MAP = {
        "auto": "en",
        "english": "en",
        "german": "de",
        "chinese": "zh",
        "japanese": "ja",
        "french": "fr",
        "spanish": "es",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
        "korean": "ko",
    }

    # Fallback regex for sentence splitting
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence end
        r'(?<=[.!?])\s*\n+|'        # Sentence end + newline
        r'\n{2,}'                    # Paragraph break
    )

    # Patterns for text cleaning
    URL_PATTERN = re.compile(r'https?://\S+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    WHITESPACE_PATTERN = re.compile(r'\s+')

    def __init__(
        self,
        language: str = "english",
        target_words: int = 1200,
        min_chunk_words: int = 50
    ):
        """
        Initialize the text chunker.

        Args:
            language: Language for sentence detection (english, german, etc.)
            target_words: Target word count per chunk (500-2000 recommended)
            min_chunk_words: Minimum words for a valid chunk
        """
        self.language = language.lower()
        self.target_words = max(100, min(target_words, 3000))
        self.min_chunk_words = min_chunk_words

        # Get language code for pySBD
        self.lang_code = self.LANGUAGE_MAP.get(self.language, "en")

        # Initialize pySBD segmenter if available
        self.segmenter = None
        if HAS_PYSBD:
            try:
                self.segmenter = pysbd.Segmenter(
                    language=self.lang_code,
                    clean=False
                )
            except Exception:
                # Fallback to English if language not supported
                self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.

        Args:
            text: Raw input text

        Returns:
            Cleaned text
        """
        # Normalize whitespace
        text = self.WHITESPACE_PATTERN.sub(' ', text.strip())

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Normalize dashes
        text = text.replace('–', '-').replace('—', '-')

        return text

    def count_words(self, text: str) -> int:
        """
        Count words in text (language-aware).

        Args:
            text: Input text

        Returns:
            Word count
        """
        if self.lang_code == "zh":
            # Chinese: count characters as "words"
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            # Also count non-Chinese words
            other_words = len(re.findall(r'[a-zA-Z]+', text))
            return chinese_chars + other_words
        elif self.lang_code == "ja":
            # Japanese: similar to Chinese
            jp_chars = len(re.findall(r'[\u3040-\u30ff\u4e00-\u9fff]', text))
            other_words = len(re.findall(r'[a-zA-Z]+', text))
            return jp_chars + other_words
        else:
            # Western languages: split by whitespace
            return len(text.split())

    def segment_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        if self.segmenter is not None:
            try:
                sentences = self.segmenter.segment(text)
                return [s.strip() for s in sentences if s.strip()]
            except Exception:
                pass

        # Fallback: regex-based splitting
        sentences = self.SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str) -> ChunkingResult:
        """
        Split text into optimal chunks respecting sentence boundaries.

        Args:
            text: Input text to chunk

        Returns:
            ChunkingResult with all chunks and statistics
        """
        # Clean text
        text = self.clean_text(text)

        if not text:
            return ChunkingResult(chunks=[], total_words=0, total_chars=0)

        # Segment into sentences
        sentences = self.segment_sentences(text)

        if not sentences:
            return ChunkingResult(chunks=[], total_words=0, total_chars=0)

        # Group sentences into chunks
        chunks: List[TextChunk] = []
        current_sentences: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = self.count_words(sentence)

            # Check if adding this sentence would exceed target
            if current_word_count + sentence_words > self.target_words and current_sentences:
                # Save current chunk
                chunk_text = ' '.join(current_sentences)
                chunks.append(TextChunk(
                    index=len(chunks),
                    text=chunk_text,
                    word_count=current_word_count,
                    char_count=len(chunk_text)
                ))
                current_sentences = []
                current_word_count = 0

            # Add sentence to current chunk
            current_sentences.append(sentence)
            current_word_count += sentence_words

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunks.append(TextChunk(
                index=len(chunks),
                text=chunk_text,
                word_count=current_word_count,
                char_count=len(chunk_text)
            ))

        # Filter out very small chunks (merge with previous if possible)
        if len(chunks) > 1:
            filtered_chunks = []
            for i, chunk in enumerate(chunks):
                if chunk.word_count >= self.min_chunk_words:
                    filtered_chunks.append(chunk)
                elif filtered_chunks:
                    # Merge with previous chunk
                    prev = filtered_chunks[-1]
                    merged_text = prev.text + ' ' + chunk.text
                    filtered_chunks[-1] = TextChunk(
                        index=prev.index,
                        text=merged_text,
                        word_count=prev.word_count + chunk.word_count,
                        char_count=len(merged_text)
                    )
                else:
                    filtered_chunks.append(chunk)

            # Re-index chunks
            for i, chunk in enumerate(filtered_chunks):
                chunk.index = i
            chunks = filtered_chunks

        return ChunkingResult(
            chunks=chunks,
            total_words=sum(c.word_count for c in chunks),
            total_chars=sum(c.char_count for c in chunks)
        )

    def preview(self, text: str, max_preview_chars: int = 100) -> str:
        """
        Generate a preview string for the UI.

        Args:
            text: Input text
            max_preview_chars: Max characters to show per chunk

        Returns:
            Formatted preview string
        """
        result = self.chunk_text(text)

        lines = [
            f"📊 {result.num_chunks} Chunks | {result.total_words} Words | {result.total_chars} Characters\n"
        ]

        for chunk in result.chunks:
            preview_text = chunk.text[:max_preview_chars]
            if len(chunk.text) > max_preview_chars:
                preview_text += "..."
            lines.append(
                f"[Chunk {chunk.index + 1}] ({chunk.word_count} words): {preview_text}"
            )

        return "\n".join(lines)
