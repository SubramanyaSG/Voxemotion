"""
utils/text_utils.py
===================
Text normalization, sentence chunking, and file text extraction.
"""

import re
import os
from config import T2_MAX_CHARS

# ── Text normalization ────────────────────────────────────────────────────────
try:
    import inflect
    from unidecode import unidecode
    _inf = inflect.engine()

    def normalize_text(text: str) -> str:
        """Normalize: unicode → ASCII, numbers → words, remove special chars."""
        text = unidecode(str(text))
        text = re.sub(r'\d+', lambda m: _inf.number_to_words(m.group()), text)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-]", '', text)
        return re.sub(r'\s+', ' ', text).strip()

except ImportError:
    def normalize_text(text: str) -> str:
        """Fallback normalization without inflect/unidecode."""
        text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-]", '', str(text))
        return re.sub(r'\s+', ' ', text).strip()


# ── Sentence chunker ──────────────────────────────────────────────────────────

def split_into_chunks(text: str, max_chars: int = T2_MAX_CHARS) -> list[str]:
    """
    Split text into sentence-level chunks each under max_chars.
    Prevents Tacotron2 from hitting its 1000-step decoder limit.

    Strategy:
      1. Split on sentence-ending punctuation (.!?)
      2. Merge short sentences greedily up to max_chars
      3. Split very long sentences on commas
      4. Hard-split as last resort
    """
    raw    = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    buf    = ''

    for sent in raw:
        sent = sent.strip()
        if not sent:
            continue
        if len(buf) + len(sent) + 1 <= max_chars:
            buf = (buf + ' ' + sent).strip() if buf else sent
        else:
            if buf:
                chunks.append(buf)
            if len(sent) > max_chars:
                # Try splitting on commas
                parts  = re.split(r'(?<=,)\s+', sent)
                sub    = ''
                for p in parts:
                    if len(sub) + len(p) + 1 <= max_chars:
                        sub = (sub + ' ' + p).strip() if sub else p
                    else:
                        if sub:
                            chunks.append(sub)
                        # Hard split as absolute last resort
                        while len(p) > max_chars:
                            chunks.append(p[:max_chars])
                            p = p[max_chars:]
                        sub = p
                if sub:
                    chunks.append(sub)
            else:
                buf = sent

    if buf:
        chunks.append(buf)

    return [c for c in chunks if c.strip()]


# ── File text extraction ──────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    try:
        from pdfminer.high_level import extract_text
        return extract_text(path)
    except Exception:
        try:
            import PyPDF2
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return '\n'.join(page.extract_text() or '' for page in reader.pages)
        except Exception as e:
            return f'[PDF extraction error: {e}]'


def extract_text_from_docx(path: str) -> str:
    """Extract text from a Word .docx file."""
    try:
        from docx import Document
        doc = Document(path)
        return '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        return f'[DOCX extraction error: {e}]'


def extract_text_from_txt(path: str) -> str:
    """Read plain text file."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f'[TXT read error: {e}]'


def extract_text_from_file(path: str) -> str:
    """Auto-detect file type and extract text."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(path)
    elif ext == '.docx':
        return extract_text_from_docx(path)
    elif ext == '.txt':
        return extract_text_from_txt(path)
    else:
        return f'[Unsupported file type: {ext}]'
