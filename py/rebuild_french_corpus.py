#!/usr/bin/env python3
"""
rebuild_french_corpus.py — Build a cleaned French corpus from Gutenberg texts.

Memory-efficient: processes files one at a time, writes paragraphs to temp file,
then does a chunked shuffle at the end.

Unlike the English corpus, this KEEPS accented characters (é, è, ê, à, ç, œ, etc.)
since they're essential for French.
"""

import os
import random
import re
import sys
import tempfile
import time

# Characters to keep for French (includes accented letters)
ALLOWED_CHARS = set(
    'abcdefghijklmnopqrstuvwxyz'
    'àâäæçéèêëîïôœùûü'  # French accented chars
    ' 0123456789\n'
    '.,;:!?\'"-()'
)

def strip_gutenberg_header_footer(text):
    """Remove Project Gutenberg header and footer."""
    lines = text.split('\n')

    header_patterns = [
        re.compile(r'\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG', re.IGNORECASE),
        re.compile(r'Produced by ', re.IGNORECASE),
    ]
    footer_patterns = [
        re.compile(r'\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG', re.IGNORECASE),
        re.compile(r'End of (?:the )?Project Gutenberg', re.IGNORECASE),
    ]

    header_end = 0
    for i, line in enumerate(lines[:200]):
        for pat in header_patterns:
            if pat.search(line):
                header_end = i + 1
                break

    footer_start = len(lines)
    for i in range(len(lines) - 1, max(len(lines) - 200, 0), -1):
        for pat in footer_patterns:
            if pat.search(lines[i]):
                footer_start = i
                break

    return '\n'.join(lines[header_end:footer_start])


def clean_text_french(text):
    """Clean text while preserving French accented characters."""
    # Lowercase
    text = text.lower()

    # Normalize some Unicode
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # smart quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2014', ' -- ').replace('\u2013', ' -- ')  # dashes
    text = text.replace('\u2026', '...')  # ellipsis
    text = text.replace('\u00a0', ' ')  # non-breaking space
    text = text.replace('\u0153', 'œ')  # ensure œ is normalized

    # Collapse runs of dashes
    text = re.sub(r'(?:\s*-\s*){3,}', ' -- ', text)
    text = re.sub(r'\s*---+\s*', ' -- ', text)

    # Remove noise characters
    text = re.sub(r'[|{}\[\]<>~^@#\\_]', '', text)
    text = text.replace('\t', ' ')

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Strip lines
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    # Keep only allowed characters (including French accents)
    text = ''.join(c for c in text if c in ALLOWED_CHARS)

    # Re-collapse spaces
    text = re.sub(r' {2,}', ' ', text)

    return text


def split_paragraphs(text, min_length=50):
    """Split text into paragraphs, filter short ones."""
    paragraphs = re.split(r'\n\n+', text)
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_length]


def main():
    texts_dir = 'gutenberg_texts'
    keep_file = 'french_keep.txt'
    output_file = 'french_corpus_cleaned_2026_04_15.txt'
    seed = 42

    # Read keep list
    with open(keep_file) as f:
        keep_files = [line.strip() for line in f if line.strip()]
    print(f"French files to process: {len(keep_files)}")

    # Process files, write paragraphs to temp file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    temp_path = temp_file.name

    total_paragraphs = 0
    total_chars = 0
    processed = 0
    start_time = time.time()

    for fn in keep_files:
        path = os.path.join(texts_dir, fn)
        if not os.path.exists(path):
            continue

        try:
            with open(path, encoding='utf-8', errors='replace') as fh:
                raw = fh.read()

            text = strip_gutenberg_header_footer(raw)
            text = clean_text_french(text)
            paragraphs = split_paragraphs(text)

            for p in paragraphs:
                temp_file.write(p + '\n\x00')  # Use null byte as separator
                total_chars += len(p)

            total_paragraphs += len(paragraphs)
            processed += 1

            if processed % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  {processed}/{len(keep_files)} files, "
                      f"{total_paragraphs} paragraphs, "
                      f"{total_chars/1e6:.1f} MB, "
                      f"{elapsed:.0f}s", flush=True)

        except Exception as e:
            print(f"  Error: {fn}: {e}", file=sys.stderr)

    temp_file.close()

    print(f"\nProcessed: {processed} files")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Total characters: {total_chars:,} ({total_chars/1e6:.1f} MB)")

    # Read paragraphs back, shuffle, write output
    print(f"\nReading paragraphs for shuffle...")
    with open(temp_path, 'r') as f:
        content = f.read()
    paragraphs = [p for p in content.split('\x00') if p.strip()]

    print(f"Shuffling {len(paragraphs)} paragraphs with seed {seed}...")
    random.seed(seed)
    random.shuffle(paragraphs)

    print(f"Writing to {output_file}...")
    with open(output_file, 'w') as out:
        for i, p in enumerate(paragraphs):
            out.write(p.strip() + '\n\n')
            if (i + 1) % 100000 == 0:
                print(f"  {i+1}/{len(paragraphs)} written", flush=True)

    # Cleanup
    os.unlink(temp_path)

    final_size = os.path.getsize(output_file)
    print(f"\nDone: {output_file} ({final_size/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
