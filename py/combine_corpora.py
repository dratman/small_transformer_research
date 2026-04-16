#!/usr/bin/env python3
"""
combine_corpora.py — Combine English and French corpora with chunked shuffling.

Memory-efficient: reads and shuffles in chunks, never loads entire corpus.
"""

import os
import random
import sys
import time

def count_paragraphs_streaming(filepath):
    """Count paragraphs without loading file into memory."""
    count = 0
    in_paragraph = False
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                if not in_paragraph:
                    count += 1
                    in_paragraph = True
            else:
                in_paragraph = False
    return count

def paragraph_generator(filepath):
    """Yield paragraphs one at a time."""
    current = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                current.append(line.rstrip())
            elif current:
                yield '\n'.join(current)
                current = []
        if current:
            yield '\n'.join(current)

def main():
    english_path = '../small_transformer_research/txt_local/corpus_cleaned_2026_04_15.txt'
    french_path = 'french_corpus_cleaned_2026_04_15.txt'
    output_path = 'corpus_bilingual_2026_04_15.txt'

    chunk_size = 50000  # Process 50K paragraphs at a time
    seed = 42
    random.seed(seed)

    print("Counting paragraphs (streaming)...")
    t0 = time.time()

    # We already know approximate counts, but let's verify
    en_count = count_paragraphs_streaming(english_path)
    fr_count = count_paragraphs_streaming(french_path)

    print(f"  English: {en_count:,} paragraphs")
    print(f"  French: {fr_count:,} paragraphs")
    print(f"  Total: {en_count + fr_count:,} paragraphs")
    print(f"  French ratio: {100*fr_count/(en_count+fr_count):.1f}%")
    print(f"  (counted in {time.time()-t0:.1f}s)")

    # Generate shuffled indices
    print(f"\nGenerating shuffle order...")
    total = en_count + fr_count
    indices = list(range(total))
    random.shuffle(indices)

    # Create index lookup: for each shuffled position, which source and offset?
    # Indices 0 to en_count-1 are English, en_count to total-1 are French
    print("Creating index mapping...")

    # Instead of random access (slow), we'll do chunked interleave
    # Read chunks from both, shuffle chunk, write

    print(f"\nProcessing in chunks of {chunk_size}...")

    en_gen = paragraph_generator(english_path)
    fr_gen = paragraph_generator(french_path)

    # Ratio for interleaving: ~7 English per 1 French
    en_per_chunk = int(chunk_size * en_count / total)
    fr_per_chunk = chunk_size - en_per_chunk

    print(f"  Per chunk: ~{en_per_chunk} English, ~{fr_per_chunk} French")

    written = 0
    start_time = time.time()

    with open(output_path, 'w') as out:
        en_done = False
        fr_done = False

        while not (en_done and fr_done):
            chunk = []

            # Read English paragraphs
            if not en_done:
                for _ in range(en_per_chunk):
                    try:
                        chunk.append(next(en_gen))
                    except StopIteration:
                        en_done = True
                        break

            # Read French paragraphs
            if not fr_done:
                for _ in range(fr_per_chunk):
                    try:
                        chunk.append(next(fr_gen))
                    except StopIteration:
                        fr_done = True
                        break

            if not chunk:
                break

            # Shuffle this chunk
            random.shuffle(chunk)

            # Write chunk
            for p in chunk:
                out.write(p + '\n\n')
                written += 1

            elapsed = time.time() - start_time
            print(f"  Written: {written:,} paragraphs ({elapsed:.0f}s)", flush=True)

    final_size = os.path.getsize(output_path)
    print(f"\nDone: {output_path}")
    print(f"  Size: {final_size/1e9:.2f} GB")
    print(f"  Paragraphs: {written:,}")

if __name__ == '__main__':
    main()
